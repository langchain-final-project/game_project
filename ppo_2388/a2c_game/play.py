"""학습한 PPO 정책을 원래 Pygame 게임 맵으로 보여 주는 관람 화면입니다.

학습·평가와 같은 UnifiedMansionEnv.step으로 정책 행동을 실행합니다.
그 안에서 발생한 실제 물리 프레임만 차례로 표시하며, 경로·중간 위치·시간을
별도로 만들어 내거나 정책을 추가 학습하지 않습니다.
"""
from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .config import load_config
from .environment import UnifiedMansionEnv
from .evaluation import isolated_rng
from .display_units import (format_steps, settings_groups, score_progress,
                            training_score_summary, load_training_monitor)

CANVAS_SIZE = (1600, 960)
GAME_SIZE = (1280, 720)


def read_json(path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {} if fallback is None else fallback


def checkpoint_entries(run):
    """실제로 저장된 초기·중간·최신 모델만 관람 목록으로 만듭니다."""
    run = Path(run)
    result = []
    if (run / 'initial_model.zip').is_file():
        result.append({'label': '학습 전 초기 모델', 'kind': 'initial',
                       'path': run / 'initial_model.zip'})
    for path in sorted((run / 'checkpoints').glob('*.zip')):
        result.append({'label': path.stem, 'kind': 'checkpoint', 'path': path})
    meta = read_json(run / 'metadata.json')
    status = read_json(run / 'status.json')
    current = meta.get('resume_checkpoint') or status.get('latest_checkpoint')
    if current and (run / current).is_file():
        result.append({'label': '현재 학습 모델', 'kind': 'latest', 'path': run / current})
    return result


def select_checkpoint(run, selection='latest'):
    entries = checkpoint_entries(run)
    if selection in ('initial', 'latest'):
        choices = [entry for entry in entries if entry['kind'] == selection]
        if not choices:
            raise ValueError(f'{selection} 체크포인트를 찾을 수 없습니다: {run}')
        return choices[-1]
    path = Path(selection)
    if not path.is_absolute():
        path = Path(run) / path
    if not path.is_file():
        raise ValueError(f'모델 파일을 찾을 수 없습니다: {path}')
    return {'label': path.stem, 'kind': 'checkpoint', 'path': path}


def load_policy(path):
    import torch
    from stable_baselines3 import PPO

    torch.set_num_threads(1)
    with isolated_rng(None):
        model = PPO.load(path, device='cpu')
    model.policy.set_training_mode(False)
    return model


class PhysicalFrameCapture:
    """게임 코어의 동작을 그대로 실행하면서 실제 물리 프레임을 모읍니다."""
    def __init__(self, core):
        self.core = core
        self.frames = deque()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def step(self, action):
        result = self.core.step(action)
        self.frames.append(deepcopy(result.snapshot))
        return result

    def step_frame(self, action):
        # 관람기는 모든 물리 프레임을 나중에 표시하므로 빌린 중간 snapshot을
        # 전달하지 않습니다. 공개 복사 경로로 실행해 프레임별 소유권을 유지합니다.
        return self.step(action)


class NativePolicyEpisode:
    """원래 출발점부터 실제 종료까지 한 판을 추론만으로 실행합니다."""
    def __init__(self, model, config, *, stage=1, seed=0, stochastic=False):
        self.model = model
        self.env = UnifiedMansionEnv(config, stages=[stage], seed=seed, training=False)
        if (model.observation_space != self.env.observation_space
                or model.action_space != self.env.action_space):
            self.env.close()
            raise ValueError('모델의 관측·행동 공간이 현재 게임과 다릅니다.')
        self.observation, self.info = self.env.reset(seed=seed, options={'stage': stage})
        self.initial_snapshot = deepcopy(self.info['snapshot'])
        self.capture = PhysicalFrameCapture(self.env.core)
        self.env.core = self.capture
        self.stochastic = bool(stochastic)
        self.done = False
        self.actions = []
        self.rendered_physical_frames = 0

    def decision(self):
        if self.done:
            raise RuntimeError('이 회차는 원래 게임에서 종료됐습니다.')
        action, _ = self.model.predict(self.observation, deterministic=not self.stochastic)
        action = int(np.asarray(action).item())
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        if truncated:
            raise RuntimeError('원래 게임 종료 전에 회차가 잘렸습니다.')
        self.done = bool(terminated)
        self.actions.append(action)
        frames = list(self.capture.frames)
        self.capture.frames.clear()
        if len(frames) != self.info['executed_frames']:
            raise RuntimeError('실제 물리 프레임 수와 표시 프레임 수가 다릅니다.')
        self.rendered_physical_frames += len(frames)
        return action, frames

    def result(self):
        if not self.done:
            return None
        episode = self.info['episode']
        return {'stage': episode['stage'], 'score': episode['score'],
                'native_outcome': deepcopy(episode['native_outcome']),
                'policy_steps': episode['policy_steps'],
                'physical_frames': episode['physical_frames'],
                'displayed_physical_frames': self.rendered_physical_frames,
                'actions': list(self.actions), 'model_steps': int(self.model.num_timesteps),
                'model_updates': int(self.model._n_updates), 'training_updates_added': 0}

    def close(self):
        self.env.close()


def _history(run, stage, mode):
    path = Path(run) / 'evaluation/history.jsonl'
    rows = []
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if row.get('stage') == stage and row.get('mode') == mode:
                    rows.append(row)
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return sorted(rows, key=lambda row: row['steps'])


def comparable_history(history):
    """평가 시드·행동 방식·판수가 같은 기록만 학습 곡선으로 연결합니다."""
    valid = [row for row in history if type(row.get('mean_score')) in (int, float)
             and math.isfinite(row['mean_score']) and 0 <= row['mean_score'] <= 100]
    if not valid:
        return []
    valid.sort(key=lambda row: row.get('steps', 0))
    latest = valid[-1]
    fields = ('stage', 'mode', 'seed', 'episodes', 'randomize_phase')
    return [row for row in valid if all(row.get(key) == latest.get(key) for key in fields)]


def _draw_panel(surface, assets, *, run, entry, model, snapshot, action,
                paused=False, stochastic=False, follow=False, history=(), metadata=None,
                training_summary=None):
    import pygame
    from kkamang.ui.theme import text, wrapped, panel, CREAM, GOLD, SAGE, MUTED

    pygame.draw.rect(surface, (31, 34, 46), (1280, 0, 320, 720))
    text(surface, assets, 'PPO 실제 게임 플레이', (1299, 20), 22, CREAM, True)
    text(surface, assets, f'Stage {snapshot["stage"]} · {snapshot["name"]}',
         (1299, 57), 17, GOLD, True)
    text(surface, assets, entry['label'], (1299, 90), 17, SAGE, True)
    text(surface, assets, f'선택 모델: {format_steps(model.num_timesteps)}', (1299, 119), 16, CREAM)
    text(surface, assets, f'옵티마이저 업데이트 {model._n_updates:,}회', (1299, 145), 15, CREAM)
    latest = (metadata or {}).get('steps')
    text(surface, assets, f'실행 전체: {format_steps(latest)}' if latest is not None else '학습 기록 대기',
         (1299, 170), 15, CREAM)
    text(surface, assets, '확률 행동' if stochastic else '최대확률 행동',
         (1299, 199), 15, CREAM)
    action_label = ('위', '아래', '왼쪽', '오른쪽', '대기')[action] if action is not None else '시작'
    text(surface, assets, f'이번 행동: {action_label}',
         (1299, 224), 15, CREAM)
    outcome = snapshot.get('outcome') or {}
    score = outcome.get('score', {}).get('total')
    text(surface, assets, f'실제 종료 점수: {score if score is not None else "아직 진행 중"}',
         (1299, 254), 18, GOLD, True)
    if outcome:
        text(surface, assets, f'{outcome.get("status")} · {outcome.get("reason")}',
             (1299, 282), 14, CREAM)
    progress = score_progress(history)
    previous, latest_mean = progress.get('previous_mean'), progress.get('latest_mean')
    show = lambda value: '—' if value is None else f'{value:.1f}'
    text(surface, assets, f'평가 {format_steps(progress.get("steps"))}: {show(previous)} → {show(latest_mean)}점',
         (1299, 317), 14, CREAM, True)
    delta, best = progress.get('delta'), progress.get('best_mean')
    delta_label = '—' if delta is None else f'{delta:+.1f}'
    text(surface, assets, f'변화 {delta_label}점 · 최고 평가 평균 {show(best)}점',
         (1299, 343), 14, SAGE)
    summary = training_summary or {}
    count = summary.get('episodes', 0)
    text(surface, assets,
         f'최근 학습 {count}판 평균 {show(summary.get("mean_score"))}점'
         if count else '최근 학습 기록 없음', (1299, 369), 14, CREAM)
    if count:
        quest = summary.get('quest_rate')
        quest_label = '—' if quest is None else f'{quest:.0%}'
        text(surface, assets,
             f'보조 진전: 퀘스트 {quest_label} · 온기 {show(summary.get("mean_warmth"))}',
             (1299, 390), 12, MUTED)
    text(surface, assets, '실제 평가 점수 변화 · 85점 미만도 표시',
         (1299, 413), 13, CREAM, True)
    graph = pygame.Rect(1304, 442, 276, 89)
    pygame.draw.rect(surface, (40, 44, 58), graph, border_radius=7)
    for value in (0, 50, 100):
        y = graph.bottom - 12 - value * (graph.height - 24) / 100
        pygame.draw.line(surface, (69, 74, 92), (graph.left + 25, y), (graph.right - 9, y))
    points = []
    graph_history = comparable_history(history)
    if graph_history:
        maximum = max(1, max(row['steps'] for row in graph_history))
        for row in graph_history:
            points.append((graph.left + 28 + row['steps'] / maximum * (graph.width - 40),
                           graph.bottom - 12 - row['mean_score'] / 100 * (graph.height - 24)))
    if len(points) > 1:
        pygame.draw.lines(surface, SAGE, False, points, 2)
    for point in points:
        pygame.draw.circle(surface, SAGE, point, 3)
    if graph_history:
        marker = graph.left + 28 + min(1., model.num_timesteps / maximum) * (graph.width - 40)
        pygame.draw.line(surface, GOLD, (marker, graph.top + 8), (marker, graph.bottom - 7), 2)
    text(surface, assets, '전체 평가 기록 · 금색선: 관람 모델', (1300, 534), 12, MUTED)
    wrapped(surface, assets,
            'L 공유 선정 모델\n← / → 저장 시점 비교\nTab 다음 맵 · Shift+Tab 이전 맵\nF1 설정 · F2 학습 현황\nSpace 일시정지 · R 다시 보기',
            (1299, 565, 286, 106), 14, CREAM, 6)
    text(surface, assets, ('일시정지' if paused else '자동 플레이')
         + (' · 새 체크포인트 따라가기' if follow else ''), (1299, 683), 13, SAGE)


def _normalise_groups(groups):
    """표시용 설정 묶음의 형식을 맞춥니다. 게임 설정은 변경하지 않습니다."""
    normalised = []
    for group in groups:
        if isinstance(group, dict):
            title = str(group.get('title', group.get('name', '설정')))
            rows = group.get('rows', group.get('items', []))
        else:
            title, rows = group
        if isinstance(rows, dict):
            rows = list(rows.items())
        normalised.append({'title': title, 'rows': [(str(label), str(value)) for label, value in rows]})
    return normalised


def _draw_settings(surface, assets, groups, *, full=False):
    import pygame
    from kkamang.ui.theme import text, fit_text, CREAM, GOLD, SAGE

    groups = _normalise_groups(groups)
    top = 0 if full else 720
    height = CANVAS_SIZE[1] - top
    pygame.draw.rect(surface, (27, 30, 42), (0, top, CANVAS_SIZE[0], height))
    title = ('전체 학습·환경 설정 — F1을 누르면 실제 게임으로 돌아갑니다' if full
             else '학습·환경 설정  |  F1: 모든 설정·이전 단위와 차원 비교')
    text(surface, assets, title, (20, top + 12), 23 if full else 16, CREAM, True)
    y_start = top + (61 if full else 43)
    row_height = 25 if full else 17
    rows_per_column = max(1, (CANVAS_SIZE[1] - y_start - 17) // row_height)
    total, shown = 0, 0
    for column, group in enumerate(groups[:4]):
        lines = [(group['title'], '', True)] + [(label, value, False) for label, value in group['rows']]
        total += len(lines)
        for row, (label, value, heading) in enumerate(lines[:rows_per_column]):
            shown += 1
            x, y = 20 + column * 400, y_start + row * row_height
            if heading:
                text(surface, assets, label, (x, y), 17 if full else 14, GOLD, True)
            else:
                fit_text(surface, assets, f'{label}: {value}', (x, y, 371, row_height),
                         16 if full else 13, CREAM, anchor='midleft')
    if total > shown:
        text(surface, assets, f'표시 {shown}/{total}항목 · F1에서 전체 설정 확인',
             (1170, top + 12), 13, SAGE)


def _draw_monitor(surface, assets, monitor, *, snapshot=None, level=None, selected_steps=None):
    """전체 Stage 학습 현황을 읽어 표시합니다. 관람 정책의 행동에는 관여하지 않습니다."""
    import pygame
    from kkamang.ui.theme import text, fit_text, CREAM, GOLD, SAGE, MUTED

    surface.fill((27, 30, 42))
    text(surface, assets, '전체 학습 현황 · Stage1~12', (25, 18), 27, CREAM, True)
    text(surface, assets, 'F2 실제 맵으로 · F1 모든 설정 · G 최대확률 / S 확률 행동'
         f' · 관람 모델 {format_steps(selected_steps)}',
         (25, 58), 16, MUTED)
    count = lambda value: '—' if value is None else f'{value:,}'
    number = lambda value, digits=1: '—' if value is None else f'{value:.{digits}f}'
    percent = lambda value: '—' if value is None else f'{value:.0%}'
    progress, cohort = monitor.get('progress', {}), monitor.get('evaluation', {})
    entropy = monitor.get('entropy_coef')
    text(surface, assets,
         f'학습 {format_steps(monitor.get("steps"))} / {format_steps(monitor.get("total_steps"))}'
         f'    업데이트 {count(monitor.get("updates"))}    상태 {monitor.get("status", "—")}',
         (25, 96), 20, GOLD, True)
    text(surface, assets,
         f'실제 엔트로피 계수 {number(entropy, 6)} · 정책 엔트로피 {number(progress.get("entropy"), 4)}'
         f' · 처리량 {number(progress.get("fps"), 0)}스텝/초', (25, 132), 17, CREAM)
    pending = ' · 다음 평가 진행 중, 완료된 평가 유지' if cohort.get('pending') else ''
    label = '12맵 G/S 평가 완료' if cohort.get('complete') else '평가 기록 미완료'
    text(surface, assets, f'{label} · 평가 모델 {format_steps(cohort.get("steps"))}{pending}',
         (25, 166), 16, SAGE)
    columns = [(25, 65, 'Stage'), (95, 180, '실제 학습량'), (285, 105, '학습 평균'),
        (400, 105, 'G 점수'), (515, 95, 'G 85점'), (620, 105, 'S 점수'),
        (735, 95, 'S 귀환'), (840, 95, 'S 85점'), (945, 95, '퀘스트'),
        (1050, 65, '온기'), (1125, 330, '실패 사유 / 판수'), (1465, 110, 'G/S/학습 n')]
    for x, width, title in columns:
        fit_text(surface, assets, title, (x, 209, width, 30), 16, GOLD, anchor='midleft')
    reasons_ko = {'timeout': '시간', 'courage': '용기'}
    for index, row in enumerate(monitor.get('stages', [])):
        y = 247 + index * 29
        if index % 2 == 0:
            pygame.draw.rect(surface, (35, 39, 52), (18, y, 1564, 29))
        greedy, stochastic = row.get('greedy') or {}, row.get('stochastic') or {}
        failures = ' · '.join(f'{reasons_ko.get(key, key)} {value}'
                              for key, value in row.get('failures', {}).items())
        values = [str(row['stage']), format_steps(row.get('learner_steps')) if row.get('trained') else '미학습 / 기록 대기',
            number((row.get('training') or {}).get('mean_score')),
            number(greedy.get('mean_score')), percent(greedy.get('star3_rate')),
            number(stochastic.get('mean_score')), percent(stochastic.get('return_rate')),
            percent(stochastic.get('star3_rate')), percent(row.get('quest_rate')),
            number(row.get('mean_warmth')), failures or ('실패 없음' if greedy or stochastic else '—'),
            f'{greedy.get("episodes", "—")}/{stochastic.get("episodes", "—")}/{(row.get("training") or {}).get("episodes", "—")}']
        for (x, width, _), value in zip(columns, values):
            fit_text(surface, assets, value, (x, y, width, 29), 16, CREAM, anchor='midleft')
    text(surface, assets, '점수는 실패0점도 포함한 실제 종료 평균입니다. 예상 점수를 성공률에 넣지 않습니다.',
         (25, 611), 15, MUTED)
    guard = monitor.get('guard', {})
    layout = monitor.get('layout', {})
    pools = layout.get('environment_stage_pools', layout.get('layout', layout.get('env_pools', layout.get('pools'))))
    if pools is None and isinstance(layout.get('environments'), list):
        pools = layout['environments']
    lines = [
        ('현재 손실', f'정책 {number(progress.get("policy_loss"), 6)} / 가치 {number(progress.get("value_loss"), 6)}'),
        ('방문 보상 현재 계수', number(progress.get('novelty_coefficient'), 6)),
        ('맵 일정 / 보상',
            (f'집중 Stage{layout.get("focus_stage", guard.get("focus_stage", "—"))}' if guard else '기본 맵 일정')
            + ' / ' + {'native_base': '기본 보상', 'allocation_only': '기본 보상 유지',
                       'return_guidance': '귀환 보조 대조',
                       'loop_guard': '수집 후 이동 순환 비용',
                       'return_loop_guard': '귀환 거리4배 · 이동 순환 비용',
                       'return_progress_guard': '귀환 거리4배 · 장기 무진전 비용'}.get(monitor.get('reward_profile'), str(monitor.get('reward_profile', '기본 보상')))),
        ('4환경 맵 배분', str(pools) if pools is not None else '기록 대기'),
        ('정체 검토 상태', str(guard.get('status', '기본 학습 · 집중 실험 미사용'))),
    ]
    if snapshot is not None and level is not None:
        from .refinement import return_diagnostics
        diagnostic = return_diagnostics(snapshot, level)
        estimate = (f'{diagnostic["estimated_return_score"]}점'
                    if snapshot['quest']['collected'] and snapshot['carried'] >= snapshot['target']
                    else '귀환 조건 미충족')
        lines.extend([
            ('현재 관람 회차', f'Stage{snapshot["stage"]} / 퀘스트 {"수집" if snapshot["quest"]["collected"] else "탐색"} / 온기 {snapshot["carried"]}'),
            ('귀환 진전 / 예상', f'길 거리 {number(diagnostic["exit_distance"])} / 귀환 준비 {"예" if diagnostic["return_ready"] else "아니오"} / 예상 {estimate}'),
        ])
    for index, (label, value) in enumerate(lines):
        fit_text(surface, assets, f'{label}: {value}', (25, 657 + index * 33, 758, 31),
                 17, CREAM, anchor='midleft')
    text(surface, assets, '실제 변경 이력', (820, 651), 20, GOLD, True)
    changes = []
    for row in monitor.get('entropy_changes', []):
        changes.append((row.get('steps', 0), f'엔트로피 {number(row.get("before"), 5)} → {number(row.get("after"), 5)}'))
    for row in monitor.get('adjustment_history', []):
        changes.append((row.get('steps', 0), f'{row.get("kind", "조정")}: {row.get("before", "—")} → {row.get("after", "—")} · {row.get("reason", "")}'))
    changes.sort(key=lambda item: item[0] or 0)
    for index, (steps, change) in enumerate(changes[-5:]):
        fit_text(surface, assets, f'{format_steps(steps)} · {change}',
                 (820, 690 + index * 33, 750, 31), 16, CREAM, anchor='midleft')
    if not changes:
        text(surface, assets, '기록된 변경 없음', (820, 693), 16, MUTED)
    text(surface, assets, '읽기 전용 관람 · 이 화면은 학습을 추가하거나 게임 규칙을 바꾸지 않습니다.',
         (25, 920), 16, MUTED)


def play(run, *, stage=1, seed=0, model=None, stochastic=False, follow=False,
         headless=False, max_episodes=None, screenshot=None, speed=1., max_frames=None,
         settings=False, monitor=False):
    """실제 게임 관람 창을 실행합니다. 이 함수는 정책을 학습하지 않습니다."""
    if type(stage) is not int or not 1 <= stage <= 12:
        raise ValueError('stage는1..12 정수여야 합니다.')
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('seed 범위를 확인하세요.')
    if type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= 20:
        raise ValueError('표시 속도는0초과20이하여야 합니다.')
    if max_episodes is not None and (type(max_episodes) is not int or max_episodes < 1):
        raise ValueError('max-episodes는 양의 정수여야 합니다.')
    if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
        raise ValueError('max-frames는 양의 정수여야 합니다.')
    if headless:
        os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
        os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
        if max_episodes is None and max_frames is None:
            max_episodes = 1
    import pygame
    import torch
    from kkamang.paths import ASSET_DIR
    from kkamang.story import STAGES
    from kkamang.ui.field import draw_game
    from kkamang.ui.theme import Assets

    if not (ASSET_DIR / 'art/character/kkamang.png').is_file():
        raise ValueError(f'원본 게임 그림이 없습니다: {ASSET_DIR}')
    run = Path(run).resolve(strict=True)
    config = load_config(run / 'config.json')
    selection = model or 'latest'
    entry = select_checkpoint(run, selection)
    if not follow:
        selection = str(entry['path'])
    policy = load_policy(entry['path'])
    torch.manual_seed(seed)
    initial_updates = int(policy._n_updates)
    episode = None
    pygame.display.init()
    pygame.font.init()
    size = CANVAS_SIZE
    if not headless:
        desktops = pygame.display.get_desktop_sizes()
        if desktops:
            ratio = min(1., (desktops[0][0] - 80) / size[0], (desktops[0][1] - 100) / size[1])
            size = (round(size[0] * ratio), round(size[1] * ratio))
    window = pygame.display.set_mode(size, pygame.RESIZABLE | (pygame.HIDDEN if headless else 0))
    pygame.display.set_caption('PPO 까망이 — 원래 게임 맵에서 학습 정책 보기')
    canvas = pygame.Surface(CANVAS_SIZE)
    assets = Assets()
    clock = pygame.time.Clock()
    queue = deque()
    current = None
    action = None
    results = []
    presented = 0
    paused = False
    need_reset = True
    running = True
    terminal_at = None
    facing = 'down'
    history = []
    metadata = read_json(run / 'metadata.json')
    metadata['progress_guard'] = read_json(run / 'progress_guard.json')
    last_metadata_refresh = 0.
    full_settings = bool(settings)
    full_monitor = bool(monitor) and not full_settings
    training_monitor = {}
    shown_groups = []
    training_summary = {}
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_F1:
                        full_settings = not full_settings
                        full_monitor = False
                    elif event.key == pygame.K_F2:
                        full_monitor = not full_monitor
                        full_settings = False
                    elif event.key == pygame.K_r:
                        need_reset = True
                    elif event.key == pygame.K_TAB:
                        direction = -1 if event.mod & pygame.KMOD_SHIFT else 1
                        stage = (stage - 1 + direction) % 12 + 1
                        need_reset = True
                    elif event.key in (pygame.K_i, pygame.K_l):
                        if event.key == pygame.K_i and not (run / 'initial_model.zip').is_file():
                            continue
                        selection = 'initial' if event.key == pygame.K_i else 'latest'
                        follow = event.key == pygame.K_l
                        need_reset = True
                    elif event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                        entries = checkpoint_entries(run)
                        index = next((i for i, candidate in enumerate(entries)
                                      if candidate['path'] == entry['path']), 0)
                        index = min(len(entries) - 1, max(0, index + (-1 if event.key == pygame.K_LEFT else 1)))
                        selection = str(entries[index]['path'])
                        follow = False
                        need_reset = True
            if not running:
                break
            if need_reset:
                if episode is not None:
                    episode.close()
                # 한 판 도중 정책을 섞지 않도록 회차를 다시 시작할 때만 모델을 교체합니다.
                entry = select_checkpoint(run, 'latest' if follow else selection)
                policy = load_policy(entry['path'])
                episode = NativePolicyEpisode(policy, config, stage=stage,
                    seed=seed + len(results), stochastic=stochastic)
                current = episode.initial_snapshot
                queue.clear()
                queue.append(current)
                terminal_at = None
                need_reset = False
                history = _history(run, stage, 'stochastic' if stochastic else 'greedy')
                action = None
            if not paused:
                if queue:
                    current = queue.popleft()
                    presented += 1
                elif episode.done:
                    if terminal_at is None:
                        result = episode.result()
                        result['checkpoint'] = str(entry['path'])
                        results.append(result)
                        terminal_at = time.monotonic()
                        print(f'Stage{stage}: {result["native_outcome"]["status"]}, '
                              f'{result["score"]}점 · 모델 {policy.num_timesteps:,}행동 · 학습 추가0회', flush=True)
                    if max_episodes is not None and len(results) >= max_episodes:
                        running = False
                    elif headless or time.monotonic() - terminal_at >= 1.5 / speed:
                        need_reset = True
                else:
                    action, frames = episode.decision()
                    if action < 4:
                        facing = ('up', 'down', 'left', 'right')[action]
                    queue.extend(frames)
                    continue
            if current is not None:
                if time.monotonic() - last_metadata_refresh >= 1:
                    metadata = read_json(run / 'metadata.json')
                    metadata['progress_guard'] = read_json(run / 'progress_guard.json')
                    history = _history(run, stage, 'stochastic' if stochastic else 'greedy')
                    training_summary = training_score_summary(run, stage, limit=100)
                    training_monitor = load_training_monitor(run)
                    last_metadata_refresh = time.monotonic()
                game_surface = canvas.subsurface((0, 0, *GAME_SIZE))
                game_surface.fill((36, 34, 49))
                draw_game(game_surface, assets, current, STAGES[stage - 1],
                    elapsed=current['motion']['frame'] / 60., facing=facing,
                    toast='PPO 자동 플레이 · 아이템 수집/귀환 자동 상호작용',
                    remaining_seconds=max(0., current['remaining_ticks'] * .3 - current['motion']['phase'] * .3))
                # 수동 게임의 E키 안내만 자동 상호작용 안내로 덮어 표시합니다.
                # 실제 수집·귀환 동작은 원래 게임 코어가 처리합니다.
                from kkamang.ui.theme import PAPER, INK, text, wrapped
                pygame.draw.rect(game_surface, PAPER, (1048, 425, 204, 180))
                text(game_surface, assets, 'PPO 자동 플레이', (1060, 433), 17, INK, True)
                wrapped(game_surface, assets,
                        '아이템은 닿으면 자동 수집\n귀환점에 닿으면 자동 귀환\n\n정책이 직접 추가 수집과\n귀환 시점을 선택합니다.',
                        (1060, 467, 175, 131), 14, INK, 6)
                _draw_panel(canvas, assets, run=run, entry=entry, model=policy,
                    snapshot=current, action=action, paused=paused,
                    stochastic=stochastic, follow=follow, history=history, metadata=metadata,
                    training_summary=training_summary)
                shown_groups = settings_groups(config, policy, {
                    **metadata, 'viewer_stage': stage, 'viewer_level': episode.env.level})
                _draw_settings(canvas, assets, shown_groups, full=full_settings)
                if full_monitor:
                    _draw_monitor(canvas, assets, training_monitor,
                                  snapshot=current, level=episode.env.level,
                                  selected_steps=int(policy.num_timesteps))
                window.blit(pygame.transform.smoothscale(canvas, window.get_size()), (0, 0))
                pygame.display.flip()
            if max_frames is not None and presented >= max_frames:
                running = False
            if not headless:
                clock.tick(max(1, round(60 * speed)))
        if screenshot is not None:
            screenshot = Path(screenshot)
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            pygame.image.save(canvas, screenshot)
        return {'episodes': results, 'displayed_frames': presented,
                'training_updates_added': 0, 'initial_model_updates': initial_updates,
                'current_model_updates': int(policy._n_updates),
                'settings_visible': full_settings,
                'monitor_visible': full_monitor,
                'settings_group_count': len(shown_groups),
                'selected_model_steps': int(policy.num_timesteps),
                'latest_learner_steps': metadata.get('steps'),
                'screenshot': str(screenshot) if screenshot is not None else None}
    finally:
        if episode is not None:
            episode.close()
        pygame.display.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model', help='initial/latest 또는 실행 폴더 기준 체크포인트 경로')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--follow', action='store_true', help='회차 종료 후 최신 학습 모델을 불러오기')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-episodes', type=int)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--speed', type=float, default=1.)
    parser.add_argument('--settings', action='store_true', help='모든 학습·환경 설정 화면으로 시작(F1 전환)')
    parser.add_argument('--monitor', action='store_true', help='Stage1~12 학습 현황으로 시작(F2 전환)')
    result = play(**vars(parser.parse_args()))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
