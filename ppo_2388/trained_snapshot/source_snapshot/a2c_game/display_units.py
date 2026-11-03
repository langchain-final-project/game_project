"""실제 맵과 웹 화면에서 함께 쓰는 학습 단위·설정·점수·현황 표시 자료를 만듭니다."""
from __future__ import annotations

import json
import math
from pathlib import Path


def format_steps(value):
    """4개 환경에서 선택한 행동의 합계를 팀 공통 단위인 100만 스텝 등으로 표시합니다."""
    if value is None:
        return '—'
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Learning steps must be a finite nonnegative count.')
    if value >= 10_000:
        number = f'{value / 10_000:,.2f}'.rstrip('0').rstrip('.')
        return f'{number}만 스텝'
    return f'{int(value):,} 스텝'


def _number(value):
    return f'{value:.9f}'.rstrip('0').rstrip('.') if isinstance(value, float) else str(value)


def _on(value):
    return '사용' if value else '미사용'


def settings_groups(config, model=None, metadata=None):
    """환경·학습·보상·평가 설정을 화면에 표시할 묶음으로 정리합니다.

    관람 중인 모델과 현재 학습기는 저장 시점이 다를 수 있습니다. 모델 객체의
    값은 선택한 모델, 메타데이터의 값은 실행 전체 현황으로 구분합니다.
    """
    meta = metadata or {}
    a, env, r = config['a2c'], config['environment'], config['reward']
    train, ev, policy = config['training'], config['evaluation'], config['policy']
    adaptive = train['entropy_adaptation']
    ppo = config.get('ppo') or {}
    ppo_rows = [
        ('PPO batch_size / epochs', f'{ppo["batch_size"]} / {ppo["n_epochs"]}'),
        ('PPO clip_range', _number(ppo['clip_range'])),
        ('PPO KL 조기 종료', '미사용'),
        ('업데이트 단위', 'Adam 갱신 / epoch · 10회 = 128스텝 rollout 1회'),
    ] if ppo else []
    n_envs, n_steps, repeat = a['n_envs'], a['n_steps'], env['action_repeat']
    navigation = env.get('observation_profile') == 'exit_distance_field_v1'
    map_channels, state_size = (15, 228) if navigation else (14, 224)
    input_size = map_channels * 9 * 16 + state_size
    selected_steps = getattr(model, 'num_timesteps', None)
    selected_updates = getattr(model, '_n_updates', None)
    selected_entropy = getattr(model, 'ent_coef', a['ent_coef'])
    pool = meta.get('stage_pool', [])
    exposed_at = (len(env['stages']) - 1) * train['stage_interval_steps']
    fresh = config.get('fresh_training') or {}
    if fresh:
        exposed_at = 0
    level = meta.get('viewer_level')
    refinement = config.get('refinement') or {}
    comparison = config.get('comparison') or {}
    all_stages = config.get('all_stage_training') or {}
    guard = meta.get('progress_guard') or {}
    refinement_learning = []
    refinement_reward = []
    if refinement:
        refinement_learning = [
            ('학습 초기화' if fresh else '진행형 이어 학습',
             '임의 가중치 · 빈 Adam · 0스텝' if fresh else '평가·정체 감지 후 집중 맵 조절'),
            ('4환경 경험 배분', '초반 전맵4 → 전맵2 · 부족맵2' if fresh else
             '복습2 · 실패 맵 순환 집중2' if all_stages else
             '복습2 · Stage3 집중2' if comparison else '복습1 · 집중2 · 다른 맵1'),
        ]
        if comparison:
            refinement_learning.append(('Stage별 경험 집계', '640만 분기 이후 추가분'))
        if all_stages:
            refinement_learning.extend([
                ('Stage별 경험 집계', '이번 전12맵 자식 실행의 추가분'),
                ('집중 맵 순환 간격', format_steps(all_stages['focus_rotate_steps'])),
                ('모델 선정', '전12맵 통과 수 → 낮은 맵 성공률 → 평균 점수'),
                ('성공 검증', '하나의 선정 모델 · 새 행동 시드2집합 · 전12맵 G/S'),
            ])
        if refinement.get('sampler_version') == 'rotating_return_focus_v1':
            refinement_learning.append(
                ('집중 맵 순환 간격', format_steps(refinement.get('focus_rotate_steps', 25_000))))
        if fresh:
            refinement_learning.extend([
                ('전맵 균등 구간', format_steps(fresh['warmup_steps'])),
                ('집중 맵 순환 간격', format_steps(fresh['focus_rotate_steps'])),
                ('모델 선정', '전12맵 통과 수 → 낮은 맵 성공률 → 평균 점수'),
                ('성능 하락 처리', '기록·최고 모델 보존; 초기 학습 계속'),
                ('현재 검토 상태', str(guard.get('status', meta.get('fresh_status', '준비')))),
            ])
        if guard and not fresh:
            refinement_learning.extend([
                ('한 구간 / 최대 구간', format_steps(guard.get('round_steps')) + f' / {guard.get("max_rounds", "—")}구간'),
                ('연속 정체 중단', f'{guard.get("patience", "—")}구간'),
                ('현재 검토 상태', str(guard.get('status', '기록 대기'))),
            ])
        refinement_reward = [
            ('귀환 거리 기울기 배율', f'기존1 → {_number(refinement.get("return_potential_gain", 4.))}배'),
            ('준비 후 정체 허용', f'{refinement.get("idle_after", 6)}스텝까지'),
            ('이후 장기 정체 비용', _number(refinement.get('idle_penalty', -.01)) + ' / 스텝'),
            ('추가 신호 적용', '귀환 준비 후 학습에만 적용'),
        ]
        if refinement.get('loop_penalty', 0.) < 0:
            refinement_reward.extend([
                ('귀환 정체 감지', '출구 최저거리 무진전'
                 if refinement.get('loop_detection') == 'no_progress' else '짧은 동일 구간 반복'),
                ('이동 순환 관찰 구간', f'{refinement["loop_window"]}스텝'),
                ('순환 비용 / 회차 상한',
                 f'{_number(refinement["loop_penalty"])} / {_number(refinement["loop_penalty_cap"])}'),
            ])
    native_rows = []
    if isinstance(level, dict):
        tick_seconds = level['tick_seconds']
        native_rows = [
            ('현재 맵 기본값', f'Stage{level["stage"]}'),
            ('게임 제한 시간', f'{level["time_budget_ticks"]}틱 / {_number(level["time_budget_ticks"] * tick_seconds)}초'),
            ('시간25점 기준', f'{level["star_fast_ticks"]}틱 이내'),
            ('기본 귀환 온기', f'{level["target_value"]}'),
            ('온기 점수 상한', f'{level["star_warmth_cap"]} / 전체{level["total_item_value"]}'),
            ('초기 용기', str(level['initial_courage'])),
            ('피격 후 보호', f'{level["protection_future_ticks"]}틱'),
            ('순찰 유령 / 샛문', f'{len(level["ghosts"])} / {len(level["shortcuts"])}개'),
            ('추격 유령', '있음' if level.get('chaser') else '없음'),
            ('자석 / 무적 아이템', ('있음' if level.get('magnet') else '없음') + ' / ' + ('있음' if level.get('invincibility') else '없음')),
        ]
    return [
        {'title': '학습 설정', 'rows': [
            ('알고리즘', '공통 PPO 모델 1개' if ppo else '원본 A2C 모델 1개'),
            ('전체 학습 목표', format_steps(train['total_steps'])),
            ('선택 모델 학습량', format_steps(selected_steps)),
            ('선택 학습 정확한 횟수', f'{selected_steps:,}스텝' if selected_steps is not None else '—'),
            ('현재 학습기 진행', format_steps(meta.get('steps'))),
            ('선택 모델 업데이트', f'{selected_updates:,}회' if selected_updates is not None else '—'),
            ('학습률', _number(a['learning_rate'])),
            ('n_steps / 병렬 환경', f'{n_steps}스텝 / {n_envs}개'),
            ('할인율 gamma', _number(a['gamma'])),
            ('GAE lambda', _number(a['gae_lambda'])),
            ('엔트로피 초기 → 선택', f'{_number(a["ent_coef"])} → {_number(float(selected_entropy))}'),
            ('현재 학습기 엔트로피 계수', _number(meta['entropy_coef']) if isinstance(meta.get('entropy_coef'), (int, float)) else '기록 대기'),
            ('엔트로피 자동 조절', f'{_on(adaptive["enabled"])} / 목표 {_number(adaptive["target"])}'),
            ('엔트로피 계수 범위', f'{_number(adaptive["min"])} ~ {_number(adaptive["max"])}'),
            ('조절 확인 간격', f'{adaptive["interval_updates"]} rollouts (각 {n_envs * n_steps}스텝)'),
            ('최적화 / 정규화', ('RMSprop' if a['use_rms_prop'] else 'Adam') + ' / Advantage 정규화 ' + _on(a['normalize_advantage'])),
            ('정책망 pi / 가치망 vf', '·'.join(map(str, a['net_arch']['pi'])) + ' / ' + '·'.join(map(str, a['net_arch']['vf']))),
            ('가치손실 계수', _number(a['vf_coef'])),
            ('기울기 제한', _number(a['max_grad_norm'])),
            ('장치 / Torch 스레드', f'{a["device"].upper()} / {a["torch_threads"]}개'),
            ('정책 logits 범위', f'±{_number(policy["logit_bound"])}'),
            ('특징 표현', 'CNN + MLP / LayerNorm·Tanh'),
            ('학습 seed', str(train['seed'])),
            ('로그 간격', f'{train["progress_every_updates"]} rollouts (각 {n_envs * n_steps}스텝)'),
            ('체크포인트 간격', format_steps(train['checkpoint_every_steps'])),
        ] + ppo_rows + refinement_learning},
        {'title': '환경·평가 설정', 'rows': [
            ('게임 환경', '실시간 RealtimeGameCore'),
            ('중간 상태 복사 최적화', _on(env.get('fast_rollout', True))),
            ('Gymnasium 연결', 'UnifiedMansionEnv'),
            ('관측 프로필', '출구 거리장 + 귀환 문맥' if navigation else '기존 지도·상태'),
            ('게임 모드', env['mode']),
            ('현재 학습 맵', ' · '.join(map(str, pool)) if pool else '실행 기록 대기'),
            ('전체 학습 범위', 'Stage1 ~ Stage12'),
            ('새 맵 경험 간격', '처음부터 전12개' if fresh else format_steps(train['stage_interval_steps'])),
            ('전12개 경험 시작', format_steps(exposed_at)),
            ('최종 목표', '실제85점 이상 귀환'),
            ('정책 행동', '상·하·좌·우·대기 5개'),
            ('행동 반복', f'최대{repeat}물리 프레임 / 스텝'),
            ('자동 수집·귀환', _on(env['auto_interaction']) + ' / E 불필요'),
            ('물리 시계', '60프레임/초 · 18프레임/틱'),
            ('시간·유령·충돌', '원래 게임 규칙 유지'),
            ('유령 초기 위상 무작위', _on(env['randomize_phase'])),
            ('외부 회차 강제 제한', '없음' if env['max_policy_steps'] is None else str(env['max_policy_steps'])),
            ('정기 평가 간격', format_steps(train['eval_every_steps'])),
            ('중간 평가 / 스테이지', f'최대확률1판 · 확률{ev["episodes"]}판'),
            ('최종 평가 / 방식·맵', f'{ev["final_episodes"]}판 × {len(ev["final_seed_sets"])}묶음'),
            ('성적 기준 점수', ' / '.join(map(str, ev['score_thresholds'])) + '점'),
            ('목표 성공률 / 배포', f'{ev["required_star3_rate"]:.0%} / 최대확률'),
            ('고정 평가 seed', str(ev['seed'])),
            ('최종 평가 seed', ' / '.join(map(str, ev['final_seed_sets']))),
        ] + native_rows},
        {'title': '학습 보상 설정', 'rows': [
            ('실제 귀환 보상', f'게임 점수 / 100 × {_number(r["score_scale"])}'),
            ('실제85점 귀환 추가', '+' + _number(r['highscore_bonus'])),
            ('퀘스트 첫 수집', '+' + _number(r['quest'])),
            ('온기 +1 실제 수집', '+' + _number(r['pickup_per_value'])),
            ('온기 +3 실제 수집', '+' + _number(3 * r['pickup_per_value'])),
            ('실제 게임 실패', _number(r['failure'])),
            ('피해 사건 1회', _number(r['damage'])),
            ('벽에서 완전 정체', _number(r['wall_stall'])),
            ('길 거리 보조 scale', _number(r['potential_scale'])),
            ('새 상태 최초 방문', '+' + _number(r['novelty_per_state'])),
            ('방문 보상 회차 상한', '+' + _number(r['novelty_cap'])),
            ('방문 보상 감쇠', '학습 진행에 따라 최대90%'),
            ('정식 평가 보상', '실제 종료 점수 / 보조 보상 제외'),
            ('실제 게임 점수', '온기60 + 시간25 + 용기15'),
            ('이전 /20와 관계', '단순 환산 아닌 보상 구조 변경'),
            ('낮은 점수 귀환', '실제 점수·COMPLETE 유지'),
            ('진행 중 점수', '미확정 / 종료 후 실제 점수'),
        ] + refinement_reward},
        {'title': '차원·단위와 기존 표현', 'rows': [
            ('팀의 학습 단위', '100만 스텝 = 행동100만회'),
            ('스텝 집계', f'{n_envs}환경 합산 / 에피소드 수 아님'),
            ('1업데이트', f'{n_envs}환경 × {n_steps}스텝 = {n_envs * n_steps}스텝'),
            ('1판(에피소드)', '원래 출발 → 실제 게임 종료'),
            ('1스텝의 게임 시간', f'최대{repeat}프레임 = {repeat / 60:g}초'),
            ('초기 실시간 관측', '623차원 = 격자620 + 3'),
            ('현재 게임 관측', '630차원 = 초기623 + 7'),
            ('추가7값의 구성', '순찰좌표4 + 추격좌표2 + 기절1'),
            ('모델 지도 입력', f'{map_channels}채널 × 9 × 16 = {map_channels * 144:,}값'),
            ('모델 상태 입력', '198 + 12 + 14' + (' + 4 = 228값' if navigation else ' = 224값')),
            ('모델 전체 입력', f'{map_channels * 144:,} + {state_size} = {input_size:,}값'),
            ('네트워크 내부 특징', '지도64 + 상태64 = 128값'),
            ('원본 정보 보존', '지도432 + 상태198 = 630'),
            ('추가 공간 표현', f'{map_channels - 3}채널 × 144칸 = {(map_channels - 3) * 144:,}값'),
            ('추가 문맥·보조', '스테이지12 + 보조14' + (' + 귀환4 = 30값' if navigation else ' = 26값')),
            (f'{input_size:,}값의 의미', '630정보 유지 + 공간·보조 재표현'),
            ('행동 표현 변화', '6개(E포함) → 자동상호작용5개'),
            ('이전6프레임 → 18', '행동 반복 시간 3배'),
            ('gamma 물리시간 환산', '0.999^(18/6) = 0.997003'),
            ('현재 관람 화면', '추론만 실행 / 학습 추가 없음'),
        ]},
    ]


def score_progress(history):
    """평가 시드·행동 방식·판수가 같을 때만 실제 평균 점수의 변화를 비교합니다."""
    valid = [row for row in history if type(row.get('mean_score')) in (int, float)
             and math.isfinite(row['mean_score']) and 0 <= row['mean_score'] <= 100]
    valid.sort(key=lambda row: row.get('steps') or 0)
    if not valid:
        return {'previous_mean': None, 'latest_mean': None, 'delta': None,
                'best_mean': None, 'episodes': 0, 'comparisons': 0}
    latest = valid[-1]
    fields = ('stage', 'mode', 'seed', 'episodes', 'randomize_phase')
    required = ('stage', 'mode', 'seed', 'episodes')
    same = ([latest] if any(latest.get(key) is None for key in required) else
            [row for row in valid if all(row.get(key) == latest.get(key) for key in fields)])
    previous = next((row for row in reversed(same[:-1])
                     if row.get('steps') != latest.get('steps')), None)
    return {'previous_mean': previous['mean_score'] if previous is not None else None,
            'latest_mean': latest['mean_score'],
            'delta': latest['mean_score'] - previous['mean_score'] if previous is not None else None,
            'best_mean': max(row['mean_score'] for row in same),
            'episodes': latest.get('episodes', 0), 'comparisons': len(same),
            'steps': latest.get('steps', 0)}


def training_score_summary(run, stage, limit=100):
    """최근 기록의 읽기 범위를 제한하고, 해당 Stage의 실패도 포함해 완료한 판을 집계합니다."""
    if type(stage) is not int or not 1 <= stage <= 12 or type(limit) is not int or limit < 1:
        raise ValueError('A real stage and positive window size are required.')
    path = Path(run) / 'train/episodes.jsonl'
    episodes = []
    try:
        with path.open('rb') as handle:
            size = path.stat().st_size
            offset = max(0, size - 2_000_000)
            handle.seek(offset)
            if offset:
                handle.readline()
            for line in handle:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if not isinstance(record, dict) or not isinstance(record.get('episode'), dict):
                    continue
                episode = record['episode']
                score = episode.get('score')
                if (episode.get('stage') == stage and type(score) in (int, float)
                        and math.isfinite(score) and 0 <= score <= 100):
                    episodes.append(episode)
    except OSError:
        pass
    episodes = episodes[-limit:]
    return {'episodes': len(episodes),
            'mean_score': sum(row['score'] for row in episodes) / len(episodes) if episodes else None,
            'best_score': max(row['score'] for row in episodes) if episodes else None,
            'latest_score': episodes[-1]['score'] if episodes else None,
            'mean_warmth': sum(row.get('warmth', 0) for row in episodes) / len(episodes) if episodes else None,
            'quest_rate': sum(bool(row.get('quest_collected')) for row in episodes) / len(episodes) if episodes else None,
            'return_rate': sum(bool(row.get('game_return')) for row in episodes) / len(episodes) if episodes else None,
            'failures_included': True, 'stage': stage}


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _read_log(path, *, max_bytes=4_000_000, max_rows=10_000):
    """읽을 분량을 제한해 완성된 기록만 읽습니다. 실행 중인 학습기를 변경하지 않습니다."""
    try:
        with Path(path).open('rb') as handle:
            handle.seek(0, 2)
            offset = max(0, handle.tell() - max_bytes)
            handle.seek(offset)
            if offset:
                handle.readline()
            lines = handle.read().splitlines()
    except OSError:
        return []
    result = []
    for line in lines[-max_rows:]:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                result.append(value)
        except (ValueError, UnicodeDecodeError):
            continue
    return result


def evaluation_cohort(history):
    """서로 다른 모델의 결과를 섞지 않고, 같은 시점의 전12맵 G/S 평가를 선택합니다."""
    cohorts = {}
    for row in history:
        if (not isinstance(row, dict) or type(row.get('stage')) is not int
                or not 1 <= row['stage'] <= 12
                or row.get('mode') not in ('greedy', 'stochastic')):
            continue
        key = (row.get('steps'), row.get('checkpoint'), row.get('evaluation_label'))
        cohorts.setdefault(key, {})[(row['stage'], row['mode'])] = row
    if not cohorts:
        return {'steps': None, 'complete': False, 'pending': False, 'rows': []}
    ordered = sorted(cohorts, key=lambda key: key[0] if type(key[0]) in (int, float) else -1)
    complete = [key for key in ordered if len(cohorts[key]) == 24]
    selected = complete[-1] if complete else ordered[-1]
    return {'steps': selected[0], 'complete': len(cohorts[selected]) == 24,
            'pending': ordered[-1] != selected,
            'rows': list(cohorts[selected].values())}


def load_training_monitor(run):
    """실제 맵 화면과 웹 대시보드가 같은 학습 현황을 사용하도록 모읍니다.

    실패한 판도 평균에 포함하고, 학습 중 평균과 고정 조건 평가를 구분합니다.
    기록이 없는 Stage·지표는 실패0점으로 만들지 않고 미확인으로 둡니다.
    """
    run = Path(run)
    metadata, status = _read_json(run / 'metadata.json'), _read_json(run / 'status.json')
    progress = _read_log(run / 'train/progress.jsonl', max_rows=2000)
    history = _read_log(run / 'evaluation/history.jsonl')
    cohort = evaluation_cohort(history)
    selected = _read_json(run / 'selected_evaluation.json')
    if selected:
        selected_rows = [{**row, 'stage': int(stage), 'mode': mode, 'steps': selected['steps']}
                         for mode, stages in selected['metrics'].items() for stage, row in stages.items()]
        cohort = {'steps': selected['steps'], 'complete': True, 'pending': False, 'rows': selected_rows}
    chosen = {(row['stage'], row['mode']): row for row in cohort['rows']}
    exposure = _read_json(run / 'stageexposure.json') or metadata.get('stage_exposure', {})
    stages = []
    latest_progress = progress[-1] if progress else {}
    training_by_stage = {row['stage']: row for row in latest_progress.get('per_stage', [])
                         if isinstance(row, dict) and type(row.get('stage')) is int}
    # Stage별 요약이 없는 예전 기록은 실제 완료 회차에서 보충합니다.
    # 로그 끝부분을 한 번만 읽어 모든 Stage에 나눠 담고 실패0점도 유지합니다.
    recent_by_stage = {}
    for record in _read_log(run / 'train/episodes.jsonl', max_bytes=4_000_000, max_rows=2000):
        episode = record.get('episode', {})
        if (not isinstance(episode, dict) or type(episode.get('stage')) is not int
                or not 1 <= episode['stage'] <= 12
                or type(episode.get('score')) not in (int, float)
                or not math.isfinite(episode['score']) or not 0 <= episode['score'] <= 100):
            continue
        recent_by_stage.setdefault(episode['stage'], []).append(episode)
    for stage, episodes in recent_by_stage.items():
        if stage in training_by_stage:
            continue
        episodes = episodes[-100:]
        average = lambda key: sum(row.get(key, 0) for row in episodes) / len(episodes)
        training_by_stage[stage] = {'stage': stage, 'episodes': len(episodes),
            'mean_score': average('score'), 'return_rate': average('game_return'),
            'star3_rate': average('highscore_success'), 'quest_rate': average('quest_collected'),
            'mean_warmth': average('warmth'), 'source': 'recent_completed_native_episodes'}
    for stage in range(1, 13):
        greedy, stochastic = chosen.get((stage, 'greedy')), chosen.get((stage, 'stochastic'))
        primary = stochastic or greedy or {}
        seen = exposure.get(str(stage), exposure.get(stage, {}))
        reasons = primary.get('reasons') or {}
        stages.append({'stage': stage, 'learner_steps': seen.get('learner_steps'),
            'trained': bool(seen.get('learner_steps', 0) > 0),
            'greedy': greedy, 'stochastic': stochastic,
            'training': training_by_stage.get(stage),
            'quest_rate': primary.get('quest_rate'), 'mean_warmth': primary.get('mean_warmth'),
            'failures': {key: value for key, value in reasons.items() if key != 'returned'},
            'failure_episodes': sum(value for key, value in reasons.items()
                                    if key != 'returned' and type(value) is int),
            'diagnostics': (metadata.get('refinement_metrics', {}).get('diagnostics', {})
                            .get(str(stage), {}))})
    entropy_changes = metadata.get('entropy_changes', [])
    changes = _read_log(run / 'adjustment_history.jsonl', max_rows=20)
    signatures = {(row.get('steps'), row.get('before'), row.get('after'))
                  for row in entropy_changes if isinstance(row, dict)}
    changes = [row for row in changes if row.get('kind') != 'entropy_coef'
               or (row.get('steps'), row.get('before'), row.get('after')) not in signatures]
    config = _read_json(run / 'config.json')
    return {'steps': status.get('steps', metadata.get('steps')),
        'updates': status.get('updates', metadata.get('updates')),
        'total_steps': status.get('total_steps', metadata.get('total_steps')),
        'status': status.get('status', metadata.get('status', '기록 대기')),
        'phase': status.get('phase'), 'stage_pool': status.get('stage_pool', metadata.get('stage_pool', [])),
        'entropy_coef': status.get('entropy_coef', metadata.get('entropy_coef')),
        'progress': latest_progress, 'evaluation': cohort,
        'stages': stages, 'layout': (_read_json(run / 'sampling.json')
                                  or _read_json(run / 'refinement_layout.json')),
        'guard': (_read_json(run / 'fresh_protocol.json') or _read_json(run / 'all_stage_protocol.json')
                  or _read_json(run / 'progress_guard.json')),
        'reward_profile': metadata.get('reward_profile')
            or config.get('refinement', {}).get('reward_profile') or 'native_base',
        'entropy_changes': entropy_changes[-6:] if isinstance(entropy_changes, list) else [],
        'adjustment_history': changes[-6:], 'live': _read_json(run / 'live.json'),
        'read_only': True, 'training_updates_added': 0}
