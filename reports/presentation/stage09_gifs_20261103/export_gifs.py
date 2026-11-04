"""선정 모델의 실제 Stage 9 최대확률 플레이를 원본 게임 렌더러로 GIF 출력합니다."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[2]
SOURCE = PROJECT / 'ppo_2388'
LOCAL_RUN = SOURCE / 'runs/seed_11_20261102_230005_281926_io_retry_v1'
RUN = LOCAL_RUN if LOCAL_RUN.is_dir() else SOURCE / 'trained_snapshot'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
sys.path.insert(0, str(SOURCE))

import pygame
from PIL import Image
import torch
from stable_baselines3 import A2C, PPO

from a2c_game.config import load_config
from a2c_game.evaluation import isolated_rng
from a2c_game.play import NativePolicyEpisode
from kkamang.story import STAGES
from kkamang.ui.field import draw_game
from kkamang.ui.theme import Assets, PAPER, INK, text, wrapped

FPS = 15
STRIDE = 60 // FPS
SIZE = (960, 600)
START_HOLD_MS = 600
END_HOLD_MS = 3000
STAGE = 9
COLORS = {'A2C': (132, 82, 189), 'PPO': (76, 148, 135)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def simulate(name, loader, folder, checkpoint, evaluation_path, replay_path):
    config = load_config(folder / 'config.json')
    evaluated = read(evaluation_path)
    expected = evaluated['metrics']['greedy'][str(STAGE)]
    replay = read(replay_path)
    checkpoint_hash = sha(checkpoint)
    model = loader.load(checkpoint, device='cpu')
    model.policy.set_training_mode(False)
    counters = (model.num_timesteps, model._n_updates)
    assert model.num_timesteps == evaluated['steps']
    samples = []
    with isolated_rng(replay['action_seed']):
        episode = NativePolicyEpisode(model, config, stage=STAGE, seed=replay['seed'], stochastic=False)
        try:
            samples.append((0, episode.initial_snapshot, 'down'))
            facing = 'down'
            while not episode.done:
                action, frames = episode.decision()
                if action < 4:
                    facing = ('up', 'down', 'left', 'right')[action]
                for snapshot in frames:
                    frame = int(snapshot['motion']['frame'])
                    if frame % STRIDE == 0 or snapshot['terminal']:
                        samples.append((frame, snapshot, facing))
            result = episode.result()
            assert result['score'] == expected['mean_score'], (name, result['score'], expected['mean_score'])
            assert result['policy_steps'] == expected['mean_policy_steps']
            assert result['physical_frames'] == expected['mean_physical_frames']
            assert result['native_outcome'] == replay['native_outcome']
            expected_actions = [frame['action'] for frame in replay['frames'][1:]]
            assert result['actions'] == expected_actions, f'{name}: replay action mismatch'
            assert counters == (model.num_timesteps, model._n_updates)
            assert sha(checkpoint) == checkpoint_hash
        finally:
            episode.close()
    print(f'{name}: replay matched, {result["score"]} points, '
          f'{result["physical_frames"]} physical frames, {len(samples)} sampled images', flush=True)
    return {'name': name, 'samples': samples, 'result': result,
            'checkpoint': str(checkpoint.relative_to(PROJECT)), 'checkpoint_sha256': checkpoint_hash,
            'seed': replay['seed'], 'action_seed': replay['action_seed']}


def render(record, sample, assets, *, ending=False):
    name = record['name']
    frame, snapshot, facing = sample
    canvas = pygame.Surface((1280, 800))
    canvas.fill((255, 249, 239))
    game = canvas.subsurface((0, 80, 1280, 720))
    draw_game(game, assets, snapshot, STAGES[STAGE - 1], elapsed=frame / 60., facing=facing,
              toast=f'{name} 자동 플레이 · 최대확률 행동 · 동일 초기 조건',
              remaining_seconds=max(0., snapshot['remaining_ticks'] * .3 - snapshot['motion']['phase'] * .3))
    # 실제 자동 상호작용 규칙에 맞게 수동 E키 안내만 교체합니다.
    pygame.draw.rect(game, PAPER, (1048, 425, 204, 180))
    text(game, assets, '정책 자동 플레이', (1060, 433), 17, INK, True)
    wrapped(game, assets, '아이템은 닿으면 자동 수집\n귀환점에 닿으면 자동 귀환\n\n정책이 직접 이동과\n귀환 시점을 선택합니다.',
            (1060, 467, 175, 131), 14, INK, 6)
    text(canvas, assets, f'{name}  |  STAGE 9', (24, 8), 29, COLORS[name], True)
    text(canvas, assets, '최대확률 행동 · 동일 초기 조건 · 실제 속도 1배속', (24, 48), 18, INK)
    text(canvas, assets, f'모델 {record["result"]["model_steps"]:,}스텝', (1256, 11), 19, INK,
         anchor='topright')
    text(canvas, assets, f'게임 경과 {frame / 60.:.1f}초', (1256, 47), 18, INK,
         anchor='topright')
    if ending:
        veil = pygame.Surface((1280, 720), pygame.SRCALPHA)
        veil.fill((25, 19, 36, 125))
        game.blit(veil, (0, 0))
        panel = pygame.Rect(260, 210, 760, 280)
        pygame.draw.rect(game, (255, 249, 239), panel, border_radius=20)
        pygame.draw.rect(game, COLORS[name], panel, width=4, border_radius=20)
        outcome = record['result']['native_outcome']
        completed = outcome['status'] == 'COMPLETE'
        heading = '귀환 성공' if completed else '시간 초과 · 귀환 실패'
        text(game, assets, f'{name}  |  {heading}', (640, 252), 31, COLORS[name], True, 'midtop')
        text(game, assets, f'최종 점수 {record["result"]["score"]}점', (640, 315), 48, INK, True, 'midtop')
        detail = '필수 기억 아이템 획득 · 출구 귀환 완료' if completed else '제한 시간 내 귀환하지 못함'
        text(game, assets, detail, (640, 401), 23, INK, anchor='midtop')
        text(game, assets, f'플레이 {frame / 60.:.2f}초 · 추가 학습 0회', (640, 442), 18, INK, anchor='midtop')
    reduced = pygame.transform.smoothscale(canvas, SIZE)
    return Image.frombytes('RGB', SIZE, pygame.image.tobytes(reduced, 'RGB'))


def encode(record, palette, assets, common_physical_frames):
    frames = []
    durations = []
    samples = record['samples']
    count = len(samples)
    for index, sample in enumerate(samples):
        terminal = index == count - 1
        rgb = render(record, sample, assets, ending=terminal)
        if index == 0:
            rgb.save(OUTPUT / f'stage09_{record["name"].lower()}_start.png')
        if terminal:
            rgb.save(OUTPUT / f'stage09_{record["name"].lower()}_result.png')
        if index == count // 2:
            rgb.save(OUTPUT / f'stage09_{record["name"].lower()}_preview.png')
        frames.append(rgb.quantize(palette=palette, dither=Image.Dither.NONE))
        frame = sample[0]
        if terminal:
            duration = round((common_physical_frames - frame) / 60 * 100) * 10 + END_HOLD_MS
        else:
            next_frame = samples[index + 1][0]
            duration = round(next_frame / 60 * 100) * 10 - round(frame / 60 * 100) * 10
            if index == 0:
                duration += START_HOLD_MS
        assert duration > 0
        durations.append(duration)
        if index % 75 == 0 or terminal:
            print(f'{record["name"]}: rendered {index + 1}/{count}', flush=True)
    output = OUTPUT / f'stage09_{record["name"].lower()}_greedy.gif'
    print(f'{record["name"]}: encoding GIF...', flush=True)
    frames[0].save(output, format='GIF', save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, disposal=1, optimize=True,
                   comment=b'Actual model inference; deterministic actions; no training; native game renderer.')
    with Image.open(output) as animation:
        total_ms = 0
        for index in range(animation.n_frames):
            animation.seek(index)
            total_ms += animation.info['duration']
        assert animation.size == SIZE
        assert total_ms == START_HOLD_MS + round(common_physical_frames / 60 * 100) * 10 + END_HOLD_MS
        assert animation.info.get('loop') == 0
        encoded_frames = animation.n_frames
    info = {'gif': output.name, 'size_bytes': output.stat().st_size,
            'resolution': list(SIZE), 'duration_ms': total_ms, 'encoded_frames': encoded_frames,
            'sampled_frames': len(frames), 'playback_speed': 1,
            'end_hold_including_wait_ms': durations[-1]}
    assert output.stat().st_size < 20 * 1024 * 1024, f'GIF is too large for convenient slide use: {output}'
    print(f'{record["name"]}: {output.stat().st_size / 1024 / 1024:.2f} MiB, {total_ms / 1000:.2f}s', flush=True)
    return info


def main():
    torch.set_num_threads(1)
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
    try:
        records = [
            simulate('A2C', A2C, SOURCE / 'shared_snapshot', SOURCE / 'shared_snapshot/model.zip',
                     SOURCE / 'shared_snapshot/selected_evaluation.json', SOURCE / 'shared_snapshot/replays/stage_09_greedy.json'),
            simulate('PPO', PPO, RUN, RUN / 'checkpoints/round_096_step_0009609344.zip',
                     RUN / 'checks/round_096.json', RUN / 'evaluation/round_096/greedy/replays/stage_09_greedy.json'),
        ]
        assets = Assets()
        # 실제 초기·중간·결과 화면으로 두 파일이 공유하는 팔레트를 구성합니다.
        palette_samples = []
        for record in records:
            samples = record['samples']
            for index in (0, len(samples) // 3, 2 * len(samples) // 3, len(samples) - 1):
                palette_samples.append(render(record, samples[index], assets, ending=index == len(samples) - 1))
        mosaic = Image.new('RGB', (SIZE[0], SIZE[1] * len(palette_samples)))
        for index, sample in enumerate(palette_samples):
            mosaic.paste(sample, (0, index * SIZE[1]))
        palette = mosaic.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        del mosaic, palette_samples
        common_physical_frames = max(record['result']['physical_frames'] for record in records)
        metadata = {'stage': STAGE, 'action_selection': 'deterministic maximum-probability',
                    'source_physics_hz': 60, 'gif_sample_fps': FPS,
                    'synchronized_duration': True, 'early_finish_holds_result_screen': True,
                    'training_updates_added': 0, 'outputs': {}}
        for record in records:
            info = encode(record, palette, assets, common_physical_frames)
            info.update({key: record[key] for key in ('checkpoint', 'checkpoint_sha256', 'seed', 'action_seed', 'result')})
            metadata['outputs'][record['name']] = info
        (OUTPUT / 'export_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
