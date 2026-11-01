"""추가 학습한 PPO를 원본 게임 그림과 물리 프레임 순서대로 재생합니다."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import EVALUATION, TRAINING
from .runtime import bootstrap, make_env, load_model


class Player:
    """행동으로 생성된 18개 물리 상태를 한 장씩 보여 줍니다."""

    def __init__(self, model, settings, stage=1, seed=11, stochastic=True):
        self.model, self.settings = model, settings
        self.stage, self.seed, self.stochastic = stage, seed, stochastic
        self.env = make_env(settings, stages=[stage], training=False)
        self.frames = deque()
        self.reset()

    def reset(self):
        seed = EVALUATION['action_seed_offset'] + self.seed
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        self.obs, info = self.env.reset(seed=self.seed, options={'stage': self.stage})
        self.snapshot = info['snapshot']
        self.frames.clear()
        self.finished = self.pending_done = False
        self.facing = 'down'
        self.episode = None
        original_step = self.env.core.step

        def captured_step(action):
            result = original_step(action)
            self.frames.append(result.snapshot)
            return result

        # 관람 프로세스의 인스턴스만 감싸며 원본 파일은 수정하지 않습니다.
        self.env.core.step = captured_step

    def advance(self):
        if self.finished:
            return
        if not self.frames:
            action, _ = self.model.predict(self.obs, deterministic=not self.stochastic)
            self.obs, _, done, truncated, info = self.env.step(int(action))
            self.pending_done = bool(done or truncated)
            self.episode = info.get('episode')
            self.facing = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}.get(int(action), self.facing)
        self.snapshot = self.frames.popleft()
        self.finished = self.pending_done and not self.frames

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description='A2C에서 전이한 PPO의 실제 게임 화면 재생')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best_model.zip')
    parser.add_argument('--stage', type=int, choices=range(1, 13), default=1)
    parser.add_argument('--seed', type=int, default=11)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--deterministic', action='store_true')
    modes.add_argument('--stochastic', action='store_true', help='기본값도 확률적 플레이입니다.')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--frames', type=int)
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error('--frames는 양의 정수여야 합니다.')
    if args.headless:
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        os.environ['SDL_AUDIODRIVER'] = 'dummy'
    bootstrap()
    import pygame
    from kkamang.realtime import PHYSICS_HZ
    from kkamang.story import STAGES
    from kkamang.ui.field import draw_game
    from kkamang.ui.theme import Assets
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    if (config.get('experiment') not in ('A2C_to_PPO', 'A2C_PPO_continuation')
            or config.get('algorithm') not in ('PPO', 'A2C')):
        raise ValueError('A2cToPpo 실행 폴더를 선택하세요.')
    torch.set_num_threads(TRAINING['torch_threads'])
    algorithm = config['algorithm']
    model = load_model(args.run / 'models' / args.checkpoint, algorithm)
    player = Player(model, config['source_environment_config'], args.stage, args.seed,
                    stochastic=not args.deterministic)
    mode = '결정적' if args.deterministic else '확률적'
    pygame.display.init()
    pygame.font.init()
    try:
        screen = pygame.display.set_mode((1280, 720))
        assets, clock = Assets(), pygame.time.Clock()
        pygame.display.set_caption(f'A2C 기반 {algorithm} | Stage {args.stage} | {mode} | 1배속')
        running, paused, presented = True, False, 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        player.reset()
                        paused = False
            if not running:
                break
            before = player.finished
            if not paused:
                player.advance()
            if player.finished and not before:
                print('종료 결과:', player.snapshot['outcome'], flush=True)
            snapshot = player.snapshot
            frame = snapshot['motion']['frame']
            remaining = max(0., (snapshot['remaining_ticks'] - snapshot['motion']['phase'])
                            * player.env.level['tick_seconds'])
            draw_game(screen, assets, snapshot, STAGES[args.stage-1], elapsed=frame/PHYSICS_HZ,
                      facing=player.facing, remaining_seconds=remaining,
                      toast=f'A2C 사전 학습 → {algorithm} | {mode} | R 재시작 · Space 일시정지 · Esc 종료')
            pygame.display.flip()
            presented += 1
            if (args.frames is not None and presented >= args.frames
                    or args.headless and player.finished):
                break
            if not args.headless:
                clock.tick(PHYSICS_HZ)
    finally:
        player.close()
        pygame.quit()


if __name__ == '__main__':
    main()
