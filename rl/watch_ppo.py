import argparse
import json
import os
from pathlib import Path

import pygame
import torch
from stable_baselines3 import PPO

from kkamang.story import STAGES
from kkamang.ui.field import draw_game
from kkamang.ui.theme import Assets
from rl.ppo_common import make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="final_model.zip")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stochastic", action="store_true", help="확률적 행동으로 관찰")
    parser.add_argument("--headless", action="store_true", help="창 없이 렌더 연결 점검")
    parser.add_argument("--frames", type=int, help="지정한 화면 프레임 후 종료")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames는 양의 정수여야 합니다.")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if args.frames is None:
            args.frames = 5

    # 1. 모델을 학습할 때 사용한 환경 설정을 읽습니다.
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if config.get("algorithm") != "PPO":
        raise ValueError("PPO 실행 폴더를 선택하세요.")

    # 2. 환경과 저장된 모델을 준비합니다.
    torch.set_num_threads(config["ppo"]["torch_threads"])
    env = make_env(config)

    try:
        model_path = args.run_dir / args.checkpoint
        model = PPO.load(model_path, env=env, device="cpu")
        torch.manual_seed(args.seed)

        # 3. 기존 게임 그림과 글꼴을 사용할 창을 만듭니다.
        pygame.display.init()
        pygame.font.init()

        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption("PPO 플레이 관찰 | R: 재시작 | ESC: 종료")

        assets = Assets()
        clock = pygame.time.Clock()

        obs, info = env.reset(seed=args.seed)
        running = True
        finished = False
        facing = "down"
        rendered_frames = 0

        directions = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
        }

        while running:
            # 4. 키보드는 재시작과 종료에만 사용합니다.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        obs, info = env.reset(seed=args.seed)
                        finished = False
                        facing = "down"

            if not running:
                break

            # 5. AI가 선택한 행동으로 환경을 한 프레임 진행합니다.
            if not finished:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                action = int(action.item())

                obs, reward, terminated, truncated, info = env.step(action)

                facing = directions.get(action, facing)
                finished = terminated or truncated

                if finished:
                    print("종료 결과:", info["outcome"])
                    print("R 키를 누르면 다시 시작합니다.")

            # 6. 방금 진행한 환경 상태를 게임 화면에 그립니다.
            snapshot = info["snapshot"]
            frame = snapshot["motion"]["frame"]
            tick_seconds = env.unwrapped.core.config["tick_seconds"]

            remaining_seconds = (
                snapshot["remaining_ticks"]
                - snapshot["motion"]["phase"]
            ) * tick_seconds

            draw_game(
                screen,
                assets,
                snapshot,
                STAGES[snapshot["stage"] - 1],
                elapsed=frame / 60,
                facing=facing,
                remaining_seconds=remaining_seconds,
            )

            pygame.display.flip()
            rendered_frames += 1
            if args.frames is not None and rendered_frames >= args.frames:
                break
            clock.tick(60)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
