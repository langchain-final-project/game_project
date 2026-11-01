import argparse
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

from kkamang.story import STAGES
from kkamang.env import RealtimeMansionEnv
from kkamang.realtime import PHYSICS_HZ
from kkamang.ui.field import draw_game
from kkamang.ui.theme import Assets
from rl.ppo_observation import (
    PPOGoalObservation, PPO_OBSERVATION_VERSION, LEGACY_PPO_OBSERVATION_VERSION,
)


class PPOPlayer:
    """학습된 모델의 행동을 지정한 물리 프레임 수만큼 유지하며 재생합니다."""

    def __init__(self, model, env, *, repeat, seed=11, stochastic=False, action_seed_base=0):
        if type(repeat) is not int or repeat <= 0:
            raise ValueError("행동 유지 프레임은 양의 정수여야 합니다.")
        self.model, self.env = model, env
        self.repeat = repeat
        self.stochastic = stochastic
        self.action_seed_base = action_seed_base
        self.seed = seed
        self.reset()

    def reset(self):
        action_seed = self.action_seed_base + self.seed
        random.seed(action_seed)
        np.random.seed(action_seed % (2**32))
        torch.manual_seed(action_seed)
        self.obs, self.info = self.env.reset(seed=self.seed)
        self.finished = False
        self.remaining_frames = 0
        self.decisions = 0
        self.facing = "down"
        self.action = None

    def advance(self):
        """화면 한 프레임에 물리 환경도 한 프레임만 진행합니다."""
        if self.finished:
            return
        if self.remaining_frames == 0:
            action, _ = self.model.predict(self.obs, deterministic=not self.stochastic)
            self.action = int(np.asarray(action).item())
            self.remaining_frames = self.repeat
            self.decisions += 1
        self.obs, _, terminated, truncated, self.info = self.env.step(self.action)
        self.remaining_frames -= 1
        self.facing = {0: "up", 1: "down", 2: "left", 3: "right"}.get(self.action, self.facing)
        self.finished = bool(terminated or truncated)


def resolve_checkpoint(run_dir, checkpoint):
    checkpoint = Path(checkpoint)
    paths = [checkpoint] if checkpoint.is_absolute() else [
        run_dir / "models" / checkpoint, run_dir / checkpoint,
    ]
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("모델 파일을 찾을 수 없습니다: " + ", ".join(map(str, paths)))


def load_player(run_dir, *, checkpoint="best_model.zip", seed=11, stochastic=False):
    """현재 학습 설정 대신 실행 폴더에 저장된 설정과 관측 버전을 사용합니다."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("algorithm") != "PPO":
        raise ValueError("PPO 실행 폴더를 선택하세요.")
    torch.set_num_threads(config.get("training", {}).get(
        "torch_threads", config.get("ppo", {}).get("torch_threads", 1),
    ))
    model_path = resolve_checkpoint(run_dir, checkpoint)
    model = PPO.load(model_path, device="cpu")
    version = getattr(model, "ppo_observation_version", None)
    if version == PPO_OBSERVATION_VERSION:
        include_movement = True
    elif version == LEGACY_PPO_OBSERVATION_VERSION:
        include_movement = False
    else:
        raise ValueError(f"이 관찰기는 해당 관측 버전을 지원하지 않습니다: {version}")
    if config.get("observation_version", version) != version:
        raise ValueError("모델과 실행 폴더의 관측 버전이 다릅니다.")

    env_config = dict(config["environment"])
    for key in ("name", "goal_features", "observation_size", "observation_version", "action_count", "action_unit"):
        env_config.pop(key, None)
    env = PPOGoalObservation(
        RealtimeMansionEnv(**env_config), include_movement=include_movement,
        star_goal=config.get("star_goal", {"enabled": False}),
        mask_ineffective_interaction=getattr(getattr(model, "policy", None),
                                           "mask_ineffective_interaction", False),
    )
    try:
        if model.observation_space != env.observation_space or model.action_space != env.action_space:
            raise ValueError("저장된 모델과 게임 환경의 관측·행동 공간이 다릅니다.")
        if getattr(model, "ppo_stage", env_config["stage"]) != env_config["stage"]:
            raise ValueError("모델과 저장된 설정의 Stage가 다릅니다.")
        repeat = config.get("action_repeat", getattr(model, "ppo_action_repeat", 1))
        if getattr(model, "ppo_action_repeat", repeat) != repeat:
            raise ValueError("모델과 저장된 설정의 행동 유지 프레임이 다릅니다.")
        player = PPOPlayer(
            model, env, repeat=repeat, seed=seed, stochastic=stochastic,
            action_seed_base=config.get("evaluation", {}).get("action_seed_base", 0),
        )
    except Exception:
        env.close()
        raise
    return player, model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", default="best_model.zip", help="models 폴더의 모델 파일명")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stochastic", action="store_true", help="확률적 행동으로 관찰")
    parser.add_argument("--headless", action="store_true", help="창 없이 렌더 연결 점검")
    parser.add_argument("--frames", type=int, help="지정한 화면 프레임 후 종료")
    parser.add_argument("--speed", type=float, default=1.0, help="재생 속도 배율, 기본 1배속")
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames는 양의 정수여야 합니다.")
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed는 0보다 큰 유한한 값이어야 합니다.")
    if not 0 <= args.seed < 2**31:
        parser.error("--seed는 0 이상 2**31 미만이어야 합니다.")
    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        os.environ["SDL_AUDIODRIVER"] = "dummy"
        if args.frames is None:
            args.frames = 5

    player, model_path = load_player(
        args.run_dir, checkpoint=args.checkpoint, seed=args.seed, stochastic=args.stochastic,
    )
    env = player.env
    mode = "확률적" if args.stochastic else "결정적"
    print(f"모델: {model_path}")
    print(f"Stage {player.info['snapshot']['stage']} | {mode} | 시드 {args.seed} | 행동 유지 {player.repeat}프레임")

    try:
        # 3. 기존 게임 그림과 글꼴을 사용할 창을 만듭니다.
        pygame.display.init()
        pygame.font.init()

        screen = pygame.display.set_mode((1280, 720))
        pygame.display.set_caption(f"Stage {player.info['snapshot']['stage']} PPO | {mode} | {model_path.name}")

        assets = Assets()
        clock = pygame.time.Clock()

        running = True
        paused = False
        rendered_frames = 0

        while running:
            # 4. 키보드로 재시작·일시정지·종료를 제어합니다.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        player.reset()
                        paused = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused

            if not running:
                break

            # 5. AI가 선택한 행동으로 환경을 한 프레임 진행합니다.
            if not player.finished and not paused:
                player.advance()
                if player.finished:
                    print("종료 결과:", player.info["outcome"])
                    print("R 키를 누르면 다시 시작합니다.")

            # 6. 방금 진행한 환경 상태를 게임 화면에 그립니다.
            snapshot = player.info["snapshot"]
            frame = snapshot["motion"]["frame"]
            tick_seconds = env.unwrapped.core.config["tick_seconds"]

            remaining_seconds = max(0.0, (
                snapshot["remaining_ticks"]
                - snapshot["motion"]["phase"]
            ) * tick_seconds)
            outcome = player.info.get("outcome") or {}
            score = outcome.get("score", {})
            result = (
                f" | {outcome.get('status')} | 점수 {score.get('total', 0)} | 별 {score.get('stars', 0)}"
                if player.finished else " | 일시정지" if paused else ""
            )

            draw_game(
                screen,
                assets,
                snapshot,
                STAGES[snapshot["stage"] - 1],
                elapsed=frame / PHYSICS_HZ,
                facing=player.facing,
                remaining_seconds=remaining_seconds,
                toast=f"{mode} AI 관찰{result} | R 재시작 · Space 일시정지 · Esc 종료",
            )

            pygame.display.flip()
            rendered_frames += 1
            if args.frames is not None and rendered_frames >= args.frames:
                break
            if args.headless and player.finished:
                break
            if not args.headless:
                clock.tick(PHYSICS_HZ * args.speed)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
