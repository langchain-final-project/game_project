"""학습 없이 물건 획득·무피해 수집의 짧은 구간을 진단합니다."""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

import numpy as np
import torch
from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.evaluate_ppo import isolated_rng
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation, PPO_OBSERVATION_VERSION


def probe_model(model, *, seeds=(11, 12, 13), frames=180, current_goals=False,
                tasks=("quest", "ghost")):
    """부분 과제 성공을 측정합니다. 게임 전체 성공률·별 획득률과 구분합니다."""
    if (getattr(model, "ppo_stage", None) != 2
            or getattr(model, "ppo_observation_version", None) != PPO_OBSERVATION_VERSION):
        raise ValueError("현재 실시간 관측으로 학습한 Stage 2 모델이 필요합니다.")
    repeat = getattr(model, "ppo_action_repeat", 1)
    if not seeds or frames <= 0 or frames % repeat:
        raise ValueError("시드가 필요하며 프레임 예산은 행동 유지 프레임의 양의 배수여야 합니다.")
    scenarios = {"magnet": (1, 2, 3, 4, 5), "quest": (6, 7, 8), "ghost": (11, 12, 13, 14)}
    if not tasks or any(task not in scenarios for task in tasks):
        raise ValueError("진단 과제는 magnet·quest·ghost 중 선택해야 합니다.")
    goals = deepcopy(cfg.STAR_GOAL_CONFIG if current_goals else model.ppo_star_goal_config)
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(full_start_probability=0, review_probability=0,
                    magnet_focus_probability=0, quest_focus_probability=0, ghost_focus_probability=0)
    environment = deepcopy(cfg.ENV_CONFIG)
    environment.update(stage=2, randomize_phase=False, max_episode_steps=None, render_mode=None)
    report = {"probe_version": "stage2_local_tasks_v1", "full_game_evaluation": False,
              "frame_budget": frames, "goal_settings": goals, "results": {}}
    was_training = model.policy.training
    try:
        with isolated_rng():
            for kind in tasks:
                starts = scenarios[kind]
                episodes = []
                for start in starts:
                    for seed in seeds:
                        action_seed = cfg.EVAL_CONFIG["action_seed_base"] + seed + start * 1000
                        random.seed(action_seed)
                        np.random.seed(action_seed % (2**32))
                        torch.manual_seed(action_seed)
                        settings["initial_frontier"] = start
                        env = ActionRepeat(PPOGoalObservation(
                            Stage2ThreeStarCurriculum(RealtimeMansionEnv(**environment), settings=settings),
                            star_goal=goals,
                            mask_ineffective_interaction=getattr(model.policy, "mask_ineffective_interaction", False),
                        ), repeat=repeat)
                        try:
                            if env.observation_space != model.observation_space or env.action_space != model.action_space:
                                raise ValueError("모델과 진단 환경의 관측·행동 공간이 다릅니다.")
                            obs, info = env.reset(seed=seed)
                            initial = info["snapshot"]
                            first_phase, first_goal = info["goal_phase"], info["goal_position"]
                            if (kind == "magnet" and initial["magnet"]["collected"]
                                    or kind == "quest" and initial["quest"]["collected"]
                                    or kind == "ghost" and (initial["carried"] >= 14 or initial["damage_count"] != 0)):
                                raise ValueError("진단 시작 상태가 부분 과제와 맞지 않습니다.")
                            success = False
                            remote_warmth = 0
                            for _ in range(frames // repeat):
                                action, _ = model.predict(obs, deterministic=False)
                                obs, _, done, truncated, info = env.step(int(np.asarray(action).item()))
                                snapshot = info["snapshot"]
                                damage = snapshot["damage_count"] - initial["damage_count"]
                                remote_warmth += sum(
                                    event["value"] for event in info["events"]
                                    if event["type"] == "collect" and event.get("magnetic", False)
                                    and event["source_position"] != event["position"]
                                )
                                if kind == "magnet":
                                    success = snapshot["magnet"]["collected"] and remote_warmth > 0
                                elif kind == "quest":
                                    success = snapshot["quest"]["collected"]
                                else:
                                    success = snapshot["carried"] >= 14 and damage == 0
                                if success or kind == "ghost" and damage > 0 or done or truncated:
                                    break
                            episodes.append({
                                "start": start, "seed": seed, "action_seed": action_seed,
                                "success": bool(success), "frames": info["physical_frames"],
                                "damage": damage, "warmth": snapshot["carried"],
                                "warmth_gain": snapshot["carried"] - initial["carried"],
                                "quest_collected": snapshot["quest"]["collected"],
                                "magnet_collected": snapshot["magnet"]["collected"],
                                "remote_warmth": remote_warmth,
                                "first_goal_phase": first_phase, "first_goal": first_goal,
                                "final_position": snapshot["motion"]["position"],
                            })
                        finally:
                            env.close()
                report["results"][kind] = {
                    "episodes": episodes,
                    "summary": {
                        "episodes": len(episodes),
                        "success_rate": float(np.mean([e["success"] for e in episodes])),
                        "mean_frames": float(np.mean([e["frames"] for e in episodes])),
                        "mean_damage": float(np.mean([e["damage"] for e in episodes])),
                    },
                }
    finally:
        model.policy.set_training_mode(was_training)
    return report


def main():
    parser = argparse.ArgumentParser(description="Stage 2 부분 과제의 학습 없는 짧은 검증")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed-base", type=int, default=11)
    parser.add_argument("--episodes-per-start", type=int, default=3)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--tasks", nargs="+", choices=("magnet", "quest", "ghost"),
                        default=["quest", "ghost"], help="진단할 부분 과제")
    parser.add_argument("--current-goals", action="store_true",
                        help="과거 모델도 현재 목표 순서로 비교합니다. 기본값은 모델에 저장된 목표입니다.")
    args = parser.parse_args()
    if (args.episodes_per_start <= 0 or args.frames <= 0 or args.seed_base < 0
            or args.seed_base + args.episodes_per_start > 2**31):
        parser.error("판수·프레임은 양수, 시드는 0 이상 2**31 미만이어야 합니다.")
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    model = PPO.load(args.model, device="cpu")
    report = probe_model(model, seeds=tuple(range(args.seed_base, args.seed_base + args.episodes_per_start)),
                         frames=args.frames, current_goals=args.current_goals, tasks=tuple(args.tasks))
    report["model"] = str(args.model)
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    out = args.output_dir or cfg.RUN_ROOT / "probes" / stamp
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for kind, result in report["results"].items():
        s = result["summary"]
        label = {"magnet": "초반 자석 원격 수집", "quest": "물건 직접 획득",
                 "ghost": "무피해 온기 14점 수집"}[kind]
        print(f"{label}: {s['success_rate']:.1%} ({s['episodes']}판), 평균 피해 {s['mean_damage']:.2f}")
    print(f"부분 과제 진단이며 게임 전체 평가와 구분합니다. 저장: {out}")


if __name__ == "__main__":
    main()
