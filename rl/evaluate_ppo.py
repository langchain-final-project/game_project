import argparse
from collections import Counter, deque
from contextlib import contextmanager
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random

from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_config import ACTION_REPEAT

import numpy as np
import torch
from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from rl.ppo_config import (
    ENV_NAME,
    ENV_CONFIG,
    EVAL_CONFIG,
    TRAIN_CONFIG,
    REWARD_VERSION,
)
from rl.ppo_observation import (
    PPOGoalObservation,
    PPO_OBSERVATION_VERSION,
)
from rl.ppo_reward import PPOReward


def make_eval_env():
    if ENV_NAME != "RealtimeMansionEnv":
        raise ValueError("RealtimeMansionEnv가 필요합니다.")

    if EVAL_CONFIG["use_curriculum"]:
        raise ValueError("평가에서는 커리큘럼을 사용하지 않습니다.")

    env = PPOReward(
        PPOGoalObservation(
            RealtimeMansionEnv(**ENV_CONFIG)
        )
    )

    return ActionRepeat(env, repeat=ACTION_REPEAT)


@contextmanager
def isolated_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def summarize(episodes):
    count = len(episodes)

    def mean(key):
        return float(np.mean([e[key] for e in episodes]))

    return {
        "episodes": count,
        "success_count": sum(e["success"] for e in episodes),
        "success_rate": mean("success"),
        "mean_score": mean("score"),
        "mean_return": mean("return"),
        "mean_steps": mean("steps"),
        "mean_decisions": mean("decision_steps"),
        "mean_damage": mean("damage_count"),
        "quest_rate": mean("quest_collected"),
        "warmth_rate": mean("warmth_met"),
        "at_least_one_star_rate": float(np.mean(
            [e["stars"] >= 1 for e in episodes]
        )),
        "at_least_two_star_rate": float(np.mean(
            [e["stars"] >= 2 for e in episodes]
        )),
        "three_star_rate": float(np.mean(
            [e["stars"] == 3 for e in episodes]
        )),
        "stall_rate": mean("stalled"),
    }


def evaluation_cases(env, smoke):
    """기본 평가와 유령 초기 상태별 평가 조건을 만듭니다."""
    groups = []

    for mode, deterministic in (
        ("deterministic", True),
        ("stochastic", False),
    ):
        seeds = list(EVAL_CONFIG[f"{mode}_seeds"])

        if smoke:
            seeds = seeds[:2]

        if not seeds:
            raise ValueError("평가 시드는 한 개 이상 필요합니다.")

        groups.append((
            mode,
            deterministic,
            [{"seed": seed, "options": None} for seed in seeds],
        ))

    if (
        ENV_CONFIG["stage"] == 2
        and EVAL_CONFIG.get("check_patrol_phases", False)
    ):
        ghost = next(
            g for g in env.unwrapped.core.config["ghosts"]
            if g["id"] == "A"
        )

        cases = []

        for index in range(len(ghost["route"])):
            for countdown in range(1, ghost["ticks_per_move"] + 1):
                cases.append({
                    "seed": 20_000 + len(cases),
                    "phase_index": index,
                    "phase_countdown": countdown,
                    "options": {
                        "phase_overrides": {
                            "A": {
                                "index": index,
                                "countdown": countdown,
                            }
                        }
                    },
                })

        groups.append((
            "patrol_phases",
            True,
            cases[:2] if smoke else cases,
        ))

    return groups


def run_episode(model, env, mode, deterministic, case):
    seed = int(case["seed"])
    action_seed = EVAL_CONFIG["action_seed_base"] + seed

    random.seed(action_seed)
    np.random.seed(action_seed % (2**32))
    torch.random.default_generator.manual_seed(action_seed)

    obs, info = env.reset(
        seed=seed,
        options=case["options"],
    )

    previous = info["snapshot"]["motion"]["position"]
    trace = deque(maxlen=120)
    counts = Counter()
    decision_counts = Counter()
    still = max_still = 0

    while True:
        action, _ = model.predict(
            obs,
            deterministic=deterministic,
        )
        action = int(np.asarray(action).item())

        obs, _, terminated, truncated, info = env.step(action)
        decision_counts[action] += 1

        for point in info["repeat_trace"]:
            position = (point["x"], point["y"])

            unchanged = max(
                abs(position[i] - previous[i])
                for i in (0, 1)
            ) < 1e-8

            still = still + 1 if unchanged else 0
            max_still = max(max_still, still)
            previous = position

            counts[action] += 1
            trace.append(point)

        if terminated or truncated:
            break

    snapshot = info["snapshot"]
    episode = info["rl_episode"]
    warmth_met = snapshot["carried"] >= snapshot["target"]

    if episode["success"]:
        failure_stage = "none"
    elif episode["reason"] == "courage":
        failure_stage = "courage"
    elif not snapshot["quest"]["collected"]:
        failure_stage = "quest_missing"
    elif not warmth_met:
        failure_stage = "warmth_missing"
    else:
        failure_stage = "return_missing"

    return {
        **episode,
        "seed": seed,
        "action_seed": action_seed,
        "mode": mode,
        "phase_index": case.get("phase_index"),
        "phase_countdown": case.get("phase_countdown"),
        "warmth_met": bool(warmth_met),
        "failure_stage": failure_stage,
        "max_stationary_frames": max_still,
        "stalled": max_still >= 120,
        "final_x": float(snapshot["motion"]["position"][0]),
        "final_y": float(snapshot["motion"]["position"][1]),
        "action_counts": dict(counts),
        "decision_counts": dict(decision_counts),
        "failure_trace": (
            [] if episode["success"] else list(trace)
        ),
    }
def evaluate_model(model, *, smoke=False):
    if (
        getattr(model, "ppo_observation_version", None)
        != PPO_OBSERVATION_VERSION
    ):
        raise ValueError("모델의 관측 버전이 다릅니다.")

    if model.device.type != "cpu":
        raise ValueError("현재 평가 코드는 CPU 모델을 사용합니다.")
    
    if getattr(model, "ppo_action_repeat", 1) != ACTION_REPEAT:
        raise ValueError("모델의 행동 반복 설정이 다릅니다.")

    env = make_eval_env()
    was_training = model.policy.training

    report = {
        "evaluation_version": "ppo_eval_phases_trace_v1",
        "observation_version": PPO_OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "environment": deepcopy(ENV_CONFIG),
        "smoke": bool(smoke),
        "results": {},
        "action_repeat": ACTION_REPEAT,
    }

    try:
        if (
            model.observation_space != env.observation_space
            or model.action_space != env.action_space
        ):
            raise ValueError(
                "모델과 환경의 관측·행동 공간이 다릅니다."
            )

        with isolated_rng():
            for mode, deterministic, cases in evaluation_cases(
                env, smoke
            ):
                episodes = [
                    run_episode(
                        model,
                        env,
                        mode,
                        deterministic,
                        case,
                    )
                    for case in cases
                ]

                report["results"][mode] = {
                    "summary": summarize(episodes),
                    "episodes": episodes,
                }

    finally:
        env.close()
        model.policy.set_training_mode(was_training)

    threshold = EVAL_CONFIG["success_rate_threshold"]

    report["passed"] = bool(
        not smoke
        and all(
            result["summary"]["success_rate"] >= threshold
            for result in report["results"].values()
        )
    )

    return report


def save_evaluation(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evaluation.json").write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    for mode, result in report["results"].items():
        rows = []

        for episode in result["episodes"]:
            row = {
                key: value
                for key, value in episode.items()
                if key != "failure_trace"
            }
            row["action_counts"] = json.dumps(
                row["action_counts"]
            )
            row["decision_counts"] = json.dumps(
                row["decision_counts"]
            )
            rows.append(row)

        with (output_dir / f"{mode}_episodes.csv").open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="실시간 PPO 평가"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(TRAIN_CONFIG["torch_threads"])

    model = PPO.load(args.model, device="cpu")
    report = evaluate_model(model, smoke=args.smoke)

    stamp = datetime.now(
        timezone(timedelta(hours=9))
    ).strftime("%Y%m%d_%H%M%S_%f")

    output_dir = args.output_dir or (
        args.model.parent
        / "evaluation"
        / f"{args.model.stem}_{stamp}"
    )

    save_evaluation(report, output_dir)

    parts = [
        f"{mode} {result['summary']['success_rate']:.1%}"
        for mode, result in report["results"].items()
    ]

    print("평가 | " + " | ".join(parts))
    print(f"통과: {report['passed']} | 저장: {output_dir}")


if __name__ == "__main__":
    main()