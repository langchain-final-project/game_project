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
    STAGNATION_CONFIG,
)
from rl.ppo_observation import (
    PPOGoalObservation,
    PPO_OBSERVATION_VERSION,
)
from rl.ppo_reward import PPOReward


def make_eval_env(*, star_goal=None, mask_ineffective_interaction=False):
    if ENV_NAME != "RealtimeMansionEnv":
        raise ValueError("RealtimeMansionEnv가 필요합니다.")

    if EVAL_CONFIG["use_curriculum"]:
        raise ValueError("평가에서는 커리큘럼을 사용하지 않습니다.")

    env = PPOReward(
        PPOGoalObservation(
            RealtimeMansionEnv(**ENV_CONFIG), star_goal=star_goal,
            mask_ineffective_interaction=mask_ineffective_interaction,
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


def post_quest_metrics(episodes):
    """직접 물건을 획득한 에피소드 중 귀환 성공률을 구합니다."""
    collected = [
        e for e in episodes
        if e["quest_collected"] and not e.get("quest_collected_at_start", False)
    ]
    count = len(collected)
    successes = sum(bool(e["success"]) for e in collected)
    return {
        "post_quest_episode_count": count,
        "post_quest_success_count": successes,
        "post_quest_success_rate": successes / count if count else None,
    }


def format_post_quest(summary):
    rate = summary["post_quest_success_rate"]
    label = f"{rate:.1%}" if rate is not None else "--"
    return (
        f"직접 획득 후 귀환 {label} "
        f"({summary['post_quest_success_count']}/{summary['post_quest_episode_count']})"
    )


def summarize(episodes):
    count = len(episodes)

    def mean(key):
        return float(np.mean([e[key] for e in episodes]))

    return {
        "episodes": count,
        "success_count": sum(e["success"] for e in episodes),
        "success_rate": mean("success"),
        **post_quest_metrics(episodes),
        "mean_score": mean("score"),
        "mean_stars": mean("stars"),
        "mean_stars_on_success": (
            float(np.mean([e["stars"] for e in episodes if e["success"]]))
            if any(e["success"] for e in episodes) else None
        ),
        "mean_return": mean("return"),
        "mean_steps": mean("steps"),
        "mean_decisions": mean("decision_steps"),
        "mean_damage": mean("damage_count"),
        "damage_free_success_rate": float(np.mean([
            bool(e["success"] and e["damage_count"] == 0) for e in episodes
        ])),
        "magnet_acquisition_rate": float(np.mean([e.get("magnet_acquired", False) for e in episodes])),
        "magnet_use_rate": float(np.mean([e.get("magnet_used", False) for e in episodes])),
        "magnet_ten_return_rate": float(np.mean([e.get("magnet_ten_return_success", False) for e in episodes])),
        "mean_magnet_remote_warmth": float(np.mean([e.get("magnet_remote_warmth", 0) for e in episodes])),
        "mean_blocked_move_frames": mean("blocked_move_frames"),
        "blocked_move_frame_rate": (
            sum(e["blocked_move_frames"] for e in episodes)
            / sum(e["physical_frames"] for e in episodes)
        ),
        "quest_rate": mean("quest_collected"),
        "invincibility_acquire_rate": float(np.mean([e.get('invincibility_acquired',False) for e in episodes])),
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
        "mean_stagnant_frames": float(np.mean([e.get("stagnant_frames", 0) for e in episodes])),
        "stagnation_frame_rate": (
            sum(e.get("stagnant_frames", 0) for e in episodes)
            / sum(e["physical_frames"] for e in episodes)
        ),
        "local_loop_rate": float(np.mean([e.get("local_loop_frames", 0) > 0 for e in episodes])),
    }


def primary_evaluation_mode(report=None):
    mode = (report or {}).get("primary_mode", EVAL_CONFIG.get("primary_mode", "all"))
    if mode not in ("stochastic", "deterministic", "all"):
        raise ValueError("primary_mode는 stochastic, deterministic, all 중 하나여야 합니다.")
    return mode


def primary_success_rate(report):
    mode = primary_evaluation_mode(report)
    results = report["results"]
    if mode == "all":
        return min(result["summary"]["success_rate"] for result in results.values())
    return results[mode]["summary"]["success_rate"]


def evaluation_passed(report):
    threshold = report.get("success_rate_threshold", EVAL_CONFIG["success_rate_threshold"])
    target_stars = report.get("target_mean_stars", EVAL_CONFIG.get("target_mean_stars", 0.0))
    mode = primary_evaluation_mode(report)
    summaries = (
        [result["summary"] for result in report["results"].values()]
        if mode == "all" else [report["results"][mode]["summary"]]
    )
    return bool(
        not report.get("smoke", False)
        and not report.get("quick", False)
        and primary_success_rate(report) >= threshold
        and all(summary.get("mean_stars", 0.0) >= target_stars for summary in summaries)
        and all(summary.get("three_star_rate", 0.0) >= report.get("target_three_star_rate", 0.0) for summary in summaries)
        and all( summary.get("magnet_use_rate", 0.0)>= report.get("minimum_magnet_use_rate", 0.0)for summary in summaries)
        and all(summary.get("mean_damage", 0.0)<= report.get("maximum_mean_damage", 1e9)for summary in summaries)
        and all(summary.get("damage_free_success_rate", 0.0) >= report.get("minimum_damage_free_success_rate", 0.0) for summary in summaries)
        and (not report.get("skill_review") or report["skill_review"]["passed"])
    )


def evaluation_cases(env, smoke, quick=False):
    """기본 평가와 유령 초기 상태별 평가 조건을 만듭니다."""
    groups = []

    for mode, deterministic in (
        ("deterministic", True),
        ("stochastic", False),
    ):
        seeds = list(EVAL_CONFIG[f"{mode}_seeds"])

        if smoke:
            seeds = seeds[:2]
        elif quick:
            seeds = seeds[:EVAL_CONFIG[f"quick_{mode}_episodes"]]

        if not seeds:
            raise ValueError("평가 시드는 한 개 이상 필요합니다.")

        groups.append((
            mode,
            deterministic,
            [{"seed": seed, "options": None} for seed in seeds],
        ))

    if (
        ENV_CONFIG["stage"] in (2, 3)
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
        "stalled": max_still >= 120 or episode.get("local_loop_frames", 0) > 0,
        "final_x": float(snapshot["motion"]["position"][0]),
        "final_y": float(snapshot["motion"]["position"][1]),
        "action_counts": dict(counts),
        "decision_counts": dict(decision_counts),
        "failure_trace": (
            [] if episode["success"] else list(trace)
        ),
    }
def evaluate_model(model, *, smoke=False, quick=False):
    if getattr(model, "ppo_stage", ENV_CONFIG["stage"]) != ENV_CONFIG["stage"]:
        raise ValueError("모델의 Stage와 현재 평가 환경의 Stage가 다릅니다.")
    if (
        getattr(model, "ppo_observation_version", None)
        != PPO_OBSERVATION_VERSION
    ):
        raise ValueError("모델의 관측 버전이 다릅니다.")

    if model.device.type != "cpu":
        raise ValueError("현재 평가 코드는 CPU 모델을 사용합니다.")
    
    if getattr(model, "ppo_action_repeat", 1) != ACTION_REPEAT:
        raise ValueError("모델의 행동 반복 설정이 다릅니다.")

    # 기존 모델은 당시의 기본 수집 안내, 새 모델은 저장된 별 목표로 평가합니다.
    star_goal = deepcopy(getattr(model, "ppo_star_goal_config", {"enabled": False}))
    env = make_eval_env(
        star_goal=star_goal,
        mask_ineffective_interaction=getattr(model.policy, "mask_ineffective_interaction", False),
    )
    was_training = model.policy.training

    report = {
        "evaluation_version": "ppo_eval_primary_mode_v5",
        "primary_mode": primary_evaluation_mode(),
        "success_rate_threshold": EVAL_CONFIG["success_rate_threshold"],
        "target_mean_stars": EVAL_CONFIG.get("target_mean_stars", 0.0),
        "target_three_star_rate": EVAL_CONFIG.get("target_three_star_rate", 0.0),
        "selection_objective": EVAL_CONFIG.get("selection_objective", "success"),
        "selection_settings": deepcopy(EVAL_CONFIG.get("selection_settings", {})),
        "safe_return_selection": deepcopy(EVAL_CONFIG.get("safe_return_selection", {})),
        "star_goal": star_goal,
        "training_task": getattr(model, "ppo_training_task", "full_game"),
        "stagnation": deepcopy(STAGNATION_CONFIG),
        "observation_version": PPO_OBSERVATION_VERSION,
        "reward_version": REWARD_VERSION,
        "environment": deepcopy(ENV_CONFIG),
        "smoke": bool(smoke),
        "quick": bool(quick),
        "results": {},
        "evaluation_seeds": {},
        "action_repeat": ACTION_REPEAT,
        "action_mask_version": getattr(model, "ppo_action_mask_version", None),
        "minimum_magnet_use_rate": EVAL_CONFIG.get("minimum_magnet_use_rate", 0.0),
        "maximum_mean_damage": EVAL_CONFIG.get("maximum_mean_damage", 1e9),
        "minimum_damage_free_success_rate": EVAL_CONFIG.get("minimum_damage_free_success_rate", 0.0),
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
                env, smoke, quick=quick
            ):
                report["evaluation_seeds"][mode] = [case["seed"] for case in cases]
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

    report["primary_success_rate"] = primary_success_rate(report)
    review_settings = getattr(model, "ppo_joint_review_config", {})
    if review_settings.get("enabled", False):
        from rl.ppo_skill_review import evaluate_skill_review
        report["skill_review"] = evaluate_skill_review(model, review_settings, smoke=smoke)
    report["passed"] = evaluation_passed(report)

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
    parser.add_argument("--seed-base", type=int,
                        help="새 평가 시드의 시작 번호. 판수는 현재 평가 설정을 유지합니다.")
    parser.add_argument("--final-stage2", action="store_true", help="저장한 Stage 2 최종 기준으로 새 시드 200판 검증")
    parser.add_argument('--final-stage3',action='store_true',help='저장한 Stage 3 최종 기준으로 새 시드 200판 검증')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.final_stage2 and args.final_stage3:
        parser.error('최종 평가 스테이지는 하나만 선택하세요.')
    if (args.final_stage2 or args.final_stage3) and (args.seed_base is not None or args.smoke or args.quick):
        parser.error("--final-stage2는 저장한 평가 조건을 사용하므로 시드·실험 모드 옵션과 함께 사용할 수 없습니다.")

    if args.seed_base is not None:
        count = max(len(EVAL_CONFIG[f"{mode}_seeds"]) for mode in ("deterministic", "stochastic"))
        if args.seed_base < 0 or args.seed_base + count > 2**31:
            parser.error("평가 시드는 0 이상 2**31 미만이어야 합니다.")
        for mode in ("deterministic", "stochastic"):
            count = len(EVAL_CONFIG[f"{mode}_seeds"])
            EVAL_CONFIG[f"{mode}_seeds"] = list(range(args.seed_base, args.seed_base + count))

    torch.set_num_threads(TRAIN_CONFIG["torch_threads"])

    stamp = datetime.now(
        timezone(timedelta(hours=9))
    ).strftime("%Y%m%d_%H%M%S_%f")

    output_dir = args.output_dir or (
        args.model.parent
        / "evaluation"
        / f"{args.model.stem}_{stamp}"
    )

    if args.final_stage2 or args.final_stage3:
        from rl.ppo_safe_return import verify_stage_model
        report = verify_stage_model(args.model, output_dir,2 if args.final_stage2 else 3)
    else:
        model = PPO.load(args.model, device="cpu")
        report = evaluate_model(model, smoke=args.smoke, quick=args.quick)
        save_evaluation(report, output_dir)

    parts = [
        f"{mode} 성공 {result['summary']['success_rate']:.1%}, "
        f"평균 별 {result['summary']['mean_stars']:.2f}, "
        f"별 3개 {result['summary']['three_star_rate']:.1%}, "
        f"{format_post_quest(result['summary'])}, "
        f"벽막힘 {result['summary']['blocked_move_frame_rate']:.1%}, "
        f"정체 {result['summary']['stagnation_frame_rate']:.1%}, "
        + (f"무적 획득 {result['summary'].get('invincibility_acquire_rate',0):.1%}, "
           if report['environment']['stage']==3 else f"자석 활용 {result['summary']['magnet_use_rate']:.1%}, ")
        + f"무피해 귀환 {result['summary']['damage_free_success_rate']:.1%}"
        for mode, result in report["results"].items()
    ]

    print("평가 | " + " | ".join(parts))
    if report.get("skill_review"):
        from rl.ppo_skill_review import format_skill_review
        print(format_skill_review(report["skill_review"]))
    print(
        f"주 평가 기준: {report['primary_mode']} | "
        f"성공률 {report['primary_success_rate']:.1%} | "
        f"통과 기준 {report['success_rate_threshold']:.1%} | "
        f"평균 별 목표 {report['target_mean_stars']:.2f}"
    )
    print(f"통과: {report['passed']} | 저장: {output_dir}")


if __name__ == "__main__":
    main()
