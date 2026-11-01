"""자석·물건 연결 PPO 연습. 전체 별 3개 평가는 별도로 수행합니다."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import ppo_config as cfg
from rl.evaluate_ppo import isolated_rng
from rl.ppo_connection_task import make_connection_env, connection_probabilities
from rl.ppo_observation import PPO_OBSERVATION_VERSION
from rl.ppo_skill_review import quest_regression_status
from rl.train_ppo import make_model, write_json
from rl.train_ppo_quest import evaluate_quest, train_partial

TASK_VERSION = "stage2_magnet_quest_connection_v1"


def setup(seed, settings=None):
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG if settings is None else settings)
    n = cfg.TRAIN_CONFIG["n_envs"]
    if cfg.ENV_CONFIG["stage"] != 2 or n < 4 or n % 4:
        raise ValueError("Stage 2 연결 과제는 4의 배수인 환경 수가 필요합니다.")
    if tuple(settings["starts"]) != (0, 1, 2, 3, 4, 5):
        raise ValueError("초반 수집 연결 과제는 시작 0~5번을 모두 포함해야 합니다.")
    if settings["frame_limit"] <= 0 or settings["evaluation_interval"] <= 0:
        raise ValueError("프레임 제한·평가 간격은 양수여야 합니다.")
    connection_probabilities(settings["starts"], settings)
    focus_starts = settings.get("focus_starts", [])
    if (any(start not in settings["starts"] for start in focus_starts)
            or not 0 < settings.get("focus_success_rate_threshold", settings["success_rate_threshold"]) <= 1):
        raise ValueError("집중 복습 시작 번호와 목표 성공률을 확인하세요.")
    if set(settings["evaluation_seeds"]) & set(settings["heldout_seeds"]):
        raise ValueError("학습 중 평가 시드와 최종 검증 시드는 겹치면 안 됩니다.")
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    settings["goals"] = deepcopy(cfg.STAR_GOAL_CONFIG)
    groups = [settings["starts"], settings["starts"], [6], [7]] * (n // 4)
    env = DummyVecEnv([lambda starts=starts: make_connection_env(starts, settings, settings["goals"])
                       for starts in groups])
    env.seed(seed)
    try:
        ppo = {**cfg.PPO_CONFIG, "learning_rate": settings["learning_rate"], "ent_coef": settings["ent_coef"]}
        settings["gamma"] = ppo["gamma"]
        model = make_model(env, seed, ppo_settings=ppo, star_goal_settings=settings["goals"],
                           transfer_settings={"enabled": True, "model_path": Path(settings["source_model"]),
                                              "action_head_scale": 1.0})
        model.ppo_training_task = TASK_VERSION
        model.ppo_connection_settings = deepcopy(settings)
        return env, model, settings
    except BaseException:
        env.close()
        raise


def evaluate_connection(model, *, seeds=None):
    if getattr(model, "ppo_training_task", None) != TASK_VERSION:
        raise ValueError("자석·물건 연결 과제 모델이 필요합니다.")
    if (getattr(model, "ppo_stage", None) != 2
            or getattr(model, "ppo_observation_version", None) != PPO_OBSERVATION_VERSION
            or getattr(model, "ppo_action_repeat", None) != cfg.ACTION_REPEAT):
        raise ValueError("연결 과제의 Stage·관측·행동 반복 설정이 다릅니다.")
    settings = deepcopy(model.ppo_connection_settings)
    seeds = list(settings["evaluation_seeds"] if seeds is None else seeds)
    if not seeds:
        raise ValueError("평가 시드가 필요합니다.")
    results = {}
    was_training = model.policy.training
    try:
        with isolated_rng():
            for start in settings["starts"]:
                env = make_connection_env([start], settings, settings["goals"])
                episodes = []
                try:
                    for seed in seeds:
                        torch.manual_seed(50_000 + seed + start * 1000)
                        obs, _ = env.reset(seed=seed)
                        while True:
                            action, _ = model.predict(obs, deterministic=False)
                            obs, _, done, truncated, info = env.step(int(np.asarray(action).item()))
                            if done or truncated:
                                break
                        episodes.append({**info["connection_episode"], "seed": int(seed)})
                finally:
                    env.close()
                results[str(start)] = {"episodes": episodes, "summary": {
                    "episodes": len(episodes), "success_rate": float(np.mean([e["success"] for e in episodes])),
                    "mean_frames": float(np.mean([e["frames"] for e in episodes]))}}
            quest_settings = deepcopy(cfg.QUEST_TASK_CONFIG)
            quest_settings["goals"] = settings["goals"]
            quest = evaluate_quest(model, seeds=seeds, task_settings=quest_settings)
    finally:
        model.policy.set_training_mode(was_training)
    return connection_report(results, quest["results"]["stochastic"], settings, seeds)


def connection_report(results, quest_review, settings, seeds):
    """모든 연결 구간과 집중 복습 구간의 개별 목표를 함께 확인합니다."""
    minimum = min(v["summary"]["success_rate"] for v in results.values())
    qmin = min(v["summary"]["success_rate"] for v in quest_review.values())
    focus = settings.get("focus_starts", [])
    thresholds = {start: (settings.get("focus_success_rate_threshold", settings["success_rate_threshold"])
                          if int(start) in focus else settings["success_rate_threshold"])
                  for start in results}
    ratios = [results[start]["summary"]["success_rate"] / threshold for start, threshold in thresholds.items()]
    focus_minimum = min((results[str(start)]["summary"]["success_rate"] for start in focus), default=minimum)
    return {"task": TASK_VERSION, "full_game_evaluation": False, "seeds": seeds,
            "results": {"stochastic": results}, "quest_review": quest_review,
            "connection_thresholds": thresholds, "focus_minimum_rate": focus_minimum,
            "connection_minimum_rate": minimum, "quest_minimum_rate": qmin,
            "minimum_target_ratio": min(*ratios, qmin / settings["quest_target"]),
            "quest_retained": qmin >= settings["quest_safety_floor"],
            "passed": all(ratio >= 1.0 for ratio in ratios) and qmin >= settings["quest_target"]}


def selection_key(report):
    return (int(report["quest_retained"]), int(report["passed"]),
            report.get("minimum_target_ratio", min(report["connection_minimum_rate"], report["quest_minimum_rate"])),
            float(np.mean([v["summary"]["success_rate"] for v in report["results"]["stochastic"].values()])))


def print_report(report, label):
    starts = " | ".join(f"{start}번 연결 {v['summary']['success_rate']:.0%}"
                        for start, v in report["results"]["stochastic"].items())
    print(f"[{label}] {starts} | 물건 6·7번 최저 {report['quest_minimum_rate']:.0%} | 부분 통과 {report['passed']}", flush=True)


def require_connection_ready(path):
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    model = PPO.load(path, device="cpu")
    if getattr(model, "ppo_training_task", None) != TASK_VERSION:
        raise ValueError("자석·물건 연결 과제 체크포인트가 필요합니다.")
    report = evaluate_connection(model, seeds=model.ppo_connection_settings["heldout_seeds"])
    print_report(report, "전체 복귀 전 새 시드 연결 검증")
    if not report["passed"]:
        raise ValueError("0~5번 연결의 구간별 목표와 6·7번 물건 각각 90%를 통과해야 합니다.")
    return report


def check_setup(seed):
    env, model, _ = setup(seed)
    try:
        obs = env.reset()
        actions, _ = model.predict(obs, deterministic=False)
        env.step(actions)
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        print(f"연결 정상 | 관측 {obs.shape} | 초반 수집 환경 절반, 물건 6·7번 복습 환경 절반")
        settings = model.ppo_connection_settings
        print(f"초반 시작 비중: {settings.get('start_weights', '균등')} | "
              f"최소 추가 학습 {settings.get('minimum_training_frames', 0):,}F | "
              f"연속 통과 {settings.get('required_consecutive_passes', 1)}회")
        print("학습·모델 저장 없음. 자석·물건 연결 과제이며 별 3개 전체 평가와 구분합니다.")
    finally:
        env.close()


def train(seed, *, quick=False):
    count = 0
    def stop_check(report):
        nonlocal count
        count, stop = quest_regression_status(count, report, cfg.CONNECTION_TASK_CONFIG["stop_after_quest_regressions"])
        if stop:
            print("물건 획득 성능이 두 번 연속 하락해 부분 과제를 중단합니다.", flush=True)
        return stop
    return train_partial(seed, quick=quick, setup_fn=setup, evaluate_fn=evaluate_connection,
                         selection_fn=selection_key, print_fn=print_report, task_version=TASK_VERSION,
                         episode_field="connection_episode", stop_check=stop_check)


def main():
    parser = argparse.ArgumentParser(description="Stage 2 자석·물건 연결 PPO")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluate", type=Path)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("seed 범위를 확인하세요.")
    if (args.heldout or args.output_dir) and not args.evaluate:
        parser.error("--heldout·--output-dir는 --evaluate와 함께 사용하세요.")
    if args.evaluate:
        torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
        model = PPO.load(args.evaluate, device="cpu")
        report = evaluate_connection(model, seeds=model.ppo_connection_settings["heldout_seeds"] if args.heldout else None)
        print_report(report, "연결 과제 평가")
        out = args.output_dir or args.evaluate.parent.parent / "connection_evaluation" / ("heldout" if args.heldout else "regular")
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "evaluation.json", report)
        print(f"저장: {out}")
    elif args.check:
        check_setup(args.seed)
    else:
        train(args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
