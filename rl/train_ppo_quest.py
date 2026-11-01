"""물건 획득 부분 과제의 PPO 학습과 구간별 평가. 전체 별 평가와 분리합니다."""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from rl import ppo_config as cfg
from rl.evaluate_ppo import isolated_rng
from rl.ppo_quest_task import make_quest_env, quest_goals
from rl.train_ppo import make_model, write_json


TASK_VERSION = "stage2_quest_acquisition_v1"


def setup(seed, settings=None):
    settings = deepcopy(cfg.QUEST_TASK_CONFIG if settings is None else settings)
    if cfg.ENV_CONFIG["stage"] != 2 or tuple(settings["starts"]) != (6, 7):
        raise ValueError("Stage 2의 6·7번 부분 과제 설정이 필요합니다.")
    if not 0 < settings["action_head_scale"] <= 1:
        raise ValueError("행동 출력 배율은 0 초과 1 이하여야 합니다.")
    if (settings["frame_limit"] <= 0 or settings["frame_limit"] % cfg.ACTION_REPEAT
            or settings["evaluation_interval"] <= 0):
        raise ValueError("부분 과제 제한·평가 간격을 확인하세요.")
    if not Path(settings["source_model"]).is_file():
        raise FileNotFoundError(settings["source_model"])
    n_envs = cfg.TRAIN_CONFIG["n_envs"]
    if n_envs < 2 or n_envs % 2:
        raise ValueError("6·7번에 같은 수의 환경을 배정하려면 짝수 환경 수가 필요합니다.")
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    settings["goals"] = quest_goals()
    starts = [settings["starts"][i % 2] for i in range(n_envs)]
    env = DummyVecEnv([lambda start=start: make_quest_env(start, settings) for start in starts])
    env.seed(seed)
    try:
        ppo = deepcopy(cfg.PPO_CONFIG)
        ppo.update(learning_rate=settings["learning_rate"], ent_coef=settings["ent_coef"],
                   gamma=settings["gamma"])
        model = make_model(env, seed, ppo_settings=ppo, star_goal_settings=settings["goals"],
                           transfer_settings={"enabled": True, "model_path": settings["source_model"],
                                              "action_head_scale": settings["action_head_scale"]})
        model.ppo_training_task = TASK_VERSION
        model.ppo_reward_version = TASK_VERSION
        model.ppo_quest_task_settings = deepcopy(settings)
        # 전체 게임의 큰 보상 척도를 부분 과제 가치 추정에 그대로 사용하지 않습니다.
        if settings.get("reset_value_head", False):
            with torch.no_grad():
                model.policy.value_net.weight.zero_()
                model.policy.value_net.bias.zero_()
        return env, model, settings
    except BaseException:
        env.close()
        raise


def evaluate_quest(model, *, seeds=None, task_settings=None):
    if getattr(model, "ppo_stage", None) != 2:
        raise ValueError("Stage 2 모델이 필요합니다.")
    if task_settings is None and getattr(model, "ppo_training_task", None) != TASK_VERSION:
        raise ValueError("물건 획득 부분 과제로 준비한 모델이 필요합니다.")
    settings = deepcopy(model.ppo_quest_task_settings if task_settings is None else task_settings)
    seeds = list(settings["evaluation_seeds"] if seeds is None else seeds)
    if not seeds:
        raise ValueError("평가 시드가 필요합니다.")
    report = {"task": TASK_VERSION, "full_game_evaluation": False,
              "success_rate_threshold": settings["success_rate_threshold"], "results": {}}
    was_training = model.policy.training
    try:
        with isolated_rng():
            for mode, deterministic in (("stochastic", False), ("deterministic", True)):
                report["results"][mode] = {}
                for start in (6, 7):
                    env = make_quest_env(start, settings)
                    episodes = []
                    initial_probabilities = None
                    try:
                        if env.observation_space != model.observation_space or env.action_space != model.action_space:
                            raise ValueError("부분 과제와 모델의 관측·행동 공간이 다릅니다.")
                        for seed in seeds if not deterministic else seeds[:3]:
                            torch.manual_seed(50_000 + seed + start * 1000)
                            obs, _ = env.reset(seed=seed)
                            if initial_probabilities is None:
                                tensor, _ = model.policy.obs_to_tensor(obs)
                                with torch.no_grad():
                                    initial_probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().tolist()
                            while True:
                                action, _ = model.predict(obs, deterministic=deterministic)
                                obs, _, done, truncated, info = env.step(int(np.asarray(action).item()))
                                if done or truncated:
                                    break
                            episodes.append({**info["quest_task_episode"], "seed": int(seed),
                                             "decisions": info["decision_steps"]})
                    finally:
                        env.close()
                    report["results"][mode][str(start)] = {
                        "summary": {"episodes": len(episodes),
                                    "success_rate": float(np.mean([e["success"] for e in episodes])),
                                    "mean_frames": float(np.mean([e["frames"] for e in episodes])),
                                    "mean_reward": float(np.mean([e["return"] for e in episodes]))},
                        "initial_action_probabilities": initial_probabilities,
                        "episodes": episodes,
                    }
    finally:
        model.policy.set_training_mode(was_training)
    report["passed"] = all(report["results"]["stochastic"][str(start)]["summary"]["success_rate"]
                           >= report["success_rate_threshold"] for start in (6, 7))
    return report


def selection_key(report):
    summaries = [report["results"]["stochastic"][str(start)]["summary"] for start in (6, 7)]
    return (min(s["success_rate"] for s in summaries),
            float(np.mean([s["success_rate"] for s in summaries])),
            -float(np.mean([s["mean_frames"] for s in summaries])))


def print_report(report, label):
    parts = []
    for start in (6, 7):
        s = report["results"]["stochastic"][str(start)]["summary"]
        parts.append(f"{start}번 획득 {s['success_rate']:.1%} ({s['episodes']}판)")
    print(f"[{label}] " + " | ".join(parts) + f" | 부분 과제 통과 {report['passed']}", flush=True)


def check_setup(seed):
    env, model, settings = setup(seed)
    try:
        obs = env.reset()
        actions, _ = model.predict(obs, deterministic=False)
        env.step(actions)
        assert obs.shape == (cfg.TRAIN_CONFIG["n_envs"], 640)
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        print(f"연결 정상 | 물건 획득 부분 과제 | 시작 6·7번 | 행동 출력 배율 {settings['action_head_scale']}")
        for start in (6, 7):
            probe = make_quest_env(start, settings)
            try:
                first, _ = probe.reset(seed=seed)
                tensor, _ = model.policy.obs_to_tensor(first)
                with torch.no_grad():
                    probabilities = model.policy.get_distribution(tensor).distribution.probs[0].cpu().tolist()
                print(f"{start}번 초기 행동 확률: 아래 {probabilities[1]:.1%}, 오른쪽 {probabilities[3]:.1%}")
            finally:
                probe.close()
        print("학습·파일 저장 없음. 전체 게임·별 성공률과 구분합니다.")
    finally:
        env.close()


class QuestEpisodes(BaseCallback):
    def __init__(self, path, episode_field="quest_task_episode"):
        super().__init__()
        self.frames = 0
        self.episodes = 0
        self.handle = path.open("w", encoding="utf-8-sig", newline="")
        self.writer = None
        self.episode_field = episode_field

    def _on_step(self):
        for info in self.locals["infos"]:
            self.frames += info["repeat_frames"]
            episode = info.get(self.episode_field)
            if episode is None:
                continue
            self.episodes += 1
            row = {"total_physical_frames": self.frames, "decision_timesteps": self.num_timesteps,
                   "decisions": info["decision_steps"], **episode}
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
        self.handle.flush()
        return True

    def close(self):
        self.handle.close()


def train(seed, *, quick=False):
    return train_partial(seed, quick=quick)


def partial_task_ready(report, physical_frames, consecutive_passes, settings):
    """출발 평가만으로 끝내지 않고 최소 학습량과 연속 통과를 확인합니다."""
    return bool(report and report["passed"]
                and physical_frames >= settings.get("minimum_training_frames", 0)
                and consecutive_passes >= settings.get("required_consecutive_passes", 1))


def train_partial(seed, *, quick=False, setup_fn=setup, evaluate_fn=evaluate_quest,
                  selection_fn=selection_key, print_fn=print_report,
                  task_version=TASK_VERSION, episode_field="quest_task_episode", stop_check=None):
    """부분 과제마다 다른 환경·평가를 사용하고 PPO 저장 루프는 공유합니다."""
    env, model, settings = setup_fn(seed)
    budget = settings["quick_timesteps"] if quick else settings["timesteps"]
    if type(budget) is not int or budget <= 0:
        env.close()
        raise ValueError("부분 과제의 추가 물리 프레임 예산을 확인하세요.")
    minimum = settings.get("minimum_training_frames", 0)
    required = settings.get("required_consecutive_passes", 1)
    if (type(minimum) is not int or not 0 <= minimum <= budget
            or type(required) is not int or required < 1):
        env.close()
        raise ValueError("최소 추가 학습량과 연속 통과 횟수를 확인하세요.")
    stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
    run = Path(settings["run_root"]) / ("quick" if quick else "train") / f"seed_{seed}_{stamp}"
    models = run / "models"
    models.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", {
        "algorithm": "PPO", "training_task": task_version, "full_game_training": False,
        "environment": deepcopy(cfg.ENV_CONFIG), "observation_version": model.ppo_observation_version,
        "star_goal": settings["goals"], "action_repeat": cfg.ACTION_REPEAT,
        "training": deepcopy(cfg.TRAIN_CONFIG), "quest_task": settings,
        "reward": deepcopy(cfg.REWARD_CONFIG),
        "network": deepcopy(cfg.NETWORK_CONFIG),
        "ppo": {**cfg.PPO_CONFIG, "learning_rate": settings["learning_rate"],
                "ent_coef": settings["ent_coef"], "gamma": settings["gamma"]},
        "planned_physical_frames": budget, "seed": seed,
    })
    callback = QuestEpisodes(run / "quest_episodes.csv", episode_field=episode_field)
    status, best_frames, best_report = "failed", 0, None
    consecutive_passes = 0
    try:
        baseline = evaluate_fn(model)
        print_fn(baseline, "학습 전 기준")
        (run / "evaluation").mkdir(exist_ok=True)
        write_json(run / "evaluation/step_0.json", baseline)
        model.save(models / "baseline_model.zip")
        model.save(models / "best_model.zip")
        best_report, best_key = baseline, selection_fn(baseline)
        consecutive_passes = int(baseline["passed"] and minimum == 0)
        next_evaluation = settings["evaluation_interval"]
        rollout = cfg.TRAIN_CONFIG["n_envs"] * model.n_steps
        print(f"부분 과제 저장: {run}", flush=True)
        while (callback.frames < budget
               and not partial_task_ready(best_report, callback.frames, consecutive_passes, settings)):
            model.learn(total_timesteps=rollout, reset_num_timesteps=False, callback=callback, log_interval=None)
            model.ppo_physical_frames = callback.frames
            if callback.frames < next_evaluation and callback.frames < budget:
                continue
            report = evaluate_fn(model)
            consecutive_passes = consecutive_passes + 1 if report["passed"] else 0
            print_fn(report, f"평가 {callback.frames:,}F")
            write_json(run / "evaluation" / f"step_{callback.frames}.json", report)
            model.save(models / f"ppo_{callback.frames}.zip")
            if selection_fn(report) > best_key:
                best_report, best_key, best_frames = report, selection_fn(report), callback.frames
                model.save(models / "best_model.zip")
            if stop_check is not None and stop_check(report):
                status = "quest_retention_stopped"
                break
            while next_evaluation <= callback.frames:
                next_evaluation += settings["evaluation_interval"]
        if status != "quest_retention_stopped":
            status = ("task_ready" if partial_task_ready(best_report, callback.frames, consecutive_passes, settings)
                      else "complete")
        model.save(models / "last_model.zip")
        shutil.copy2(models / "best_model.zip", models / "final_model.zip")
    except KeyboardInterrupt:
        status = "interrupted"
        model.save(models / "interrupted_model.zip")
        raise
    finally:
        write_json(run / "training_summary.json", {
            "status": status, "training_task": task_version, "full_game_evaluated": False,
            "physical_frames": callback.frames, "decision_timesteps": model.num_timesteps,
            "episodes": callback.episodes, "selected_physical_frames": best_frames,
            "task_passed": status == "task_ready",
            "selected_evaluation_passed": bool(best_report and best_report["passed"]),
            "minimum_training_frames": minimum,
            "required_consecutive_passes": required,
            "consecutive_passes": consecutive_passes,
            "selected_stochastic": best_report["results"]["stochastic"] if best_report else None,
            "selected_quest_review": best_report.get("quest_review") if best_report else None,
        })
        callback.close()
        env.close()
    print(f"부분 과제 종료: {status} | 결과: {run}")
    return run


def require_full_return_ready(path):
    """전체 과제로 돌아가기 전 두 시작점의 새 시드 성공률을 확인합니다."""
    if cfg.ENV_CONFIG["stage"] != 2:
        raise ValueError("부분 과제 모델의 전체 복귀는 Stage 2에서만 가능합니다.")
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    model = PPO.load(path, device="cpu")
    if getattr(model, "ppo_training_task", None) != TASK_VERSION:
        raise ValueError("물건 획득 부분 과제 체크포인트가 아닙니다.")
    report = evaluate_quest(model, seeds=model.ppo_quest_task_settings["heldout_seeds"])
    print_report(report, "전체 복귀 전 새 시드 검증")
    if not report["passed"]:
        raise ValueError("6·7번이 각각 기준을 통과해야 전체 과제로 복귀할 수 있습니다.")
    return report


def main():
    parser = argparse.ArgumentParser(description="Stage 2 물건 획득 전용 PPO")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluate", type=Path, help="부분 과제 모델 평가만 실행")
    parser.add_argument("--heldout", action="store_true", help="새 시드로 구간별 평가")
    parser.add_argument("--output-dir", type=Path, help="부분 과제 평가 결과 저장 위치")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("seed 범위를 확인하세요.")
    if (args.heldout or args.output_dir) and not args.evaluate:
        parser.error("--heldout·--output-dir는 --evaluate와 함께 사용하세요.")
    if args.evaluate:
        torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
        model = PPO.load(args.evaluate, device="cpu")
        seeds = model.ppo_quest_task_settings["heldout_seeds"] if args.heldout else None
        report = evaluate_quest(model, seeds=seeds)
        report["model"] = str(args.evaluate)
        print_report(report, "부분 과제 평가")
        stamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
        out = args.output_dir or args.evaluate.parent.parent / "quest_evaluation" / stamp
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "evaluation.json", report)
        print(f"부분 과제 평가 저장: {out}")
    elif args.check:
        check_setup(args.seed)
    else:
        train(args.seed, quick=args.quick)


if __name__ == "__main__":
    main()
