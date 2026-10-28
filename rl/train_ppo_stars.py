"""성공 PPO 모델을 불러와 별 수집을 추가 학습한다. 기존 파일은 수정하지 않는다."""
import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import time

import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.ppo_stars_common import (
    ROOT, build_config, make_env, load_initial_model, runtime_metadata,
    training_targets, write_json, EXPLORATION_PRESETS,
)
from rl.train_ppo import TrainingLog, KST
from rl.evaluate_ppo_stars import evaluate_both, log_evaluation


class StarTrainingLog(TrainingLog):
    def __init__(self, directory, *, warmth_goal, source_steps):
        super().__init__(directory)
        self.warmth_goal = warmth_goal
        self.source_steps = source_steps
        self.star1 = self.star2 = self.star3 = self.goal_met = self.early_returns = 0
        self.warmth_total = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            episode = info.get("rl_episode")
            if episode is None:
                continue
            success = bool(episode["success"])
            star1 = success and episode["stars"] >= 1
            goal_met = episode["warmth"] >= self.warmth_goal
            early = success and not goal_met
            self.star1 += int(star1)
            self.star2 += int(success and episode["stars"] >= 2)
            self.star3 += int(success and episode["stars"] >= 3)
            self.goal_met += int(goal_met)
            self.early_returns += int(early)
            self.warmth_total += episode["warmth"]
            info["rl_episode"] = {
                **episode, "star1": star1, "warmth_goal_met": goal_met,
                "early_return": early,
            }
            # 학습용 시작 상태가 없으므로 모든 에피소드는 원래 시작점(0)이다.
            info["curriculum_start"] = 0
            info["curriculum_frontier"] = 0
        previous_progress = self.last_progress
        result = super()._on_step()
        if self.last_progress != previous_progress:
            print(f"별 수집: 별1+ {self.star1}/{self.episodes}, "
                  f"목표 온기 충족 {self.goal_met}, 조기 귀환 {self.early_returns}", flush=True)
        return result

    def summary(self, timesteps):
        return {
            **super().summary(timesteps),
            "cumulative_timesteps": self.source_steps + timesteps,
            "curriculum": None, "warmth_goal": self.warmth_goal,
            "at_least_one_star_count": self.star1,
            "at_least_two_star_count": self.star2,
            "three_star_count": self.star3,
            "warmth_goal_met_count": self.goal_met,
            "early_return_count": self.early_returns,
            "mean_warmth": self.warmth_total / self.episodes if self.episodes else None,
        }


def train(seed=11, *, smoke=False, initial_model=None, output_root=None, config=None,
          exploration=None):
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed는 0 이상 2**31 미만의 정수여야 합니다.")
    if config is not None and (initial_model is not None or exploration is not None):
        raise ValueError("config와 initial_model/exploration을 동시에 전달하지 마세요.")
    config = build_config(initial_model, exploration=exploration) if config is None else deepcopy(config)
    targets = training_targets(config, smoke=smoke)
    ppo = config["ppo"]
    torch.set_num_threads(ppo["torch_threads"])
    torch.use_deterministic_algorithms(True)
    env = DummyVecEnv([lambda: Monitor(make_env(config, training=True)) for _ in range(ppo["n_envs"])])
    model = logger = callback = None
    run_dir = None
    metadata = report = None
    started = time.perf_counter()
    training_seconds = evaluation_seconds = 0.0
    try:
        # 여기서 실제 가중치와 Adam 상태를 불러온다. 새 PPO 모델을 만들지 않는다.
        model, source = load_initial_model(config, env, seed)
        config["execution"] = {
            "run_kind": "smoke" if smoke else "full",
            "training_mode": "finetune", "timesteps_unit": "additional_environment_steps",
        }
        root = Path(output_root) if output_root else ROOT / "runs" / (
            "ppo_star_smoke" if smoke else config["experiment"]
        )
        now = datetime.now(KST)
        run_dir = root / f"seed_{seed}_{now:%Y%m%d_%H%M%S_%f}"
        run_dir.mkdir(parents=True, exist_ok=False)
        for name in ("train", "checkpoints", "evaluation"):
            (run_dir / name).mkdir()
        write_json(run_dir / "config.json", config)
        provenance = runtime_metadata(config)
        metadata = {
            **provenance, "training_seed": seed, "started_at": now.isoformat(),
            "status": "running", "completed_timesteps": 0, "total_timesteps": targets[-1],
            "cumulative_completed_timesteps": source["source_cumulative_timesteps"],
            "initial_model": source, "checkpoint_results": {},
        }
        report = {
            "schema_version": 1, "config": config["evaluation"], "training_seed": seed,
            "status": "running", "evaluator": provenance,
        }
        write_json(run_dir / "metadata.json", metadata)
        write_json(run_dir / "evaluation" / "evaluation.json", report)
        print(f"결과 폴더: {run_dir}\n원본 모델: {source['path']}\n"
              f"추가 학습: {targets[-1]:,}스텝 / 온기 목표 {config['environment']['warmth_goal']}\n"
              f"탐색 조건: {config.get('exploration', 'legacy')} / ent_coef={ppo['ent_coef']}", flush=True)
        callback = StarTrainingLog(
            run_dir / "train", warmth_goal=config["environment"]["warmth_goal"],
            source_steps=source["source_cumulative_timesteps"],
        )
        logger = configure(str(run_dir / "train"), ["csv", "tensorboard"])
        model.set_logger(logger)

        def evaluate_checkpoint(path, steps):
            nonlocal evaluation_seconds
            start = time.perf_counter()
            try:
                summaries = evaluate_both(
                    model, run_dir / "evaluation" / f"step_{steps:07d}",
                    training_seed=seed, model_steps=steps, smoke=smoke,
                    model_path=str(path.resolve()), config=config, record_scalar=logger.record,
                )
                log_evaluation(summaries, logger.record)
                logger.dump(step=steps)
                metadata["checkpoint_results"][str(steps)] = {
                    s["evaluation_mode"]: {key: s[key] for key in (
                        "success_rate", "at_least_one_star_rate", "mean_score", "mean_warmth",
                        "warmth_goal_rate", "early_return_rate", "star_training_pass",
                    )} for s in summaries
                }
            finally:
                evaluation_seconds += time.perf_counter() - start
            write_json(run_dir / "metadata.json", metadata)

        initial = run_dir / "checkpoints" / "model_0000000.zip"
        model.save(initial)
        evaluate_checkpoint(initial, 0)
        for target in targets:
            start = time.perf_counter()
            try:
                model.learn(target - model.num_timesteps, reset_num_timesteps=False,
                            callback=callback, log_interval=100)
            finally:
                training_seconds += time.perf_counter() - start
            checkpoint = run_dir / "checkpoints" / f"model_{model.num_timesteps:07d}.zip"
            model.save(checkpoint)
            metadata.update(
                completed_timesteps=int(model.num_timesteps), training_seconds=training_seconds,
                cumulative_completed_timesteps=source["source_cumulative_timesteps"] + int(model.num_timesteps),
            )
            write_json(run_dir / "metadata.json", metadata)
            evaluate_checkpoint(checkpoint, int(model.num_timesteps))
        model.save(run_dir / "final_model.zip")
        metadata["status"] = "complete"
    except BaseException as exc:
        if metadata is not None:
            metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                            error=f"{type(exc).__name__}: {exc}")
            if model is not None:
                model.save(run_dir / "interrupted_model.zip")
        raise
    finally:
        env.close()
        if callback is not None:
            callback.close()
        if logger is not None:
            logger.close()
        if metadata is not None:
            steps = int(model.num_timesteps) if model is not None else 0
            summary = callback.summary(steps) if callback is not None else {}
            metadata.update(
                completed_timesteps=steps, training_episodes=summary,
                cumulative_completed_timesteps=metadata["initial_model"]["source_cumulative_timesteps"] + steps,
                training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                elapsed_seconds=time.perf_counter() - started,
                finished_at=datetime.now(KST).isoformat(),
            )
            report["status"] = metadata["status"]
            write_json(run_dir / "metadata.json", metadata)
            write_json(run_dir / "train" / "progress.json", summary)
            write_json(run_dir / "evaluation" / "evaluation.json", report)
    print(f"완료: {run_dir}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="성공한 PPO 모델을 불러와 별 수집을 추가 학습합니다.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--init-model", type=Path, help="기본 성공 모델 대신 사용할 PPO 모델")
    parser.add_argument("--exploration", choices=tuple(EXPLORATION_PRESETS),
                        help="탐색 강도 비교 조건. 기본값 entropy03")
    parser.add_argument("--show-config", action="store_true",
                        help="설정만 출력하고 종료. 학습과 결과 폴더 생성 없음")
    args = parser.parse_args()
    if args.show_config:
        print(json.dumps(build_config(args.init_model, exploration=args.exploration),
                         ensure_ascii=False, indent=2))
        return
    train(args.seed, smoke=args.smoke, initial_model=args.init_model,
          exploration=args.exploration)


if __name__ == "__main__":
    main()
