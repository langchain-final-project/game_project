# Stage 1 공통 PPO를 seed만 달리 학습하고 체크포인트·그래프·평가·실행 정보를 저장합니다.
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from rl.ppo_common import (ROOT, build_config, make_env,
                       runtime_metadata, training_targets, write_json)
from rl.evaluate_ppo import evaluate_both, log_evaluation

# Windows에 별도 시간대 데이터가 없어도 한국 시간으로 실행 시각을 기록합니다.
KST = timezone(timedelta(hours=9))


class TrainingLog(BaseCallback):
    """벡터 환경의 자동 reset 전에 info에 담긴 종료 통계를 CSV와 그래프에 남깁니다."""

    def __init__(self, directory):
        super().__init__()
        self.directory = Path(directory)
        self.handle = None
        self.writer = None
        self.last_progress = 0
        self.episodes = self.successes = self.full_episodes = self.full_successes = 0
        self.frontiers = {}

    def _on_training_start(self):
        if self.handle is None:
            self.handle = (self.directory / "episodes.csv").open("w", newline="", encoding="utf-8-sig")

    def _on_step(self):
        # num_timesteps는 환경 4개 합계입니다. 콜백 호출 횟수와 혼동하지 않습니다.
        for rank, info in enumerate(self.locals["infos"]):
            episode = info.get("rl_episode")
            if episode is None:
                continue
            row = {"total_timesteps": self.num_timesteps, "env_rank": rank, **episode, "curriculum_start": info.get("curriculum_start", -1),
                "curriculum_frontier": info.get("curriculum_frontier", -1),}
            if self.writer is None:
                self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.handle.flush()
            self.episodes += 1
            self.successes += int(episode["success"])
            start = info.get("curriculum_start", 0)
            if start == 0:
                self.full_episodes += 1
                self.full_successes += int(episode["success"])
            self.frontiers[rank] = info.get("curriculum_frontier", -1)
            for name in ("success", "score", "perfect", "quest_collected", "warmth", "blocked"):
                self.logger.record_mean(f"episode/{name}", float(episode[name]))
            self.logger.record_mean("episode/three_stars", float(episode["stars"] == 3))
        if self.num_timesteps - self.last_progress >= 10000:
            print(
                f"PPO 학습 {self.num_timesteps:,} 스텝 | "
                f"학습 성공 {self.successes}/{self.episodes} | "
                f"원래 시작점 성공 {self.full_successes}/{self.full_episodes} | "
                f"환경별 단계 {self.frontiers}", flush=True,
            )
            write_json(self.directory / "progress.json", self.summary(self.num_timesteps))
            self.last_progress = self.num_timesteps
        return True

    def _on_rollout_end(self):
        # SB3의 entropy_loss는 음수 엔트로피이므로, 이해하기 쉬운 양수 그래프도 제공합니다.
        loss = self.logger.name_to_value.get("train/entropy_loss")
        if loss is not None:
            self.logger.record("train/policy_entropy", -float(loss))

    def close(self):
        if self.handle is not None:
            self.handle.close()

    def summary(self, timesteps):
        return {
            "completed_timesteps": timesteps,
            "episodes": self.episodes, "successes": self.successes,
            "full_start_episodes": self.full_episodes,
            "full_start_successes": self.full_successes,
            "frontiers": self.frontiers,
        }


def _prepare_run(config, seed, targets, output_root):
    """학습 로직과 무관한 결과 폴더와 최초 실행 기록을 준비합니다."""
    kind = config['execution']['run_kind']
    root = Path(output_root) if output_root else ROOT / 'runs' / ('ppo_smoke' if kind == 'smoke' else config['experiment'])
    now = datetime.now(KST)
    run_dir = root / f"seed_{seed}_{now:%Y%m%d_%H%M%S_%f}"
    run_dir.mkdir(parents=True, exist_ok=False)
    for name in ("train", "checkpoints", "evaluation"):
        (run_dir / name).mkdir()
    write_json(run_dir / "config.json", config)
    provenance = runtime_metadata(config)
    metadata = {**provenance, "training_seed": seed,
                "started_at": now.isoformat(), "status": "running", "completed_timesteps": 0,
                "total_timesteps": targets[-1]}
    write_json(run_dir / "metadata.json", metadata)
    evaluation_report = {"schema_version": 1, "config": config["evaluation"], "evaluator": provenance,
                         "training_seed": seed, "status": "running"}
    write_json(run_dir / "evaluation" / "evaluation.json", evaluation_report)
    print(f"결과 폴더: {run_dir}\n종류: {kind} / seed={seed}", flush=True)
    return run_dir, metadata, evaluation_report


def _finish_run(run_dir, metadata, evaluation_report, *, started, training_seconds, evaluation_seconds):
    """완료·중단·오류 모두 같은 형식으로 시간과 상태를 기록합니다."""
    metadata.update(elapsed_seconds=time.perf_counter() - started,
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
                    finished_at=datetime.now(KST).isoformat())
    write_json(run_dir / "metadata.json", metadata)
    evaluation_report["status"] = metadata["status"]
    write_json(run_dir / "evaluation" / "evaluation.json", evaluation_report)


def train(seed: int, *, smoke: bool = False, output_root: Path | None = None, config=None) -> Path:
    """공통 설정 → 환경·모델 생성 → 구간별 학습·평가 → 최종 저장 순서로 실행합니다."""
    if not 0 <= seed < 2**31:
        raise ValueError("seed는 0 이상 2**31 미만의 정수여야 합니다.")
    config = build_config() if config is None else deepcopy(config)
    if config['algorithm'] != 'PPO':
        raise ValueError('이 파일은 PPO 학습 전용입니다.')
    ppo = config['ppo']
    targets = training_targets(config, smoke=smoke)
    # 저장 시점·평가 횟수는 공통 설정에서 계산하므로 실행 정보에는 종류만 따로 남깁니다.
    config['execution'] = {'run_kind': 'smoke' if smoke else 'full'}
    run_dir, metadata, evaluation_report = _prepare_run(config, seed, targets, output_root)

    # CPU 스레드 수도 같게 맞춥니다. SB3 seed가 모델 초기화·행동 샘플링과 환경 seed를 설정합니다.
    env = None
    callback = TrainingLog(run_dir / "train")
    model = None
    logger = None
    started = time.perf_counter()
    training_seconds = evaluation_seconds = 0.0
    try:
        torch.set_num_threads(ppo["torch_threads"])
        torch.use_deterministic_algorithms(True)
        env = DummyVecEnv([lambda: Monitor(make_env(config, training=True)) for _ in range(ppo["n_envs"])])
        activations = {"Tanh": torch.nn.Tanh, "ReLU": torch.nn.ReLU}
        if ppo["activation"] not in activations:
            raise ValueError('지원하는 활성화 함수는 Tanh 또는 ReLU입니다.')
        options = {key: ppo[key] for key in (
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
            "clip_range", "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef",
            "max_grad_norm", "target_kl", "device")}
        model = PPO(ppo["policy"], env, seed=seed, verbose=0,
                    policy_kwargs={"net_arch": ppo["net_arch"],
                                   "activation_fn": activations[ppo["activation"]]}, **options)
        logger = configure(str(run_dir / "train"), ["csv", "tensorboard"])
        model.set_logger(logger)

        def evaluate_checkpoint(path, steps):
            nonlocal evaluation_seconds
            evaluation_start = time.perf_counter()
            summaries = evaluate_both(model, run_dir / "evaluation" / f"step_{steps:07d}",
                                      training_seed=seed, model_steps=steps, smoke=smoke,
                                      model_path=str(path.resolve()), config=config,
                                      record_scalar=logger.record)
            log_evaluation(summaries, logger.record)
            logger.dump(step=steps)
            evaluation_seconds += time.perf_counter() - evaluation_start

        initial = run_dir / "checkpoints" / "model_0000000.zip"
        model.save(initial)
        evaluate_checkpoint(initial, 0)
        for target in targets:
            training_start = time.perf_counter()
            # n_envs × n_steps의 배수에서 끝냅니다.
            # reset_num_timesteps=False로 누적 스텝·환경 상태·모델을 이어갑니다. 학습률은 고정입니다.
            try:
                model.learn(total_timesteps=target - model.num_timesteps,
                            reset_num_timesteps=False, callback=callback, log_interval=100)
            finally:
                # 사용자가 중간에 멈춰도 진행 중이던 학습 구간의 시간을 누락하지 않습니다.
                training_seconds += time.perf_counter() - training_start
            checkpoint = run_dir / "checkpoints" / f"model_{model.num_timesteps:07d}.zip"
            model.save(checkpoint)
            metadata.update(completed_timesteps=model.num_timesteps, training_seconds=training_seconds)
            write_json(run_dir / "metadata.json", metadata)
            evaluate_checkpoint(checkpoint, model.num_timesteps)
        model.save(run_dir / "final_model.zip")
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        if model is not None:
            metadata["completed_timesteps"] = model.num_timesteps
            model.save(run_dir / "interrupted_model.zip")
        raise
    finally:
        step = int(model.num_timesteps) if model is not None else 0
        metadata["training_episodes"] = callback.summary(step)
        write_json(run_dir / "train" / "progress.json", callback.summary(step))
        callback.close()
        if env is not None:
            env.close()
        if logger is not None:
            logger.close()
        _finish_run(run_dir, metadata, evaluation_report, started=started,
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds)
    print(f"완료: {run_dir}", flush=True)
    print(f"학습 성공 {callback.successes}/{callback.episodes}, "
          f"원래 시작점 성공 {callback.full_successes}/{callback.full_episodes}", flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="PPO 전용 설정·보상으로 Stage 1을 학습합니다.")
    parser.add_argument("--seed", type=int, required=True, help="학습 seed 예: 11, 22, 33, 44")
    parser.add_argument("--smoke", action="store_true", help="본 실험과 분리한 2,000스텝 연결 점검")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**31:
        parser.error("seed는 0 이상 2**31 미만이어야 합니다.")
    train(args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()
