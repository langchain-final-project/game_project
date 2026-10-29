import argparse
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
import shutil
import sys

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.ppo_action_repeat import ActionRepeat

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import (
    Stage1RouteCurriculum,
    Stage2RouteCurriculum,
)
from rl.ppo_observation import (
    PPOGoalObservation,
    PPO_OBSERVATION_VERSION,
)
from rl.ppo_reward import PPOReward
from rl.evaluate_ppo import (
    evaluate_model,
    save_evaluation,
    isolated_rng,
)


CURRICULUMS = {
    1: Stage1RouteCurriculum,
    2: Stage2RouteCurriculum,
}


def training_targets(smoke):
    t, p = cfg.TRAIN_CONFIG, cfg.PPO_CONFIG

    positive = (
        t["n_envs"],
        t["torch_threads"],
        p["n_steps"],
        p["batch_size"],
        p["n_epochs"],
    )

    if any(type(v) is not int or v <= 0 for v in positive):
        raise ValueError(
            "환경·스레드·수집·배치·반복 횟수를 확인하세요."
        )

    if p["device"] != "cpu":
        raise ValueError("현재 코드는 CPU 설정을 사용합니다.")

    rollout = t["n_envs"] * p["n_steps"]

    if p["batch_size"] < 2 or rollout % p["batch_size"]:
        raise ValueError(
            "batch_size는 2 이상이며 rollout을 나누어야 합니다."
        )

    total = t[
        "smoke_timesteps" if smoke else "total_timesteps"
    ]

    targets = [total] if smoke else sorted(
        set([*t["checkpoint_steps"], total])
    )

    if any(
        type(n) is not int
        or n <= 0
        or n > total
        # or n % rollout
        for n in targets
    ):
        raise ValueError(
            "학습량과 저장 시점은 양의 물리 프레임 수여야 합니다."
        )

    if cfg.TRANSFER_CONFIG["enabled"]:
        path = cfg.TRANSFER_CONFIG["model_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        if not 0 < cfg.TRANSFER_CONFIG["action_head_scale"] <= 1:
            raise ValueError(
                "action_head_scale은 0 초과 1 이하여야 합니다."
            )

    s = cfg.STABILITY_CONFIG

    if (
        not 0 <= s["activation_success_rate"] <= 1
        or not 0 < s["max_success_drop"] <= 1
        or type(s["patience"]) is not int
        or s["patience"] < 1
        or s["ent_coef_after_success"] < 0
    ):
        raise ValueError("성능 하락 감시 설정을 확인하세요.")

    return targets


def make_env():
    if cfg.ENV_NAME != "RealtimeMansionEnv":
        raise ValueError("RealtimeMansionEnv가 필요합니다.")

    env = RealtimeMansionEnv(**cfg.ENV_CONFIG)

    if cfg.CURRICULUM_CONFIG["enabled"]:
        curriculum_class = CURRICULUMS[
            cfg.ENV_CONFIG["stage"]
        ]
        env = curriculum_class(env)

    env = PPOReward(PPOGoalObservation(env))
    return ActionRepeat(env, repeat=cfg.ACTION_REPEAT)


def make_model(env, seed):
    source = None

    if cfg.TRANSFER_CONFIG["enabled"]:
        with isolated_rng():
            source = PPO.load(
                cfg.TRANSFER_CONFIG["model_path"],
                device="cpu",
                verbose=0,
            )

        if (
            getattr(source, "ppo_observation_version", None)
            != PPO_OBSERVATION_VERSION
        ):
            raise ValueError("전이 모델의 관측 버전이 다릅니다.")

        if (
            source.observation_space != env.observation_space
            or source.action_space != env.action_space
        ):
            raise ValueError(
                "전이 모델과 환경의 관측·행동 공간이 다릅니다."
            )

    activation = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
    }[cfg.NETWORK_CONFIG["activation"]]

    # 새 최적화기와 0스텝 카운터로 시작합니다.
    model = PPO(
        env=env,
        seed=seed,
        policy_kwargs={
            "net_arch": deepcopy(
                cfg.NETWORK_CONFIG["net_arch"]
            ),
            "activation_fn": activation,
        },
        **deepcopy(cfg.PPO_CONFIG),
    )

    if source is not None:
        if source.policy.activation_fn != activation:
            raise ValueError(
                "전이 모델과 활성화 함수가 다릅니다."
            )

        # 정책망·가치망 가중치만 가져옵니다.
        model.policy.load_state_dict(
            source.policy.state_dict()
        )

        # 새로운 이동 방향을 탐색하도록 행동 출력을 완화합니다.
        with torch.no_grad():
            scale = cfg.TRANSFER_CONFIG["action_head_scale"]
            model.policy.action_net.weight.mul_(scale)
            model.policy.action_net.bias.mul_(scale)

    model.ppo_observation_version = PPO_OBSERVATION_VERSION
    model.ppo_reward_version = cfg.REWARD_VERSION
    model.ppo_stage = cfg.ENV_CONFIG["stage"]
    model.ppo_action_repeat = cfg.ACTION_REPEAT
    model.ppo_physical_frames = 0

    model.transfer_source = (
        str(cfg.TRANSFER_CONFIG["model_path"])
        if source else None
    )
    model.transfer_source_timesteps = (
        int(source.num_timesteps) if source else 0
    )

    return model


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def evaluation_key(report):
    """가장 낮은 평가 성공률부터 우선 비교합니다."""
    results = report["results"]

    d = results["deterministic"]["summary"]
    s = results["stochastic"]["summary"]
    phase = results.get("patrol_phases", {}).get(
        "summary", d
    )

    rates = (
        d["success_rate"],
        s["success_rate"],
        phase["success_rate"],
    )

    return (
        min(rates),
        *rates,
        d["mean_score"],
        -s["mean_steps"],
    )


def update_stability(peak, bad_count, rate):
    settings = cfg.STABILITY_CONFIG
    peak = max(peak, rate)

    dropped = (
        peak >= settings["activation_success_rate"]
        and peak - rate
        > settings["max_success_drop"] + 1e-9
    )

    bad_count = bad_count + 1 if dropped else 0

    return (
        peak,
        bad_count,
        bad_count >= settings["patience"],
    )


class EpisodeLogger(BaseCallback):
    def __init__(self, total_steps, csv_path=None):
        super().__init__()

        self.total_steps = total_steps
        self.last_print = 0
        self.physical_frames = 0

        self.recent = deque(maxlen=100)
        self.full_recent = deque(maxlen=100)

        self.episodes = self.successes = 0
        self.full_episodes = self.full_successes = 0

        self.handle = (
            csv_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            if csv_path is not None else None
        )
        self.writer = None

    def _on_step(self):
        self.physical_frames += sum(
            info["repeat_frames"]
            for info in self.locals["infos"]
        )

        for rank, info in enumerate(self.locals["infos"]):
            episode = info.get("rl_episode")

            if episode is None:
                continue

            success = bool(episode["success"])
            self.episodes += 1
            self.successes += int(success)
            self.recent.append(success)

            start = info.get("curriculum_start", 0)

            if start == 0:
                self.full_episodes += 1
                self.full_successes += int(success)
                self.full_recent.append(success)

            row = {
                "total_timesteps": self.physical_frames,
                "decision_timesteps": self.num_timesteps,
                "env_rank": rank,
                "curriculum_start": start,
                "curriculum_frontier": info.get(
                    "curriculum_frontier", 0
                ),
                "curriculum_next_frontier": info.get(
                    "curriculum_next_frontier", 0
                ),
                **episode,
            }

            if self.handle is not None:
                if self.writer is None:
                    self.writer = csv.DictWriter(
                        self.handle,
                        fieldnames=list(row),
                    )
                    self.writer.writeheader()

                self.writer.writerow(row)
                self.handle.flush()

            self.logger.record(
                "task/train_success_rate_100",
                float(np.mean(self.recent)),
            )

            if self.full_recent:
                self.logger.record(
                    "task/full_start_success_rate_100",
                    float(np.mean(self.full_recent)),
                )

            self.logger.record(
                f"curriculum/env_{rank}",
                row["curriculum_next_frontier"],
            )

        if (
            self.physical_frames - self.last_print >= 10_000
            or (
                self.physical_frames >= self.total_steps
                and self.last_print < self.total_steps
            )
        ):
            train_rate = (
                f"{np.mean(self.recent):.1%}"
                if self.recent else "--"
            )
            full_rate = (
                f"{np.mean(self.full_recent):.1%}"
                if self.full_recent else "--"
            )
            levels = "/".join(
                str(info.get("curriculum_next_frontier", "-"))
                for info in self.locals["infos"]
            )

            print(
                f"[{self.physical_frames:,}/{self.total_steps:,}F] "
                f"학습 {train_rate} | "
                f"처음부터 {full_rate} | 단계 {levels}",
                flush=True,
            )

            self.last_print = self.physical_frames

        return True

    def close(self):
        if self.handle is not None:
            self.handle.close()


def make_vec(seed):
    return make_vec_env(
        make_env,
        n_envs=cfg.TRAIN_CONFIG["n_envs"],
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )


def check_setup(seed):
    training_targets(False)
    torch.set_num_threads(
        cfg.TRAIN_CONFIG["torch_threads"]
    )

    env = make_vec(seed)

    try:
        model = make_model(env, seed)
        obs = env.reset()

        actions, _ = model.predict(
            obs,
            deterministic=True,
        )
        next_obs, _, _, _ = env.step(actions)

        assert obs.shape == (
            cfg.TRAIN_CONFIG["n_envs"],
            cfg.OBSERVATION_CONFIG["final_size"],
        )
        assert next_obs.shape == obs.shape
        assert (
            env.action_space.n
            == cfg.OBSERVATION_CONFIG["action_count"]
        )
        assert model.num_timesteps == 0
        assert not model.policy.optimizer.state

        print(
            f"연결 정상 | Stage {model.ppo_stage} | "
            f"관측 {obs.shape} | "
            f"전이 원본 {model.transfer_source_timesteps:,}스텝 | "
            f"새 학습 0스텝"
        )
        print("학습·파일 저장 없음")

    finally:
        env.close()


def train(seed, smoke=False):
    targets = training_targets(smoke)
    torch.set_num_threads(
        cfg.TRAIN_CONFIG["torch_threads"]
    )

    stamp = datetime.now(
        timezone(timedelta(hours=9))
    ).strftime("%Y%m%d_%H%M%S_%f")

    run_dir = (
        cfg.RUN_ROOT
        / ("smoke" if smoke else "train")
        / f"seed_{seed}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    model_dir = run_dir / "models"
    model_dir.mkdir()

    env = model = logger = callback = None
    status = "failed"

    best_key = None
    best_step = None
    best_passed = False
    peak = 0.0
    bad_count = 0

    try:
        env = make_vec(seed)
        model = make_model(env, seed)

        if cfg.LOG_CONFIG["save_config"]:
            write_json(run_dir / "config.json", {
                "algorithm": "PPO",
                "seed": seed,
                "smoke": smoke,
                "planned_timesteps": targets[-1],
                "action_repeat": cfg.ACTION_REPEAT,
                "timestep_unit": "physical_frame",
                "environment_name": cfg.ENV_NAME,
                "environment": cfg.ENV_CONFIG,
                "observation": cfg.OBSERVATION_CONFIG,
                "observation_version": PPO_OBSERVATION_VERSION,
                "curriculum": cfg.CURRICULUM_CONFIG,
                "curriculum_version": CURRICULUMS[
                    cfg.ENV_CONFIG["stage"]
                ].version,
                "network": cfg.NETWORK_CONFIG,
                "ppo": cfg.PPO_CONFIG,
                "training": cfg.TRAIN_CONFIG,
                "evaluation": cfg.EVAL_CONFIG,
                "reward": cfg.REWARD_CONFIG,
                "reward_version": cfg.REWARD_VERSION,
                "transfer": cfg.TRANSFER_CONFIG,
                "transfer_source_timesteps": (
                    model.transfer_source_timesteps
                ),
                "stability": cfg.STABILITY_CONFIG,
                "logging": cfg.LOG_CONFIG,
                "python_executable": sys.executable,
                "versions": {
                    name: version(name)
                    for name in (
                        "torch",
                        "stable-baselines3",
                        "gymnasium",
                        "numpy",
                    )
                },
            })

        formats = ["csv"]

        if cfg.LOG_CONFIG["tensorboard"]:
            formats.append("tensorboard")

        logger = configure(
            str(run_dir / "logs"),
            formats,
        )
        model.set_logger(logger)

        callback = EpisodeLogger(
            targets[-1],
            run_dir / "train_episodes.csv"
            if cfg.LOG_CONFIG["save_episode_csv"] else None,
        )

        print(f"저장 위치: {run_dir}", flush=True)

        rollout = (
            cfg.TRAIN_CONFIG["n_envs"]
            * cfg.PPO_CONFIG["n_steps"]
        )
        eval_index = 0

        while callback.physical_frames < targets[-1]:
            previous_frames = callback.physical_frames

            model.learn(
                total_timesteps=rollout,
                reset_num_timesteps=False,
                callback=callback,
                log_interval=None,
            )

            frames = callback.physical_frames

            if frames <= previous_frames:
                raise RuntimeError(
                    "물리 프레임이 진행되지 않았습니다."
                )

            model.ppo_physical_frames = frames

            logger.record("time/physical_frames", frames)
            logger.record(
                "time/decision_timesteps",
                model.num_timesteps,
            )

            if frames < targets[eval_index]:
                logger.dump(step=frames)
                continue

            while (
                eval_index < len(targets)
                and targets[eval_index] <= frames
            ):
                eval_index += 1

            target = frames
            model.save(model_dir / f"ppo_{target}.zip")

            print(
                f"[평가 {target:,}] 진행 중...",
                end="\r",
                flush=True,
            )

            report = evaluate_model(model, smoke=smoke)
            report["training_timesteps"] = target
            report["training_seed"] = seed
            report["decision_timesteps"] = int(model.num_timesteps)
            report["timestep_unit"] = "physical_frame"

            save_evaluation(
                report,
                run_dir / "evaluation" / f"step_{target}",
            )

            labels = {
                "deterministic": "결정적",
                "stochastic": "확률적",
                "patrol_phases": "유령 상태",
            }
            parts = []

            for mode, result in report["results"].items():
                summary = result["summary"]

                parts.append(
                    f"{labels.get(mode, mode)} "
                    f"{summary['success_rate']:.1%}"
                )

                for name, value in summary.items():
                    logger.record(
                        f"eval/{mode}/{name}",
                        value,
                    )

            logger.dump(step=target)

            print(
                f"[평가 {target:,}] " + " | ".join(parts),
                flush=True,
            )

            # 모든 평가를 통과하면 탐색 강도를 낮춥니다.
            if not smoke and report["passed"]:
                model.ent_coef = cfg.STABILITY_CONFIG[
                    "ent_coef_after_success"
                ]

            key = evaluation_key(report)

            if best_key is None or key > best_key:
                best_key = key
                best_step = target
                best_passed = report["passed"]
                model.save(model_dir / "best_model.zip")

            if not smoke:
                rate = report["results"]["deterministic"][
                    "summary"
                ]["success_rate"]

                peak, bad_count, stop = update_stability(
                    peak,
                    bad_count,
                    rate,
                )

                if stop:
                    status = "early_stopped"
                    print(
                        "[중단] 성공률 하락이 연속 발생해 "
                        "최적 모델을 선택합니다.",
                        flush=True,
                    )
                    break

        else:
            status = "complete"

        # 마지막 시점의 모델과 평가상 최적 모델을 구분합니다.
        model.save(model_dir / "last_model.zip")

        shutil.copyfile(
            model_dir / "best_model.zip",
            model_dir / "final_model.zip",
        )

    except KeyboardInterrupt:
        status = "interrupted"

        if model is not None:
            model.save(model_dir / "interrupted_model.zip")

        raise

    except Exception:
        status = "failed"
        raise

    finally:
        try:
            write_json(run_dir / "training_summary.json", {
                "status": status,
                "smoke": smoke,
                "total_timesteps": (
                    callback.physical_frames if callback else 0
                ),
                "decision_timesteps": (
                    int(model.num_timesteps) if model else 0
                ),
                "action_repeat": cfg.ACTION_REPEAT,
                "selected_model_timesteps": best_step,
                "best_passed": best_passed,
                "peak_deterministic_success_rate": peak,
                "consecutive_drops": bad_count,
                "episodes": (
                    callback.episodes if callback else 0
                ),
                "successes": (
                    callback.successes if callback else 0
                ),
                "full_start_episodes": (
                    callback.full_episodes if callback else 0
                ),
                "full_start_successes": (
                    callback.full_successes if callback else 0
                ),
            })

        finally:
            if callback is not None:
                callback.close()
            if env is not None:
                env.close()
            if logger is not None:
                logger.close()

    print(
        f"[완료] 실행 {callback.physical_frames:,}프레임 | "
        f"선택 {best_step:,}프레임 | {status}"
    )

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="실시간 PPO 전이 학습"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.TRAIN_CONFIG["default_seed"],
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.seed < 2**31:
        parser.error("seed는 0 이상 2**31 미만이어야 합니다.")

    if args.check:
        check_setup(args.seed)
    else:
        train(args.seed, smoke=args.smoke)


if __name__ == "__main__":
    main()