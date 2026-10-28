"""PPO 전용 실험 설정과 환경 생성. rl.common/rl.reward에 의존하지 않는다."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import sys

import numpy as np
import torch

from kkamang.env import RealtimeMansionEnv
from kkamang.realtime import PHYSICS_HZ, FRAMES_PER_TICK
from rl.curriculum_route import Stage1RouteCurriculum
from rl.task_goal_observation import (
    TaskGoalObservation, TASK_GOAL_SIZE, TASK_GOAL_VERSION,
)
from rl.ppo_reward import PPOReward, REWARD_CONFIG, REWARD_VERSION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
BASELINE_VERSION = "ppo_realtime_stage1_route_v3_goal_v1"
GRAPH_METRICS = (
    "success_rate", "at_least_one_star_rate", "at_least_two_star_rate",
    "three_star_rate", "mean_score", "perfect_rate", "quest_rate", "warmth_rate",
)

# 실제 비교 조건: 기존 A2C와 같은 게임/관측/보상/커리큘럼/총 환경 스텝.
# PPO에서만 사용하는 batch_size, n_epochs, clip_range를 별도로 명시한다.
CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "experiment": "RTenv_s01_ppo_route_v3_goal_v1_lr2e-4_ns100_600k",
    "algorithm": "PPO",
    "curriculum": "stage1_route_v3",
    "environment": {
        "name": "RealtimeMansionEnv",
        "stage": 1,
        "mode": "basic",
        "randomize_phase": False,
        "max_episode_steps": None,
        "goal_features": True,
    },
    "ppo": {
        "policy": "MlpPolicy",
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation": "Tanh",
        "n_envs": 4,
        "learning_rate": 0.0002,
        "n_steps": 100,
        "batch_size": 100,
        "n_epochs": 10,
        "gamma": 0.9994,
        "gae_lambda": 1.0,
        "clip_range": 0.2,
        "clip_range_vf": None,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "device": "cpu",
        "torch_threads": 1,
        "total_timesteps": 600000,
        "checkpoints": [200000, 400000, 600000],
    },
    "evaluation": {
        "version": "ppo_stage1_eval_v1",
        "deterministic_seeds": list(range(10000, 10050)),
        "stochastic_seeds": list(range(10000, 10100)),
        "action_seed_base": 50000,
        "primary_mode": "deterministic",
        "success_rate_threshold": 0.9,
        "minimum_training_seeds": 3,
    },
}


def build_config():
    config = deepcopy(CONFIG)
    config["environment"].update(
        observation_size=TASK_GOAL_SIZE,
        observation_version=TASK_GOAL_VERSION,
        action_count=6,
        action_unit="physics_frame",
    )
    config["reward"] = {"version": REWARD_VERSION, **deepcopy(REWARD_CONFIG)}
    validate_config(config)
    return config


def validate_config(config):
    if config.get("schema_version") != SCHEMA_VERSION or config.get("algorithm") != "PPO":
        raise ValueError("PPO 전용 설정이 필요합니다.")
    env = config["environment"]
    expected = {
        "name": "RealtimeMansionEnv", "observation_size": TASK_GOAL_SIZE,
        "observation_version": TASK_GOAL_VERSION, "action_count": 6,
        "action_unit": "physics_frame", "goal_features": True,
    }
    if any(env.get(key) != value for key, value in expected.items()):
        raise ValueError("PPO 환경은 목표 정보가 포함된 636차원 실시간 관측을 사용합니다.")
    if env["mode"] not in ("basic", "relaxed"):
        raise ValueError("mode는 basic 또는 relaxed여야 합니다.")
    if config.get("curriculum") not in (None, "stage1_route_v3"):
        raise ValueError("알 수 없는 curriculum 이름입니다. 설정을 확인하세요.")
    if config.get("curriculum") and (env["stage"] != 1 or env["randomize_phase"]):
        raise ValueError("현재 커리큘럼은 고정 위상의 Stage 1 전용입니다.")
    ppo = config["ppo"]
    positive = lambda value: type(value) is int and value > 0
    if any(not positive(ppo[key]) for key in (
        "n_envs", "n_steps", "batch_size", "n_epochs", "torch_threads", "total_timesteps",
    )):
        raise ValueError("스텝 수, 환경 수, 배치, epoch, 스레드는 양의 정수여야 합니다.")
    rollout = ppo["n_envs"] * ppo["n_steps"]
    if ppo["batch_size"] < 2 or rollout % ppo["batch_size"]:
        raise ValueError("batch_size는 2 이상이며 n_envs × n_steps를 나누어야 합니다.")
    if ppo["device"] != "cpu" or ppo["activation"] not in ("Tanh", "ReLU"):
        raise ValueError("이 비교 실험은 CPU와 Tanh/ReLU를 지원합니다.")
    if not (ppo["learning_rate"] > 0 and 0 < ppo["gamma"] <= 1
            and 0 <= ppo["gae_lambda"] <= 1 and ppo["clip_range"] > 0):
        raise ValueError("학습률, gamma, gae_lambda, clip_range를 확인하세요.")
    total, points = ppo["total_timesteps"], ppo["checkpoints"]
    if (total % rollout or points != sorted(set(points))
            or any(not positive(p) or p > total or p % rollout for p in points)):
        raise ValueError("학습량과 체크포인트는 rollout 배수이며 중복 없이 오름차순이어야 합니다.")
    reward = config["reward"]
    if set(reward) != {"version", *REWARD_CONFIG}:
        raise ValueError("저장된 PPO 보상 설정의 항목을 확인하세요.")
    if not 0 < reward["warmth_cap"] <= 14 or reward["score_divisor"] <= 0:
        raise ValueError("온기 보상 상한과 점수 나눗값을 확인하세요.")
    evaluation = config["evaluation"]
    if evaluation["primary_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("primary_mode를 확인하세요.")
    for mode in ("deterministic", "stochastic"):
        seeds = evaluation[f"{mode}_seeds"]
        if (not seeds or len(seeds) != len(set(seeds))
                or any(type(seed) is not int or seed < 0 for seed in seeds)):
            raise ValueError("평가 seed는 중복 없는 0 이상의 정수여야 합니다.")
    if (type(evaluation["action_seed_base"]) is not int
            or evaluation["action_seed_base"] < 0
            or not 0 <= evaluation["success_rate_threshold"] <= 1
            or not positive(evaluation["minimum_training_seeds"])):
        raise ValueError("평가 기준과 행동 seed를 확인하세요.")


def training_targets(config, smoke=False):
    validate_config(config)
    ppo = config["ppo"]
    if smoke:
        limit = min(2000, ppo["total_timesteps"])
        if limit % (ppo["n_envs"] * ppo["n_steps"]):
            raise ValueError("smoke의 2,000스텝도 rollout 배수여야 합니다.")
        return [limit]
    return sorted({*ppo["checkpoints"], ppo["total_timesteps"]})


def evaluation_cases(config=None, smoke=False):
    evaluation = (config or build_config())["evaluation"]
    return [
        {"name": mode, "deterministic": mode == "deterministic",
         "seeds": tuple(evaluation[f"{mode}_seeds"][:2] if smoke
                        else evaluation[f"{mode}_seeds"])}
        for mode in ("deterministic", "stochastic")
    ]


def make_env(config=None, *, training=False):
    config = config or build_config()
    validate_config(config)
    settings = config["environment"]
    env = RealtimeMansionEnv(
        stage=settings["stage"], mode=settings["mode"],
        randomize_phase=settings["randomize_phase"],
        max_episode_steps=settings["max_episode_steps"], safety_coefficient=0.0,
    )
    if training and config.get("curriculum") == "stage1_route_v3":
        env = Stage1RouteCurriculum(env)
    env = TaskGoalObservation(env)
    reward = config["reward"]
    return PPOReward(env, reward_config={k: v for k, v in reward.items() if k != "version"},
                     reward_version=reward["version"])


@contextmanager
def isolated_action_rng(seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        try:
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed % 2**32)
                torch.random.default_generator.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def load_model(path, config=None):
    config = config or build_config()
    validate_config(config)
    from stable_baselines3 import PPO
    return PPO.load(path, device=config["ppo"]["device"])


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    if rows:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_metadata(config=None):
    config = config or build_config()
    env = make_env(config)
    try:
        _, info = env.reset(seed=0)
        tick_seconds = env.unwrapped.core.config["tick_seconds"]
    finally:
        env.close()
    sources = [
        "kkamang/env.py", "kkamang/core.py", "kkamang/realtime.py",
        "kkamang/levels.py", "kkamang/scoring.py", "rl/curriculum_route.py",
        "rl/task_goal_observation.py", "rl/observation.py",
        "rl/ppo_common.py", "rl/ppo_reward.py", "rl/train_ppo.py",
        "rl/evaluate_ppo.py", "rl/watch_ppo.py",
        "data/levels/realtime_observation_contract.json",
    ]
    return {
        "schema_version": SCHEMA_VERSION, "baseline_version": BASELINE_VERSION,
        "algorithm": "PPO", "config_digest": config_digest(config),
        "source_hashes": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
        "versions": {name: metadata.version(name) for name in (
            "gymnasium", "numpy", "pygame-ce", "stable-baselines3", "torch", "tensorboard", "pillow",
        )},
        "python": platform.python_version(), "platform": platform.platform(),
        "executable": sys.executable, "config_hash": info["config_hash"],
        "observation_version": info["observation_version"],
        "simulation": {"action_unit": "physics_frame", "physics_hz": PHYSICS_HZ,
                       "frames_per_tick": FRAMES_PER_TICK, "tick_seconds": tick_seconds},
    }


def validate_run(config, run_metadata):
    validate_config(config)
    if (run_metadata.get("schema_version") != SCHEMA_VERSION
            or run_metadata.get("algorithm") != "PPO"
            or run_metadata.get("config_digest") != config_digest(config)):
        raise ValueError("PPO 실행 정보와 저장 설정이 일치하지 않습니다.")
    current = runtime_metadata(config)
    for key in ("config_hash", "observation_version", "versions"):
        if current[key] != run_metadata[key]:
            raise ValueError(f"저장 당시와 현재 조건이 다릅니다: {key}")
    # A2C 설정 파일을 수정해도 PPO 재평가는 영향을 받지 않는다.
    return current
