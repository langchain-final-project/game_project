"""귀환 성공 PPO에서 별 수집을 추가 학습하는 독립 설정과 모델 로딩."""
from copy import deepcopy
import hashlib
from importlib import metadata
import json
import math
import platform
import sys
from pathlib import Path

from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from kkamang.realtime import PHYSICS_HZ, FRAMES_PER_TICK
from rl import ppo_common as base
from rl.ppo_common import (
    ROOT, SCHEMA_VERSION, GRAPH_METRICS, isolated_action_rng,
    write_json, write_csv, config_digest, evaluation_cases,
)
from rl.ppo_reward import PPOReward
from rl.ppo_star_observation import PPOStarObservation, observation_version
from rl.task_goal_observation import TASK_GOAL_SIZE, TASK_GOAL_VERSION

BASELINE_VERSION = "ppo_star_goal7_entropy_comparison_v2"
# 탐색 강도만 비교한다. 보상/학습률/시작 모델은 각 조건에서 같다.
EXPLORATION_PRESETS = {
    "baseline": 0.01,
    "entropy03": 0.03,
    "entropy05": 0.05,
}
CONFIG = deepcopy(base.CONFIG)
CONFIG.update(
    experiment="RTenv_s01_ppo_star1_goal7_{exploration}_ft200k_v2",
    exploration="entropy03",
    curriculum=None,
    initial_model=(
        "runs/RTenv_s01_ppo_route_v3_goal_v1_lr2e-4_ns100_600k/"
        "seed_11_20261029_001614_962501/final_model.zip"
    ),
)
CONFIG["environment"]["warmth_goal"] = 7
CONFIG["ppo"].update(
    learning_rate=0.00005,
    ent_coef=EXPLORATION_PRESETS[CONFIG["exploration"]],
    total_timesteps=200000,
    checkpoints=[50000, 100000, 150000, 200000],
)
CONFIG["evaluation"].update(version="ppo_star_eval_v1", star_rate_threshold=0.8)


def model_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def source_config_path(path):
    return (path.parent.parent if path.parent.name == "checkpoints" else path.parent) / "config.json"


def read_source(path):
    path = model_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"추가 학습할 PPO 모델이 없습니다: {path}")
    config = json.loads(source_config_path(path).read_text(encoding="utf-8"))
    if (config.get("algorithm") != "PPO"
            or config["environment"].get("observation_size") != TASK_GOAL_SIZE):
        raise ValueError("636차원 관측의 PPO 모델만 추가 학습할 수 있습니다.")
    return path, config


def build_config(initial_model=None, *, exploration=None):
    config = deepcopy(CONFIG)
    preset = exploration if exploration is not None else config["exploration"]
    if preset not in EXPLORATION_PRESETS:
        raise ValueError(f"탐색 조건은 {tuple(EXPLORATION_PRESETS)} 중에서 선택하세요.")
    config["exploration"] = preset
    config["ppo"]["ent_coef"] = EXPLORATION_PRESETS[preset]
    config["experiment"] = config["experiment"].format(exploration=preset)
    path, source = read_source(initial_model or config["initial_model"])
    config["initial_model"] = str(path.resolve())
    settings = config["environment"]
    settings.update(
        observation_size=TASK_GOAL_SIZE,
        observation_version=observation_version(settings["warmth_goal"]),
        action_count=6, action_unit="physics_frame",
    )
    # 원본 실행에 저장된 보상을 사용한다. 현재 기본 계수를 바꿔도 원본 보상은 보존된다.
    config["reward"] = deepcopy(source["reward"])
    validate_config(config)
    return config


def compatible_base_config(config):
    proxy = deepcopy(config)
    proxy["environment"]["observation_version"] = TASK_GOAL_VERSION
    return proxy


def validate_config(config):
    goal = config["environment"].get("warmth_goal")
    if type(goal) is not int or not 5 <= goal <= 14:
        raise ValueError("별 수집 목표는 5~14의 정수여야 합니다.")
    if config["environment"].get("observation_version") != observation_version(goal):
        raise ValueError("수집 목표와 관측 버전이 일치하지 않습니다.")
    if config.get("curriculum") is not None:
        raise ValueError("별 수집 실험은 원래 시작점만 사용합니다. curriculum=None이 필요합니다.")
    if config["environment"]["stage"] != 1 or config["environment"]["randomize_phase"]:
        raise ValueError("첫 별 수집 실험은 고정 위상의 Stage 1을 사용합니다.")
    if not config.get("initial_model"):
        raise ValueError("처음부터 학습하지 않습니다. initial_model을 지정하세요.")
    if not 0 <= config["evaluation"]["star_rate_threshold"] <= 1:
        raise ValueError("별 획득률 기준은 0~1이어야 합니다.")
    entropy = config["ppo"]["ent_coef"]
    if (isinstance(entropy, bool) or not isinstance(entropy, (int, float))
            or not math.isfinite(entropy) or entropy < 0):
        raise ValueError("ent_coef는 0 이상의 유한한 수여야 합니다.")
    # v1 실행에는 exploration 항목이 없으므로 저장된 ent_coef 그대로 재평가한다.
    preset = config.get("exploration")
    if preset is not None:
        if preset not in EXPLORATION_PRESETS or not math.isclose(
            entropy, EXPLORATION_PRESETS[preset], rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError("탐색 조건 이름과 ent_coef가 일치하지 않습니다.")
    base.validate_config(compatible_base_config(config))


def training_targets(config, smoke=False):
    validate_config(config)
    return base.training_targets(compatible_base_config(config), smoke=smoke)


def make_env(config=None, *, training=False):
    config = config or build_config()
    validate_config(config)
    settings = config["environment"]
    env = RealtimeMansionEnv(
        stage=settings["stage"], mode=settings["mode"],
        randomize_phase=settings["randomize_phase"],
        max_episode_steps=settings["max_episode_steps"], safety_coefficient=0.0,
    )
    # 학습과 평가 모두 원래 시작점이다. 준비 경로는 실행하지 않는다.
    env = PPOStarObservation(env, warmth_goal=settings["warmth_goal"])
    reward = config["reward"]
    return PPOReward(env, reward_config={k: v for k, v in reward.items() if k != "version"},
                     reward_version=reward["version"])


def load_initial_model(config, env, seed):
    validate_config(config)
    path, source = read_source(config["initial_model"])
    for key in ("name", "stage", "mode", "randomize_phase"):
        if source["environment"][key] != config["environment"][key]:
            raise ValueError(f"원본 모델과 게임 조건이 다릅니다: {key}")
    for key in ("policy", "net_arch", "activation"):
        if source["ppo"][key] != config["ppo"][key]:
            raise ValueError(f"원본 모델과 신경망 설정이 다릅니다: {key}")
    ppo = config["ppo"]
    keys = (
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
        "clip_range", "clip_range_vf", "normalize_advantage", "ent_coef", "vf_coef",
        "max_grad_norm", "target_kl",
    )
    model = PPO.load(path, env=env, device=ppo["device"], seed=seed,
                     force_reset=True, **{key: ppo[key] for key in keys})
    original_steps = int(model.num_timesteps)
    metadata_path = source_config_path(path).with_name("metadata.json")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    cumulative_source_steps = original_steps
    if "initial_model" in source_metadata:
        cumulative_source_steps = source_metadata["initial_model"]["source_cumulative_timesteps"] + original_steps
    elif path.name in ("final_model.zip", "interrupted_model.zip"):
        cumulative_source_steps = source_metadata.get("cumulative_completed_timesteps", original_steps)
    provenance = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(source_config_path(path).read_bytes()).hexdigest(),
        "source_model_timesteps": original_steps,
        "source_cumulative_timesteps": cumulative_source_steps,
        "source_training_seed": source_metadata.get("training_seed"),
        "source_observation_version": source["environment"]["observation_version"],
        "new_observation_version": config["environment"]["observation_version"],
        "optimizer_state_retained": True,
    }
    # 가중치와 Adam 상태는 유지하고, 이번 추가 학습 카운터만 0부터 기록한다.
    model.num_timesteps = 0
    model._last_obs = None
    model.ep_info_buffer = None
    model.ep_success_buffer = None
    for group in model.policy.optimizer.param_groups:
        group["lr"] = ppo["learning_rate"]
    return model, provenance


def load_model(path, config=None):
    config = config or build_config()
    validate_config(config)
    return PPO.load(path, device=config["ppo"]["device"])


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
        "kkamang/levels.py", "kkamang/scoring.py", "rl/observation.py",
        "rl/task_goal_observation.py", "rl/ppo_reward.py", "rl/ppo_common.py",
        "rl/ppo_star_observation.py", "rl/ppo_stars_common.py",
        "rl/train_ppo_stars.py", "rl/evaluate_ppo_stars.py", "rl/watch_ppo_stars.py",
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
    if (run_metadata.get("algorithm") != "PPO"
            or run_metadata.get("config_digest") != config_digest(config)):
        raise ValueError("별 수집 실행 정보와 저장 설정이 다릅니다.")
    current = runtime_metadata(config)
    for key in ("config_hash", "observation_version", "versions"):
        if current[key] != run_metadata[key]:
            raise ValueError(f"저장 당시와 현재 조건이 다릅니다: {key}")
    return current
