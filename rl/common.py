# 공통 설정·환경 선택·난수 격리·모델 로딩과 실험 조건 검증을 한곳에서 관리합니다.
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
import subprocess
import sys

import numpy as np
import torch

from kkamang.env import (
    MansionEnv,
    RealtimeMansionEnv,
    OBSERVATION_SIZE,
    REALTIME_OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    REALTIME_OBSERVATION_VERSION,
)
from rl.reward import ProposedReward, REWARD_CONFIG, REWARD_VERSION
from rl.observation import (InteractionReadyObservation, INTERACTION_READY_SIZE,
                            INTERACTION_READY_VERSION)
from kkamang.realtime import PHYSICS_HZ, FRAMES_PER_TICK

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
# 학습·환경의 동작을 바꾸면 이 버전을 올림. 파일 해시는 실행 당시 코드를 추적할 때만 사용
BASELINE_VERSION = "stage1_a2c_score20_lr0003_nsteps20_interaction_ready_v1"
BASE_OBSERVATION_BASELINE_VERSION = "stage1_a2c_score20_lr0003_nsteps20_v1"
ENVIRONMENTS = {"MansionEnv": MansionEnv, "RealtimeMansionEnv": RealtimeMansionEnv}
COMPARISON_PACKAGES = ("gymnasium", "numpy", "torch", "stable-baselines3")
GAME_SOURCES = (
    "kkamang/env.py",
    "kkamang/core.py",
    "kkamang/realtime.py",
    "kkamang/levels.py",
    "kkamang/scoring.py",
    "rl/reward.py",
)
EVAL_SEEDS = tuple(range(10000, 10050))
STOCHASTIC_EVAL_SEEDS = tuple(range(10000, 10100))
EVALUATION_CONFIG = {
    "version": "stage1_eval_v1",
    "deterministic_seeds": list(EVAL_SEEDS),
    "stochastic_seeds": list(STOCHASTIC_EVAL_SEEDS),
    "action_seed_base": 50000,
    "primary_mode": "deterministic",  # 팀 공식 평가. stochastic은 보조 결과
    "success_rate_threshold": 0.9,
    "minimum_training_seeds": 3,
}
SUMMARY_METRICS = (
    "success_rate",
    "at_least_one_star_rate",
    "at_least_two_star_rate",
    "three_star_rate",
    "perfect_rate",
    "mean_score",
    "mean_success_score",
    "quest_rate",
    "warmth_rate",
    "conditions_rate",
    "return_after_conditions_rate",
    "mean_steps",
    "mean_ticks",
    "mean_return",
    *(f"star_{i}_count" for i in range(4)),
    *(f"star_{i}_mean_score" for i in range(4)),
)
GRAPH_METRICS = (
    "success_rate",
    "at_least_one_star_rate",
    "at_least_two_star_rate",
    "three_star_rate",
    "mean_score",
    "perfect_rate",
    "quest_rate",
    "warmth_rate",
)

# --seed만 바꾸기, 보상은 reward.py, 나머지 실행 설정은 여기서 관리 하기
CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "experiment": "stage1_a2c_score20_lr0002_nsteps20_warmthReward",
    "algorithm": "A2C",
    "environment": {
        "name": "MansionEnv",
        "stage": 1,
        "mode": "basic",
        "randomize_phase": False,
        "max_episode_steps": None,
        "interaction_ready": True,
    },
    "a2c": {
        "policy": "MlpPolicy",
        "net_arch": {"pi": [64, 64], "vf": [64, 64]},
        "activation": "Tanh",
        "n_envs": 4,
        "learning_rate": 0.0002,
        "n_steps": 20,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "use_rms_prop": True,
        "rms_prop_eps": 1e-5,
        "normalize_advantage": False,
        "device": "cpu",
        "torch_threads": 1,
        "total_timesteps": 1000000,
        "checkpoints": list(range(50000, 1000001, 50000)),
    },
    "evaluation": EVALUATION_CONFIG,
}


def environment_contract(name, *, interaction_ready=False):
    """환경 종류에서 관측 차원·버전·행동 단위를 자동으로 가져옵니다."""
    if name not in ENVIRONMENTS:
        raise ValueError(f"지원하지 않는 환경입니다: {name}")
    if type(interaction_ready) is not bool:
        raise ValueError("interaction_ready는 bool이어야 합니다.")
    if interaction_ready:
        if name != "MansionEnv":
            raise ValueError("interaction_ready 실험은 격자 MansionEnv에서만 지원합니다.")
        return {
            "observation_size": INTERACTION_READY_SIZE,
            "observation_version": INTERACTION_READY_VERSION,
            "action_count": 6,
            "action_unit": "one_world_tick",
        }
    realtime = name == "RealtimeMansionEnv"
    return {
        "observation_size": REALTIME_OBSERVATION_SIZE if realtime else OBSERVATION_SIZE,
        "observation_version": REALTIME_OBSERVATION_VERSION
        if realtime
        else OBSERVATION_VERSION,
        "action_count": 6,
        "action_unit": "physics_frame" if realtime else "one_world_tick",
    }


def build_config():
    """관측 계약과 보상은 실제 코드에서 가져와 설정값이 따로 어긋나지 않게 합니다."""
    config = deepcopy(CONFIG)
    settings = config["environment"]
    settings.update(environment_contract(
        settings["name"], interaction_ready=settings.get("interaction_ready", False),
    ))
    config["reward"] = {"version": REWARD_VERSION, **deepcopy(REWARD_CONFIG)}
    return config


def validate_config(config):
    """잘못된 설정을 학습 시작 전에 거부합니다. 새 형식 한 가지만 지원합니다."""
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("지원하지 않는 설정 형식입니다. 공통 초기 설정을 사용하세요.")
    env = config["environment"]
    for key, value in environment_contract(
        env["name"], interaction_ready=env.get("interaction_ready", False),
    ).items():
        if env.get(key) != value:
            raise ValueError(f"환경의 관측·행동 계약이 다릅니다: {key}")
    if config["reward"] != {"version": REWARD_VERSION, **REWARD_CONFIG}:
        raise ValueError("설정과 실제 보상 코드가 다릅니다.")
    if env["mode"] not in ("basic", "relaxed"):
        raise ValueError("지원하지 않는 게임 모드입니다.")
    options = config[config["algorithm"].lower()]
    if options["device"] != "cpu":
        raise ValueError("현재 공통 학습·평가와 난수 격리는 CPU 기준입니다.")
    total, points = options["total_timesteps"], options["checkpoints"]
    positive = lambda value: (
        isinstance(value, int) and not isinstance(value, bool) and value > 0
    )
    if (
        not positive(total)
        or not positive(options["n_steps"])
        or not positive(options["n_envs"])
    ):
        raise ValueError("학습량·수집 길이·환경 수는 양의 정수여야 합니다.")
    batch = options["n_steps"] * options["n_envs"]
    if total % batch or any(not positive(p) or p > total or p % batch for p in points):
        raise ValueError(
            "학습량·저장 시점은 수집 배치의 배수이며 저장 시점은 총 학습량 이하여야 합니다."
        )
    if points != sorted(set(points)):
        raise ValueError("저장 시점은 중복 없이 오름차순으로 지정하세요.")
    evaluation = config["evaluation"]
    if evaluation["primary_mode"] not in ("deterministic", "stochastic"):
        raise ValueError("공식 평가 방식을 확인하세요.")
    for mode in ("deterministic", "stochastic"):
        seeds = evaluation[f"{mode}_seeds"]
        if (
            not seeds
            or len(seeds) != len(set(seeds))
            or any(
                not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds
            )
        ):
            raise ValueError("평가 seed는 중복되지 않는 0 이상의 정수여야 합니다.")
    if not 0 <= evaluation["success_rate_threshold"] <= 1 or not positive(
        evaluation["minimum_training_seeds"]
    ):
        raise ValueError("귀환 성공률 기준과 학습 seed 수를 확인하세요.")


def training_targets(config, smoke=False):
    """종료점은 total_timesteps 하나로 정하고 저장 목록에 없어도 최종 모델은 남깁니다."""
    validate_config(config)
    options = config[config["algorithm"].lower()]
    total = options["total_timesteps"]
    if smoke:
        limit = min(2000, total)
        if limit % (options["n_envs"] * options["n_steps"]):
            raise ValueError("짧은 점검 학습량도 수집 배치의 배수여야 합니다.")
        return [limit]
    return sorted(set([*options["checkpoints"], total]))


def evaluation_cases(config=None, smoke=False):
    evaluation = (config or build_config())["evaluation"]
    return [
        {
            "name": mode,
            "deterministic": mode == "deterministic",
            "seeds": tuple(
                evaluation[f"{mode}_seeds"][:2]
                if smoke
                else evaluation[f"{mode}_seeds"]
            ),
        }
        for mode in ("deterministic", "stochastic")
    ]


def make_env(config=None):
    """학습·평가·재생이 같은 설정으로 환경을 만들고, 기본은 격자 Stage 1"""
    config = config or build_config()
    validate_config(config)
    settings = config["environment"]
    env = ENVIRONMENTS[settings["name"]](
        stage=settings["stage"],
        mode=settings["mode"],
        randomize_phase=settings["randomize_phase"],
        max_episode_steps=settings["max_episode_steps"],
        safety_coefficient=0.0,
    )
    if settings.get("interaction_ready", False):
        env = InteractionReadyObservation(env)
    return ProposedReward(env)


@contextmanager
def isolated_action_rng(seed=None):
    """CPU 평가 후 학습의 Python·NumPy·Torch 난수 흐름을 원상 복원합니다."""
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
    """알고리즘별 로딩 진입점으로, A3C 구현 시 이곳에 로더를 추가해 진행"""
    algorithm = (config or build_config())["algorithm"]
    if algorithm != "A2C":
        raise ValueError(f"{algorithm} 모델 로더는 아직 구현되지 않았습니다.")
    from stable_baselines3 import A2C

    return A2C.load(path, device="cpu")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def config_digest(config):
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def runtime_metadata(config=None):
    config = config or build_config()
    versions = {
        name: metadata.version(name)
        for name in (
            "gymnasium",
            "numpy",
            "pygame-ce",
            "stable-baselines3",
            "torch",
            "tensorboard",
            "pillow",
        )
    }
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    # 원본 해시는 기록만 합니다. 비교는 설정·기준 버전·패키지·레벨·관측 계약으로 판단합니다.
    sources = sorted(
        {*GAME_SOURCES, *(str(p.relative_to(ROOT)) for p in (ROOT / "rl").glob("*.py"))}
    )
    raw_files = [
        *sources,
        "requirements-a2c.txt",
        "data/levels/observation_contract.json",
        "data/levels/realtime_observation_contract.json",
        "data/levels/interaction_ready_observation_contract.json",
    ]
    env = make_env(config)
    try:
        _, info = env.reset(seed=0)
        realtime = config["environment"]["name"] == "RealtimeMansionEnv"
        simulation = {
            "action_unit": config["environment"]["action_unit"],
            "tick_seconds": env.unwrapped.core.config["tick_seconds"],
            "physics_hz": PHYSICS_HZ if realtime else None,
            "frames_per_tick": FRAMES_PER_TICK if realtime else None,
        }
    finally:
        env.close()
    # 直前の620次元対照実験も、その保存設定で引き続き検証できます。
    baseline_version = BASELINE_VERSION
    if (config["experiment"] == BASE_OBSERVATION_BASELINE_VERSION
            and not config["environment"].get("interaction_ready", False)):
        baseline_version = BASE_OBSERVATION_BASELINE_VERSION
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline_version": baseline_version,
        "algorithm": config["algorithm"],
        "config_digest": config_digest(config),
        "git_commit": commit,
        "source_hashes": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in raw_files
        },
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "simulation": simulation,
        "executable": sys.executable,
        "config_hash": info["config_hash"],
        "observation_version": info["observation_version"],
    }


def comparison_key(config, run_metadata):
    """seed만 다른 실험인지 확인할 공통 조건입니다. 주석·서식 변경은 비교를 막지 않습니다."""
    return {
        "config": config,
        "baseline_version": run_metadata.get("baseline_version"),
        "versions": {
            name: run_metadata["versions"].get(name) for name in COMPARISON_PACKAGES
        },
        "python": ".".join(run_metadata["python"].split(".")[:2]),
        "config_hash": run_metadata["config_hash"],
        "observation_version": run_metadata["observation_version"],
    }


def validate_run(config, run_metadata):
    """저장 설정의 무결성과 현재 실행 조건의 호환성을 한 번씩 확인합니다."""
    if run_metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("새 초기 파일로 생성한 실행만 지원합니다.")
    if (
        run_metadata["config_digest"] != config_digest(config)
        or run_metadata["algorithm"] != config["algorithm"]
    ):
        raise ValueError("저장 설정과 실행 정보가 일치하지 않습니다.")
    current = runtime_metadata(config)
    recorded, actual = (
        comparison_key(config, run_metadata),
        comparison_key(config, current),
    )
    changed = [key for key in actual if recorded[key] != actual[key]]
    if changed:
        raise ValueError(f"실행 당시와 현재 공통 조건이 다릅니다: {', '.join(changed)}")
    return current  # 재평가에서 같은 실행 정보를 다시 수집하지 않도록 재사용합니다.
