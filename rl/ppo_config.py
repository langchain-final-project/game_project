from pathlib import Path


# 프로젝트와 Stage 2 결과 저장 위치
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RUN_ROOT = PROJECT_ROOT / "reports" / "local" / "ppo_realtime_stage2_repeat9"


# Stage 1에서 가져올 모델
TRANSFER_CONFIG = {
    "enabled": True,
    "model_path": (
        PROJECT_ROOT
        / "reports"
        / "local"
        / "ppo_realtime_stage1"
        / "train"
        / "seed_11_20261029_174009_109300"
        / "models"
        / "ppo_400000.zip"
    ),

    # 새로운 방향을 탐색할 수 있도록 행동 출력층을 완화합니다.
    "action_head_scale": 0.1,
}


# 학습 환경
ENV_NAME = "RealtimeMansionEnv"

ENV_CONFIG = {
    "stage": 2,
    "mode": "basic",
    "randomize_phase": False,
    "max_episode_steps": None,
    "render_mode": None,
    "safety_coefficient": 0.0,
}


# 관측과 행동
OBSERVATION_CONFIG = {
    "base_size": 630,
    "goal_feature_size": 6,
    "final_size": 636,
    "action_count": 6,
}


# Stage 2 커리큘럼
CURRICULUM_CONFIG = {
    "enabled": True,
    "name": "stage2_safe_route_v1",
    "history_size": 20,
    "required_successes": 16,
    "full_start_probability": 0.10,
    "review_probability": 0.20,
    "final_full_start_probability": 0.80,
}


# 신경망
NETWORK_CONFIG = {
    "net_arch": {
        "pi": [64, 64],
        "vf": [64, 64],
    },
    "activation": "Tanh",
}

ACTION_REPEAT = 9
# PPO 알고리즘
PPO_CONFIG = {
    "policy": "MlpPolicy",
    "learning_rate": 0.00005,
    "n_steps": 100,
    "batch_size": 100,
    "n_epochs": 5,
    "gamma": 0.9994 ** ACTION_REPEAT,
    "gae_lambda": 1.0,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.01,
    "device": "cpu",
    "verbose": 0,
}


# Stage 2에서 추가로 학습할 양
TRAIN_CONFIG = {
    "n_envs": 4,
    "torch_threads": 1,
    "default_seed": 11,
    "training_seeds": [11, 22, 33],
    "total_timesteps": 1_000_000,
    "checkpoint_steps": list(range(20_000, 1_000_001, 20_000)),
    "smoke_timesteps": 2_000,
}


# 보상
REWARD_VERSION = "ppo_collection_return_score20_v1"

REWARD_CONFIG = {
    "quest": 1.0,
    "warmth_per_point": 0.1,
    "warmth_cap": 14,
    "surplus_warmth_per_point": 0.3,
    "success": 10.0,
    "score_divisor": 20.0,
    "star_bonus": 2.0,
    "failure": -2.0,
    "damage": -0.1,
}


# 원래 시작점에서 평가
EVAL_CONFIG = {
    "use_curriculum": False,
    "deterministic_seeds": list(range(10_000, 10_050)),
    "stochastic_seeds": list(range(10_000, 10_100)),
    "action_seed_base": 50_000,
    "success_rate_threshold": 0.90,
    "minimum_training_seeds": 3,

    # 유령 A의 초기 위치·카운트다운 8가지 조합도 검사합니다.
    "check_patrol_phases": True,
}


# 성능 하락 감시
STABILITY_CONFIG = {
    # 결정적 평가 성공률이 90%에 도달하면 감시를 활성화합니다.
    "activation_success_rate": 0.90,

    # 최고 성공률보다 10%p 넘게 낮아진 평가를 하락으로 봅니다.
    "max_success_drop": 0.10,

    # 하락이 두 번 연속 발생하면 학습을 중단하고 최적 모델을 복원합니다.
    "patience": 2,

    # 기본 과제를 익힌 이후에는 탐색 강도를 낮춥니다.
    "ent_coef_after_success": 0.001,
}


# 기록
LOG_CONFIG = {
    "save_episode_csv": True,
    "save_config": True,
    "tensorboard": True,
}