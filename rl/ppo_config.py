from pathlib import Path


# 프로젝트와 목표 정보 분리 실험의 결과 저장 위치
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 확정 모델과 새 학습 결과를 분리합니다.
RUN_ROOT = PROJECT_ROOT / "reports" / "local" / "experiments" / "stage3_return_connection"

# Stage 3에서 익힌 수집·물건 획득을 보존하고 긴 귀환 구간을 보완합니다.
TRANSFER_CONFIG = {
    "enabled": True,
    "model_path": (
        PROJECT_ROOT / "reports/local/experiments/stage3_efficient_three_stars/quick"
        / "seed_11_20261101_123909_108588/models/best_model.zip"
    ),
    "action_head_scale": 1.0,
}

# 별 3개 전체 학습과 별개인 물건 획득 연습입니다.
QUEST_TASK_CONFIG = {
    "name": "stage2_quest_acquisition_v1",
    "run_root": PROJECT_ROOT / "reports/local/experiments/stage2_quest_acquisition",
    "source_model": (
        PROJECT_ROOT / "reports/local/experiments/stage2_quest_first/quick"
        / "seed_11_20261101_013936_887142/models/best_model.zip"
    ),
    "starts": [6, 7],
    "action_head_scale": 0.3,
    "reset_value_head": True,
    "learning_rate": 0.0003,
    "ent_coef": 0.03,
    "gamma": 0.99,
    "frame_limit": 180,
    "timesteps": 90_000,
    "quick_timesteps": 90_000,
    "evaluation_interval": 9_000,
    "success_bonus": 10.0,
    "progress_per_cell": 1.0,
    "time_per_frame": -0.005,
    "damage": -1.0,
    "timeout": -2.0,
    "evaluation_seeds": list(range(10_000, 10_020)),
    "heldout_seeds": list(range(30_000, 30_050)),
    "success_rate_threshold": 0.90,
}

# 초반 수집과 물건 획득을 연결한 뒤 전체 별 과제로 돌아갑니다.
CONNECTION_TASK_CONFIG = {
    "name": "stage2_magnet_quest_connection_v1",
    "profile": "early_review_v2",
    "run_root": PROJECT_ROOT / "reports/local/experiments/stage2_magnet_connection_early_review",
    "source_model": (
        PROJECT_ROOT / "reports/local/experiments/stage2_magnet_connection/quick"
        / "seed_11_20261101_031141_705856/models/best_model.zip"
    ),
    "starts": [0, 1, 2, 3, 4, 5],
    # 초반 수집 환경 안에서 0·1번에 80%, 나머지 구간에 20%를 배정합니다.
    "start_weights": [0.40, 0.40, 0.05, 0.05, 0.05, 0.05],
    "focus_starts": [0, 1],
    "focus_success_rate_threshold": 0.90,
    "frame_limit": 360,
    "success_bonus": 5.0,
    "timeout": -2.0,
    "learning_rate": 0.00005,
    "ent_coef": 0.01,
    "timesteps": 150_000,
    "quick_timesteps": 150_000,
    "evaluation_interval": 15_000,
    "evaluation_seeds": list(range(20_000, 20_050)),
    "heldout_seeds": list(range(40_000, 40_050)),
    "minimum_training_frames": 30_000,
    "required_consecutive_passes": 2,
    "success_rate_threshold": 0.80,
    "quest_safety_floor": 0.80,
    "quest_target": 0.90,
    "stop_after_quest_regressions": 2,
}

JOINT_REVIEW_CONFIG = {
    "enabled": True,
    "seeds": list(range(10_000, 10_020)),
    "frame_limit": 180,
    "quest_safety_floor": 0.80,
    "quest_target": 0.90,
    "magnet_target": 0.80,
    "stop_after_quest_regressions": 2,
    "require_quest_source_ready": True,
    "require_connection_source_ready": True,
}

FINAL_STAGE2_CONFIG = {
    "enabled": True,
    "minimum_training_frames": 100_000,
    "required_consecutive_passes": 2,
    "regression_patience": 3,
    "heldout_seed_base": 60_000,
    "heldout_episodes": 200,
}

STAGE3_TRAINING_CONFIG = {
    "enabled": True,
    "minimum_training_frames": 200_000,
    "required_consecutive_passes": 2,
    "heldout_seed_base": 70_000,
    "heldout_episodes": 200,
}


# 학습 환경
ENV_NAME = "RealtimeMansionEnv"

ENV_CONFIG = {
    "stage": 3,
    "mode": "basic",
    "randomize_phase": False,
    "max_episode_steps": None,
    "render_mode": None,
    "safety_coefficient": 0.0,
}


# 관측과 행동
OBSERVATION_CONFIG = {
    "base_size": 630,
    "movement_feature_size": 4,
    "goal_feature_size": 6,
    "final_size": 640,
    "action_count": 6,
}


# 진행 구간 외에도 물건 획득 전과 유령 근처 수집을 별도 비중으로 연습합니다.
CURRICULUM_CONFIG = {
    "enabled": True,
    "name": "stage3_efficient_three_star_v2",
    "initial_frontier": 18,
    "quick_initial_frontier": 18,
    "history_size": 20,
    "required_successes": 16,
    "full_start_probability": 0.05,
    "review_probability": 0.05,
    "final_full_start_probability": 0.80,
    "stage3_collect_probability": 0.05,
    "stage3_collect_starts": [1, 2, 3, 4, 5, 6, 7, 8],
    "stage3_quest_probability": 0.05,
    "stage3_quest_starts": [9, 10, 11, 12],
    "stage3_return_probability": 0.40,
    "stage3_return_starts": [13, 14, 15, 16, 17, 18],
    "magnet_focus_probability": 0.05,
    "magnet_focus_starts": [1, 2, 3, 4, 5],
    "quest_focus_probability": 0.10,
    "quest_focus_starts": [6, 7],
    "ghost_focus_probability": 0.35,
    "ghost_focus_starts": [9, 10, 11, 12, 13, 14],
    "ghost_wait_frames": list(range(0, 144, 9)),
    "ghost_max_wait_frames": 144,
    "ghost_safe_preparation": True,
    "keep_focus_at_frontier_zero": True,
}


# 신경망
NETWORK_CONFIG = {
    "features_extractor": "world_goal_branches_v1",
    "feature_branch_size": 32,
    "net_arch": {
        "pi": [64, 64],
        "vf": [64, 64],
    },
    "activation": "Tanh",
}

ACTION_REPEAT = 9
ACTION_MASK_CONFIG = {"mask_ineffective_interaction": True}

# 물리 프레임 기준. 대기·상호작용과 좁은 구간 왕복도 포함합니다.
STAGNATION_CONFIG = {
    "stationary_grace_frames": 30,
    "local_window_frames": 120,
    "local_max_span": 0.75,
}
# PPO 알고리즘
PPO_CONFIG = {
    "policy": "MovementAwareActorCriticPolicy",
    "learning_rate": 0.0001,
    "n_steps": 128,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.9994 ** ACTION_REPEAT,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "clip_range_vf": None,
    "normalize_advantage": True,
    # 수집 경로를 새로 탐색하되 효과 없는 상호작용은 행동 마스크로 제외합니다.
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.02,
    "device": "cpu",
    "verbose": 0,
}


# 단위는 환경 4개를 합한 물리 프레임입니다. 9프레임마다 행동을 선택합니다.
TRAIN_CONFIG = {
    "n_envs": 4,
    "torch_threads": 1,
    "default_seed": 11,
    "training_seeds": [11, 22, 33],
    "total_timesteps": 1_000_000,
    "checkpoint_steps": list(range(50_000, 1_000_001, 50_000)),
    "smoke_timesteps": 2_000,
    "quick_timesteps": 300_000,
    "quick_checkpoint_interval": 25_000,
}


# 보상
REWARD_VERSION = "ppo_stage3_return_connection_v2"

# 게임의 필수 온기 4점은 유지하고 목표 안내만 추가 수집으로 확장합니다.
STAR_GOAL_CONFIG = {
    "enabled": True,
    "direction_encoding": "local_clipped_v1",
    "warmth_target": 12,
    "stage1_three_star": False,
    "stage2_three_star": False,
    "stage3_three_star": True,
    "stage3_route_plan": True,
    "stage3_score_aware_warmth": True,
    "stage3_waypoint_mode": "corridor_lookahead_v1",
    "stage3_return_reserve_ticks": 3,
    "magnet_min_warmth": 2,
    "quest_after_magnet": True,
    "magnet_pickup_targets": True,
    "collection_path_reward": True,
    "path_waypoints": True,
    "min_remaining_ticks": 12,
    "min_courage": 1,
}

REWARD_CONFIG = {
    "quest": 1.0,
    "stage2_quest": 5.0,
    "stage3_quest": 5.0,
    "stage3_quest_progress_per_cell": 1.0,
    "stage3_basic_warmth_per_point": 0.5,
    "stage3_time_per_frame": -0.002,
    "stage3_return_progress_per_cell": 1.0,
    "stage3_invincibility_pickup": 1.0,
    "stage3_damage_free_success": 2.0,
    "stage2_quest_progress_per_cell": 1.0,
    "warmth_per_point": 0.1,
    "stage2_basic_warmth_per_point": 0.5,
    "stage2_magnet_pickup": 1.0,
    "warmth_cap": 14,
    "surplus_warmth_per_point": 1.0,
    "success": 10.0,
    "score_divisor": 10.0,
    "star_bonus": 6.0,
    # Stage 2 성공 시 별 0/1/2/3개에 대한 보상 조정.
    # 시간·용기 부족 시 낮은 별에 대한 감점은 면제합니다.
    "star_completion_adjustments": [-8.0, -6.0, 6.0, 10.0],
    "warmth_progress_per_cell": 0.25,
    "failure": -2.0,
    "damage": -0.5,
    # Stage 1에서는 별 0~2개 조기 귀환과 별 3개 귀환을 구분합니다.
    "stage1_star_completion_adjustments": [-8.0, -6.0, -3.0, 10.0],
    "stage2_star_completion_adjustments": [-8.0, -6.0, -3.0, 10.0],
    # Stage 2 전용: 원격 수집분은 기존 온기 보상을 이 배율로 대체합니다.
    "magnet_remote_warmth_multiplier": 1.5,
    "stage2_damage": -6.0,
    "stage2_ghost_risk_per_frame": -0.02,
    "stage2_ghost_risk_distance": 1.1,
    "stage3_damage": -1.5,
    "stage3_star_completion_adjustments": [-8.0, -6.0, -3.0, 10.0],
    "stage2_damage_free_success": 8.0,
    "stage2_damage_free_requires_three_stars": True,
    # 별 3개 목표에서는 벽을 반영한 경로 거리 증감으로 귀환을 안내합니다.
    "return_progress_per_cell": 0.5,
    # 이동 명령(0~3)을 내렸으나 실제로 움직이지 못한 물리 프레임당 벌점.
    "blocked_move_per_frame": -0.01,
    # 수집 안내 중에는 물건 접근 보상을 중단해 목표 간 충돌을 막습니다.
    "quest_progress_per_cell": 0.5,
    # 모든 물리 프레임에 적용하며 대기·상호작용에도 시간이 소모됩니다.
    "time_per_frame": -0.001,
    "stagnation_per_frame": -0.01,
    "magnet_ten_success": 0.0,
}


# 원래 시작점에서 평가
EVAL_CONFIG = {
    "use_curriculum": False,
    # stochastic / deterministic / all 중 주 평가 기준을 선택합니다.
    "primary_mode": "stochastic",
    # 실패한 판의 별 0개도 포함한 전체 평균을 사용합니다.
    "selection_objective": "three_stars",
    "safe_return_selection": {
        "minimum_success_rate": 0.90,
        "minimum_three_star_rate": 0.80,
        "minimum_mean_stars": 2.5,
        "minimum_magnet_use_rate": 0.90,
        "damage_weight": 0.20,
        "damage_free_weight": 0.10,
    },
    # 성공률·별 개수 하한을 지킨 후보에서 자석 활용과 피해를 비교합니다.
    "selection_settings": {
        "minimum_success_rate": 0.97,
        "maximum_stagnation_frame_rate": 0.06,
        "require_regression_seed_success": True,
        "regression_seed": 11,
        "minimum_mean_stars": 1.6,
        "magnet_use_weight": 0.5,
        "remote_warmth_weight": 0.1,
        "damage_weight": 0.5,
        "stagnation_weight": 0.25,
    },
    "evaluate_transfer_baseline": True,
    # 실패를 포함한 별 3개 귀환율 90%와 평균 별 2.7개를 목표로 합니다.
    "target_mean_stars": 2.7,
    "target_three_star_rate": 0.90,
    "deterministic_seeds": [11, *range(10_000, 10_019)],
    "stochastic_seeds": [11, *range(10_000, 10_099)],
    "action_seed_base": 50_000,
    "success_rate_threshold": 0.90,
    "minimum_training_seeds": 3,
    # 짧은 실험도 실제 시작점에서 같은 확률적 100판을 평가합니다.
    "quick_deterministic_episodes": 20,
    "quick_stochastic_episodes": 100,

    # 유령 상태별 결정적 평가는 진단용으로 계속 기록합니다.
    "check_patrol_phases": True,

    "minimum_magnet_use_rate": 0.0,
    "maximum_mean_damage": 1.0,
    "minimum_damage_free_success_rate": 0.0,
}


# 성능 하락 감시
STABILITY_CONFIG = {
    # 주 평가 기준의 성공률이 90%에 도달하면 감시를 활성화합니다.
    "activation_success_rate": 0.90,

    # 최고 성공률보다 10%p 넘게 낮아진 평가를 하락으로 봅니다.
    "max_success_drop": 0.10,

    # 하락이 두 번 연속 발생하면 학습을 중단하고 최적 모델을 복원합니다.
    "patience": 2,

    # 확률적 플레이의 행동 다양성을 유지합니다. 필요할 때만 명시적으로 켭니다.
    "reduce_entropy_after_success": False,
    "ent_coef_after_success": 0.001,
}


# 기록
LOG_CONFIG = {
    "save_episode_csv": True,
    "save_config": True,
    "tensorboard": True,
}
