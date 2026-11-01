"""같은 기준 가중치에서 A2C·PPO를 보수적인 공통 설정으로 비교합니다."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = ROOT / '_vendor' / 'a2c'
RUN_ROOT = ROOT / 'runs'
SOURCE_ARCHIVE = Path.home() / 'Downloads' / 'a2c_compact.zip'
SOURCE_CHECKPOINT = 'shared_snapshot/checkpoints/step_0006200068.zip'
BASELINE_CHECKPOINT = ROOT / 'runs/seed_11_20261101_161343_544404/models/baseline_model.zip'
BASELINE_SHA256 = '2b7387396de5b757dceb39598b807448f0e43fcc8fdd51fc0b839eae1d7d4f19'

TRAINING = {
    'seed': 11,
    'physical_frames': 300_000,
    'n_envs': 4,
    'torch_threads': 1,
    'evaluation_interval_frames': 100_000,
    'progress_interval_frames': 25_000,
    'stages': list(range(1, 13)),
}

# 두 알고리즘에 같은 값을 적용합니다. 효과는 새 비교 실험에서 검증합니다.
COMMON_UPDATES = {
    'learning_rate': 0.00005,
    'ent_coef': 0.005,
    'n_steps': 128,
}

PPO_UPDATES = {
    'batch_size': 128,
    'n_epochs': 4,
    'clip_range': 0.1,
    'target_kl': 0.02,
}

EVALUATION = {
    'stages': list(range(1, 13)),
    'periodic_stochastic_episodes': 50,
    'periodic_deterministic_episodes': 1,
    'periodic_seed_base': 10_000,
    'final_stochastic_episodes': 100,
    'final_deterministic_episodes': 10,
    'final_seed_base': 30_000,
    'action_seed_offset': 50_000_000,
    'target_success_rate': 0.90,
    'target_three_star_rate': 0.90,
    'protected_baseline_rate': 0.80,
    'maximum_stage_drop': 0.10,
    'protected_metrics': ['success_rate', 'three_star_rate', 'quest_rate'],
    'require_all_stages_passed_for_adoption': True,
}

# benchmark는 기존 seed 11 실험과 같은 조건입니다. 귀환 보강은 별도 프로필입니다.
PROFILES = ('benchmark', 'return_practice')
BENCHMARK_REFERENCE = ROOT / 'runs/comparisons/seed_11_20261101_172005_698191.json'
RETURN_PRACTICE = {
    'version': 'native_return_practice_v1',
    'bank_file': str(ROOT / 'practice_states' / 'return_v1.json'),
    'stages': [3, 4, 9, 10],
    'preparation_seed': 11,
    'distance_bands': [1, 3, 6, 10, 16, 24, 32],
    'practice_probability': 0.70,
    'history_size': 8,
    'required_successes': 6,
    'review_probability': 0.20,
    'return_progress_per_cell': 0.10,
    'time_per_frame': -0.0002,
    'three_star_return_bonus': 0.5,
}
