"""환경·A2C·보상·평가의 공통 설정과 유효성 검사를 모읍니다.

현재 값은 비교 실험을 위한 설정이며 모든 Stage의 최적값이라는 뜻은 아닙니다.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

VERSION = 'native_all_stages_single_a2c_v1'
ROLLOUT_STEPS = 128
# 환경 4개 × 각 환경 32행동을 모아 한 번 갱신합니다. 학습 스텝은 네 환경의 합계입니다.
DEFAULT_CONFIG = {
    'schema_version': 1, 'experiment': VERSION,
    'environment': {'stages': list(range(1, 13)), 'mode': 'basic',
        'action_repeat': 18, 'randomize_phase': False, 'auto_interaction': True,
        'max_policy_steps': None},
    'a2c': {'learning_rate': .0002, 'n_steps': 32, 'n_envs': 4,
        'gamma': .999 ** 3, 'gae_lambda': 1., 'ent_coef': .03,
        'vf_coef': .5, 'max_grad_norm': .5, 'use_rms_prop': False,
        'normalize_advantage': True, 'device': 'cpu', 'torch_threads': 1,
        'net_arch': {'pi': [128, 128], 'vf': [128, 128]}},
    'policy': {'logit_bound': 4., 'features_version': 'small_map_layernorm_tanh_v1',
        'safeguards_are_unvalidated_hypotheses': True},
    # 실제 게임 점수는 그대로 두고 아래 항목을 학습용 보상으로 사용합니다.
    'reward': {'quest': .05, 'pickup_per_value': .05, 'score_scale': 1.,
        'highscore_bonus': 1., 'failure': -1., 'damage': -.02,
        'wall_stall': -.001, 'potential_scale': .2,
        'novelty_per_state': .01, 'novelty_cap': .2},
    # Stage는 성공 여부와 별개로 25만 스텝마다 추가합니다. 전12개 노출은 275만부터입니다.
    'training': {'seed': 11, 'total_steps': 10_000_000,
        'stage_interval_steps': 250_000, 'eval_every_steps': 100_000,
        'progress_every_updates': 25, 'checkpoint_every_steps': 100_000,
        'entropy_adaptation': {'enabled': True, 'min': .01, 'max': .1,
            'target': .35, 'interval_updates': 100}},
    # 정기 평가는 최대확률 행동 1판·확률 행동 5판이며 최종 평가는 별도 시드를 씁니다.
    'evaluation': {'episodes': 5, 'seed': 10_000, 'final_episodes': 40,
        'final_seed_sets': [70_000_000, 80_000_000], 'required_star3_rate': .9,
        'modes': ['greedy', 'stochastic'], 'score_thresholds': [40, 45, 65, 85]},
}


def _number(value, name, *, integer=False, zero=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (not zero and value == 0)
            or (integer and type(value) is not int)):
        raise ValueError(f'Invalid positive setting: {name}')


def validate_config(config):
    """공통 실험의 환경·입력·학습 단위를 바꾸는 잘못된 설정을 실행 전에 막습니다."""
    if (not isinstance(config, dict) or type(config.get('schema_version')) is not int
            or config.get('schema_version') != 1 or config.get('experiment') != VERSION):
        raise ValueError('Unsupported unified-A2C configuration version.')
    env = config['environment']
    if env.get('observation_profile', 'native') not in ('native', 'exit_distance_field_v1'):
        raise ValueError('Unsupported environment observation_profile.')
    if type(env.get('fast_rollout', True)) is not bool:
        raise ValueError('environment.fast_rollout must be boolean.')
    if (env['stages'] != list(range(1, 13))
            or any(type(stage) is not int for stage in env['stages']) or env['mode'] != 'basic'
            or type(env['action_repeat']) is not int
            or env['action_repeat'] != 18 or env['auto_interaction'] is not True
            or env['randomize_phase'] is not False or env['max_policy_steps'] is not None):
        raise ValueError('Preserve the original twelve basic stages, repeat18, and native endings.')
    a2c = config['a2c']
    for key in ('n_envs', 'n_steps', 'torch_threads'):
        _number(a2c[key], f'a2c.{key}', integer=True)
    if (a2c['n_envs'] != 4 or a2c['n_steps'] != 32 or a2c['device'] != 'cpu'
            or a2c['torch_threads'] != 1 or a2c['use_rms_prop'] is not False
            or a2c['normalize_advantage'] is not True):
        raise ValueError('This experiment uses four32-step CPU/Adam A2C environments.')
    for key in ('learning_rate', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'max_grad_norm'):
        _number(a2c[key], f'a2c.{key}', zero=key in ('gae_lambda', 'ent_coef'))
    if not 0 < a2c['gamma'] <= 1 or not 0 <= a2c['gae_lambda'] <= 1:
        raise ValueError('Discount/GAE must be within their probability ranges.')
    for branch in ('pi', 'vf'):
        if not isinstance(a2c['net_arch'].get(branch), list) or not a2c['net_arch'][branch]:
            raise ValueError('Actor and critic hidden widths are required.')
        for width in a2c['net_arch'][branch]:
            _number(width, f'net_arch.{branch}', integer=True)
    _number(config['policy']['logit_bound'], 'policy.logit_bound')
    if config['policy'].get('features_version') != 'small_map_layernorm_tanh_v1':
        raise ValueError('Unsupported feature encoder.')
    reward = config['reward']
    if set(reward) != set(DEFAULT_CONFIG['reward']):
        raise ValueError('Reward terms must match the declared native85 experiment.')
    for key, value in reward.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f'Nonfinite reward: {key}')
        if (key in ('failure', 'damage', 'wall_stall') and value > 0
                or key not in ('failure', 'damage', 'wall_stall') and value < 0):
            raise ValueError(f'Invalid reward sign: {key}')
    training = config['training']
    for key in ('total_steps', 'stage_interval_steps', 'eval_every_steps',
                'progress_every_updates', 'checkpoint_every_steps'):
        _number(training[key], f'training.{key}', integer=True)
    if training['total_steps'] % ROLLOUT_STEPS or training['total_steps'] > 10_000_000:
        raise ValueError('Aggregate learner budget must be a128 multiple up to10,000,000.')
    if type(training['seed']) is not int or not 0 <= training['seed'] < 2**31 - 4:
        raise ValueError('Training seed must be a nonnegative32-bit integer.')
    adaptive = training['entropy_adaptation']
    if type(adaptive['enabled']) is not bool:
        raise ValueError('Entropy adaptation enabled must be boolean.')
    for key in ('min', 'max', 'target'):
        _number(adaptive[key], f'entropy_adaptation.{key}', zero=key == 'min')
    _number(adaptive['interval_updates'], 'entropy interval', integer=True)
    if not adaptive['min'] <= a2c['ent_coef'] <= adaptive['max'] or adaptive['target'] > math.log(5):
        raise ValueError('Entropy bounds must contain the starting coefficient; target <=log5.')
    evaluation = config['evaluation']
    for key in ('episodes', 'final_episodes'):
        _number(evaluation[key], f'evaluation.{key}', integer=True)
    if type(evaluation['seed']) is not int or not 0 <= evaluation['seed'] < 2**31 - 13_000_000:
        raise ValueError('Invalid evaluation seed.')
    seeds = evaluation['final_seed_sets']
    if (not isinstance(seeds, list) or len(seeds) != 2 or len(set(seeds)) != 2
            or any(type(seed) is not int or not 0 <= seed < 2**31 - 13_000_000 for seed in seeds)):
        raise ValueError('Two distinct final evaluation seed sets are required.')
    if (evaluation['modes'] != ['greedy', 'stochastic']
            or evaluation['score_thresholds'] != [40, 45, 65, 85]
            or type(evaluation['required_star3_rate']) not in (int, float)
            or not math.isfinite(evaluation['required_star3_rate'])
            or not 0 < evaluation['required_star3_rate'] <= 1):
        raise ValueError('Keep fixed score thresholds and both evaluation modes.')
    if 'initialization' in config or 'frozen_prefix' in config:
        raise ValueError('Historical controllers and prepared starts are not part of this experiment.')
    return config


def default_config(seed=11, total_steps=10_000_000):
    """기본 설정을 복사해 실행별 학습 시드와 총 스텝만 지정합니다."""
    config = deepcopy(DEFAULT_CONFIG)
    config['training'].update(seed=seed, total_steps=total_steps)
    return validate_config(config)


def load_config(path=None):
    return validate_config(json.loads(Path(path).read_text(encoding='utf-8'))
                           if path is not None else default_config())


def stage_pool(config, steps):
    """누적 학습 스텝으로 노출 Stage를 정합니다. 85점 통과 판정과는 별개입니다."""
    if type(steps) is not int or steps < 0:
        raise ValueError('Learner steps must be a nonnegative integer.')
    last_stage = min(12, 1 + steps // config['training']['stage_interval_steps'])
    return list(range(1, last_stage + 1))


stage_pool_for_steps = stage_pool
