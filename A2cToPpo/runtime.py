"""독립 게임 코드의 로딩과 공통 관측·신경망·가중치 전이를 담당합니다."""

from copy import deepcopy
import json
from pathlib import Path
import sys

from . import config as cfg
from .config import VENDOR_ROOT
from .prepare import sha256


def bootstrap():
    manifest_path = VENDOR_ROOT / 'transfer_manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('먼저 python -m A2cToPpo.prepare 명령으로 원본을 준비하세요.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = VENDOR_ROOT / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError('복사한 원본이 변경되었습니다: ' + name)
    root = str(VENDOR_ROOT)
    loaded = sys.modules.get('a2c_game')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(VENDOR_ROOT.resolve()):
        raise RuntimeError('다른 a2c_game이 먼저 로드되었습니다. 별도 Python 프로세스를 사용하세요.')
    if root not in sys.path:
        sys.path.insert(0, root)
    from a2c_game.paths import ensure_game_path
    ensure_game_path()
    return manifest


def source_config():
    return json.loads((VENDOR_ROOT / 'shared_snapshot/config.json').read_text(encoding='utf-8'))


def profile_settings(profile):
    if profile not in cfg.PROFILES:
        raise ValueError('지원하지 않는 실험 프로필입니다: ' + profile)
    settings = source_config()
    if profile == 'return_practice':
        from .return_practice import ensure_bank
        recipe = deepcopy(cfg.RETURN_PRACTICE)
        _, digest = ensure_bank(settings,recipe)
        settings['return_practice'] = {**recipe,'enabled':True,'bank_sha256':digest}
    return settings


def make_progress(settings):
    recipe = settings.get('return_practice',{})
    if not recipe.get('enabled',False):
        return None
    from .return_practice import ReturnProgress, ensure_bank
    bank,digest = ensure_bank(settings,recipe)
    if digest != recipe['bank_sha256']:
        raise ValueError('귀환 준비 상태 파일이 변경되었습니다.')
    return ReturnProgress(bank,recipe)


def make_env(settings, *, stages=None, seed=11, training=False, progress=0.0, practice_progress=None):
    from a2c_game.environment import UnifiedMansionEnv
    env = UnifiedMansionEnv(deepcopy(settings), stages=stages, seed=seed, training=training)
    env.set_training_progress(progress)
    recipe = settings.get('return_practice',{})
    if training and recipe.get('enabled',False):
        from .return_practice import ReturnPractice, ensure_bank
        bank,digest = ensure_bank(settings,recipe)
        if digest != recipe['bank_sha256']:
            raise ValueError('귀환 준비 상태 파일이 변경되었습니다.')
        env = ReturnPractice(env,settings,bank,recipe,practice_progress)
    return env


def source_steps(model):
    return int(getattr(model, 'a2c_to_ppo', {}).get('source_decision_steps', model.num_timesteps))


def load_model(path, algorithm):
    from stable_baselines3 import A2C, PPO
    algorithms = {'PPO': PPO, 'A2C': A2C}
    if algorithm not in algorithms:
        raise ValueError('알고리즘은 PPO 또는 A2C여야 합니다.')
    return algorithms[algorithm].load(path, device='cpu')


def load_baseline():
    path = Path(cfg.BASELINE_CHECKPOINT)
    if not path.is_file() or sha256(path) != cfg.BASELINE_SHA256:
        raise ValueError('고정한 학습 전 baseline_model.zip이 없거나 변경되었습니다: ' + str(path))
    # 보관 형식은 PPO이며, 가중치는 추가 학습 전 A2C 정책 그대로입니다.
    source = load_model(path, 'PPO')
    if source.num_timesteps != 0 or source.a2c_to_ppo.get('physical_frames') != 0:
        raise ValueError('추가 학습 전 0프레임 기준 모델만 사용할 수 있습니다.')
    if source.a2c_to_ppo.get('source_sha256') != sha256(VENDOR_ROOT / cfg.SOURCE_CHECKPOINT):
        raise ValueError('기준 모델의 출처가 보관한 A2C 선정 모델과 다릅니다.')
    return source


def transfer_model(vec_env, seed=11, *, algorithm='PPO'):
    import torch
    from stable_baselines3 import A2C, PPO
    algorithm = algorithm.upper()
    if algorithm not in ('PPO', 'A2C'):
        raise ValueError('알고리즘은 PPO 또는 A2C여야 합니다.')
    source = load_baseline()
    if source.observation_space != vec_env.observation_space or source.action_space != vec_env.action_space:
        raise ValueError('A2C 원본과 새 환경의 관측·행동 공간이 다릅니다.')
    shared = {
        'gamma': source.gamma, 'gae_lambda': source.gae_lambda,
        'vf_coef': source.vf_coef,
        'max_grad_norm': source.max_grad_norm,
        'normalize_advantage': source.normalize_advantage,
        **cfg.COMMON_UPDATES,
    }
    constructor, options = ((PPO, deepcopy(cfg.PPO_UPDATES)) if algorithm == 'PPO'
                            else (A2C, {'use_rms_prop': False}))
    model = constructor(source.policy_class, vec_env, seed=seed, device='cpu', verbose=0,
                        policy_kwargs=deepcopy(source.policy_kwargs), **shared, **options)
    model.policy.load_state_dict(source.policy.state_dict(), strict=True)
    if not all(torch.equal(value, model.policy.state_dict()[name])
               for name, value in source.policy.state_dict().items()):
        raise RuntimeError('전이 후 가중치가 원본과 다릅니다.')
    model.a2c_to_ppo = {
        'algorithm': algorithm, 'initialization': 'A2C_pretrained_weights',
        'source_checkpoint': str(cfg.BASELINE_CHECKPOINT), 'source_sha256': cfg.BASELINE_SHA256,
        'source_decision_steps': source_steps(source),
        'original_hyperparameters': {'learning_rate': float(source.lr_schedule(1.0)),
                                     'ent_coef': source.ent_coef},
        'physical_frames': 0, 'shared_hyperparameters': shared,
        'algorithm_updates': options,
    }
    return source, model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=str), encoding='utf-8')
