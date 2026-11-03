"""2,388 입력 단일 PPO를 임의 가중치·빈 Adam·0스텝부터 학습합니다."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import time

import torch

from . import train as runner
from .learning_utils import coverage_rank, development_goal, StageProgressLog
from .config import default_config, load_config, validate_config
from .environment import UnifiedMansionEnv
from .evaluation import evaluate
from .file_io import replace_with_retry
from .navigation_observation import NAVIGATION_PROFILE, OBSERVATION_DIMENSIONS
from .paths import PROJECT_ROOT
from .policy import UnifiedA2CPolicy
from .refinement import RefinementEnv, reward_profile_settings
from .runtime import KST, resolve_run, set_latest_run
from .visual import frame_from_snapshot


PROTOCOL = 'fresh_2388_all12_v1'
STAGES = list(range(1, 13))
TERMINAL = {'goal_achieved', 'complete_not_passed'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fresh_config(seed=11, total_steps=10_000_000):
    config = default_config(seed=seed, total_steps=total_steps)
    config['environment'].update(observation_profile=NAVIGATION_PROFILE, fast_rollout=True)
    config['evaluation'].update(episodes=20, final_episodes=40,
        final_seed_sets=[270_000_000, 280_000_000])
    config['refinement'] = {'version': PROTOCOL,
        **reward_profile_settings('return_progress_guard'), 'sampler_version': PROTOCOL}
    config['fresh_training'] = {'version': PROTOCOL, 'input_dimensions': 2388,
        'warmup_steps': 200_000, 'focus_rotate_steps': 25_000,
        'round_steps': 100_000, 'required_star3_rate': .9,
        'selection': 'all12_coverage_then_worst_stage_then_macro',
        'retention_behavior': 'record_regression_and_preserve_best; continue_learning',
        'scope': 'fixed native ghost phase; heldout action seeds'}
    config['ppo'] = runner.ppo_settings(config)
    return validate_config(config)


def stage_layout(config, steps, metrics):
    settings = config['fresh_training']
    if steps < settings['warmup_steps']:
        return [STAGES.copy() for _ in range(4)]
    weak = [s for s in STAGES
        if metrics.get('greedy', {}).get(str(s), {}).get('mean_score', 0) < 85
        or metrics.get('stochastic', {}).get(str(s), {}).get('star3_rate', 0) < .9]
    weak = weak or STAGES
    epoch = max(0, steps - settings['warmup_steps']) // settings['focus_rotate_steps']
    return [STAGES.copy(), STAGES.copy(),
            [weak[(2 * epoch) % len(weak)]], [weak[(2 * epoch + 1) % len(weak)]]]


def save_control(run, control):
    run = Path(run)
    runner.write_json(run / 'fresh_protocol.json', control)
    state = read(run / 'metadata.json')
    state.update(fresh_status=control['status'], goal_achieved=control['goal_achieved'])
    if control.get('selected'):
        state['best'] = {**control['selected'], 'sha256': control['selected']['checkpoint_sha256']}
    runner.write_json(run / 'metadata.json', state)
    runner.write_json(run / 'status.json', {
        **{k: state[k] for k in ('algorithm', 'steps', 'updates', 'rollout_updates', 'total_steps', 'stage_pool')},
        'status': control['status'], 'phase': 'fresh_2388_all12', 'input_dimensions': 2388,
        'initialization': state['initialization'], 'goal_achieved': control['goal_achieved'],
        'latest_checkpoint': state['resume_checkpoint'], 'best_checkpoint': 'selected_model.zip',
        'best': state.get('best'), 'dashboard_url': state.get('dashboard_url')})


def seed_preview(run, config):
    frames = []
    for rank, stage in enumerate(STAGES[:4]):
        env = UnifiedMansionEnv(config, stages=[stage], training=False)
        try:
            _, info = env.reset(seed=10_000 + stage)
            frames.append({'env_rank': rank, 'stage': stage,
                           'frame': frame_from_snapshot(info['snapshot'])})
        finally:
            env.close()
    runner.write_json(Path(run) / 'live.json', {'mode': 'prepared', 'steps': 0,
        'stage': 1, 'frame': frames[0]['frame'], 'environments': frames})


def prepare(config_path=None, *, output_root=None):
    config = load_config(config_path) if config_path else fresh_config()
    config['ppo'] = runner.ppo_settings(config)
    if (config.get('fresh_training', {}).get('version') != PROTOCOL
            or config['environment'].get('observation_profile') != NAVIGATION_PROFILE
            or any(k in config for k in ('comparison', 'all_stage_training'))):
        raise ValueError('Use the clean fresh_2388 configuration.')
    torch.set_num_threads(1)
    vec = runner.DummyVecEnv([lambda i=i: RefinementEnv(config, stages=STAGES,
        seed=config['training']['seed'] + i, training=True) for i in range(4)])
    try:
        a = config['a2c']
        model = runner.PPO(UnifiedA2CPolicy, vec, seed=config['training']['seed'],
            device='cpu', verbose=0, learning_rate=a['learning_rate'], n_steps=a['n_steps'],
            gamma=a['gamma'], gae_lambda=a['gae_lambda'], ent_coef=a['ent_coef'],
            vf_coef=a['vf_coef'], max_grad_norm=a['max_grad_norm'],
            normalize_advantage=a['normalize_advantage'], **config['ppo'],
            policy_kwargs={'net_arch': a['net_arch'], 'activation_fn': torch.nn.Tanh,
                           'logit_bound': config['policy']['logit_bound']})
        runner.assert_integrity(model, config['training']['total_steps'])
        if model.num_timesteps != 0 or model._n_updates != 0 or model.policy.optimizer.state:
            raise ValueError('Fresh preparation must have zero steps, zero updates and empty Adam.')
        root = Path(output_root or PROJECT_ROOT / 'runs').resolve()
        if (root / 'latest').exists() and not (root / 'latest').is_symlink():
            raise ValueError('The latest alias is occupied by a real folder.')
        run = root / f'seed_{config["training"]["seed"]}_{datetime.now(KST):%Y%m%d_%H%M%S_%f}'
        run.mkdir(parents=True, exist_ok=False)
        runner.write_json(run / 'config.json', config)
        model.save(run / 'initial_model.zip')
        for name in ('resume_A.zip', 'selected_model.zip'):
            shutil.copyfile(run / 'initial_model.zip', run / name)
        hashes = runner.source_manifest()
        for name, digest in hashes.items():
            target = run / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / name, target)
            if runner.sha(target) != digest:
                raise ValueError(f'Source changed during preparation: {name}')
        state = {'version': runner.VERSION, 'algorithm': 'PPO', 'single_policy': True,
            'initialization': 'fresh_random_parameters_and_empty_Adam',
            'teacher_actions': False, 'injected_game_states': False, 'frozen_prefix': False,
            'status': 'prepared', 'steps': 0, 'updates': 0, 'rollout_updates': 0,
            'total_steps': config['training']['total_steps'], 'seed': config['training']['seed'],
            'config_sha256': runner.sha(run / 'config.json'), 'source_hashes': hashes,
            'resume_checkpoint': 'resume_A.zip', 'checkpoint_sha256': runner.sha(run / 'resume_A.zip'),
            'resume_count': 0, 'stage_pool': STAGES, 'best': None, 'evaluations': [],
            'next_eval_steps': 100_000, 'pending_evaluation': None, 'final_validation': None,
            'goal_achieved': False, 'entropy_coef': config['a2c']['ent_coef'],
            'fresh_version': PROTOCOL, 'reward_profile': 'return_progress_guard',
            'stage_exposure_scope': 'this fresh experiment from zero',
            'stage_exposure': {str(s): dict.fromkeys(('learner_steps', 'physical_frames',
                'episodes', 'returns', 'highscores'), 0) for s in STAGES},
            'versions': {p: version(p) for p in ('stable-baselines3', 'torch', 'gymnasium', 'numpy')},
            'resume_restores_weights_adam_counters': True,
            'resume_restores_native_episode_and_rng': False}
        runner.write_json(run / 'metadata.json', state)
        runner.write_json(run / 'stageexposure.json', state['stage_exposure'])
        control = {'version': PROTOCOL, 'status': 'prepared', 'baseline': None,
            'completed_rounds': [], 'pending_round': None, 'latest_metrics': {},
            'selected': None, 'final_validation': None, 'goal_achieved': False}
        save_control(run, control)
        seed_preview(run, config)
        set_latest_run(run)
        print(f'2,388 입력 새 공통 모델 준비: {run}\n0스텝·0업데이트·빈 Adam. 추가 학습0.', flush=True)
        return run
    finally:
        vec.close()


def verify(run):
    run = resolve_run(run)
    state, config = read(run / 'metadata.json'), load_config(run / 'config.json')
    if (state.get('version') != runner.VERSION or state.get('algorithm') != 'PPO'
            or config.get('ppo') != runner.ppo_settings(config)
            or state.get('fresh_version') != PROTOCOL
            or config.get('fresh_training', {}).get('version') != PROTOCOL
            or config['environment'].get('observation_profile') != NAVIGATION_PROFILE):
        raise ValueError('This is not a fresh 2,388-input PPO experiment.')
    if runner.sha(run / 'config.json') != state['config_sha256']:
        raise ValueError('Saved configuration changed.')
    for name, digest in state['source_hashes'].items():
        if runner.sha(PROJECT_ROOT / name) != digest or runner.sha(run / 'source_snapshot' / name) != digest:
            raise ValueError(f'Source differs from the saved experiment: {name}')
    if runner.sha(run / state['resume_checkpoint']) != state['checkpoint_sha256']:
        raise ValueError('Committed checkpoint changed.')
    model = runner.load_candidate(run / state['resume_checkpoint'])
    runner.assert_integrity(model, state['total_steps'])
    if (model.num_timesteps, model._n_updates) != (state['steps'], state['updates']):
        raise ValueError('Checkpoint counters differ from metadata.')
    dimensions = sum(math.prod(space.shape) for space in model.observation_space.spaces.values())
    if dimensions != OBSERVATION_DIMENSIONS or dimensions != 2388:
        raise ValueError('The fresh model must have exactly 2,388 inputs.')
    if state['steps'] == 0 and (state['updates'] or model.policy.optimizer.state):
        raise ValueError('A zero-step fresh model must retain empty Adam.')
    return run, config, state, model


@contextmanager
def training_hooks(run, metrics):
    previous = runner.UnifiedMansionEnv, runner.update_sampling, runner.LearningLog
    def sampling(vec, config, steps):
        layout = stage_layout(config, steps, metrics)
        for env, stages in zip(vec.envs, layout, strict=True):
            env.set_stage_pool(stages)
            env.set_training_progress(min(1., steps / config['training']['total_steps']))
        record = {'steps': steps, 'environment_stage_pools': layout, 'sampler_version': PROTOCOL}
        path = Path(run) / 'sampling.json'
        if not path.is_file() or read(path).get('environment_stage_pools') != layout:
            runner.append_json(Path(run) / 'sampling_history.jsonl', record)
        runner.write_json(path, record)
        return STAGES.copy()
    try:
        runner.UnifiedMansionEnv, runner.update_sampling = RefinementEnv, sampling
        runner.LearningLog = StageProgressLog
        yield
    finally:
        runner.UnifiedMansionEnv, runner.update_sampling, runner.LearningLog = previous


def evaluate_round(run, label):
    run = Path(run)
    state, config = read(run / 'metadata.json'), load_config(run / 'config.json')
    output = run / 'checks' / f'{label}.json'
    if output.is_file():
        result = read(output)
        if result['steps'] != state['steps'] or runner.sha(run / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('A completed evaluation belongs to a different checkpoint.')
        return result
    checkpoint = f'checkpoints/{label}_step_{state["steps"]:010d}.zip'
    (run / 'checkpoints').mkdir(exist_ok=True)
    shutil.copyfile(run / state['resume_checkpoint'], run / checkpoint)
    model, metrics = runner.load_candidate(run / checkpoint), {}
    published = {(r.get('evaluation_label'), r.get('stage'), r.get('mode'))
                 for r in _history(run)}
    for mode in ('greedy', 'stochastic'):
        print(f'전12맵 평가 {label}: {mode} / 추가 학습0', flush=True)
        value = evaluate(model, config, stages=STAGES, episodes=1 if mode == 'greedy' else config['evaluation']['episodes'],
            seed=config['evaluation']['seed'] + (1_000_000 if mode == 'stochastic' else 0),
            mode=mode, steps=state['steps'], checkpoint=checkpoint,
            output_dir=run / 'evaluation' / label / mode)
        metrics[mode] = deepcopy(value['per_stage'])
        for stage, row in metrics[mode].items():
            row['replay'] = Path(row['replay']).relative_to(run).as_posix()
        for row in value['history_rows']:
            row['replay'] = Path(row['replay']).relative_to(run).as_posix()
            if (label, row['stage'], row['mode']) not in published:
                runner.append_json(run / 'evaluation/history.jsonl', {**row, 'evaluation_label': label})
    result = {'label': label, 'steps': state['steps'], 'updates': state['updates'],
        'checkpoint': checkpoint, 'checkpoint_sha256': runner.sha(run / checkpoint),
        'metrics': metrics, 'scope': 'development G1/S20; native original starts; fixed ghost phase'}
    runner.write_json(output, result)
    return result


def _history(run):
    path = Path(run) / 'evaluation/history.jsonl'
    rows = []
    for line in path.read_text().splitlines() if path.is_file() else []:
        try:
            row = json.loads(line)
            if isinstance(row, dict): rows.append(row)
        except json.JSONDecodeError:
            pass
    return rows


def select_model(run, result, control):
    run = Path(run)
    source = run / result['checkpoint']
    if runner.sha(source) != result['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint changed.')
    pending = run / '.selected_pending.zip'
    shutil.copyfile(source, pending)
    replace_with_retry(pending, run / 'selected_model.zip')
    control['selected'] = {k: result[k] for k in ('steps', 'updates', 'checkpoint', 'checkpoint_sha256', 'label')}
    control['selected']['rank'] = coverage_rank(result)
    runner.write_json(run / 'selection.json', control['selected'])


def final_validation(run, config, control, expired):
    locked = control.get('final_validation')
    if locked is None:
        locked = {'sha256': control['selected']['checkpoint_sha256'],
            'steps': control['selected']['steps'], 'results': {},
            'scope': 'one checkpoint; heldout action seeds; fixed native ghost phase'}
        control.update(final_validation=locked, status='final_evaluation')
        save_control(run, control)
    if runner.sha(Path(run) / 'selected_model.zip') != locked['sha256']:
        raise ValueError('The checkpoint locked for final validation changed.')
    model = runner.load_candidate(Path(run) / 'selected_model.zip')
    for seed in config['evaluation']['final_seed_sets']:
        for mode in ('greedy', 'stochastic'):
            key = f'{seed}_{mode}'
            if key in locked['results']:
                continue
            if expired():
                return False
            value = evaluate(model, config, stages=STAGES,
                episodes=1 if mode == 'greedy' else config['evaluation']['final_episodes'],
                seed=seed, mode=mode, steps=locked['steps'], checkpoint='selected_model.zip',
                output_dir=Path(run) / 'evaluation/final' / key)
            locked['results'][key] = value['per_stage']
            save_control(run, control)
    required = config['fresh_training']['required_star3_rate']
    control['goal_achieved'] = all(set(rows) == {str(s) for s in STAGES} and all(
        row['star3_rate'] >= required and row['return_rate'] >= required for row in rows.values())
        for rows in locked['results'].values())
    control['status'] = 'goal_achieved' if control['goal_achieved'] else 'complete_not_passed'
    return True


@contextmanager
def learner_lock(run):
    """같은 실행의 중복 학습을 막고 종료하면 OS가 잠금을 해제합니다."""
    with (Path(run) / '.learner.lock').open('a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            handle.seek(0); handle.write(b'0'); handle.flush(); handle.seek(0)
            lock, unlock = lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1), lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            lock, unlock = lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB), lambda: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        try:
            lock()
        except OSError as error:
            raise RuntimeError('This experiment already has an active learner.') from error
        try:
            yield
        finally:
            unlock()


def train(run, *, max_hours=2.):
    if type(max_hours) not in (int, float) or not math.isfinite(max_hours) or max_hours <= 0:
        raise ValueError('hours must be finite and positive.')
    run = resolve_run(run)
    with learner_lock(run):
        return _train(run, max_hours)


def _train(run, max_hours):
    run, config, state, _ = verify(run)
    control = read(run / 'fresh_protocol.json')
    if control['version'] != PROTOCOL:
        raise ValueError('Invalid fresh protocol.')
    if control['status'] in TERMINAL:
        print(f'종료된 실험: {control["status"]}. 결과를 보존합니다.', flush=True)
        return run
    started = time.monotonic()
    expired = lambda: time.monotonic() - started >= max_hours * 3600
    try:
        if control['baseline'] is None:
            baseline = evaluate_round(run, 'baseline')
            control.update(baseline={k: baseline[k] for k in ('steps', 'checkpoint', 'checkpoint_sha256')},
                           latest_metrics=baseline['metrics'])
            select_model(run, baseline, control)
            save_control(run, control)
        if control.get('final_validation') is not None:
            final_validation(run, config, control, expired)
            return run
        while not expired():
            state = read(run / 'metadata.json')
            if control['pending_round'] is None:
                updates = min(math.ceil(config['fresh_training']['round_steps'] / 128),
                              (state['total_steps'] - state['steps']) // 128)
                if updates <= 0:
                    final_validation(run, config, control, expired)
                    break
                control['pending_round'] = {'number': len(control['completed_rounds']) + 1,
                    'start_steps': state['steps'],
                    'target_updates': state['updates'] + updates * config['ppo']['n_epochs']}
                save_control(run, control)
            pending = control['pending_round']
            remaining = (pending['target_updates'] - state['updates']) // config['ppo']['n_epochs']
            if remaining > 0:
                control['status'] = 'training'
                save_control(run, control)
                with training_hooks(run, control['latest_metrics']):
                    runner.train(resume=run, session_updates=remaining, dashboard=False,
                        managed_evaluation=True, max_hours=max(.000001,
                            max_hours - (time.monotonic() - started) / 3600))
            state = read(run / 'metadata.json')
            if state['updates'] < pending['target_updates']:
                control['status'] = 'time_limit'
                break
            result = evaluate_round(run, f'round_{pending["number"]:03d}')
            rank = coverage_rank(result)
            previous = control['latest_metrics']
            regressed = [s for s in STAGES if previous.get('greedy', {}).get(str(s), {}).get('mean_score', 0) >= 85
                         and result['metrics']['greedy'][str(s)]['mean_score'] < 85]
            control['completed_rounds'].append({**pending, 'end_steps': state['steps'],
                'rank': rank, 'regressed_greedy_stages': regressed,
                'evaluation': f'checks/round_{pending["number"]:03d}.json'})
            control.update(pending_round=None, latest_metrics=result['metrics'], status='reviewed')
            if rank > control['selected']['rank']:
                select_model(run, result, control)
            save_control(run, control)
            print(f'전12맵: G85점 {rank[1]}/12 · G/S90% {rank[0]}/12 · {state["steps"]:,}스텝', flush=True)
            if development_goal(result) or state['total_steps'] - state['steps'] < 128:
                final_validation(run, config, control, expired)
                break
        else:
            control['status'] = 'time_limit'
    except KeyboardInterrupt:
        control['status'] = 'interrupted'
        raise
    except BaseException as error:
        control.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save_control(run, control)
        print(f'저장 상태: {control["status"]}\n{run}', flush=True)
    return run


def check(run, *, baseline=True):
    run, config, state, model = verify(run)
    before = (model.num_timesteps, model._n_updates, runner.sha(run / state['resume_checkpoint']))
    stages = []
    for stage in STAGES:
        env = RefinementEnv(config, stages=[stage], training=True)
        try:
            obs, _ = env.reset(seed=stage)
            if env.observation_space != model.observation_space or not env.observation_space.contains(obs):
                raise ValueError(f'Stage{stage} input mismatch.')
            for _ in range(3):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(int(action))
                if done: break
            stages.append(stage)
        finally:
            env.close()
    control = read(run / 'fresh_protocol.json')
    if baseline and state['steps'] == 0 and control['baseline'] is None:
        result = evaluate_round(run, 'baseline')
        control.update(baseline={k: result[k] for k in ('steps', 'checkpoint', 'checkpoint_sha256')},
                       latest_metrics=result['metrics'])
        select_model(run, result, control)
        save_control(run, control)
    after = (model.num_timesteps, model._n_updates, runner.sha(run / state['resume_checkpoint']))
    if before != after:
        raise ValueError('Read-only verification changed the model.')
    report = {'status': 'passed', 'input_dimensions': 2388, 'checked_stages': stages,
        'steps': state['steps'], 'updates': state['updates'], 'training_updates_added': 0,
        'fresh_initialization': state['initialization'], 'checkpoint_sha256_before': before[2],
        'checkpoint_sha256_after': after[2], 'empty_Adam': not bool(model.policy.optimizer.state)}
    runner.write_json(run / 'verification/check.json', report)
    return report
