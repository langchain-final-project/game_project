"""하나의 SB3 PPO를 학습·평가·저장하고 학습 화면에 실제 진행 상황을 전달합니다.

환경 4개에서 모은 새 경험 128스텝마다 같은 정책과 Adam을 10 epochs 갱신합니다.
Stage가 추가되어도 모델은 교체하지 않으며, 원래 게임 점수와 귀환으로 평가합니다.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from copy import deepcopy
from datetime import datetime
import hashlib
from importlib import metadata as packages
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from .config import default_config, load_config, stage_pool, validate_config
from .environment import UnifiedMansionEnv
from .evaluation import evaluate, isolated_rng
from .file_io import LiveJSONWriter, replace_with_retry, write_json
from .paths import PROJECT_ROOT
from .policy import UnifiedA2CPolicy
from .visual import frame_from_snapshot
from .display_units import format_steps
from .runtime import KST, set_latest_run

VERSION = 'shared_native_game_ppo_v1'
ROLLOUT = 128
PPO_SETTINGS = {'batch_size': ROLLOUT, 'n_epochs': 10, 'clip_range': .2,
                'clip_range_vf': None, 'target_kl': None}


def ppo_settings(config):
    """128개 전체 배치·고정 epochs를 저장해 재개와 Adam 검증의 전제를 유지합니다."""
    settings = config.get('ppo', PPO_SETTINGS)
    if settings != PPO_SETTINGS:
        raise ValueError('This comparison requires the fixed full-batch PPO settings.')
    return settings.copy()


def completed_rollouts(model):
    """SB3의 epoch 누적 횟수를 완료한 128스텝 경험 수집 횟수로 환산합니다."""
    return model._n_updates // model.n_epochs


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with path.open('rb+') as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b'\n':
                handle.seek(0, os.SEEK_END)
                handle.write(b'\n')
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_manifest():
    """실행에 사용한 코드·게임 설정의 해시를 기록해 재개 시 변경을 확인합니다."""
    paths = [*PROJECT_ROOT.glob('a2c_game/*.py'),
             *PROJECT_ROOT.glob('game_source/kkamang/*.py'),
             *PROJECT_ROOT.glob('game_source/data/levels/*.json'),
             PROJECT_ROOT / 'game_source_manifest.json',
             PROJECT_ROOT / 'run.py', PROJECT_ROOT / 'requirements.txt']
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha(p) for p in sorted(paths) if p.is_file()}


def assert_integrity(model, total_steps):
    """PPO 종류·누적 스텝·완료 epochs와 Adam의 상태가 서로 맞는지 검사합니다."""
    if (not isinstance(model, PPO) or not isinstance(model.policy, UnifiedA2CPolicy)
            or model.n_envs != 4 or model.n_steps != 32
            or model.batch_size != ROLLOUT or model.n_epochs != PPO_SETTINGS['n_epochs']
            or model.target_kl is not None or model.clip_range_vf is not None
            or model._n_updates < 0 or model._n_updates % model.n_epochs
            or not 0 <= model.num_timesteps <= total_steps
            or model.num_timesteps % 4
            or model.num_timesteps < completed_rollouts(model) * ROLLOUT):
        raise ValueError('PPO type, rollout counters or total budget do not match.')
    optimizer = model.policy.optimizer
    if not isinstance(optimizer, torch.optim.Adam):
        raise ValueError('The saved shared PPO must retain its Adam state.')
    if model._n_updates == 0:
        if optimizer.state:
            raise ValueError('A fresh model must start with empty Adam state.')
    elif (len(optimizer.state) != len(list(model.policy.parameters())) or any(
            int(s.get('step', -1)) != model._n_updates or 'exp_avg' not in s or 'exp_avg_sq' not in s
            for s in optimizer.state.values())):
        raise ValueError('Adam steps/moments do not match completed PPO epochs.')


def save_resume(model, run, state):
    """두 저장 슬롯을 번갈아 써 마지막 일관된 모델과 메타데이터를 보존합니다."""
    assert_integrity(model, state['total_steps'])
    committed_path = run / 'metadata.json'
    committed = json.loads(committed_path.read_text()) if committed_path.exists() else {}
    slot = 'resume_B.zip' if committed.get('resume_checkpoint') == 'resume_A.zip' else 'resume_A.zip'
    pending = run / '.checkpoint_pending.zip'
    model.save(pending)
    replace_with_retry(pending, run / slot)
    candidate = {**state, 'resume_checkpoint': slot, 'checkpoint_sha256': sha(run / slot),
                 'steps': int(model.num_timesteps), 'updates': int(model._n_updates),
                 'rollout_updates': completed_rollouts(model),
                 'entropy_coef': float(model.ent_coef),
                 'discarded_rollout_decisions': int(model.num_timesteps - ROLLOUT * completed_rollouts(model))}
    write_json(run / 'metadata.json', candidate)
    state.update(resume_checkpoint=slot, checkpoint_sha256=candidate['checkpoint_sha256'],
                 steps=int(model.num_timesteps), updates=int(model._n_updates),
                 rollout_updates=completed_rollouts(model),
                 entropy_coef=float(model.ent_coef),
                 discarded_rollout_decisions=candidate['discarded_rollout_decisions'])


def load_candidate(path):
    # PPO.load도 전역 난수 시드를 바꾸므로 읽기 전후 난수 상태를 복원합니다.
    with isolated_rng(None):
        return PPO.load(path, device='cpu')


def update_sampling(vec, config, steps):
    pool = stage_pool(config, steps)
    # 환경 1개는 이전 맵을 복습하고 나머지 3개는 최신 맵의 추출 비중을 높입니다.
    # 모두 원래 출발점에서 새로 수집하는 경험이며 과거 경험 버퍼를 쓰지 않습니다.
    older = pool[:-1] or pool
    vec.envs[0].set_stage_pool(older)
    weights = [1. for _ in pool]
    weights[-1] = 2.
    for env in vec.envs[1:]:
        env.set_stage_pool(pool, weights)
    for env in vec.envs:
        env.set_training_progress(min(1., steps / config['training']['total_steps']))
    return pool


def adapt_entropy(model, config, state, entropy):
    """측정 엔트로피의 이동평균에 따라 설정된 범위 안에서 탐색 계수를 조절합니다."""
    settings = config['training']['entropy_adaptation']
    previous = state.get('entropy_ema')
    state['entropy_ema'] = entropy if previous is None else .8 * previous + .2 * entropy
    if (not settings['enabled']
            or (model._n_updates - state.get('entropy_checked_update', 0)) / model.n_epochs
            < settings['interval_updates']):
        return
    state['entropy_checked_update'] = int(model._n_updates)
    coefficient = float(model.ent_coef)
    if state['entropy_ema'] < settings['target']:
        coefficient *= 1.25
    elif state['entropy_ema'] > settings['target'] * 2:
        coefficient *= .98
    coefficient = min(settings['max'], max(settings['min'], coefficient))
    if coefficient != model.ent_coef:
        state.setdefault('entropy_changes', []).append({'steps': int(model.num_timesteps),
            'entropy': state['entropy_ema'], 'before': float(model.ent_coef), 'after': coefficient})
        model.ent_coef = coefficient


class LearningLog(BaseCallback):
    """게임별 실제 성적, Stage별 경험량, 화면에 보낼 현재 위치를 모읍니다."""
    def __init__(self, run, state, config, expired):
        super().__init__()
        self.run, self.state, self.config, self.expired = run, state, config, expired
        self.recent = deque(maxlen=200)
        self.stage_recent = defaultdict(lambda: deque(maxlen=100))
        self.count = Counter()
        self.stopped_by_time = False
        self.live_episode = 0
        self.live_writer = LiveJSONWriter(self.run / 'live.json')

    def _on_step(self):
        # 한 호출에는 네 환경 각각의 결과가 들어옵니다. 종료된 판만 성적에 집계합니다.
        exposure = self.state['stage_exposure']
        for rank, info in enumerate(self.locals['infos']):
            stage = str(info['stage'])
            row = exposure[stage]
            row['learner_steps'] += 1
            row['physical_frames'] += int(info['executed_frames'])
            episode = info.get('episode')
            if episode is not None:
                row['episodes'] += 1
                row['returns'] += int(episode['game_return'])
                row['highscores'] += int(episode['highscore_success'])
                self.recent.append(episode)
                self.stage_recent[int(stage)].append(episode)
                append_json(self.run / 'train/episodes.jsonl',
                            {'steps': self.num_timesteps, 'env_rank': rank, 'episode': episode})
                self.count['episodes'] += 1
                if rank == 0:
                    self.live_episode += 1
        if self.n_calls % 8 == 0 and self.live_writer.ready():
            info = self.locals['infos'][0]
            frame = frame_from_snapshot(info['snapshot'], info.get('policy_action'))
            self.live_writer.write({'stage': info['stage'], 'steps': self.num_timesteps,
                'mode': 'training', 'episode_index': self.live_episode, 'frame': frame,
                'environments': [{'env_rank': rank, 'stage': row['stage'],
                    'frame': frame_from_snapshot(row['snapshot'], row.get('policy_action'))}
                    for rank, row in enumerate(self.locals['infos'])]})
        self.stopped_by_time = self.expired()
        final_partial = (self.num_timesteps >= self.state['total_steps']
                         and self.locals.get('n_steps', 0) + 1 < self.model.n_steps)
        return not (self.stopped_by_time or final_partial)


def train(*, resume=None, seed=None, total_steps=None, max_hours=8., session_updates=None,
          check_only=False, dashboard=True, port=8765, output_root=None, config_path=None,
          managed_evaluation=False):
    """새 학습 또는 저장한 PPO 재개를 실행하고 시간·학습량 예산 안에서 진행합니다."""
    if (type(max_hours) not in (int, float) or not math.isfinite(max_hours) or max_hours <= 0
            or session_updates is not None and (type(session_updates) is not int or session_updates < 1)):
        raise ValueError('max-hours and session-updates must be positive.')
    if check_only and resume is not None:
        raise ValueError('Use check-only with a fresh model, not resume.')
    if resume is None:
        # 새 학습은 임의 가중치와 빈 Adam에서 시작하며 실행 당시 소스도 함께 보존합니다.
        config = load_config(config_path) if config_path else default_config(
            seed=11 if seed is None else seed, total_steps=10_000_000 if total_steps is None else total_steps)
        if config_path and (seed is not None or total_steps is not None):
            raise ValueError('Use either a complete config file or seed/total-steps overrides.')
        config['ppo'] = ppo_settings(config)
        validate_config(config)
        root = (Path(output_root) if output_root is not None else PROJECT_ROOT / 'runs').resolve()
        if check_only:
            root /= 'connection_checks'
        run = root / f'seed_{config["training"]["seed"]}_{datetime.now(KST):%Y%m%d_%H%M%S_%f}'
        run.mkdir(parents=True, exist_ok=False)
        if not check_only:
            set_latest_run(run)
        write_json(run / 'config.json', config)
        hashes = source_manifest()
        for name, expected in hashes.items():
            target = run / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / name, target)
            if sha(target) != expected:
                raise ValueError('Source changed while copying the run snapshot.')
        state = {'version': VERSION, 'algorithm': 'PPO', 'single_policy': True,
            'initialization': 'fresh_random_parameters_and_empty_Adam',
            'teacher_actions': False, 'frozen_prefix': False, 'injected_game_states': False,
            'status': 'training', 'steps': 0, 'updates': 0, 'rollout_updates': 0,
            'total_steps': config['training']['total_steps'], 'seed': config['training']['seed'],
            'config_sha256': sha(run / 'config.json'), 'source_hashes': hashes,
            'resume_count': 0, 'stage_pool': [1], 'best': None, 'evaluations': [],
            'next_eval_steps': 0, 'pending_evaluation': None, 'final_validation': None,
            'goal_achieved': False, 'entropy_coef': config['a2c']['ent_coef'],
            'stage_exposure': {str(s): {'learner_steps': 0, 'physical_frames': 0,
                'episodes': 0, 'returns': 0, 'highscores': 0} for s in range(1, 13)},
            'versions': {p: packages.version(p) for p in ('stable-baselines3','torch','gymnasium','numpy')},
            'resume_restores_weights_adam_counters': True,
            'resume_restores_native_episode_and_rng': False}
    else:
        # 일반 재개는 저장 당시 설정·소스·가중치가 같은지 확인한 뒤 누적 스텝을 잇습니다.
        if any(value is not None for value in (seed, total_steps, config_path)):
            raise ValueError('A resumed run uses its saved seed, total budget and complete config.')
        run = Path(resume).resolve(strict=True)
        state = json.loads((run / 'metadata.json').read_text())
        if (state['version'] != VERSION or state.get('algorithm') != 'PPO'
                or state.get('check_only')):
            raise ValueError('Only a genuine training run of this version can resume.')
        config = load_config(run / 'config.json')
        if sha(run / 'config.json') != state['config_sha256']:
            raise ValueError('Saved training configuration changed.')
        for name, expected in state['source_hashes'].items():
            if sha(PROJECT_ROOT / name) != expected or sha(run / 'source_snapshot' / name) != expected:
                raise ValueError(f'Source differs from this run: {name}')
        if sha(run / state['resume_checkpoint']) != state['checkpoint_sha256']:
            raise ValueError('Resume checkpoint changed.')
        if state['status'] == 'complete':
            print(f'이미 종료된 실행: {run}', flush=True)
            return run
        state.update(resume_count=state['resume_count'] + 1, status='training')
    # 과거 엔트로피 조정은 원래 스텝에 남겨 두고 새 변경처럼 다시 출력하지 않습니다.
    state.setdefault('logged_entropy_changes', len(state.get('entropy_changes', [])))
    if config.get('ppo') != ppo_settings(config):
        raise ValueError('The saved PPO run must include its complete PPO settings.')
    torch.set_num_threads(config['a2c']['torch_threads'])
    started = time.monotonic()
    expired = lambda: time.monotonic() - started >= max_hours * 3600
    model = vec = logger = server = callback = None
    ready = False
    total_steps = state['total_steps']
    published = set()
    history_path = run / 'evaluation/history.jsonl'
    if history_path.exists():
        for line in history_path.read_text().splitlines():
            try:
                row = json.loads(line)
                published.add((row.get('evaluation_label'), row['stage'], row['mode']))
            except (json.JSONDecodeError, KeyError):
                continue

    def publish(row, label):
        key = (label, row['stage'], row['mode'])
        if key not in published:
            append_json(history_path, {**row, 'evaluation_label': label})
            published.add(key)

    def save():
        if not ready:
            return
        state['invocation_elapsed_seconds'] = time.monotonic() - started
        save_resume(model, run, state)
        write_json(run / 'stageexposure.json', state['stage_exposure'])
        write_json(run / 'status.json', {**{k: state[k] for k in
            ('algorithm', 'status', 'steps', 'updates', 'rollout_updates', 'total_steps', 'stage_pool')},
            'phase': 'shared_stage_exposure', 'run_name': run.name,
            'learning_rate': config['a2c']['learning_rate'], 'entropy_coef': float(model.ent_coef),
            'latest_checkpoint': state['resume_checkpoint'], 'best': state['best'],
            'best_checkpoint': (state['best'] or {}).get('checkpoint'),
            'dashboard_url': state.get('dashboard_url'), 'goal_achieved': state['goal_achieved']})

    def evaluation_round(label, seed_base, final=False):
        # 평가할 정책을 고정하고 Stage별 결과를 저장해 중단 후에도 완료 지점부터 잇습니다.
        pending = state['pending_evaluation']
        if pending is None:
            checkpoint = f'checkpoints/step_{model.num_timesteps:010d}.zip'
            (run / 'checkpoints').mkdir(exist_ok=True)
            if not (run / checkpoint).exists():
                model.save(run / checkpoint)
            pending = {'label': label, 'steps': int(model.num_timesteps),
                'updates': int(model._n_updates), 'seed': seed_base, 'checkpoint': checkpoint,
                'sha256': sha(run / checkpoint), 'results': {}, 'final': final}
            if label == 'baseline':
                pending.update(checkpoint='initial_model.zip', steps=0, updates=0,
                               sha256=sha(run / 'initial_model.zip'))
            if final:
                locked = state.get('locked_selection')
                if locked is None:
                    selected = state['best']
                    if selected is not None:
                        checkpoint = selected['checkpoint']
                        if sha(run / checkpoint) != selected['sha256']:
                            raise ValueError('Selected best checkpoint changed before final validation.')
                    chosen = load_candidate(run / checkpoint)
                    shutil.copyfile(run / checkpoint, run / 'selected_model.zip')
                    locked = {'checkpoint': 'selected_model.zip', 'sha256': sha(run / 'selected_model.zip'),
                              'steps': int(chosen.num_timesteps), 'updates': int(chosen._n_updates)}
                    state['locked_selection'] = locked
                if sha(run / locked['checkpoint']) != locked['sha256']:
                    raise ValueError('The model locked for both independent final evaluations changed.')
                pending.update(locked)
            state['pending_evaluation'] = pending
            save()
        if sha(run / pending['checkpoint']) != pending['sha256']:
            raise ValueError('The candidate locked for evaluation changed.')
        candidate = load_candidate(run / pending['checkpoint'])
        modes = ('greedy', 'stochastic')
        for mode_index, mode in enumerate(modes):
            mode_results = pending['results'].setdefault(mode, {})
            episodes = (config['evaluation']['final_episodes'] if final else
                        1 if mode == 'greedy' else config['evaluation']['episodes'])
            for stage in range(1, 13):
                if str(stage) in mode_results:
                    publish(mode_results[str(stage)], label)
                    continue
                if expired():
                    return False
                print(f'평가 {label}: Stage{stage:02d} {mode} {episodes}회 / 추가 학습0', flush=True)
                output = run / 'evaluation' / label / mode / f'stage_{stage:02d}'
                result = evaluate(candidate, config, stages=[stage], episodes=episodes,
                    seed=seed_base + mode_index * 1_000_000, mode=mode,
                    steps=pending['steps'], checkpoint=pending['checkpoint'], output_dir=output)
                metrics = result['per_stage'][str(stage)]
                row = {'steps': pending['steps'], 'updates': pending['updates'], 'stage': stage,
                    'mode': mode, 'seed': seed_base + mode_index * 1_000_000,
                    'checkpoint': pending['checkpoint'], **metrics,
                    'replay': str(Path(metrics['replay']).relative_to(run))}
                # Stage 평가 결과를 먼저 저장한 뒤 학습 곡선에 공개해 중복·누락을 줄입니다.
                mode_results[str(stage)] = row
                save()
                publish(row, label)
                print(f'  결과 Stage{stage:02d} {mode}: {row["mean_score"]:.1f}점 / '
                      f'귀환 {row.get("return_rate", 0):.0%} / '
                      f'85점 {row.get("star3_rate", 0):.0%}', flush=True)
        greedy = pending['results']['greedy']
        # 최고 모델은 최소 85점률→평균 85점률→평균 점수→퀘스트율 순서로 고릅니다.
        rank = [min(r['star3_rate'] for r in greedy.values()),
                sum(r['star3_rate'] for r in greedy.values()) / 12,
                sum(r['mean_score'] for r in greedy.values()) / 12,
                sum(r['quest_rate'] for r in greedy.values()) / 12]
        if not final and (state['best'] is None or tuple(rank) > tuple(state['best']['rank'])):
            state['best'] = {'rank': rank, 'checkpoint': pending['checkpoint'],
                             'sha256': pending['sha256'], 'steps': pending['steps']}
        completed = deepcopy(pending)
        state['evaluations'].append({'label': label, 'steps': pending['steps'], 'rank': rank})
        state['pending_evaluation'] = None
        if final:
            state.setdefault('final_results', {})[label] = completed
        else:
            interval = config['training']['eval_every_steps']
            state['next_eval_steps'] = (pending['steps'] // interval + 1) * interval
        save()
        print(f'평가 완료 {label}: 전12개 평균 {rank[2]:.1f}점 / 최소85점 귀환률 {rank[0]:.0%}', flush=True)
        return True

    try:
        pool = stage_pool(config, state['steps'])
        vec = DummyVecEnv([lambda i=i: UnifiedMansionEnv(config, stages=pool,
            seed=state['seed'] + i, training=True) for i in range(4)])
        if resume is None:
            a = config['a2c']
            # PPO 손실·rollout 수집·역전파는 SB3가 담당하고 이 파일은 실행을 관리합니다.
            model = PPO(UnifiedA2CPolicy, vec, seed=state['seed'], device='cpu', verbose=0,
                learning_rate=a['learning_rate'], n_steps=a['n_steps'], gamma=a['gamma'],
                gae_lambda=a['gae_lambda'], ent_coef=a['ent_coef'], vf_coef=a['vf_coef'],
                max_grad_norm=a['max_grad_norm'], normalize_advantage=a['normalize_advantage'],
                **config['ppo'], policy_kwargs={'net_arch': a['net_arch'],
                    'activation_fn': nn.Tanh, 'logit_bound': config['policy']['logit_bound']})
            model.save(run / 'initial_model.zip')
        else:
            model = PPO.load(run / state['resume_checkpoint'], env=vec, device='cpu', force_reset=True)
            if model.num_timesteps != state['steps'] or model._n_updates != state['updates']:
                raise ValueError('Model and metadata counters differ.')
        assert_integrity(model, total_steps)
        invocation_start_steps = model.num_timesteps
        state['stage_pool'] = update_sampling(vec, config, model.num_timesteps)
        write_json(run / 'sampling.json', {'steps': int(model.num_timesteps),
            'environment_stage_pools': [list(env._pending_stages) for env in vec.envs],
            'weights': [env._pending_probabilities.tolist() for env in vec.envs]})
        ready = True
        if dashboard and not check_only:
            from .dashboard import start_dashboard
            try:
                server = start_dashboard(run, port=port)
                state['dashboard_url'] = server.dashboard_url
                print(f'학습 화면: {server.dashboard_url}', flush=True)
            except OSError as error:
                print(f'화면 포트를 열지 못했습니다({error}). 학습은 계속하며 별도 dashboard 명령으로 볼 수 있습니다.', flush=True)
        save()
        print(f'실행 폴더: {run}\n공통 PPO {format_steps(model.num_timesteps)} / {format_steps(total_steps)} '
              f'(정확히 {model.num_timesteps:,}/{total_steps:,}스텝); '
              f'{completed_rollouts(model):,}rollouts / {model._n_updates:,}epochs; '
              f'스테이지 {state["stage_pool"]}', flush=True)
        if check_only:
            before = deepcopy(model.policy.state_dict())
            observation = vec.reset()
            for _ in range(4):
                actions, _ = model.predict(observation)
                observation, _, _, _ = vec.step(actions)
            model.save(run / 'roundtrip.zip')
            restored = load_candidate(run / 'roundtrip.zip')
            assert_integrity(restored, total_steps)
            if any(not torch.equal(tensor, restored.policy.state_dict()[key]) for key, tensor in before.items()):
                raise RuntimeError('Zero-learning model roundtrip changed parameters.')
            for stage in range(1, 13):
                env = UnifiedMansionEnv(config, stages=[stage], training=False)
                obs, _ = env.reset(seed=10000 + stage)
                if not env.observation_space.contains(obs):
                    raise RuntimeError(f'Invalid native observation: stage {stage}')
                model.predict(obs)
                env.close()
            state.update(status='connection_checked', check_only=True, check_learning_updates=0)
            return run
        logger = configure(str(run / 'train' / f'session_{state["resume_count"]:03d}'), ['csv'])
        model.set_logger(logger)
        callback = LearningLog(run, state, config, expired)
        # session_updates와 로그 간격은 원본과 같은 128스텝 rollout 단위입니다.
        session_end = (math.inf if session_updates is None
                       else model._n_updates + session_updates * model.n_epochs)
        optimizer = model.policy.optimizer
        if state['pending_evaluation'] is not None:
            pending = state['pending_evaluation']
            if not evaluation_round(pending['label'], pending['seed'], pending['final']):
                state['status'] = 'paused'
                return run
        # 짧은 연결 검증만 전체 초기 평가를 생략합니다. 일반 학습은 시작 전 평가도 기록합니다.
        if session_updates is None and state['next_eval_steps'] == 0:
            if not evaluation_round('baseline', config['evaluation']['seed']):
                state['status'] = 'paused'
                return run
        while model.num_timesteps < total_steps and model._n_updates < session_end and not expired():
            previous_pool = state['stage_pool']
            state['stage_pool'] = update_sampling(vec, config, model.num_timesteps)
            if previous_pool != state['stage_pool']:
                print(f'학습 스테이지 확장: {state["stage_pool"]} (성공 판정과 별개)', flush=True)
                append_json(run / 'adjustment_history.jsonl', {'steps': int(model.num_timesteps),
                    'kind': 'stage_pool', 'before': previous_pool, 'after': state['stage_pool'],
                    'reason': '고정 노출 일정에 따른 새 맵 추가; 성공 판정과 별개'})
                write_json(run / 'sampling.json', {'steps': int(model.num_timesteps),
                    'environment_stage_pools': [list(env._pending_stages) for env in vec.envs],
                    'weights': [env._pending_probabilities.tolist() for env in vec.envs]})
            end_eval = state['next_eval_steps'] or config['training']['eval_every_steps']
            remaining = total_steps - model.num_timesteps
            until_eval = remaining if session_updates is not None else max(4, end_eval - model.num_timesteps)
            updates = min(config['training']['progress_every_updates'],
                          (session_end - model._n_updates) / model.n_epochs,
                          math.ceil(remaining / ROLLOUT), math.ceil(until_eval / ROLLOUT))
            updates_before = model._n_updates
            # 환경 4개 × n_steps 32 = 128개 새 경험마다 PPO를 10 epochs 학습합니다.
            model.learn(min(int(updates) * ROLLOUT, remaining), reset_num_timesteps=False, callback=callback)
            if model.policy.optimizer is not optimizer:
                raise RuntimeError('Stage expansion replaced the shared optimizer.')
            values = model.logger.name_to_value
            gradient_update = model._n_updates > updates_before
            # 시간 종료로 rollout이 미완료인 경우 갱신하지 않은 손실을 새 값처럼 기록하지 않습니다.
            entropy = float(-values['train/entropy_loss']) if gradient_update else None
            if gradient_update:
                adapt_entropy(model, config, state, entropy)
            changes = state.get('entropy_changes', [])
            for change in changes[state.get('logged_entropy_changes', 0):]:
                append_json(run / 'adjustment_history.jsonl', {**change, 'kind': 'entropy_coef',
                    'reason': '관측한 정책 엔트로피에 따른 선언된 자동 조정'})
                print(f'  엔트로피 계수 변경: {change["before"]:.5f} → {change["after"]:.5f} '
                      f'(측정 엔트로피 {change["entropy"]:.3f})', flush=True)
            state['logged_entropy_changes'] = len(changes)
            episodes = list(callback.recent)
            per_stage = []
            for stage, recent in sorted(callback.stage_recent.items()):
                n = len(recent)
                average = lambda key: sum(e.get(key, 0) for e in recent) / n
                per_stage.append({'stage': stage, 'episodes': n, 'mean_score': average('score'),
                    'return_rate': average('game_return'), 'star3_rate': average('highscore_success'),
                    'quest_rate': average('quest_collected'), 'mean_warmth': average('warmth'),
                    'mean_damage_count': average('damage_count'),
                    'mean_wall_stall_decisions': average('wall_stall_decisions'),
                    'reasons': dict(Counter(e['reason'] for e in recent))})
            row = {'steps': int(model.num_timesteps), 'updates': int(model._n_updates),
                 'rollout_updates': completed_rollouts(model),
                'elapsed_seconds': time.monotonic() - started,
                'fps': (model.num_timesteps - invocation_start_steps)
                       / max(.001, time.monotonic() - started),
                'entropy': entropy, 'entropy_coef': float(model.ent_coef),
                'gradient_update': gradient_update,
                 'policy_loss': float(values['train/policy_gradient_loss']) if gradient_update else None,
                'value_loss': float(values['train/value_loss']) if gradient_update else None,
                'stage_pool': state['stage_pool'], 'episodes': len(episodes), 'per_stage': per_stage,
                'novelty_coefficient': config['reward']['novelty_per_state']
                    * max(.1, 1. - model.num_timesteps / total_steps),
                'mean_score': np.mean([e['score'] for e in episodes]).item() if episodes else None,
                'return_rate': sum(e['game_return'] for e in episodes) / len(episodes) if episodes else None,
                'highscore_rate': sum(e['highscore_success'] for e in episodes) / len(episodes) if episodes else None}
            append_json(run / 'train/progress.jsonl', row)
            model.logger.dump(step=model.num_timesteps)
            checkpoint_interval = config['training']['checkpoint_every_steps']
            if model.num_timesteps >= state.get('next_checkpoint_steps', checkpoint_interval):
                candidate_path = run / 'checkpoints' / f'step_{model.num_timesteps:010d}.zip'
                candidate_path.parent.mkdir(exist_ok=True)
                if not candidate_path.exists():
                    model.save(candidate_path)
                state['next_checkpoint_steps'] = (model.num_timesteps // checkpoint_interval + 1) * checkpoint_interval
            save()
            entropy_display = f'{entropy:.3f}' if entropy is not None else '미완료 rollout'
            mean = '아직 종료한 판 없음' if row['mean_score'] is None else f'{row["mean_score"]:.1f}점'
            print(f'PPO 학습 {format_steps(model.num_timesteps)} / {format_steps(total_steps)}; '
                  f'{completed_rollouts(model):,}rollouts / {model._n_updates:,}epochs; '
                  f'Stage1~{state["stage_pool"][-1]}; '
                  f'최근 학습 평균={mean}; entropy={entropy_display}; 계수={model.ent_coef:.5f}; '
                  f'속도={row["fps"]:.0f}스텝/초', flush=True)
            for item in per_stage:
                failures = {key: count for key, count in item['reasons'].items() if key != 'returned'}
                print(f'  Stage{item["stage"]:02}: 최근{item["episodes"]}판 '
                      f'{item["mean_score"]:.1f}점 / 귀환 {item["return_rate"]:.0%} / '
                      f'85점 {item["star3_rate"]:.0%} / 퀘스트 {item["quest_rate"]:.0%} / '
                      f'온기 {item["mean_warmth"]:.1f} / 실패사유 {failures or "없음"}', flush=True)
            if callback.stopped_by_time:
                break
            if session_updates is None and model.num_timesteps >= (state['next_eval_steps'] or end_eval):
                label = f'step_{model.num_timesteps:010d}'
                if not evaluation_round(label, config['evaluation']['seed']):
                    break
        if model.num_timesteps >= total_steps and not managed_evaluation:
            # 총 예산을 마친 뒤 선정 모델을 고정해 별도 두 시드 집합으로 최종 평가합니다.
            if not any(e['steps'] == model.num_timesteps and e['label'] != 'baseline'
                       and not e['label'].startswith('final_') for e in state['evaluations']):
                if not evaluation_round(f'step_{model.num_timesteps:010d}', config['evaluation']['seed']):
                    state['status'] = 'paused'
                    return run
            state['status'] = 'final_evaluation'
            for i, seed_base in enumerate(config['evaluation']['final_seed_sets']):
                label = f'final_{i + 1}'
                if label not in state.get('final_results', {}):
                    if not evaluation_round(label, seed_base, final=True):
                        state['status'] = 'paused'
                        return run
            state['goal_achieved'] = all(
                row['star3_rate'] >= config['evaluation']['required_star3_rate']
                for result in state['final_results'].values()
                for row in result['results']['greedy'].values())
            state['status'] = 'complete'
        else:
            state['status'] = 'paused'
        return run
    except BaseException as error:
        state.update(status='paused' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if ready:
                try:
                    assert_integrity(model, total_steps)
                except ValueError:
                    # 옵티마이저 갱신 도중 중단되었다면 마지막으로 검증된 저장 상태를 유지합니다.
                    committed = json.loads((run / 'metadata.json').read_text())
                    state.clear()
                    state.update(committed, status='paused', error='Interrupted during an incomplete optimizer update; retained last consistent checkpoint.')
                    write_json(run / 'metadata.json', state)
                else:
                    save()
        finally:
            if vec is not None:
                vec.close()
            if logger is not None:
                logger.close()
            if server is not None:
                server.shutdown()
                server.server_close()
            print(f'저장 완료: {run} ({state["status"]})', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--total-steps', type=int)
    parser.add_argument('--max-hours', type=float, default=8.)
    parser.add_argument('--session-updates', type=int)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--no-dashboard', dest='dashboard', action='store_false')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--config', dest='config_path', type=Path)
    train(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
