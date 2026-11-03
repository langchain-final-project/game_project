"""원래 출발점에서 추가 학습 없이 평가하고 실패를 포함해 점수·재생을 기록합니다."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import csv
import math
from pathlib import Path
import random

import numpy as np
import torch

from .config import validate_config
from .environment import UnifiedMansionEnv, validate_stages
from .file_io import write_json
from .visual import frame_from_snapshot


@contextmanager
def isolated_rng(seed=None):
    """평가 전후 난수 상태를 복원해 평가가 진행 중인 학습의 난수 흐름을 바꾸지 않게 합니다."""
    states = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.random.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def _distribution_metrics(model, observation):
    # 행동 분포의 엔트로피와 최대 확률을 읽어 한 행동에 쏠리는 정도를 관찰합니다.
    policy = getattr(model, 'policy', None)
    if policy is None or not hasattr(policy, 'obs_to_tensor') or not hasattr(policy, 'get_distribution'):
        return None, None
    tensor, _ = policy.obs_to_tensor(observation)
    with torch.no_grad():
        distribution = policy.get_distribution(tensor)
        entropy = float(distribution.entropy().mean().cpu())
        categorical = getattr(distribution, 'distribution', None)
        maximum = (float(categorical.probs.max().cpu())
                   if hasattr(categorical, 'probs') else None)
    if not math.isfinite(entropy) or maximum is not None and not math.isfinite(maximum):
        raise RuntimeError('Nonfinite policy probabilities during evaluation.')
    return entropy, maximum


def evaluate(model, config, stages=tuple(range(1, 13)), episodes=5, seed=10000,
             mode='greedy', steps=0, checkpoint='', output_dir=None):
    """준비 상태 인계·재시도·성공 필터링 없이 모든 평가 회차를 집계합니다.

    각 Stage의 첫 회차는 성공 여부와 무관하게 재생을 저장합니다.
    history_rows는 평가가 끝난 뒤 학습기가 학습 곡선 기록에 추가할 수 있습니다.
    """
    validate_config(config)
    stages = list(stages)
    validate_stages(stages)
    if (type(episodes) is not int or episodes < 1 or type(seed) is not int
            or not 0 <= seed < 2**31 - 13_000_000 or mode not in ('greedy', 'stochastic')
            or type(steps) is not int or steps < 0):
        raise ValueError('Invalid evaluation episodes, seed, mode, or learner steps.')
    deterministic = mode == 'greedy'
    # 최대확률(G)은 가장 높은 확률의 행동, 확률 행동(S)은 정책 분포에서 뽑은 행동입니다.
    output = Path(output_dir).resolve() if output_dir is not None else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
    policy = getattr(model, 'policy', None)
    was_training = getattr(policy, 'training', None)
    records, metrics, replays = [], {}, {}
    try:
        with isolated_rng(seed):
            if policy is not None and hasattr(policy, 'set_training_mode'):
                policy.set_training_mode(False)
            for stage in stages:
                env = UnifiedMansionEnv(config, stages=[stage], seed=seed, training=False)
                stage_rows, entropies, maximums = [], [], []
                if (hasattr(model, 'observation_space') and model.observation_space != env.observation_space
                        or hasattr(model, 'action_space') and model.action_space != env.action_space):
                    env.close()
                    raise ValueError('Evaluation observation/action spaces differ from the saved model.')
                try:
                    for episode_index in range(episodes):
                        env_seed = seed + stage * 10_000 + episode_index
                        action_seed = seed + stage * 1_000_000 + episode_index
                        with isolated_rng(action_seed):
                            observation, info = env.reset(seed=env_seed, options={'stage': stage})
                            frames = []
                            if episode_index == 0:
                                frame = frame_from_snapshot(info['snapshot'])
                                frame['stage'] = stage
                                frames.append(frame)
                            terminal = truncated = False
                            rewards = 0.
                            decisions = 0
                            while not (terminal or truncated):
                                entropy, maximum = _distribution_metrics(model, observation)
                                if entropy is not None: entropies.append(entropy)
                                if maximum is not None: maximums.append(maximum)
                                action, _ = model.predict(observation, deterministic=deterministic)
                                action = int(np.asarray(action).item())
                                observation, reward, terminal, truncated, info = env.step(action)
                                rewards += reward
                                decisions += 1
                                if episode_index == 0:
                                    frame = frame_from_snapshot(info['snapshot'], action=action)
                                    frame['stage'] = stage
                                    frames.append(frame)
                                if decisions > 10_000:
                                    raise RuntimeError('A native finite-time episode did not terminate.')
                        if truncated or not terminal or not info.get('native_terminated'):
                            raise RuntimeError('Evaluation must reach the real native game ending.')
                        episode = info['episode']
                        native = episode['native_outcome']
                        returned = native.get('status') == 'COMPLETE'
                        # 귀환 실패는 실제 점수 0점으로 포함하며 보상 합계와 분리합니다.
                        score = int(native.get('score', {}).get('total', 0)) if returned else 0
                        row = {'stage': stage, 'mode': mode, 'episode': episode_index,
                            'seed': env_seed, 'action_seed': action_seed, 'score': score,
                            'game_return': returned, 'star1': returned and score >= 45,
                            'star2': returned and score >= 65, 'star3': returned and score >= 85,
                            'score40': returned and score >= 40,
                            'quest_collected': bool(episode['quest_collected']),
                            'warmth': int(episode['warmth']), 'damage_count': int(episode['damage_count']),
                            'policy_steps': int(episode['policy_steps']),
                            'physical_frames': int(episode['physical_frames']),
                            'wall_stall_decisions': int(episode['wall_stall_decisions']),
                            'visited_cells': int(episode['visited_cells']),
                            'reason': episode['reason'], 'native_status': native['status'],
                            'evaluation_return': float(rewards)}
                        records.append(row)
                        stage_rows.append(row)
                        if episode_index == 0:
                            replay = {'schema_version': 1, 'stage': stage, 'mode': mode,
                                'steps': steps, 'seed': env_seed, 'action_seed': action_seed,
                                'checkpoint': str(checkpoint), 'original_native_start': True,
                                'selection': 'first_episode_unfiltered',
                                'score': score, 'native_outcome': native, 'frames': frames,
                                'sample_unit': 'initial_state_then_every_policy_decision'}
                            replays[str(stage)] = replay
                            if output is not None:
                                write_json(output/'replays'/f'stage_{stage:02d}_{mode}.json', replay)
                finally:
                    env.close()
                average = lambda key: sum(row[key] for row in stage_rows) / episodes
                metrics[str(stage)] = {
                    'stage': stage, 'episodes': episodes, 'mean_score': average('score'),
                    'return_rate': average('game_return'), 'score40_rate': average('score40'),
                    'star1_rate': average('star1'), 'star2_rate': average('star2'),
                    'star3_rate': average('star3'), 'quest_rate': average('quest_collected'),
                    'mean_warmth': average('warmth'), 'mean_damage_count': average('damage_count'),
                    'mean_policy_steps': average('policy_steps'),
                    'mean_physical_frames': average('physical_frames'),
                    'mean_wall_stall_decisions': average('wall_stall_decisions'),
                    'mean_visited_cells': average('visited_cells'),
                    'entropy': sum(entropies)/len(entropies) if entropies else None,
                    'mean_max_action_probability': sum(maximums)/len(maximums) if maximums else None,
                    'score_distribution': dict(Counter(str(row['score']) for row in stage_rows)),
                    'reasons': dict(Counter(row['reason'] for row in stage_rows)),
                    'replay': str(output/'replays'/f'stage_{stage:02d}_{mode}.json') if output else None}
    finally:
        if was_training is not None and hasattr(policy, 'set_training_mode'):
            policy.set_training_mode(was_training)
    # 전체 성적은 Stage별 지표의 평균입니다. 평가 반복 횟수는 독립 학습 시드 수가 아닙니다.
    macro_keys = ('mean_score', 'return_rate', 'score40_rate', 'star1_rate',
                  'star2_rate', 'star3_rate', 'quest_rate')
    result = {'schema_version': 1, 'algorithm': 'PPO', 'single_policy': True,
        'original_native_start': True, 'steps': steps,
        'updates': int(getattr(model, '_n_updates', 0)), 'checkpoint': str(checkpoint),
        'mode': mode, 'seed': seed, 'episodes_per_stage': episodes,
        'total_episodes': len(records), 'per_stage': metrics,
        'macro': {key: sum(row[key] for row in metrics.values())/len(metrics) for key in macro_keys},
        'native_score_thresholds': [40, 45, 65, 85],
        'training_auxiliary_rewards_enabled': False,
        'randomize_phase': config['environment']['randomize_phase'],
        'greedy_repetitions_are_independent_training_seeds': False,
        'history_rows': [{'steps': steps, 'updates': int(getattr(model, '_n_updates', 0)),
            'mode': mode, 'seed': seed, 'checkpoint': str(checkpoint), **row}
            for row in metrics.values()]}
    if output is not None:
        with (output/'episodes.csv').open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        write_json(output/'summary.json', result)
    return result
