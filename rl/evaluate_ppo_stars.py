# 공식·보조 평가의 귀환·별·점수와 실제 재생·고정 기준 상태의 행동 확률을 기록합니다.
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from rl.ppo_stars_common import (GRAPH_METRICS, build_config, evaluation_cases, isolated_action_rng,
                       load_model, make_env, validate_run, write_csv, write_json)


ACTION_PROBABILITY_KEYS = ('p_up', 'p_down', 'p_left', 'p_right', 'p_interact', 'p_wait')


def policy_record(model, observation, snapshot):
    """PPO의 행동 추첨 없이 현재 입력에 대한 6개 행동 확률을 조회합니다."""
    policy = model.policy
    was_training = policy.training
    try:
        policy.set_training_mode(False)
        with torch.no_grad():
            tensor, _ = policy.obs_to_tensor(observation)
            probabilities = policy.get_distribution(tensor).distribution.probs[0].cpu().tolist()
    finally:
        policy.set_training_mode(was_training)
    return {
        # 좌표만 같아도 시간·수집 상태는 다를 수 있어 전체 입력의 해시를 함께 비교합니다.
        'observation_hash': hashlib.sha256(np.asarray(observation, dtype='<f4').tobytes()).hexdigest(),
        'tick': snapshot['tick'], 'remaining_ticks': snapshot['remaining_ticks'],
        'x': snapshot['position'][0], 'y': snapshot['position'][1],
        'warmth': snapshot['carried'], 'quest_collected': snapshot['quest']['collected'],
        'courage': snapshot['courage'], 'greedy_action': int(np.argmax(probabilities)),
        **dict(zip(
            ('goal_quest', 'goal_warmth', 'goal_return', 'goal_dx', 'goal_dy', 'interaction_ready'),
            map(float, observation[-6:]),
        )),
        **dict(zip(ACTION_PROBABILITY_KEYS, probabilities)),
    }


def annotate_replay(model, env, replay):
    """대표 재생의 원래 행동열을 다시 실행해 행동 직전 상태의 확률을 붙입니다."""
    with isolated_action_rng():
        observation, info = env.reset(seed=replay['seed'])
        for frame in replay['frames']:
            frame['decision'] = policy_record(model, observation, info['snapshot'])
            # 원래 행동을 그대로 실행합니다. 정책 행동을 새로 추첨하지 않습니다.
            observation, _, _, _, info = env.step(frame['action'])


def rollout_episode(model, env, *, seed, deterministic=True, action_seed=None, config=None):
    """학습 알고리즘과 분리된 예측 인터페이스로 한 회차를 실행합니다. 가중치는 바꾸지 않습니다."""
    config = build_config() if config is None else config
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError('모델과 선택한 환경의 관측·행동 공간이 다릅니다.')
    if deterministic:
        action_seed = None
    elif action_seed is None:
        action_seed = config['evaluation']['action_seed_base'] + seed
    policy = getattr(model, 'policy', None)
    was_training = getattr(policy, 'training', None)
    try:
        # 환경 초기화 seed와 정책의 행동 추첨 seed를 구분하고 평가 후 학습 RNG를 복구합니다.
        with isolated_action_rng(action_seed):
            obs, info = env.reset(seed=seed)
            initial_tick = info['snapshot']['tick']
            replay = {
                'schema_version': 1, 'seed': seed, 'action_seed': action_seed,
                'deterministic': deterministic,
                'evaluation_mode': 'deterministic' if deterministic else 'stochastic',
                'environment': config['environment']['name'],
                'observation_size': int(env.observation_space.shape[0]),
                'action_unit': config['environment']['action_unit'],
                'config_hash': info['config_hash'], 'observation_version': info['observation_version'],
                'initial': info['snapshot'], 'actions': [], 'frames': [],
            }
            terminated = truncated = False
            cumulative_reward = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(np.asarray(action).item())
                obs, reward, terminated, truncated, info = env.step(action)
                cumulative_reward += float(reward)
                replay['actions'].append(action)
                replay['frames'].append({
                    'action': action, 'reward': float(reward), 'cumulative_reward': cumulative_reward,
                    'snapshot': info['snapshot'], 'events': info['events'],
                    'reward_terms': info['reward_terms'],
                })
            # 실시간 환경에서는 행동 수와 세계 틱 수가 다르므로 둘 다 기록합니다.
            episode = {**info['rl_episode'], 'ticks': info['snapshot']['tick'] - initial_tick}
            replay.update(episode=episode, terminated=terminated, truncated=truncated)
            return replay
    finally:
        # SB3 정책과 일반 PyTorch 정책 모두 모드를 되돌릴 수 있게 예측 인터페이스를 분리합니다.
        setter = getattr(policy, 'set_training_mode', None) or getattr(policy, 'train', None)
        if isinstance(was_training, bool) and callable(setter):
            setter(was_training)


def summarize_episodes(rows):
    """성공·실패를 먼저 구분하고 별 1개 이상→2개 이상→3개 및 점수를 계산합니다."""
    if not rows:
        raise ValueError('집계할 평가 회차가 없습니다.')
    n = len(rows)
    successful = [row for row in rows if row['success']]
    ready = [row for row in rows if row.get('conditions_met', False)]
    metrics = {
        'success_count': len(successful), 'failed_count': n - len(successful),
        'success_rate': len(successful) / n,
        'mean_score': mean(row['score'] for row in rows),
        'mean_success_score': mean(row['score'] for row in successful) if successful else None,
        # 분모는 전 평가 회차입니다. 실패를 제외해 성공률이 낮은 모델을 유리하게 만들지 않습니다.
        'at_least_one_star_rate': sum(row['stars'] >= 1 for row in successful) / n,
        'at_least_two_star_rate': sum(row['stars'] >= 2 for row in successful) / n,
        'three_star_rate': sum(row['stars'] == 3 for row in successful) / n,
        'perfect_rate': sum(row['score'] == 100 for row in successful) / n,
        'quest_rate': sum(row.get('quest_collected', False) for row in rows) / n,
        'warmth_rate': sum(row.get('warmth', 0) >= 4 for row in rows) / n,
        'conditions_rate': len(ready) / n,
        'return_after_conditions_rate': len(successful) / len(ready) if ready else None,
        'mean_steps': mean(row.get('steps', 0) for row in rows),
        'mean_ticks': mean(row.get('ticks', 0) for row in rows),
        'mean_return': mean(row.get('return', 0) for row in rows),
        'reasons': dict(Counter(row.get('reason', 'unknown') for row in rows)),
    }
    for stars in range(4):
        # 성공했지만 45점 미만인 별 0개와 실패(점수·별 0)를 섞지 않습니다.
        subset = [row for row in successful if row['stars'] == stars]
        metrics[f'star_{stars}_count'] = len(subset)
        metrics[f'star_{stars}_mean_score'] = mean(row['score'] for row in subset) if subset else None
    return metrics


def evaluate_model(model, output_dir: Path, *, training_seed: int, model_steps: int,
                   seeds=None, model_path=None, deterministic=True, action_seed_base=None,
                   config=None) -> dict:
    """한 평가 방식을 수행합니다. 공식·보조 결과는 서로 다른 폴더에 저장합니다."""
    config = build_config() if config is None else config
    evaluation = config['evaluation']
    mode = 'deterministic' if deterministic else 'stochastic'
    if seeds is None:
        seeds = evaluation[f'{mode}_seeds']
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('중복되지 않는 0 이상의 평가 seed를 지정하세요.')
    action_seed_base = evaluation['action_seed_base'] if action_seed_base is None else action_seed_base
    if action_seed_base < 0:
        raise ValueError('행동 seed 기준값은 0 이상이어야 합니다.')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_env(config)
    started = perf_counter()
    rows, trajectories, saved_kinds, decisions = [], set(), set(), []
    identity = {'algorithm': config['algorithm'], 'training_seed': training_seed,
                'model_steps': model_steps, 'evaluation_mode': mode, 'deterministic': deterministic}
    try:
        for index, seed in enumerate(seeds):
            replay = rollout_episode(model, env, seed=seed, deterministic=deterministic,
                                     action_seed=None if deterministic else action_seed_base + seed,
                                     config=config)
            row = {
                **replay['episode'], **identity, 'episode': index,
                'evaluation_seed': seed, 'action_seed': replay['action_seed'],
                'terminated': replay['terminated'], 'truncated': replay['truncated'],
                'config_hash': replay['config_hash'], 'observation_version': replay['observation_version'],
            }
            rows.append(row)
            trajectories.add(hashlib.sha256(bytes(replay['actions'])).hexdigest())
            kind = 'success' if row['success'] else 'failure'
            if kind not in saved_kinds:
                annotate_replay(model, env, replay)
                replay.update(identity, model_path=model_path)
                write_json(output_dir / 'replays' / f'{kind}.json', replay)
                decisions.extend({**identity, 'representative_kind': kind, 'evaluation_seed': seed,
                                  'action_seed': replay['action_seed'], 'step_before': step,
                                  **frame['decision'], 'chosen_action': frame['action']}
                                 for step, frame in enumerate(replay['frames']))
                saved_kinds.add(kind)
    finally:
        env.close()

    metrics = summarize_episodes(rows)
    warmth_goal = config['environment']['warmth_goal']
    metrics.update(
        mean_warmth=mean(row['warmth'] for row in rows),
        warmth_goal_rate=sum(row['warmth'] >= warmth_goal for row in rows) / len(rows),
        early_return_rate=sum(row['success'] and row['warmth'] < warmth_goal for row in rows) / len(rows),
        star_training_pass=(
            metrics['success_rate'] >= evaluation['success_rate_threshold']
            and metrics['at_least_one_star_rate'] >= evaluation['star_rate_threshold']
        ),
    )
    summary = {
        'schema_version': 1, 'evaluation_version': evaluation['version'], **identity,
        'episodes': len(rows), 'official': mode == evaluation['primary_mode'],
        'evaluation_seconds': perf_counter() - started,
        'evaluation_seeds': list(seeds), 'action_seeds': [row['action_seed'] for row in rows],
        'config_hash': rows[0]['config_hash'], 'observation_version': rows[0]['observation_version'],
        **metrics,
        'unique_trajectories': len(trajectories),
        'deterministic_repeat': deterministic and len(trajectories) == 1 and len(rows) > 1,
        'return_pass': metrics['success_rate'] >= evaluation['success_rate_threshold'],
    }
    write_csv(output_dir / 'episodes.csv', rows)
    # 대표 성공·실패의 모든 행동을 저장해 같은 경로의 중복 기록을 줄입니다.
    write_csv(output_dir / 'action_probabilities.csv', decisions)
    write_json(output_dir / 'summary.json', summary)
    write_csv(output_dir / 'summary.csv', [{key: value for key, value in summary.items()
                                           if not isinstance(value, (dict, list))}])
    return summary


def evaluate_both(model, output_dir, *, training_seed, model_steps, model_path=None,
                  smoke=False, config=None, record_scalar=None):
    config = build_config() if config is None else config
    summaries = []
    for case in evaluation_cases(config, smoke=smoke):
        summary = evaluate_model(model, Path(output_dir) / case['name'], training_seed=training_seed,
                                 model_steps=model_steps, model_path=model_path, config=config,
                                 deterministic=case['deterministic'], seeds=case['seeds'])
        summaries.append(summary)
        label = '공식' if summary['official'] else '보조'
        print(f"평가 {model_steps:,}스텝 [{case['name']}, {summary['episodes']}회, {label}]: "
              f"성공률={summary['success_rate']:.0%}, 별1+={summary['at_least_one_star_rate']:.0%}, "
              f"별2+={summary['at_least_two_star_rate']:.0%}, 별3={summary['three_star_rate']:.0%}, "
              f"평균 점수={summary['mean_score']:.1f}, "
              f"평균 온기={summary['mean_warmth']:.1f}, "
              f"목표 온기 충족={summary['warmth_goal_rate']:.0%}", flush=True)
    return summaries


def log_evaluation(summaries, record_scalar):
    """학습·재평가가 같은 그래프 지표를 기록합니다. 기록기는 호출자가 관리합니다."""
    for summary in summaries:
        for key in GRAPH_METRICS:
            if summary.get(key) is not None:
                record_scalar(f"eval/{summary['evaluation_mode']}/{key}", summary[key])


def evaluate_run(run_dir: Path) -> list[dict]:
    """새 표준 형식의 저장 모델을 같은 조건으로 다시 평가합니다."""
    from torch.utils.tensorboard import SummaryWriter

    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    torch.set_num_threads(config['ppo']['torch_threads'])
    metadata = json.loads((run_dir / 'metadata.json').read_text(encoding='utf-8'))
    evaluator = validate_run(config, metadata)
    root = run_dir / 'evaluation'
    report = {'schema_version': 1, 'config': config['evaluation'], 'training_seed': metadata['training_seed'],
              'status': 'running', 'evaluator': evaluator}
    models = sorted((run_dir / 'checkpoints').glob('*.zip'))
    final = run_dir / 'final_model.zip'
    if final.exists():
        models.append(final)
    if not models:
        raise ValueError(f'저장 모델이 없습니다: {run_dir}')
    write_json(root / 'evaluation.json', report)
    summaries, seen_steps = [], set()
    writer = SummaryWriter(log_dir=str(root / 'tensorboard'))
    try:
        for path in models:
            model = load_model(path, config)
            if getattr(model, 'seed', metadata['training_seed']) != metadata['training_seed']:
                raise ValueError(f'기록과 모델의 학습 seed가 다릅니다: {path}')
            steps = int(model.num_timesteps)
            if steps in seen_steps:
                continue
            seen_steps.add(steps)
            batch = evaluate_both(model, root / f'step_{steps:07d}',
                                  training_seed=metadata['training_seed'], model_steps=steps,
                                  model_path=str(path.resolve()), config=config,
                                  smoke=config.get('execution', {}).get('run_kind') == 'smoke',
                                  record_scalar=lambda tag, value: writer.add_scalar(tag, value, steps))
            summaries.extend(batch)
            log_evaluation(batch, lambda tag, value: writer.add_scalar(tag, value, steps))
        report['status'] = 'complete'
    except BaseException:
        report['status'] = 'error'
        raise
    finally:
        writer.close()
        write_json(root / 'evaluation.json', report)
    return summaries


def main():
    parser = argparse.ArgumentParser(description='공식·보조 평가로 귀환·별·점수를 기록합니다.')
    parser.add_argument('--run-dir', type=Path, required=True, help='표준 config.json과 모델이 있는 실행 폴더')
    evaluate_run(parser.parse_args().run_dir)


if __name__ == '__main__':
    main()
