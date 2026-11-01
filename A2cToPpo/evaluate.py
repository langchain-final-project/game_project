"""같은 게임·행동 시드에서 모든 스테이지의 실제 귀환과 별을 비교합니다."""

import argparse
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import random
from statistics import mean

import numpy as np
import torch

from .config import EVALUATION, TRAINING
from .runtime import bootstrap, make_env, write_json, load_model


@contextmanager
def isolated_rng():
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def summarize(episodes):
    if not episodes:
        raise ValueError('평가할 에피소드가 필요합니다.')
    return {
        'episodes': len(episodes),
        'success_rate': mean(float(e['game_return']) for e in episodes),
        'three_star_rate': mean(float(e['stars'] == 3) for e in episodes),
        'mean_stars': mean(e['stars'] for e in episodes),
        'mean_score': mean(e['score'] for e in episodes),
        'quest_rate': mean(float(e['quest_collected']) for e in episodes),
        'mean_warmth': mean(e['warmth'] for e in episodes),
        'mean_damage': mean(e['damage_count'] for e in episodes),
        'mean_physical_frames': mean(e['physical_frames'] for e in episodes),
        'mean_decisions': mean(e['policy_steps'] for e in episodes),
    }


def evaluate(model, settings, *, stages=None, stochastic_episodes=10,
             deterministic_episodes=1, seed_base=10_000, evaluation_settings=None):
    protocol = deepcopy(EVALUATION if evaluation_settings is None else evaluation_settings)
    stages = list(protocol['stages'] if stages is None else stages)
    if not stages or any(type(s) is not int or not 1 <= s <= 12 for s in stages):
        raise ValueError('평가 스테이지는 1~12여야 합니다.')
    if stochastic_episodes < 1 or deterministic_episodes < 0:
        raise ValueError('확률적 평가는 1판 이상, 결정적 평가는 0판 이상이어야 합니다.')
    report = {'algorithm': type(model).__name__, 'primary_mode': 'stochastic',
              'seed_base': seed_base, 'stages': stages,
              'action_repeat': settings['environment']['action_repeat'],
              'original_native_start': True, 'evaluation_settings': protocol, 'modes': {}}
    was_training = model.policy.training
    with isolated_rng():
        model.policy.set_training_mode(False)
        try:
            for mode, count in [('deterministic', deterministic_episodes), ('stochastic', stochastic_episodes)]:
                if not count:
                    continue
                episodes = []
                for stage in stages:
                    env = make_env(settings, stages=[stage], training=False)
                    try:
                        for index in range(count):
                            seed = seed_base + stage * 1000 + index
                            action_seed = protocol['action_seed_offset'] + seed
                            random.seed(action_seed)
                            np.random.seed(action_seed % 2**32)
                            torch.manual_seed(action_seed)
                            obs, _ = env.reset(seed=seed, options={'stage': stage})
                            while True:
                                action, _ = model.predict(obs, deterministic=mode == 'deterministic')
                                obs, _, done, truncated, info = env.step(int(action))
                                if done or truncated:
                                    episode = dict(info['episode'])
                                    episode.update(mode=mode, seed=seed, action_seed=action_seed)
                                    episodes.append(episode)
                                    break
                    finally:
                        env.close()
                per_stage = {str(s): summarize([e for e in episodes if e['stage'] == s]) for s in stages}
                report['modes'][mode] = {'overall': summarize(episodes),
                                         'per_stage': per_stage, 'episodes': episodes}
        finally:
            model.policy.set_training_mode(was_training)
    primary = report['modes']['stochastic']['per_stage']
    report['all_stages_passed'] = all(
        s['success_rate'] >= protocol['target_success_rate']
        and s['three_star_rate'] >= protocol['target_three_star_rate']
        for s in primary.values())
    return report


def preservation_guard(candidate, baseline):
    protocol = baseline.get('evaluation_settings', EVALUATION)
    current = candidate['modes']['stochastic']['per_stage']
    reference = baseline['modes']['stochastic']['per_stage']
    failures = []
    for stage, original in reference.items():
        for metric in protocol.get('protected_metrics', ['success_rate', 'three_star_rate']):
            if metric not in original:
                continue
            rate = current.get(stage, {}).get(metric, 0.0)
            if (original[metric] >= protocol['protected_baseline_rate']
                    and rate + 1e-9 < original[metric] - protocol['maximum_stage_drop']):
                failures.append({'stage': int(stage), 'metric': metric,
                                 'baseline': original[metric], 'candidate': rate})
    return {'passed': not failures, 'regressions': failures}


def adoption_decision(candidate, baseline):
    protocol = baseline.get('evaluation_settings', EVALUATION)
    guard = preservation_guard(candidate, baseline)
    checks = {
        'preservation': guard['passed'],
        'improves_over_baseline': selection_key(candidate) > selection_key(baseline),
        'all_stage_targets': (candidate.get('all_stages_passed', False)
                              or not protocol.get('require_all_stages_passed_for_adoption', True)),
    }
    return {'accepted': all(checks.values()), 'checks': checks,
            'preservation_guard': guard,
            'failed_checks': [key for key, passed in checks.items() if not passed]}


def selection_key(report):
    summary = report['modes']['stochastic']['overall']
    return (summary['three_star_rate'], summary['success_rate'], summary['mean_score'],
            -summary['mean_damage'], -summary['mean_physical_frames'])


def print_report(report, label='평가', detailed=False):
    primary = report['modes']['stochastic']
    summary = primary['overall']
    print(f"[{label}] 확률적 성공 {summary['success_rate']:.1%} | "
          f"별 3개 {summary['three_star_rate']:.1%} | 평균 별 {summary['mean_stars']:.2f} | "
          f"평균 피해 {summary['mean_damage']:.2f}", flush=True)
    if detailed:
        for stage, value in primary['per_stage'].items():
            print(f"  Stage {stage}: 성공 {value['success_rate']:.1%} | "
                  f"별 3개 {value['three_star_rate']:.1%} | 점수 {value['mean_score']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description='분리된 PPO 모델을 실제 시작점에서 평가합니다.')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', default='best_model.zip')
    parser.add_argument('--episodes', type=int, default=EVALUATION['final_stochastic_episodes'])
    parser.add_argument('--deterministic-episodes', type=int, default=EVALUATION['final_deterministic_episodes'])
    args = parser.parse_args()
    bootstrap()
    torch.set_num_threads(TRAINING['torch_threads'])
    config = json.loads((args.run / 'config.json').read_text(encoding='utf-8'))
    if (config.get('algorithm') not in ('PPO', 'A2C')
            or config.get('experiment') not in ('A2C_to_PPO', 'A2C_PPO_continuation')):
        raise ValueError('A2cToPpo 실행 폴더를 선택하세요.')
    model = load_model(args.run / 'models' / args.checkpoint, config['algorithm'])
    protocol = config.get('evaluation', EVALUATION)
    report = evaluate(model, config['source_environment_config'],
                      stochastic_episodes=args.episodes,
                      deterministic_episodes=args.deterministic_episodes,
                      seed_base=protocol['final_seed_base'], evaluation_settings=protocol)
    folder = args.run / 'evaluations' / ('manual_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    write_json(folder / 'report.json', report)
    print_report(report, detailed=True)
    print('저장:', folder)


if __name__ == '__main__':
    main()
