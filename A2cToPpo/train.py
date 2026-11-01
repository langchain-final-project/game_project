"""같은 기준 가중치에서 A2C 또는 PPO를 새 최적화기로 추가 학습합니다."""

import argparse
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from . import config as cfg
from .evaluate import evaluate, preservation_guard, selection_key, print_report
from .runtime import (bootstrap, profile_settings, make_progress, make_env,
                      transfer_model, write_json, source_steps)


class FrameCounter(BaseCallback):
    """4개 환경의 실제 진행 프레임을 합산하고 종료한 판만 기록합니다."""

    def __init__(self, writer=None, stream=None):
        super().__init__()
        self.frames = 0
        self.episodes = 0
        self.writer, self.stream = writer, stream
        self.recent = deque(maxlen=100)

    def _on_step(self):
        infos = self.locals['infos']
        self.frames += sum(info['executed_frames'] for info in infos)
        for rank, info in enumerate(infos):
            if 'episode' not in info:
                continue
            episode = info['episode']
            self.episodes += 1
            self.recent.append(episode)
            if self.writer:
                row = {'physical_frames_total': self.frames,
                       'decision_steps_total': self.num_timesteps, 'env_rank': rank}
                row.update({key: episode.get(key) for key in EPISODE_FIELDS})
                row['start_type'] = episode.get('start_type','full_start')
                self.writer.writerow(row)
        if self.stream:
            self.stream.flush()
        return True


EPISODE_FIELDS = ('stage', 'game_return', 'score', 'stars', 'quest_collected',
                  'warmth', 'damage_count', 'policy_steps', 'physical_frames',
                  'wall_stall_decisions', 'reason', 'learning_return',
                  'start_type','start_distance','prepared_frames','quest_collected_at_start',
                  'damage_at_start','base_learning_return','return_extra_reward')


def verify_transfer(source, model, vec):
    observation = vec.reset()
    source.policy.set_training_mode(False)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        tensor = source.policy.obs_to_tensor(observation)[0]
        before = source.policy.get_distribution(tensor).distribution.probs
        after = model.policy.get_distribution(tensor).distribution.probs
        value_before = source.policy.predict_values(tensor)
        value_after = model.policy.predict_values(tensor)
    report = {
        'algorithm': type(model).__name__, 'initialization': 'A2C_pretrained_weights',
        'source_decision_steps': source_steps(source),
        'observation_shapes': {key: list(value.shape) for key, value in observation.items()},
        'action_count': int(model.action_space.n), 'action_repeat': 18,
        'identical_action_probabilities': torch.equal(before, after),
        'identical_value_predictions': torch.equal(value_before, value_after),
        'optimizer_entries': len(model.policy.optimizer.state),
        'new_decision_steps': int(model.num_timesteps),
        'ppo_decision_steps': int(model.num_timesteps),
        'shared_hyperparameters': deepcopy(model.a2c_to_ppo['shared_hyperparameters']),
        'algorithm_updates': deepcopy(model.a2c_to_ppo['algorithm_updates']),
        'baseline_checkpoint': str(cfg.BASELINE_CHECKPOINT),
        'baseline_sha256': cfg.BASELINE_SHA256,
    }
    if (not report['identical_action_probabilities'] or not report['identical_value_predictions']
            or report['optimizer_entries'] or report['ppo_decision_steps']):
        raise RuntimeError('가중치 전이 검증에 실패했습니다.')
    return report


def periodic_evaluation(model, settings):
    return evaluate(model, settings,
                    stochastic_episodes=cfg.EVALUATION['periodic_stochastic_episodes'],
                    deterministic_episodes=cfg.EVALUATION['periodic_deterministic_episodes'],
                    seed_base=cfg.EVALUATION['periodic_seed_base'])


def train(*, seed=11, physical_frames=300_000, check=False, algorithm='PPO',profile='benchmark'):
    if type(physical_frames) is not int or physical_frames <= 0:
        raise ValueError('물리 프레임 예산은 양의 정수여야 합니다.')
    algorithm = algorithm.upper()
    if algorithm not in ('PPO', 'A2C'):
        raise ValueError('알고리즘은 PPO 또는 A2C여야 합니다.')
    manifest = bootstrap()
    torch.set_num_threads(cfg.TRAINING['torch_threads'])
    settings = profile_settings(profile)
    practice_progress = make_progress(settings)
    source_total = settings['training']['total_steps']
    progress = 0.0
    vec = DummyVecEnv([
        lambda rank=rank: make_env(settings, stages=cfg.TRAINING['stages'], seed=seed+rank,
                                   training=True, progress=progress, practice_progress=practice_progress)
        for rank in range(cfg.TRAINING['n_envs'])
    ])
    try:
        source, model = transfer_model(vec, seed, algorithm=algorithm)
        original_steps = source_steps(source)
        progress = min(1., original_steps / source_total)
        vec.env_method('set_training_progress', progress)
        if any(gamma != model.gamma for gamma in vec.get_attr('gamma')):
            raise ValueError('환경 보상 할인과 PPO gamma가 다릅니다.')
        verification = verify_transfer(source, model, vec)
        verification['profile'] = profile
        if check:
            import json
            print(json.dumps(verification, ensure_ascii=False, indent=2))
            print('연결 정상 | 학습·새 모델 저장 없음 | 기존 rl 설정 변경 없음')
            return verification
        prefix = '' if profile=='benchmark' else profile+'_'
        run = cfg.RUN_ROOT / f"{algorithm.lower()}_{prefix}seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        models = run / 'models'
        models.mkdir(parents=True)
        run_config = {
            'experiment': 'A2C_PPO_continuation', 'algorithm': algorithm,
            'profile':profile,
            'initialization': 'A2C_pretrained_weights',
            'source_environment_config': settings,
            'training': {**cfg.TRAINING, 'seed': seed, 'physical_frames': physical_frames},
            'common_updates': deepcopy(model.a2c_to_ppo['shared_hyperparameters']),
            'algorithm_updates': deepcopy(model.a2c_to_ppo['algorithm_updates']),
            'evaluation': deepcopy(cfg.EVALUATION),
            'source': {'archive_sha256': manifest['archive_sha256'], **model.a2c_to_ppo},
            'units': {'budget': 'aggregate_physical_frames',
                      'sb3_num_timesteps': 'aggregate_action_decisions', 'action_repeat': 18},
        }
        write_json(run / 'config.json', run_config)
        write_json(run / 'transfer_check.json', verification)
        model.a2c_to_ppo['environment_config'] = deepcopy(settings)
        model.save(models / 'baseline_model.zip')
        model.save(models / 'best_model.zip')
        model.save(models / 'candidate_model.zip')
        model.set_logger(configure(str(run / 'logs'), ['csv']))
        print('실행 폴더:', run, flush=True)
        print(f'{algorithm} | 추가 예산 {physical_frames:,} 물리 프레임 | 18프레임 행동 | Stage 1~12', flush=True)
        print('추가 학습 전 기준 모델을 같은 평가 시드에서 평가합니다.', flush=True)
        baseline = periodic_evaluation(model, settings)
        baseline['model_role'] = 'A2C_pretrained_policy_before_PPO_updates'
        write_json(run / 'evaluations' / 'baseline.json', baseline)
        print_report(baseline, '학습 전 기준', detailed=True)
        best_report, best_frames = baseline, 0
        counter = FrameCounter()
        status = 'complete'
        next_eval = cfg.TRAINING['evaluation_interval_frames']
        next_progress = cfg.TRAINING['progress_interval_frames']
        try:
            with (run / 'train_episodes.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=(
                    'physical_frames_total', 'decision_steps_total', 'env_rank', *EPISODE_FIELDS))
                writer.writeheader()
                counter.writer, counter.stream = writer, stream
                while counter.frames < physical_frames:
                    progress = min(1., (original_steps + model.num_timesteps) / source_total)
                    vec.env_method('set_training_progress', progress)
                    # 선택한 알고리즘으로 수집 묶음을 갱신한 다음 평가합니다.
                    model.learn(total_timesteps=model.n_steps * vec.num_envs,
                                reset_num_timesteps=False, callback=counter,
                                log_interval=1, progress_bar=False)
                    model.a2c_to_ppo['physical_frames'] = counter.frames
                    if counter.frames >= next_progress:
                        print(f'[{counter.frames:,}/{physical_frames:,}F] '
                              f'행동 선택 {model.num_timesteps:,}회 | 종료 {counter.episodes}판', flush=True)
                        next_progress = (counter.frames // cfg.TRAINING['progress_interval_frames'] + 1) * cfg.TRAINING['progress_interval_frames']
                    if counter.frames < next_eval and counter.frames < physical_frames:
                        continue
                    checkpoint = f'{algorithm.lower()}_frames_{counter.frames:010d}.zip'
                    model.save(models / checkpoint)
                    model.save(models / 'last_model.zip')
                    report = periodic_evaluation(model, settings)
                    report.update(physical_frames=counter.frames,
                                  decision_steps=int(model.num_timesteps), checkpoint=checkpoint)
                    guard = preservation_guard(report, baseline)
                    report['preservation_guard'] = guard
                    if practice_progress is not None:
                        report['return_curriculum'] = practice_progress.report()
                    selected = guard['passed'] and selection_key(report) > selection_key(best_report)
                    if selected:
                        model.save(models / 'candidate_model.zip')
                        best_report, best_frames = report, counter.frames
                    report['selected_as_candidate'] = selected
                    write_json(run / 'evaluations' / f'frames_{counter.frames:010d}.json', report)
                    print_report(report, f'평가 {counter.frames:,}F')
                    print(f"기존 성능 보존: {guard['passed']} | 검증 대기 후보 갱신: {selected}", flush=True)
                    next_eval = (counter.frames // cfg.TRAINING['evaluation_interval_frames'] + 1) * cfg.TRAINING['evaluation_interval_frames']
        except KeyboardInterrupt:
            status = 'interrupted'
            print('학습 중단 요청: 마지막 상태와 기존 최적 모델을 따로 보관합니다.', flush=True)
        model.a2c_to_ppo['physical_frames'] = counter.frames
        model.save(models / 'last_model.zip')
        write_json(run / 'candidate_evaluation.json', best_report)
        write_json(run / 'best_evaluation.json', baseline)
        summary = {'status': 'verifying' if status == 'complete' else status,
                   'algorithm': algorithm, 'initialization': 'A2C_pretrained_weights',
                   'profile':profile,
                   'run_dir': str(run),
                   'physical_frames': counter.frames, 'decision_steps': int(model.num_timesteps),
                   'selected_model_physical_frames': 0,
                   'candidate_model_physical_frames': best_frames,
                   'candidate_accepted': False, 'selected_model_origin': 'baseline',
                   'episodes': counter.episodes, 'final_verification': None}
        if practice_progress is not None:
            summary['return_curriculum'] = practice_progress.report()
        write_json(run / 'training_summary.json', summary)
        if status == 'complete':
            print('새 시드로 기준·후보·마지막 모델을 평가한 뒤 최종 채택 여부를 판단합니다.', flush=True)
            try:
                from .selection import finalize_run
                decision = finalize_run(run, run_config, best_frames, evaluate_fn=evaluate)
                summary.update(decision)
                summary['status'] = 'complete'
            except KeyboardInterrupt:
                summary['status'] = 'final_verification_interrupted'
                print('최종 평가가 중단되었습니다. 저장된 최적 모델은 보존됩니다.', flush=True)
            except Exception as error:
                summary.update(status='final_verification_failed', error=str(error))
                write_json(run / 'training_summary.json', summary)
                raise
            write_json(run / 'training_summary.json', summary)
        print('저장:', run, flush=True)
        return summary
    finally:
        vec.close()


def positive_integer(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('양의 정수를 입력하세요.')
    return number


def main():
    parser = argparse.ArgumentParser(description='A2C 사전 학습 가중치에서 PPO 추가 학습')
    parser.add_argument('--seed', type=int, default=cfg.TRAINING['seed'])
    parser.add_argument('--frames', type=positive_integer, default=cfg.TRAINING['physical_frames'])
    parser.add_argument('--algorithm', type=str.lower, choices=('ppo', 'a2c', 'both'), default='ppo',
                        help='같은 기준에서 PPO, A2C 또는 두 실험을 순서대로 실행')
    parser.add_argument('--profile',choices=cfg.PROFILES,default='benchmark')
    parser.add_argument('--check', action='store_true', help='가중치·환경 연결만 검증하고 학습하지 않음')
    args = parser.parse_args()
    algorithms = ('PPO', 'A2C') if args.algorithm == 'both' else (args.algorithm.upper(),)
    results = []
    for algorithm in algorithms:
        result = train(seed=args.seed, physical_frames=args.frames, check=args.check,
                       algorithm=algorithm,profile=args.profile)
        results.append(result)
        if not args.check and result.get('status') != 'complete':
            break
    if (args.algorithm == 'both' and not args.check and len(results) == 2
            and all(r.get('final_verification') for r in results)):
        from .compare import compare_runs
        report = compare_runs(results[0]['run_dir'], results[1]['run_dir'])
        print('대조 실험 비교 저장:', report['report_file'], flush=True)


if __name__ == '__main__':
    main()
