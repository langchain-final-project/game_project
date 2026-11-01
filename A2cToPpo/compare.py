"""동일한 출발점·학습 조건·평가 시드를 사용한 두 추가 학습 결과를 비교합니다."""

import argparse
from datetime import datetime
import json
from pathlib import Path

from .config import RUN_ROOT
from .runtime import write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare_runs(ppo_run, a2c_run):
    runs = {'PPO': Path(ppo_run), 'A2C': Path(a2c_run)}
    configs = {name: read_json(path / 'config.json') for name, path in runs.items()}
    for algorithm, config in configs.items():
        if config['algorithm'] != algorithm or config['experiment'] != 'A2C_PPO_continuation':
            raise ValueError('새 대조 실험으로 만든 PPO·A2C 실행 폴더를 지정하세요.')
    first, second = configs['PPO'], configs['A2C']
    if first.get('profile','benchmark') != second.get('profile','benchmark'):
        raise ValueError('서로 다른 실험 프로필은 비교할 수 없습니다.')
    for key in ('common_updates', 'source_environment_config', 'evaluation', 'training'):
        if first[key] != second[key]:
            raise ValueError('비교 조건이 다릅니다: ' + key)
    if first['source']['source_sha256'] != second['source']['source_sha256']:
        raise ValueError('출발 가중치가 다른 실험은 같은 출발점 대조 실험으로 비교할 수 없습니다.')
    reports = {}
    for algorithm, path in runs.items():
        summary = read_json(path / 'training_summary.json')
        if summary.get('status') != 'complete' or not summary.get('final_verification'):
            raise ValueError('최종 검증을 완료한 실행만 비교할 수 있습니다.')
        reports[algorithm] = {
            role: read_json(path / 'final_verification' / f'{role}.json')
            for role in ('baseline', 'candidate', 'last', 'best')
        }
        protocol = configs[algorithm]['evaluation']
        for report in reports[algorithm].values():
            if (report.get('seed_base') != protocol['final_seed_base']
                    or report.get('evaluation_settings') != protocol):
                raise ValueError('최종 평가 시드 또는 평가 조건이 다릅니다.')
            stages = report['modes']['stochastic']['per_stage']
            if (set(stages) != {str(s) for s in protocol['stages']}
                    or any(stage['episodes'] != protocol['final_stochastic_episodes']
                           for stage in stages.values())):
                raise ValueError('최종 확률적 평가의 스테이지 또는 평가 판 수가 부족합니다.')
    baseline_first = reports['PPO']['baseline']['modes']['stochastic']['overall']
    baseline_second = reports['A2C']['baseline']['modes']['stochastic']['overall']
    if baseline_first != baseline_second:
        raise ValueError('동일해야 할 학습 전 평가 결과가 다릅니다. 초기화·평가 조건을 확인하세요.')
    if (reports['PPO']['baseline']['modes']['stochastic']['per_stage']
            != reports['A2C']['baseline']['modes']['stochastic']['per_stage']):
        raise ValueError('학습 전 스테이지별 평가 결과가 다릅니다.')
    result = {
        'comparison_type': 'matched_A2C_pretrained_continuation',
        'profile':first.get('profile','benchmark'),
        'runs': {name: str(path) for name, path in runs.items()},
        'requested_physical_frames': first['training']['physical_frames'],
        'baseline_sha256': first['source']['source_sha256'],
        'seed': first['training']['seed'], 'baseline': baseline_first,
        'results': {},
    }
    for algorithm, roles in reports.items():
        summary = read_json(runs[algorithm] / 'training_summary.json')
        result['results'][algorithm] = {
            'actual_physical_frames': summary['physical_frames'],
            'decision_steps': summary['decision_steps'],
            'candidate_accepted': summary['candidate_accepted'],
            **{role: report['modes']['stochastic']['overall'] for role, report in roles.items()},
            'per_stage': {role: report['modes']['stochastic']['per_stage'] for role, report in roles.items()},
        }
    # 같은 예산의 마지막 모델과 평가로 선택한 후보를 함께 보여 줍니다.
    # 기준으로 되돌린 best만 비교해서 실제 학습의 성능 하락을 숨기지 않습니다.
    for role in ('last', 'candidate', 'best'):
        for algorithm in ('A2C', 'PPO'):
            metrics = result['results'][algorithm][role]
            print(f"{role} | {algorithm}: 성공 {metrics['success_rate']:.1%} | "
                  f"별 3개 {metrics['three_star_rate']:.1%} | 점수 {metrics['mean_score']:.2f}", flush=True)
    file = RUN_ROOT / 'comparisons' / f"seed_{result['seed']}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
    result['report_file'] = str(file)
    write_json(file, result)
    return result


def main():
    parser = argparse.ArgumentParser(description='같은 조건의 A2C·PPO 추가 학습 결과 비교')
    parser.add_argument('--ppo-run', type=Path, required=True)
    parser.add_argument('--a2c-run', type=Path, required=True)
    args = parser.parse_args()
    result = compare_runs(args.ppo_run, args.a2c_run)
    print('비교 저장:', result['report_file'])


if __name__ == '__main__':
    main()
