"""학습 시드 반복 실행과 같은 프로필 안에서의 평균·표준편차 집계입니다."""

import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from . import config as cfg
from .compare import compare_runs, read_json
from .runtime import write_json
from .train import train, positive_integer

METRICS = ('success_rate','three_star_rate','mean_score','quest_rate','mean_damage')


def comparison_protocol(report):
    configs = {name:read_json(Path(run)/'config.json') for name,run in report['runs'].items()}
    config = configs['PPO']
    training = deepcopy(config['training'])
    training.pop('seed')
    return {'profile':config.get('profile','benchmark'), 'baseline_sha256':report['baseline_sha256'],
            'source_environment_config':config['source_environment_config'],
            'common_updates':config['common_updates'], 'training':training,
            'evaluation':config['evaluation'],
            'algorithm_updates':{name:c['algorithm_updates'] for name,c in configs.items()}}


def validate_comparisons(paths):
    reports = [read_json(path) for path in paths]
    if not reports:
        raise ValueError('집계할 비교 보고서가 필요합니다.')
    if len({r['seed'] for r in reports}) != len(reports):
        raise ValueError('같은 학습 시드가 중복되었습니다.')
    protocol = comparison_protocol(reports[0])
    for report in reports:
        if comparison_protocol(report) != protocol:
            raise ValueError('프로필·학습 예산·계수·평가 조건이 다른 실험을 합칠 수 없습니다.')
    return reports, protocol


def stats(values):
    return {'mean':mean(values), 'std':stdev(values) if len(values)>1 else None,
            'values':list(values)}


def aggregate(paths, output):
    reports,protocol = validate_comparisons(paths)
    result = {'profile':protocol['profile'], 'protocol':protocol,
              'training_seeds':[r['seed'] for r in reports], 'training_seed_count':len(reports),
              'comparison_files':[str(p) for p in paths], 'algorithms':{},'paired_ppo_minus_a2c':{}}
    for role in ('last','candidate','best'):
        result['paired_ppo_minus_a2c'][role] = {
            metric:stats([r['results']['PPO'][role][metric]-r['results']['A2C'][role][metric]
                          for r in reports]) for metric in METRICS}
    for algorithm in ('PPO','A2C'):
        result['algorithms'][algorithm] = {}
        for role in ('last','candidate','best'):
            overall = {metric:stats([r['results'][algorithm][role][metric] for r in reports]) for metric in METRICS}
            per_stage = {str(stage):{
                metric:stats([r['results'][algorithm]['per_stage'][role][str(stage)][metric] for r in reports])
                for metric in METRICS} for stage in protocol['evaluation']['stages']}
            result['algorithms'][algorithm][role] = {'overall':overall,'per_stage':per_stage}
    write_json(Path(output)/'summary.json',result)
    lines = ['# 반복 시드 비교 결과','',f"프로필: {result['profile']}",
             f"학습 시드: {result['training_seeds']}",'',
             '표준편차는 학습 시드 간 변동입니다. 평가 판 수를 독립 학습 횟수로 세지 않습니다.','',
             '| 모델 | 알고리즘 | 성공률 평균 ± 표준편차 | 별 3개 평균 ± 표준편차 | 평균 점수 |',
             '|---|---|---:|---:|---:|']
    for role in ('last','candidate','best'):
        for algorithm in ('PPO','A2C'):
            data = result['algorithms'][algorithm][role]['overall']
            def percent(metric):
                value = data[metric]
                deviation = f"{value['std']*100:.2f}%p" if value['std'] is not None else '시드 1개'
                return f"{value['mean']*100:.2f}% ± {deviation}"
            lines.append(f"| {role} | {algorithm} | {percent('success_rate')} | "
                         f"{percent('three_star_rate')} | {data['mean_score']['mean']:.2f} |")
    lines += ['', '같은 A2C 사전 학습 가중치에서 추가 학습한 결과입니다.',
              'benchmark와 return_practice 결과는 서로 합치지 않습니다.','']
    (Path(output)/'summary.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def run_experiments(*,profile='benchmark',seeds=(22,33),frames=300000,
                    include=None,check=False,aggregate_only=False):
    if len(set(seeds)) != len(seeds):
        raise ValueError('반복할 학습 시드가 중복되었습니다.')
    references = list(include) if include is not None else (
        [cfg.BENCHMARK_REFERENCE] if profile=='benchmark' and cfg.BENCHMARK_REFERENCE.is_file() else [])
    if references:
        reports,protocol = validate_comparisons(references)
        if protocol['profile'] != profile or (not aggregate_only and protocol['training']['physical_frames'] != frames):
            raise ValueError('포함한 기존 보고서의 프로필·프레임 예산이 이번 실험과 다릅니다.')
        if not aggregate_only:
            current_training = {**cfg.TRAINING,'physical_frames':frames}
            current_training.pop('seed')
            if (protocol['training'] != current_training or protocol['evaluation'] != cfg.EVALUATION
                    or protocol['baseline_sha256'] != cfg.BASELINE_SHA256
                    or protocol['algorithm_updates']['PPO'] != cfg.PPO_UPDATES
                    or any(protocol['common_updates'].get(k)!=v for k,v in cfg.COMMON_UPDATES.items())):
                raise ValueError('현재 설정이 기존 비교 보고서와 다릅니다. 반복 전에 설정을 맞추세요.')
        if not aggregate_only and {r['seed'] for r in reports} & set(seeds):
            raise ValueError('기존 보고서에 포함한 시드를 다시 실행하도록 요청했습니다.')
    if check:
        for seed in seeds:
            for algorithm in ('PPO','A2C'):
                train(seed=seed,physical_frames=frames,check=True,algorithm=algorithm,profile=profile)
        return {'status':'checked','profile':profile,'training_performed':False}
    output = cfg.RUN_ROOT/'suites'/f"{profile}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    comparisons = list(references)
    if not aggregate_only:
        for seed in seeds:
            runs = {}
            for algorithm in ('PPO','A2C'):
                result = train(seed=seed,physical_frames=frames,algorithm=algorithm,profile=profile)
                if result['status']!='complete':
                    write_json(output/'progress.json',{'status':'interrupted','comparisons':[str(p) for p in comparisons]})
                    return {'status':'interrupted','output':str(output)}
                runs[algorithm] = result['run_dir']
            paired = compare_runs(runs['PPO'],runs['A2C'])
            comparisons.append(Path(paired['report_file']))
            write_json(output/'progress.json',{'status':'running','comparisons':[str(p) for p in comparisons]})
    result = aggregate(comparisons,output)
    write_json(output/'progress.json',{'status':'complete','comparisons':[str(p) for p in comparisons]})
    print('반복 시드 집계 저장:',output,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description='같은 프로필로 A2C·PPO 학습 시드 반복 및 집계')
    parser.add_argument('--profile',choices=cfg.PROFILES,default='benchmark')
    parser.add_argument('--seeds',nargs='+',type=int,default=[22,33])
    parser.add_argument('--frames',type=positive_integer,default=cfg.TRAINING['physical_frames'])
    parser.add_argument('--include',nargs='*',type=Path,default=None)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--aggregate-only',action='store_true')
    args=parser.parse_args()
    run_experiments(profile=args.profile,seeds=args.seeds,frames=args.frames,
                    include=args.include,check=args.check,aggregate_only=args.aggregate_only)


if __name__=='__main__':
    main()
