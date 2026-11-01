"""학습 시드 집계와 프로필 분리를 검증합니다."""

import json
from pathlib import Path

import pytest

from test_transfer import report
from test_comparison import prepare_pair


def paired(tmp_path,monkeypatch,seed,profile='benchmark'):
    from A2cToPpo import compare
    from A2cToPpo.runtime import write_json
    paths=prepare_pair(tmp_path/str(seed))
    for path in paths.values():
        file=path/'config.json';config=json.loads(file.read_text(encoding='utf-8'))
        config['training']['seed']=seed;config['profile']=profile
        config['algorithm_updates']={'n_epochs':4} if config['algorithm']=='PPO' else {'use_rms_prop':False}
        write_json(file,config)
    if seed in (22,33):
        file=paths['PPO']/'final_verification/last.json'
        original=json.loads(file.read_text(encoding='utf-8'))
        changed=report([(True,3),(False,0)] if seed==22 else [(True,3),(True,3)])
        original['modes']=changed['modes'];write_json(file,original)
    monkeypatch.setattr(compare,'RUN_ROOT',tmp_path/'comparisons')
    return Path(compare.compare_runs(paths['PPO'],paths['A2C'])['report_file'])


def test_aggregate_uses_independent_training_seeds_and_paired_differences(tmp_path,monkeypatch):
    from A2cToPpo.experiments import aggregate
    paths=[paired(tmp_path,monkeypatch,s) for s in [11,22,33]]
    result=aggregate(paths,tmp_path/'summary')
    assert result['training_seed_count']==3 and result['training_seeds']==[11,22,33]
    ppo=result['algorithms']['PPO']['last']['overall']['success_rate']
    assert ppo['mean']==.5 and ppo['std']==.5
    a2c=result['algorithms']['A2C']['last']['overall']['success_rate']
    assert a2c['mean']==.5 and a2c['std']==0
    assert result['paired_ppo_minus_a2c']['last']['success_rate']['mean']==0
    assert (tmp_path/'summary/summary.md').is_file()


def test_aggregate_rejects_duplicate_seeds_and_different_profiles(tmp_path,monkeypatch):
    from A2cToPpo.experiments import aggregate
    benchmark=paired(tmp_path,monkeypatch,11)
    practice=paired(tmp_path,monkeypatch,22,'return_practice')
    with pytest.raises(ValueError):aggregate([benchmark,benchmark],tmp_path/'duplicate')
    with pytest.raises(ValueError):aggregate([benchmark,practice],tmp_path/'mixed')
    assert not (tmp_path/'duplicate').exists() and not (tmp_path/'mixed').exists()


def test_repeated_runner_uses_both_algorithms_for_each_seed_without_changing_budget(tmp_path,monkeypatch):
    from A2cToPpo import experiments
    monkeypatch.setattr(experiments.cfg,'RUN_ROOT',tmp_path/'runs')
    calls=[]
    def completed(**kwargs):
        calls.append(kwargs)
        return {'status':'complete','run_dir':f"{kwargs['algorithm']}_{kwargs['seed']}"}
    comparisons=[]
    def compare(first,second):
        comparisons.append((first,second))
        return {'report_file':str(tmp_path/f'{len(comparisons)}.json')}
    monkeypatch.setattr(experiments,'train',completed)
    monkeypatch.setattr(experiments,'compare_runs',compare)
    monkeypatch.setattr(experiments,'aggregate',lambda paths,out:{'training_seed_count':len(paths)})
    result=experiments.run_experiments(seeds=[22,33],frames=300000,include=[])
    assert [(c['seed'],c['algorithm']) for c in calls]==[(22,'PPO'),(22,'A2C'),(33,'PPO'),(33,'A2C')]
    assert all(c['physical_frames']==300000 and c['profile']=='benchmark' for c in calls)
    assert result['training_seed_count']==2


def test_suite_check_never_starts_optimization_or_creates_results(tmp_path,monkeypatch):
    from A2cToPpo import experiments
    monkeypatch.setattr(experiments.cfg,'RUN_ROOT',tmp_path/'runs')
    calls=[]
    def check(**kwargs):
        calls.append(kwargs)
        assert kwargs['check']
        return {}
    monkeypatch.setattr(experiments,'train',check)
    result=experiments.run_experiments(profile='return_practice',seeds=[11],include=[],check=True)
    assert len(calls)==2 and not result['training_performed']
    assert not (tmp_path/'runs').exists()
