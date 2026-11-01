"""공통 초기화와 최종 채택·복원 절차를 실제 학습 없이 검증합니다."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_transfer import child, report


def test_both_algorithms_share_weights_hyperparameters_and_fresh_adam():
    child("""
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from A2cToPpo.runtime import bootstrap, source_config, make_env, transfer_model
from A2cToPpo.train import verify_transfer
bootstrap(); torch.set_num_threads(1)
vec = DummyVecEnv([lambda:make_env(source_config(),stages=[1],training=True) for _ in range(4)])
source, ppo = transfer_model(vec,11,algorithm='PPO')
_, a2c = transfer_model(vec,11,algorithm='A2C')
assert ppo.learning_rate == a2c.learning_rate == .00005
assert ppo.ent_coef == a2c.ent_coef == .005
assert ppo.n_steps == a2c.n_steps == 128
assert ppo.gamma == a2c.gamma == source.gamma
assert ppo.gae_lambda == a2c.gae_lambda == source.gae_lambda
assert type(ppo.policy.optimizer) is type(a2c.policy.optimizer) is torch.optim.Adam
assert not ppo.policy.optimizer.state and not a2c.policy.optimizer.state
assert ppo.num_timesteps == a2c.num_timesteps == 0
assert all(torch.equal(v,a2c.policy.state_dict()[k]) for k,v in ppo.policy.state_dict().items())
for model in [ppo,a2c]:
 result = verify_transfer(source,model,vec)
 assert result['identical_action_probabilities'] and result['identical_value_predictions']
 assert result['source_decision_steps'] == 6200068
vec.close()
""")


@pytest.mark.parametrize('update_hash', [False, True])
def test_last_model_cannot_replace_frozen_baseline(update_hash):
    child("""
import sys
from A2cToPpo import runtime,config
from A2cToPpo.prepare import sha256
runtime.bootstrap()
config.BASELINE_CHECKPOINT = config.BASELINE_CHECKPOINT.parent/'last_model.zip'
if sys.argv[1] == 'True': config.BASELINE_SHA256=sha256(config.BASELINE_CHECKPOINT)
try:runtime.load_baseline()
except ValueError:pass
else:raise AssertionError('last model was accepted as the initial baseline')
""", update_hash)


def test_quest_ability_is_protected_even_when_original_return_rate_is_zero():
    from A2cToPpo.config import EVALUATION
    from A2cToPpo.evaluate import preservation_guard
    baseline = report([(False, 0)])
    baseline['evaluation_settings'] = deepcopy(EVALUATION)
    candidate = deepcopy(baseline)
    candidate['modes']['stochastic']['per_stage']['1']['quest_rate'] = .1
    guard = preservation_guard(candidate, baseline)
    assert not guard['passed']
    assert guard['regressions'][0]['metric'] == 'quest_rate'


@pytest.mark.parametrize('case', ['regression', 'incomplete', 'tie', 'accepted'])
def test_final_adoption_copies_only_a_verified_candidate(tmp_path, monkeypatch, case):
    from A2cToPpo import selection
    from A2cToPpo.config import EVALUATION
    config = {'algorithm': 'PPO', 'source_environment_config': {}, 'evaluation': deepcopy(EVALUATION)}
    baseline = report([(True, 3), (False, 0), (False, 0)])
    if case == 'regression':
        candidate = report([(False, 0), (True, 3), (True, 3)])
    elif case == 'incomplete':
        candidate = report([(True, 3), (True, 3), (False, 0)])
    elif case == 'tie':
        candidate = deepcopy(baseline)
    else:
        candidate = report([(True, 3), (True, 3), (True, 3)])
    baseline.update(all_stages_passed=False, evaluation_settings=deepcopy(EVALUATION))
    candidate.update(all_stages_passed=case == 'accepted', evaluation_settings=deepcopy(EVALUATION))
    models = tmp_path / 'models'
    models.mkdir()
    for role, content in [('baseline', b'original'), ('candidate', b'candidate'), ('last', b'last'), ('best', b'original')]:
        (models / f'{role}_model.zip').write_bytes(content)
    monkeypatch.setattr(selection, 'load_model', lambda path, algorithm: SimpleNamespace(role=path.stem.split('_')[0]))
    calls = []
    def evaluate(model, settings, **kwargs):
        calls.append((model.role, kwargs))
        return deepcopy(baseline if model.role == 'baseline' else candidate)
    result = selection.finalize_run(tmp_path, config, 302830, evaluate_fn=evaluate)
    accepted = case == 'accepted'
    assert result['candidate_accepted'] == accepted
    assert result['selected_model_physical_frames'] == (302830 if accepted else 0)
    assert (models/'best_model.zip').read_bytes() == (b'candidate' if accepted else b'original')
    assert (models/'baseline_model.zip').read_bytes() == b'original'
    assert (models/'candidate_model.zip').read_bytes() == b'candidate'
    assert (models/'last_model.zip').read_bytes() == b'last'
    assert [role for role, _ in calls] == ['baseline', 'candidate', 'last']
    assert all(options['seed_base'] == EVALUATION['final_seed_base'] for _, options in calls)
    adopted = json.loads((tmp_path/'best_evaluation.json').read_text(encoding='utf-8'))
    assert adopted['model_role'] == ('candidate' if accepted else 'baseline')


def test_interrupted_verification_keeps_baseline_as_best(tmp_path, monkeypatch):
    from A2cToPpo import selection
    from A2cToPpo.config import EVALUATION
    models = tmp_path/'models'
    models.mkdir()
    for role in ['baseline', 'candidate', 'last', 'best']:
        (models/f'{role}_model.zip').write_bytes(b'baseline' if role in ('baseline','best') else role.encode())
    monkeypatch.setattr(selection, 'load_model', lambda path, algorithm: None)
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        selection.finalize_run(tmp_path, {'algorithm':'PPO','evaluation':deepcopy(EVALUATION),
                                        'source_environment_config':{}}, 1, evaluate_fn=interrupted)
    assert (models/'best_model.zip').read_bytes() == b'baseline'


def test_a2c_training_path_creates_new_optimizer_without_updating_weights(tmp_path):
    child("""
import sys
from pathlib import Path
import numpy as np,torch
from stable_baselines3 import A2C
from A2cToPpo import train as module
module.cfg.RUN_ROOT=Path(sys.argv[1])/'runs'
def no_update(self,*,callback,**kwargs):
 if self._last_obs is None:self._last_obs=self.env.reset()
 self._last_obs,_,_,infos=self.env.step(np.full(self.env.num_envs,4,dtype=np.int64))
 self.num_timesteps+=self.env.num_envs
 callback.init_callback(self);callback.num_timesteps=self.num_timesteps;callback.locals={'infos':infos}
 callback._on_step();return self
A2C.learn=no_update
def fake(*args,**kwargs):
 summary={'episodes':1,'success_rate':0.,'three_star_rate':0.,'mean_stars':0.,'mean_score':0.,
          'mean_damage':0.,'mean_physical_frames':1800.}
 return {'all_stages_passed':False,'modes':{'stochastic':{'overall':dict(summary),
         'per_stage':{str(s):dict(summary) for s in range(1,13)}}}}
module.periodic_evaluation=fake;module.evaluate=fake
result=module.train(physical_frames=1,algorithm='A2C')
assert result['physical_frames']==72 and result['decision_steps']==4
assert result['algorithm']=='A2C' and result['selected_model_origin']=='baseline'
run=Path(result['run_dir']);baseline=A2C.load(run/'models/baseline_model.zip',device='cpu')
last=A2C.load(run/'models/last_model.zip',device='cpu')
assert not last.policy.optimizer.state
assert all(torch.equal(v,last.policy.state_dict()[k]) for k,v in baseline.policy.state_dict().items())
""", tmp_path)


def prepare_pair(tmp_path):
    from A2cToPpo.config import EVALUATION
    from A2cToPpo.runtime import write_json
    paths = {algorithm: tmp_path/algorithm.lower() for algorithm in ['PPO','A2C']}
    baseline = report([(True,3),(False,0)])
    protocol = deepcopy(EVALUATION)
    protocol.update(stages=[1,2],final_stochastic_episodes=1,final_deterministic_episodes=0)
    for algorithm, path in paths.items():
        config = {'algorithm':algorithm,'experiment':'A2C_PPO_continuation',
                  'common_updates':{'learning_rate':.00005,'ent_coef':.005,'n_steps':128},
                  'source_environment_config':{'environment':{'action_repeat':18}},
                  'evaluation':protocol,'training':{'seed':11,'physical_frames':300000},
                  'source':{'source_sha256':'same_baseline'}}
        write_json(path/'config.json',config)
        write_json(path/'training_summary.json',{'status':'complete','final_verification':{'done':True},
                    'physical_frames':300000,'decision_steps':16700,'candidate_accepted':False})
        for role in ['baseline','candidate','last','best']:
            current = deepcopy(baseline)
            if algorithm=='PPO' and role=='last':current=report([(False,0),(False,0)])
            current.update(seed_base=protocol['final_seed_base'],evaluation_settings=protocol)
            write_json(path/'final_verification'/f'{role}.json',current)
    return paths


def test_comparison_keeps_raw_last_results_when_both_adopt_baseline(tmp_path,monkeypatch):
    from A2cToPpo import compare
    paths=prepare_pair(tmp_path)
    monkeypatch.setattr(compare,'RUN_ROOT',tmp_path/'output')
    result=compare.compare_runs(paths['PPO'],paths['A2C'])
    assert result['results']['PPO']['best']==result['results']['A2C']['best']
    assert result['results']['PPO']['last']['success_rate']==0
    assert result['results']['A2C']['last']['success_rate']==.5
    assert Path(result['report_file']).is_file()


@pytest.mark.parametrize('mismatch',['common_updates','source_environment_config','evaluation','training','source','partial'])
def test_comparison_rejects_unmatched_or_incomplete_runs(tmp_path,monkeypatch,mismatch):
    from A2cToPpo import compare
    from A2cToPpo.runtime import write_json
    paths=prepare_pair(tmp_path)
    monkeypatch.setattr(compare,'RUN_ROOT',tmp_path/'output')
    file=paths['A2C']/('training_summary.json' if mismatch=='partial' else 'config.json')
    data=json.loads(file.read_text(encoding='utf-8'))
    if mismatch=='partial':data['status']='interrupted'
    elif mismatch=='source':data['source']['source_sha256']='different'
    else:data[mismatch]['changed']=True
    write_json(file,data)
    with pytest.raises(ValueError):compare.compare_runs(paths['PPO'],paths['A2C'])
    assert not (tmp_path/'output').exists()


def test_both_command_stops_before_a2c_when_ppo_is_interrupted(monkeypatch):
    import sys
    from A2cToPpo import train
    calls=[]
    def interrupted(**kwargs):
        calls.append(kwargs)
        return {'status':'interrupted','final_verification':None}
    monkeypatch.setattr(train,'train',interrupted)
    monkeypatch.setattr(sys,'argv',['train','--algorithm','both','--frames','1'])
    train.main()
    assert len(calls)==1 and calls[0]['algorithm']=='PPO'


def test_both_command_uses_same_budget_and_compares_completed_runs(monkeypatch):
    import sys
    from A2cToPpo import train,compare
    calls=[]
    def completed(**kwargs):
        calls.append(kwargs)
        return {'status':'complete','final_verification':{'done':True},'run_dir':kwargs['algorithm']}
    comparisons=[]
    def compare_pair(*args):
        comparisons.append(args)
        return {'report_file':'test.json'}
    monkeypatch.setattr(train,'train',completed)
    monkeypatch.setattr(compare,'compare_runs',compare_pair)
    monkeypatch.setattr(sys,'argv',['train','--algorithm','both','--seed','22','--frames','300000'])
    train.main()
    assert [c['algorithm'] for c in calls]==['PPO','A2C']
    assert all(c['seed']==22 and c['physical_frames']==300000 for c in calls)
    assert comparisons==[('PPO','A2C')]
