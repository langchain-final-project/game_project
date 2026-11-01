"""원본 보존·전이·평가·관람을 실제 최적화 없이 검증합니다."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[2]


def child(code, *args):
    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', code, *map(str, args)],
                            cwd=PROJECT, capture_output=True, text=True, encoding='utf-8',
                            timeout=90, env={**os.environ, 'PYGAME_HIDE_SUPPORT_PROMPT': '1'})
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_transfer_uses_real_observations_and_saves_nothing():
    output = child("""
from A2cToPpo.train import train
from A2cToPpo.config import RUN_ROOT, VENDOR_ROOT, SOURCE_CHECKPOINT
from A2cToPpo.prepare import sha256
before = sha256(VENDOR_ROOT / SOURCE_CHECKPOINT)
files = set(RUN_ROOT.rglob('*')) if RUN_ROOT.exists() else set()
result = train(check=True)
assert result['identical_action_probabilities'] and result['identical_value_predictions']
assert result['optimizer_entries'] == result['ppo_decision_steps'] == 0
assert result['observation_shapes'] == {'map':[4,14,9,16], 'state':[4,224]}
assert result['action_count'] == 5 and result['action_repeat'] == 18
assert before == sha256(VENDOR_ROOT / SOURCE_CHECKPOINT)
assert files == (set(RUN_ROOT.rglob('*')) if RUN_ROOT.exists() else set())
""")
    assert 'A2C_pretrained_weights' in output


def test_native_physics_and_viewer_present_every_physical_frame():
    child("""
from copy import deepcopy
from A2cToPpo.runtime import bootstrap, source_config, make_env
bootstrap()
from kkamang.realtime import RealtimeGameCore
from A2cToPpo.watch import Player
settings = source_config()
env = make_env(settings, stages=[1])
env.reset(seed=11)
core = RealtimeGameCore(deepcopy(env.level), seed=11, randomize_phase=False, auto_interaction=True)
for _ in range(18): core.step(3)
obs, _, _, _, info = env.step(3)
assert info['executed_frames'] == 18
assert info['snapshot']['motion']['position'] == core.snapshot()['motion']['position']
assert info['snapshot']['motion']['frame'] == 18
env.close()
class Policy:
 def predict(self, obs, deterministic=False): return 3, None
player = Player(Policy(), settings, stage=1)
player.advance()
assert player.snapshot['motion']['frame'] == 1
assert player.env.core.snapshot()['motion']['frame'] == 18
for _ in range(17): player.advance()
assert player.snapshot['motion']['frame'] == 18 and not player.frames
player.advance()
assert player.snapshot['motion']['frame'] == 19
player.reset()
player.advance()
assert player.snapshot['motion']['frame'] == 1
player.close()
""")


def test_evaluation_restores_rng_and_policy_mode():
    child("""
import random, numpy as np, torch
from A2cToPpo.runtime import bootstrap, source_config
bootstrap()
from A2cToPpo.evaluate import evaluate
class Policy:
 training = True
 def set_training_mode(self, value): self.training = value
class Model:
 policy = Policy()
 def predict(self, obs, deterministic=False): return 4, None
model = Model()
random.seed(71); np.random.seed(72); torch.manual_seed(73)
states = random.getstate(), np.random.get_state(), torch.get_rng_state()
report = evaluate(model, source_config(), stages=[1], stochastic_episodes=2,
                  deterministic_episodes=1, seed_base=1000)
assert model.policy.training
assert random.getstate() == states[0]
after = np.random.get_state()
assert after[0] == states[1][0] and np.array_equal(after[1], states[1][1])
assert after[2:] == states[1][2:]
assert torch.equal(torch.get_rng_state(), states[2])
assert report['modes']['stochastic']['overall']['episodes'] == 2
assert report['modes']['stochastic']['overall']['success_rate'] == 0
assert all(e['reason'] == 'timeout' for e in report['modes']['stochastic']['episodes'])
""")


def report(stages):
    from A2cToPpo.evaluate import summarize
    rows = [{'game_return': returned, 'stars': stars, 'score': stars*30,
             'quest_collected': True, 'warmth': 14, 'damage_count': 0,
             'physical_frames': 900, 'policy_steps': 50} for returned, stars in stages]
    summaries = {str(i+1): summarize([row]) for i, row in enumerate(rows)}
    return {'modes': {'stochastic': {'overall': summarize(rows), 'per_stage': summaries}}}


def test_model_selection_protects_mastered_stages():
    from A2cToPpo.evaluate import preservation_guard, selection_key
    baseline = report([(True, 3), (False, 0), (False, 0)])
    candidate = report([(False, 0), (True, 3), (True, 3)])
    assert selection_key(candidate) > selection_key(baseline)
    assert not preservation_guard(candidate, baseline)['passed']
    improved = report([(True, 3), (True, 3), (False, 0)])
    assert preservation_guard(improved, baseline)['passed']
    assert selection_key(improved) > selection_key(baseline)


def test_training_orchestration_without_optimizer_updates(tmp_path):
    child("""
import sys
from pathlib import Path
import torch, numpy as np
from stable_baselines3 import PPO
from A2cToPpo import train as module
from A2cToPpo.prepare import sha256
from A2cToPpo.config import VENDOR_ROOT, SOURCE_CHECKPOINT
module.cfg.RUN_ROOT = Path(sys.argv[1]) / 'runs'
def no_optimizer_update(self, *, callback, **kwargs):
 if self._last_obs is None: self._last_obs = self.env.reset()
 self._last_obs, _, _, infos = self.env.step(np.full(self.env.num_envs,4,dtype=np.int64))
 self.num_timesteps += self.env.num_envs
 callback.init_callback(self)
 callback.num_timesteps = self.num_timesteps
 callback.locals = {'infos':infos}
 callback._on_step()
 return self
PPO.learn = no_optimizer_update
def fake_evaluation(*args, **kwargs):
 summary = {'episodes':10,'success_rate':0.,'three_star_rate':0.,'mean_stars':0.,
            'mean_score':0.,'mean_damage':0.,'mean_physical_frames':1800.}
 return {'all_stages_passed':False,'modes':{'stochastic':{
  'overall':dict(summary),'per_stage':{str(s):dict(summary) for s in range(1,13)}}}}
module.periodic_evaluation = fake_evaluation
module.evaluate = fake_evaluation
digest = sha256(VENDOR_ROOT / SOURCE_CHECKPOINT)
summary = module.train(physical_frames=1)
assert summary['status'] == 'complete' and summary['physical_frames'] == 72
assert summary['decision_steps'] == 4 and summary['selected_model_physical_frames'] == 0
run = next(module.cfg.RUN_ROOT.iterdir())
baseline = PPO.load(run/'models/baseline_model.zip',device='cpu')
last = PPO.load(run/'models/last_model.zip',device='cpu')
assert not last.policy.optimizer.state
assert all(torch.equal(v,last.policy.state_dict()[k]) for k,v in baseline.policy.state_dict().items())
assert digest == sha256(VENDOR_ROOT / SOURCE_CHECKPOINT)
assert (run/'final_verification/comparison.json').is_file()
""", tmp_path)


def test_corrupt_source_is_rejected_without_overwriting(tmp_path):
    child("""
import json,sys
from pathlib import Path
from A2cToPpo import runtime
root = Path(sys.argv[1])
(root/'sample.txt').write_text('changed',encoding='utf-8')
(root/'transfer_manifest.json').write_text(json.dumps({'files':{'sample.txt':'0'*64}}),encoding='utf-8')
runtime.VENDOR_ROOT = root
try: runtime.bootstrap()
except ValueError: pass
else: raise AssertionError('corrupt source was accepted')
assert (root/'sample.txt').read_text(encoding='utf-8') == 'changed'
""", tmp_path)
