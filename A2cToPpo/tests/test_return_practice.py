"""실제 경로 재생·준비 상태 보존·귀환 보상·평가 분리를 확인합니다."""

from test_transfer import child


def test_every_prepared_state_replays_and_finishes_with_three_stars():
    child("""
from A2cToPpo.runtime import bootstrap,source_config,profile_settings,make_env
from A2cToPpo.return_practice import ensure_bank,snapshot_hash
bootstrap()
settings=profile_settings('return_practice');recipe=settings['return_practice']
bank,_=ensure_bank(settings,recipe)
assert set(s['stage'] for s in bank['states'])=={3,4,9,10}
for state in bank['states']:
 env=make_env(source_config(),stages=[state['stage']])
 env.reset(seed=state['reset_seed'],options={'stage':state['stage']})
 for action in state['prefix']:
  result=env.core.step(action)
  assert not result.terminated
 snapshot=env.core.snapshot()
 assert snapshot_hash(snapshot)==state['snapshot_sha256']
 assert snapshot['quest']['collected'] and snapshot['carried']==14
 assert snapshot['motion']['frame']==state['native_frame']
 for action in state['continuation']:
  result=env.core.step(action)
  if result.terminated:break
 assert result.terminated and result.snapshot['outcome']['score']['stars']==3
 env.close()
""")


def test_return_reset_keeps_native_time_and_does_not_pay_prepared_pickups():
    child("""
from copy import deepcopy
from A2cToPpo.runtime import bootstrap,source_config,profile_settings,make_env
from A2cToPpo.return_practice import ensure_bank,ReturnProgress,ReturnPractice
bootstrap();settings=profile_settings('return_practice');recipe=deepcopy(settings['return_practice'])
recipe['practice_probability']=1.
bank,_=ensure_bank(settings,recipe);progress=ReturnProgress(bank,recipe)
state=next(s for s in bank['states'] if s['stage']==3 and s['distance']==6)
progress.choose=lambda rng:state
env=ReturnPractice(make_env(source_config(),stages=[3],training=True),settings,bank,recipe,progress)
_,info=env.reset(seed=11)
assert not info['original_native_start'] and info['quest_collected_at_start']
assert info['snapshot']['motion']['frame']==state['native_frame']
assert info['snapshot']['remaining_ticks']==state['remaining_ticks']
assert env.env.physical_frames==env.env.policy_steps==0
info=env.step(4)[4]
assert info['reward_terms']['quest_reward']==info['reward_terms']['pickup_reward']==0
assert info['executed_frames']==18 and env.env.physical_frames==18
assert info['snapshot']['motion']['frame']==state['native_frame']+18
env.reset(seed=11)
action=state['continuation'][0]
first=env.step(action)[4]['reward_terms']['return_progress_reward']
opposite={0:1,1:0,2:3,3:2}[action]
second=env.step(opposite)[4]['reward_terms']['return_progress_reward']
assert first>0 and abs(first+second)<1e-8
env.close()
""")


def test_shared_progress_promotes_only_current_tasks_and_is_fresh_for_each_algorithm():
    child("""
import numpy as np
from A2cToPpo.runtime import bootstrap,profile_settings,make_progress,make_env
bootstrap();settings=profile_settings('return_practice')
first,second=make_progress(settings),make_progress(settings)
assert first is not second
stage=3;state=first.states[stage][0]
for i in range(8):first.record(state,i>=2)
assert first.frontier[stage]==1 and second.frontier[stage]==0
for _ in range(8):first.record(state,True)
assert first.frontier[stage]==1
env=make_env(settings,stages=[3],training=False)
_,info=env.reset(seed=11,options={'stage':3})
assert info['original_native_start'] and info['snapshot']['motion']['frame']==0
assert not info['snapshot']['quest']['collected'] and info['snapshot']['carried']==0
env.close()
""")


def test_training_profile_mixes_original_starts_and_prepared_returns():
    child("""
from A2cToPpo.runtime import bootstrap,profile_settings,make_env,make_progress
bootstrap();settings=profile_settings('return_practice')
env=make_env(settings,training=True,practice_progress=make_progress(settings))
count=0;native_stages=set()
for i in range(500):
 _,info=env.reset(seed=11 if i==0 else None)
 if info['start_type']=='return_practice':
  count+=1
  assert info['snapshot']['stage'] in [3,4,9,10] and info['snapshot']['quest']['collected']
 else:
  native_stages.add(info['snapshot']['stage'])
  assert info['snapshot']['motion']['frame']==0
assert .64<count/500<.76 and native_stages==set(range(1,13))
env.close()
""")


def test_prepared_return_episode_is_marked_separately_and_updates_shared_progress():
    child("""
from copy import deepcopy
from A2cToPpo.runtime import bootstrap,source_config,profile_settings,make_env
from A2cToPpo.return_practice import ensure_bank,ReturnProgress,ReturnPractice
bootstrap();settings=profile_settings('return_practice');recipe=deepcopy(settings['return_practice'])
recipe['practice_probability']=1.
bank,_=ensure_bank(settings,recipe);tracker=ReturnProgress(bank,recipe)
state=next(s for s in bank['states'] if s['stage']==3 and s['distance']==1)
tracker.choose=lambda rng:state
env=ReturnPractice(make_env(source_config(),stages=[3],training=True),settings,bank,recipe,tracker)
env.reset(seed=11)
_,reward,done,tr,info=env.step(state['continuation'][0])
assert done and not tr and info['episode']['stars']==3
assert info['episode']['start_type']=='return_practice' and info['episode']['quest_collected_at_start']
assert info['episode']['physical_frames']==info['executed_frames']<18
assert info['episode']['prepared_frames']==state['native_frame']
assert info['reward_terms']['return_success_bonus']==.5
assert tracker.attempts[3]==tracker.successes[3]==1
assert not info['original_native_start']
env.close()
""")
