"""Stage 3 목표 계획·역순 커리큘럼·전이와 최종 검증을 학습 없이 확인합니다."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage3EfficientCurriculum, Stage3ThreeStarCurriculum
from rl.ppo_goal import select_goal, bonus_collection_safe
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.ppo_stage3_plan import plan_target, stage3_warmth_target
from rl.train_ppo import make_env, make_model, training_curriculum_class


def test_tour_avoids_nearby_dead_end_when_quest_and_return_make_other_item_cheaper():
    snapshot = {'grid':['#########','#.......#','#.......#','#.......#','#########'],
                'motion':{'position':[1,2]}, 'carried':0,
                'items':[{'id':'a','active':True,'value':1,'position':[1,1]},
                         {'id':'b','active':True,'value':3,'position':[5,2]}],
                'quest':{'position':[7,2],'collected':False},'shortcuts':[]}
    assert plan_target(snapshot,3,[1,2]) == [5,2]
    assert plan_target(snapshot,5,[1,2]) is None


def test_sampler_reviews_collection_and_quest_while_expanding_backwards():
    env = Stage3EfficientCurriculum(RealtimeMansionEnv(stage=3))
    try:
        _, info = env.reset(seed=11)
        assert info['curriculum_start']==18
        assert len(env.prefixes)==23
        assert training_curriculum_class() is Stage3EfficientCurriculum
        env.frontier = 16
        counts = Counter()
        for _ in range(20000):
            start = env._choose_start();counts[env._phase]+=1
            if env._phase=='stage3_collect_focus':assert start in range(1,9)
            if env._phase=='stage3_quest_focus':assert start in (9,10,11,12)
            if env._phase=='stage3_return_focus':assert start in (13,14,15,16,17,18)
        for phase, expected in {'full_start':.05,'stage3_collect_focus':.05,
                                'stage3_quest_focus':.05,'stage3_return_focus':.4,
                                'review':.05,'frontier':.4}.items():
            assert counts[phase]/20000 == pytest.approx(expected,abs=.015)
    finally:
        env.close()


@pytest.mark.parametrize('change',[{'stage3_collect_probability':.9},
                                   {'stage3_quest_starts':[]},{'stage3_collect_starts':[99]}])
def test_invalid_stage3_mix_is_rejected(change):
    settings = deepcopy(cfg.CURRICULUM_CONFIG);settings.update(change)
    env=RealtimeMansionEnv(stage=3)
    try:
        with pytest.raises(ValueError):Stage3EfficientCurriculum(env,settings=settings)
    finally:env.close()


def test_goal_guide_has_feasible_three_star_path_without_running_ppo():
    env=PPOGoalObservation(RealtimeMansionEnv(stage=3))
    try:
        _,info=env.reset(seed=11)
        for _ in range(1800):
            position=info['snapshot']['motion']['position'];goal=info['goal_position']
            dx,dy=goal[0]-position[0],goal[1]-position[1]
            moves=[(abs(dx),3 if dx>0 else 2),(abs(dy),1 if dy>0 else 0)]
            candidates=[(v,a) for v,a in moves if v>1e-8 and env.unwrapped.core.can_move(a)]
            action=max(candidates)[1] if candidates else 5
            _,_,done,tr,info=env.step(action)
            if done or tr:break
        assert info['outcome']['score']['stars']==3
        assert info['snapshot']['quest']['collected']
        assert info['snapshot']['carried']>=12
        assert info['snapshot']['motion']['frame']<1000
        # 규칙 행동은 목표 안내의 실현 가능성 점검이며 PPO 성능 수치가 아닙니다.
    finally:env.close()


def test_two_hits_raise_warmth_reserve_and_do_not_pay_return_progress_early():
    settings=deepcopy(cfg.CURRICULUM_CONFIG);settings['initial_frontier']=9
    env=PPOReward(PPOGoalObservation(Stage3EfficientCurriculum(RealtimeMansionEnv(stage=3),settings=settings)))
    try:
        _,info=env.reset(seed=11)
        snapshot=deepcopy(info['snapshot'])
        snapshot['quest']['collected']=True
        snapshot['carried']=13
        snapshot['damage_count']=2;snapshot['courage']=1
        original=deepcopy(snapshot)
        assert bonus_collection_safe(snapshot,cfg.STAR_GOAL_CONFIG)
        assert stage3_warmth_target(snapshot,cfg.STAR_GOAL_CONFIG,True)==14
        assert select_goal(snapshot,cfg.STAR_GOAL_CONFIG)[0]==1
        assert env._return_distance(snapshot) is None
        assert snapshot==original
        snapshot['remaining_ticks']=10
        assert not bonus_collection_safe(snapshot,cfg.STAR_GOAL_CONFIG)
        assert select_goal(snapshot,cfg.STAR_GOAL_CONFIG)[0]==2
    finally:env.close()


def test_legacy_stage3_model_does_not_use_new_tour_planner(monkeypatch):
    from rl import ppo_stage3_plan
    def unexpected(*args):raise AssertionError('new planner used for legacy model')
    monkeypatch.setattr(ppo_stage3_plan,'efficient_goal',unexpected)
    settings=deepcopy(cfg.STAR_GOAL_CONFIG);settings.pop('stage3_route_plan')
    env=RealtimeMansionEnv(stage=3)
    try:
        snapshot=env.reset(seed=11)[1]['snapshot']
        assert select_goal(snapshot,settings)[0]==1
    finally:env.close()


def test_source_weights_are_preserved_but_stage3_metadata_and_optimizer_are_new(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG,'enabled',False)
    env=make_env()
    try:
        source=make_model(env,11);source.ppo_stage=2
        path=tmp_path/'stage2.zip';source.save(path)
        scale=cfg.TRANSFER_CONFIG['action_head_scale']
        model=make_model(env,11,transfer_settings={'enabled':True,'model_path':path,'action_head_scale':scale})
        assert all(torch.equal(v*scale if k.startswith('action_net.') else v,model.policy.state_dict()[k])
                   for k,v in source.policy.state_dict().items())
        assert model.ppo_stage==3 and model.ppo_star_goal_config['stage3_route_plan']
        assert not hasattr(model,'ppo_joint_review_config')
        assert not hasattr(model,'ppo_final_stage2_config')
        assert model.ppo_final_stage3_config['heldout_seed_base']==70000
        assert model.num_timesteps==0 and not model.policy.optimizer.state
    finally:env.close()


def test_final_stage3_evaluation_uses_saved_stage_and_restores_current_config(monkeypatch,tmp_path):
    from stable_baselines3 import PPO
    from rl import evaluate_ppo
    from rl.ppo_safe_return import verify_stage_model
    settings={**cfg.STAGE3_TRAINING_CONFIG,'environment':deepcopy(cfg.ENV_CONFIG),
              'evaluation':deepcopy(cfg.EVAL_CONFIG)}
    model=SimpleNamespace(ppo_stage=3,ppo_final_stage3_config=settings)
    monkeypatch.setattr(PPO,'load',lambda *args,**kwargs:model)
    original=deepcopy(cfg.EVAL_CONFIG)
    def evaluate(model):
        assert cfg.EVAL_CONFIG['stochastic_seeds']==list(range(70000,70200))
        assert cfg.ENV_CONFIG['stage']==3
        return {'passed':False}
    monkeypatch.setattr(evaluate_ppo,'evaluate_model',evaluate)
    monkeypatch.setattr(evaluate_ppo,'save_evaluation',lambda *args:None)
    report=verify_stage_model('unused.zip',tmp_path,3)
    assert report['final_stage3_verification'] and not report['passed']
    assert cfg.EVAL_CONFIG==original


def test_saved_stage2_final_recipe_still_works_from_stage3_workspace(monkeypatch,tmp_path):
    from stable_baselines3 import PPO
    from rl import evaluate_ppo
    from rl.ppo_safe_return import verify_final_model
    evaluation=deepcopy(cfg.EVAL_CONFIG)
    evaluation['minimum_magnet_use_rate']=.8
    environment={**cfg.ENV_CONFIG,'stage':2}
    model=SimpleNamespace(ppo_stage=2,ppo_final_stage2_config={**cfg.FINAL_STAGE2_CONFIG,
                            'environment':environment,'evaluation':evaluation})
    monkeypatch.setattr(PPO,'load',lambda *args,**kwargs:model)
    original_stage=cfg.ENV_CONFIG['stage']
    def evaluate(model):
        assert cfg.ENV_CONFIG['stage']==2
        assert cfg.EVAL_CONFIG['minimum_magnet_use_rate']==.8
        return {'passed':False}
    monkeypatch.setattr(evaluate_ppo,'evaluate_model',evaluate)
    monkeypatch.setattr(evaluate_ppo,'save_evaluation',lambda *args:None)
    assert verify_final_model('unused.zip',tmp_path)['final_stage2_verification']
    assert cfg.ENV_CONFIG['stage']==original_stage==3
    assert cfg.EVAL_CONFIG['minimum_magnet_use_rate']==0
