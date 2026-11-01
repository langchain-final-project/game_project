"""무피해 복습·피해 보상·성능 보존·최종 검증을 학습 없이 확인합니다."""

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl import ppo_safe_return as safe
from rl.ppo_curriculum import Stage2SafeReturnCurriculum
from rl.ppo_reward import PPOReward
from rl.train_ppo import evaluation_key, training_curriculum_class


@pytest.fixture(autouse=True)
def stage2_safe_return_profile(monkeypatch):
    # 현재 실행 스테이지와 분리해 당시 Stage 2 조건을 검증합니다.
    monkeypatch.setitem(cfg.ENV_CONFIG, 'stage', 2)
    for key, value in {
        'name': 'stage2_safe_return_v1', 'initial_frontier': 9,
        'full_start_probability': .4, 'review_probability': .05,
        'magnet_focus_probability': .05, 'quest_focus_probability': .1,
        'ghost_focus_probability': .35, 'keep_focus_at_frontier_zero': True,
    }.items():
        monkeypatch.setitem(cfg.CURRICULUM_CONFIG, key, value)
    for key, value in {'warmth_target': 14, 'stage2_three_star': True,
                       'stage3_three_star': False}.items():
        monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, key, value)


def test_full_play_and_ghost_focus_are_both_retained_at_frontier_zero():
    env = Stage2SafeReturnCurriculum(RealtimeMansionEnv(stage=2))
    try:
        env.reset(seed=11)
        assert training_curriculum_class() is Stage2SafeReturnCurriculum
        for frontier in (9, 0):
            env.frontier = frontier
            counts = Counter()
            for _ in range(20000):
                env._choose_start()
                counts[env._phase] += 1
            for phase, expected in {"full_start": .4, "magnet_focus": .05, "quest_focus": .1,
                                    "ghost_focus": .35, "review": .05, "frontier": .05}.items():
                assert counts[phase] / 20000 == pytest.approx(expected, abs=.015)
    finally:
        env.close()


@pytest.mark.parametrize("start", [9, 10, 11, 12, 13, 14])
def test_ghost_preparation_preserves_real_time_without_starting_with_damage(start):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(full_start_probability=0, magnet_focus_probability=0,
                    quest_focus_probability=0, review_probability=0,
                    ghost_focus_probability=1, ghost_focus_starts=[start])
    env = Stage2SafeReturnCurriculum(RealtimeMansionEnv(stage=2), settings=settings)
    try:
        env.reset(seed=11)
        waits = set()
        for _ in range(40):
            _, info = env.reset()
            wait = info["curriculum_ghost_wait_frames"]
            waits.add(wait)
            snapshot = info["snapshot"]
            assert snapshot["damage_count"] == 0 and snapshot["courage"] == 3
            assert snapshot["quest"]["collected"]
            assert snapshot["motion"]["frame"] == env.prefix_frames[start] + wait
            assert env.env._steps == 0
            assert env._prepared[start].snapshot()["motion"]["frame"] == env.prefix_frames[start]
        assert max(waits) > 18 and len(waits) >= 8
    finally:
        env.close()


def test_scripted_feasible_route_can_complete_with_three_stars_and_no_damage():
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings["initial_frontier"] = 9
    env = Stage2SafeReturnCurriculum(RealtimeMansionEnv(stage=2), settings=settings)
    try:
        env.reset(seed=11)
        for action in env.route[env.prefix_frames[9]:]:
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert info["outcome"]["score"]["stars"] == 3
        assert info["snapshot"]["damage_count"] == 0
        assert info["curriculum_task_success"]
    finally:
        env.close()


def test_damaged_three_star_clear_does_not_promote_curriculum(monkeypatch):
    env = Stage2SafeReturnCurriculum(RealtimeMansionEnv(stage=2))
    try:
        obs, info = env.reset(seed=11)
        snapshot = deepcopy(info["snapshot"])
        snapshot["damage_count"] = 1
        result = {**info, "snapshot": snapshot, "outcome": {"status": "COMPLETE", "score": {"stars": 3}}}
        monkeypatch.setattr(env.env, "step", lambda action: (obs, 0., True, False, result))
        assert not env.step(5)[4]["curriculum_task_success"]
        assert list(env.history) == [False]
    finally:
        env.close()


class RewardEvents(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(640,), dtype=np.float32)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, snapshot, events, stars=None):
        self.core = SimpleNamespace(config={"stage": 2})
        self.snapshot, self.events, self.stars = snapshot, events, stars

    def info(self):
        return {"snapshot": deepcopy(self.snapshot), "events": self.events,
                "goal_phase": 2, "goal_position": self.snapshot["exit"],
                "outcome": None if self.stars is None else {"status": "COMPLETE",
                    "score": {"total": 90, "stars": self.stars}}}

    def reset(self, *, seed=None, options=None):
        return np.zeros(640, dtype=np.float32), self.info()

    def step(self, action):
        return np.zeros(640, dtype=np.float32), 0., self.stars is not None, False, self.info()


@pytest.fixture
def snapshot():
    env = RealtimeMansionEnv(stage=2)
    try:
        return env.reset(seed=11)[1]["snapshot"]
    finally:
        env.close()


@pytest.mark.parametrize("stars,damage,bonus", [(3, 0, 8.), (2, 0, 0.), (3, 1, 0.)])
def test_damage_cost_and_bonus_require_damage_free_three_star_clear(snapshot, stars, damage, bonus):
    snapshot["damage_count"] = damage
    env = PPOReward(RewardEvents(snapshot, [{"type": "damage"}] * damage, stars))
    try:
        env.reset()
        terms = env.step(5)[4]["reward_terms"]
        assert terms["damage_penalty"] == -6. * damage
        assert terms["damage_free_success_reward"] == bonus
    finally:
        env.close()


@pytest.mark.parametrize("distance,protection,negative", [(.7, 0, True), (2., 0, False), (.7, 3, False)])
def test_risk_penalty_uses_physical_proximity_and_respects_protection(snapshot, distance, protection, negative):
    x, y = snapshot["motion"]["position"]
    snapshot["motion"]["ghost_positions"] = {"A": [x + distance, y]}
    snapshot["protection"] = protection
    env = PPOReward(RewardEvents(snapshot, []))
    try:
        env.reset()
        value = env.step(5)[4]["reward_terms"]["ghost_risk_penalty"]
        assert (value < 0) == negative
        assert -.02 <= value <= 0
    finally:
        env.close()


def report(success=.96, three=.82, damage=1.16, damage_free=.05, stars=2.63):
    summary = {"success_rate": success, "three_star_rate": three, "mean_stars": stars,
               "mean_damage": damage, "damage_free_success_rate": damage_free,
               "magnet_use_rate": .97, "mean_steps": 490}
    return {"primary_mode": "stochastic", "selection_objective": "safe_three_stars", "quick": True,
            "success_rate_threshold": .9, "target_three_star_rate": .9, "target_mean_stars": 2.7,
            "maximum_mean_damage": .1, "minimum_magnet_use_rate": .8,
            "minimum_damage_free_success_rate": .8,
            "results": {"stochastic": {"summary": summary}}}


def test_selection_rewards_damage_reduction_above_performance_floors():
    baseline = report()
    safer = report(three=.84, damage=.3, damage_free=.7, stars=2.65)
    assert evaluation_key(safer) > evaluation_key(baseline)
    assert evaluation_key(report(three=.79, damage=0., damage_free=.9)) < evaluation_key(baseline)
    assert evaluation_key(report(success=.89, three=.95, damage=0., damage_free=.89)) < evaluation_key(baseline)


def test_final_criteria_and_consecutive_passes_cannot_be_faked_by_quick_flag():
    good = report(three=.92, damage=.05, damage_free=.9, stars=2.8)
    assert safe.performance_ready(good)
    assert not safe.performance_ready(report(three=.92, damage=.11, damage_free=.9, stars=2.8))
    assert safe.final_training_status(good, 100000, 1, 0, cfg.FINAL_STAGE2_CONFIG)[2]
    assert not safe.final_training_status(good, 99999, 1, 0, cfg.FINAL_STAGE2_CONFIG)[2]
    assert not safe.final_training_status(good, 100000, 0, 0, cfg.FINAL_STAGE2_CONFIG)[2]
    bad = report(success=.8)
    passes, regressions, ready, stop = safe.final_training_status(bad, 100000, 1, 2, cfg.FINAL_STAGE2_CONFIG)
    assert (passes, regressions, ready, stop) == (0, 3, False, True)


@pytest.mark.parametrize("fail", [False, True])
def test_final_verification_uses_new_seeds_and_restores_selection_settings(monkeypatch, tmp_path, fail):
    from stable_baselines3 import PPO
    from rl import evaluate_ppo
    settings = {**deepcopy(cfg.FINAL_STAGE2_CONFIG), "evaluation": deepcopy(cfg.EVAL_CONFIG),
                "environment": deepcopy(cfg.ENV_CONFIG)}
    model = SimpleNamespace(ppo_stage=2, ppo_final_stage2_config=settings)
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 3)
    monkeypatch.setitem(cfg.EVAL_CONFIG, "target_three_star_rate", .1)
    original = deepcopy(cfg.EVAL_CONFIG)
    original_environment = deepcopy(cfg.ENV_CONFIG)
    monkeypatch.setattr(PPO, "load", lambda *args, **kwargs: model)
    saved = []
    def evaluate(model):
        assert cfg.EVAL_CONFIG["stochastic_seeds"] == list(range(60000, 60200))
        assert cfg.EVAL_CONFIG["target_three_star_rate"] == .9
        assert cfg.ENV_CONFIG["stage"] == 2
        if fail:
            raise RuntimeError("verification failed")
        return {"passed": True}
    monkeypatch.setattr(evaluate_ppo, "evaluate_model", evaluate)
    monkeypatch.setattr(evaluate_ppo, "save_evaluation", lambda report, out: saved.append(report))
    if fail:
        with pytest.raises(RuntimeError):
            safe.verify_final_model("unused.zip", tmp_path)
    else:
        assert safe.verify_final_model("unused.zip", tmp_path)["final_stage2_verification"]
        assert len(saved) == 1
    assert cfg.EVAL_CONFIG == original
    assert cfg.ENV_CONFIG == original_environment
