"""학습 없이 목표 순서·시작 상태 혼합·유령 대기·부분 진단을 검증합니다."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
import torch

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.probe_stage2 import probe_model
from rl.train_ppo import make_env, make_model


@pytest.fixture(autouse=True)
def stage2_profile(monkeypatch):
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 2)
    for key, value in {
        "name": "stage2_quest_first_mixed_v2", "initial_frontier": 9,
        "quick_initial_frontier": 9, "full_start_probability": .1,
        "magnet_focus_probability": 0,
        "review_probability": .1, "quest_focus_probability": .4,
        "quest_focus_starts": [6, 7, 8], "ghost_focus_probability": .2,
        "ghost_focus_starts": [11, 12, 13, 14], "ghost_wait_frames": [0, 9, 18],
    }.items():
        monkeypatch.setitem(cfg.CURRICULUM_CONFIG, key, value)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "stage2_three_star", True)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "quest_after_magnet", True)


def prepared(start, *, goals=None):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings["initial_frontier"] = start
    return PPOGoalObservation(Stage2ThreeStarCurriculum(
        RealtimeMansionEnv(stage=2), settings=settings), star_goal=goals)


@pytest.mark.parametrize("start,phase", [(3, 1), (6, 0), (7, 0), (8, 0), (9, 1), (16, 2)])
def test_magnet_then_quest_then_extra_collection_then_return(start, phase):
    env = prepared(start)
    try:
        obs, info = env.reset(seed=11)
        assert env.observation_space.contains(obs)
        assert info["goal_phase"] == phase
        if start in (6, 7, 8):
            assert info["snapshot"]["magnet"]["collected"]
            assert not info["snapshot"]["quest"]["collected"]
    finally:
        env.close()


def test_old_model_keeps_its_original_collect_first_order():
    old = deepcopy(cfg.STAR_GOAL_CONFIG)
    old.pop("quest_after_magnet")
    env = prepared(6, goals=old)
    try:
        assert env.reset(seed=11)[1]["goal_phase"] == 1
    finally:
        env.close()


def test_pre_quest_progress_rewards_correct_downward_movement():
    env = PPOReward(prepared(7))
    try:
        env.reset(seed=11)
        info = env.step(1)[4]
        assert info["reward_terms"]["quest_progress_reward"] > 0
        assert info["reward_terms"]["warmth_progress_reward"] == 0
        assert info["reward_terms"]["return_progress_reward"] == 0
    finally:
        env.close()


def test_sampler_reserves_pre_quest_and_ghost_experience():
    env = Stage2ThreeStarCurriculum(RealtimeMansionEnv(stage=2))
    try:
        env.reset(seed=123)
        counts = Counter()
        for _ in range(20000):
            start = env._choose_start()
            counts[env._phase] += 1
            if env._phase == "quest_focus":
                assert start in (6, 7, 8)
            if env._phase == "ghost_focus":
                assert start in (11, 12, 13, 14)
        for phase, expected in {"full_start": .1, "quest_focus": .4,
                                "ghost_focus": .2, "review": .1, "frontier": .2}.items():
            assert counts[phase] / 20000 == pytest.approx(expected, abs=.015)
    finally:
        env.close()


def test_ghost_preparation_uses_real_time_and_does_not_corrupt_cached_state():
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(full_start_probability=0, review_probability=0,
                    quest_focus_probability=0, ghost_focus_probability=1,
                    ghost_focus_starts=[14])
    env = Stage2ThreeStarCurriculum(RealtimeMansionEnv(stage=2), settings=settings)
    try:
        env.reset(seed=11)
        waits = set()
        for _ in range(30):
            _, info = env.reset()
            s = info["snapshot"]
            wait = info["curriculum_ghost_wait_frames"]
            waits.add(wait)
            assert info["curriculum_phase"] == "ghost_focus"
            assert s["motion"]["frame"] == env.prefix_frames[14] + wait
            assert s["damage_count"] == 0
            assert env.env._steps == 0
            assert env._prepared[14].snapshot()["motion"]["frame"] == env.prefix_frames[14]
        assert waits == {0, 9, 18}
    finally:
        env.close()


@pytest.mark.parametrize("start", range(34))
def test_prepared_route_still_finishes_with_three_stars_and_no_damage(start):
    env = prepared(start)
    try:
        env.reset(seed=11)
        route = Stage2ThreeStarCurriculum.route
        for action in route[Stage2ThreeStarCurriculum.prefix_frames[start]:]:
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done and not truncated
        assert info["outcome"]["score"]["total"] == 100
        assert info["snapshot"]["damage_count"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("change", [
    {"quest_focus_probability": .9}, {"ghost_focus_starts": []},
    {"quest_focus_starts": [99]}, {"ghost_wait_frames": [-1]},
])
def test_invalid_mixtures_are_rejected(change):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(change)
    raw = RealtimeMansionEnv(stage=2)
    try:
        with pytest.raises(ValueError):
            Stage2ThreeStarCurriculum(raw, settings=settings)
    finally:
        raw.close()


def test_probe_is_partial_and_does_not_learn_or_change_rng():
    torch.set_num_threads(1)
    env = make_env()
    try:
        model = make_model(env, 11)
        weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = probe_model(model, seeds=(11,), frames=180)
        assert not report["full_game_evaluation"]
        assert report["results"]["quest"]["summary"]["episodes"] == 3
        assert report["results"]["ghost"]["summary"]["episodes"] == 4
        assert all(e["first_goal_phase"] == 0 for e in report["results"]["quest"]["episodes"])
        for result in report["results"].values():
            assert all(e["frames"] <= 180 for e in result["episodes"])
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(v, weights[k]) for k, v in model.policy.state_dict().items())
    finally:
        env.close()
