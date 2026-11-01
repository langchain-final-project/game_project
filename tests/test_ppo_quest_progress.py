"""물건 접근 보상, 시간 비용, 짧은 실험의 시작 구간을 학습 없이 검증합니다."""

from copy import deepcopy

import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage2RouteCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward, REWARD_TERM_NAMES
from rl.train_ppo import effective_curriculum_settings, make_vec


@pytest.fixture(autouse=True)
def base_collection_goal(monkeypatch):
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "enabled", False)


def make_start(start=7):
    settings = effective_curriculum_settings(quick=True)
    settings.update(initial_frontier=start, full_start_probability=0.0, review_probability=0.0)
    return ActionRepeat(PPOReward(PPOGoalObservation(Stage2RouteCurriculum(
        RealtimeMansionEnv(stage=2, safety_coefficient=0.0), settings=settings,
    ))), repeat=9)


@pytest.fixture
def env():
    wrapped = make_start()
    wrapped.reset(seed=11)
    try:
        yield wrapped
    finally:
        wrapped.close()


def test_approach_and_retreat_cancel_progress_but_accumulate_time_cost(env):
    _, _, _, _, retreat = env.step(2)
    _, _, _, _, approach = env.step(3)
    assert not approach["snapshot"]["quest"]["collected"]
    assert retreat["reward_terms"]["quest_progress_reward"] == pytest.approx(-0.25)
    assert approach["reward_terms"]["quest_progress_reward"] == pytest.approx(0.25)
    assert sum(info["reward_terms"]["quest_progress_reward"] for info in (retreat, approach)) == pytest.approx(0.0)
    assert sum(info["reward_terms"]["time_penalty"] for info in (retreat, approach)) == pytest.approx(-0.018)


def test_acquisition_frame_gets_progress_then_return_is_not_charged(env):
    _, _, _, _, info = env.step(3)
    assert info["snapshot"]["quest"]["collected"]
    assert info["reward_terms"]["quest_progress_reward"] == pytest.approx(0.25)
    assert info["reward_terms"]["quest_reward"] == 1.0
    assert info["goal_phase"] == 2
    _, _, _, _, info = env.step(2)
    assert info["reward_terms"]["quest_progress_reward"] == 0.0


@pytest.mark.parametrize("action", [0, 1, 2, 3, 4, 5])
def test_time_cost_applies_to_every_action_and_physical_frame(env, action):
    _, _, _, _, info = env.step(action)
    assert info["reward_terms"]["time_penalty"] == pytest.approx(
        cfg.REWARD_CONFIG["time_per_frame"] * info["repeat_frames"],
    )


def test_reset_restores_distance_baseline(env):
    env.step(2)
    _, info = env.reset(seed=11)
    assert info["curriculum_start"] == 7
    _, _, _, _, info = env.step(3)
    assert info["reward_terms"]["quest_progress_reward"] == pytest.approx(0.25)


def test_precollected_quest_has_no_collection_progress_reward():
    env = make_start(8)
    try:
        env.reset(seed=11)
        _, _, _, _, info = env.step(2)
        assert info["reward_terms"]["quest_progress_reward"] == 0.0
    finally:
        env.close()


def test_success_episode_records_terminal_frame_time_and_all_reward_terms(env):
    route = [3, 2] + [0] * 10 + [2] * 4
    total = 0.0
    for action in route:
        _, reward, done, truncated, info = env.step(action)
        total += reward
        if done or truncated:
            break
    episode = info["rl_episode"]
    assert episode["success"]
    assert episode["quest_progress_reward"] == pytest.approx(0.25)
    assert episode["time_penalty"] == pytest.approx(-0.001 * episode["physical_frames"])
    assert total == pytest.approx(episode["return"])
    assert total == pytest.approx(sum(episode[name] for name in REWARD_TERM_NAMES))


def test_quick_start_override_does_not_mutate_normal_training_config(monkeypatch):
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 2)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "name", "stage2_safe_route_v1")
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "enabled", True)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "initial_frontier", 15)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "quick_initial_frontier", 7)
    original = deepcopy(cfg.CURRICULUM_CONFIG)
    assert effective_curriculum_settings()["initial_frontier"] == 15
    assert effective_curriculum_settings(quick=True)["initial_frontier"] == 7
    assert cfg.CURRICULUM_CONFIG == original
    env = make_vec(11, quick=True)
    try:
        env.reset()
        for info in env.reset_infos:
            assert info["curriculum_start"] == 7
            assert not info["snapshot"]["quest"]["collected"]
    finally:
        env.close()
