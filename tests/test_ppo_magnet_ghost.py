"""자석 원격 수집과 유령 피해를 보상·관측·평가에 연결하는지 검증합니다."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from kkamang.env import RealtimeMansionEnv, encode_realtime_observation
from rl import ppo_config as cfg
from rl.evaluate_ppo import summarize
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward


def initial_snapshot():
    env = RealtimeMansionEnv(stage=2)
    try:
        return deepcopy(env.reset(seed=11)[1]["snapshot"])
    finally:
        env.close()


class ScriptedEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, shape=(640,), dtype=np.float32)
    action_space = gym.spaces.Discrete(6)

    def __init__(self, initial, current, events, status=None):
        self.initial, self.current = initial, current
        self.events, self.status = events, status

    def reset(self, *, seed=None, options=None):
        return np.zeros(640, dtype=np.float32), {"snapshot": deepcopy(self.initial)}

    def step(self, action):
        outcome = None if self.status is None else {
            "status": self.status, "reason": "returned" if self.status == "COMPLETE" else "timeout",
            "score": {"total": 70, "stars": 2},
        }
        return np.zeros(640, dtype=np.float32), 0., bool(outcome), False, {
            "snapshot": deepcopy(self.current), "events": deepcopy(self.events), "outcome": outcome,
        }


def collect(value, *, magnetic, remote):
    return {"type": "collect", "value": value, "magnetic": magnetic,
            "position": [4, 1], "source_position": [5, 1] if remote else [4, 1]}


@pytest.mark.parametrize("events,gain,ordinary,magnetic", [
    ([collect(1, magnetic=True, remote=True)], 1, 0., .9),
    ([collect(1, magnetic=True, remote=False)], 1, .6, 0.),
    ([collect(3, magnetic=False, remote=False)], 3, 1.8, 0.),
    ([collect(1, magnetic=True, remote=True), collect(3, magnetic=False, remote=False)], 4, 1.8, .9),
])
def test_remote_reward_replaces_its_base_share(events, gain, ordinary, magnetic):
    s = initial_snapshot()
    s["carried"] = 4
    current = deepcopy(s)
    current["carried"] += gain
    env = PPOReward(ScriptedEnv(s, current, events))
    try:
        env.reset()
        _, reward, _, _, info = env.step(5)
        assert info["reward_terms"]["warmth_reward"] == pytest.approx(ordinary)
        assert info["reward_terms"]["magnet_warmth_reward"] == pytest.approx(magnetic)
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
    finally:
        env.close()


def test_magnet_pickup_alone_has_no_reward_and_resets_metrics():
    s = initial_snapshot()
    env = PPOReward(ScriptedEnv(s, s, [{"type": "magnet_collect"}], status="FAILED"))
    try:
        env.reset()
        info = env.step(5)[4]
        assert info["reward_terms"]["magnet_warmth_reward"] == 0
        assert info["rl_episode"]["magnet_acquired"]
        assert not info["rl_episode"]["magnet_used"]
        env.env.events = []
        env.reset()
        assert not env.step(5)[4]["rl_episode"]["magnet_acquired"]
    finally:
        env.close()


def test_magnetic_reward_is_zero_after_warmth_cap():
    s = initial_snapshot()
    s["carried"] = cfg.REWARD_CONFIG["warmth_cap"]
    current = deepcopy(s)
    current["carried"] += 1
    env = PPOReward(ScriptedEnv(s, current, [collect(1, magnetic=True, remote=True)], status="COMPLETE"))
    try:
        env.reset()
        info = env.step(5)[4]
        assert info["reward_terms"]["warmth_reward"] == 0
        assert info["reward_terms"]["magnet_warmth_reward"] == 0
        assert info["rl_episode"]["magnet_remote_warmth"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("status,damage,expected", [("COMPLETE", 0, 1.5), ("COMPLETE", 1, 0), ("FAILED", 0, 0)])
def test_no_damage_bonus_requires_actual_success(status, damage, expected):
    s = initial_snapshot()
    current = deepcopy(s)
    current["damage_count"] = damage
    events = [{"type": "damage"}] * damage
    env = PPOReward(ScriptedEnv(s, current, events, status=status))
    try:
        env.reset()
        info = env.step(5)[4]
        assert info["reward_terms"]["damage_free_success_reward"] == expected
        assert info["reward_terms"]["damage_penalty"] == -.5 * damage
        assert info["rl_episode"]["damage_free_success"] == (status == "COMPLETE" and damage == 0)
    finally:
        env.close()


def test_protected_contact_is_not_damage_and_stage1_keeps_previous_cost():
    s = initial_snapshot()
    s["stage"] = 1
    for events, penalty in [([{"type": "contact", "protected": True}], 0), ([{"type": "damage"}], -.1)]:
        env = PPOReward(ScriptedEnv(s, s, events, status="COMPLETE"))
        try:
            env.reset()
            info = env.step(5)[4]
            assert info["reward_terms"]["damage_penalty"] == penalty
            assert info["reward_terms"]["damage_free_success_reward"] == 0
        finally:
            env.close()


def test_existing_observation_includes_magnet_and_ghost_state():
    env = PPOGoalObservation(RealtimeMansionEnv(stage=2))
    try:
        obs, info = env.reset(seed=11)
        s = info["snapshot"]
        magnet = s["magnet"]
        assert obs.shape == (640,)
        np.testing.assert_allclose(obs[610:615], [1, 0, magnet["position"][0] / 15,
                                               magnet["position"][1] / 8, 0])
        changed = deepcopy(s)
        changed["magnet"].update(collected=True, remaining_ticks=1)
        changed["ghosts"][0]["direction"] *= -1
        encoded = encode_realtime_observation(changed)
        assert encoded[611] == 1 and encoded[614] > 0
        assert encoded[481 + 51] != obs[481 + 51]
        center = s["motion"]["ghost_positions"][s["ghosts"][0]["id"]]
        np.testing.assert_allclose(obs[623:625], [center[0] / 15, center[1] / 8])
    finally:
        env.close()


def test_evaluation_metrics_include_failures_in_denominator():
    base = {"success": True, "quest_collected": True, "quest_collected_at_start": False,
            "stars": 2, "score": 70, "return": 1, "steps": 200, "decision_steps": 23,
            "damage_count": 0, "blocked_move_frames": 0, "physical_frames": 200,
            "warmth_met": True, "stalled": False,
            "magnet_acquired": True, "magnet_used": True, "magnet_remote_warmth": 3}
    failure = {**base, "success": False, "damage_count": 1, "stars": 0,
               "magnet_acquired": False, "magnet_used": False, "magnet_remote_warmth": 0}
    summary = summarize([base, failure])
    assert summary["damage_free_success_rate"] == .5
    assert summary["magnet_acquisition_rate"] == .5
    assert summary["magnet_use_rate"] == .5
    assert summary["mean_magnet_remote_warmth"] == 1.5
