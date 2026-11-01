"""벽 정체 재현에서 발견한 무효 상호작용과 작은 왕복을 검증합니다."""

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_observation import PPOGoalObservation, MOVEMENT_MASK_SLICE
from rl.ppo_features import WorldGoalFeaturesExtractor
from rl.ppo_policy import MovementAwareActorCriticPolicy
from rl.ppo_reward import PPOReward
from rl.ppo_stagnation import StagnationTracker


def test_all_actions_count_as_stationary_after_grace_and_reset_on_collection():
    tracker = StagnationTracker([1, 1])
    grace = cfg.STAGNATION_CONFIG["stationary_grace_frames"]
    for _ in range(grace - 1):
        assert tracker.update([1, 1]) == (False, False)
    assert tracker.update([1, 1]) == (True, False)
    assert tracker.stationary_frames == grace
    assert tracker.update([1, 1], made_progress=True) == (False, False)
    assert tracker.stationary_streak == 0


def test_small_oscillations_are_detected_even_without_continuous_stillness():
    tracker = StagnationTracker([1, 1])
    window = cfg.STAGNATION_CONFIG["local_window_frames"]
    for i in range(window):
        stagnant, local_loop = tracker.update([1 + (i % 2) * 0.5, 1])
    assert stagnant and local_loop
    assert tracker.max_stationary_frames < 2
    assert tracker.update([2.5, 1]) == (False, False)


def test_continuous_travel_is_not_stagnation():
    tracker = StagnationTracker([1, 1])
    for i in range(1, 250):
        assert tracker.update([1 + i / 18, 1]) == (False, False)
    assert tracker.stagnant_frames == 0


@pytest.mark.parametrize("action", [4, 5])
def test_wait_and_interaction_penalty_is_logged_without_becoming_wall_collision(action):
    env = ActionRepeat(PPOReward(PPOGoalObservation(RealtimeMansionEnv(stage=2))), repeat=9)
    try:
        env.reset(seed=11)
        for _ in range(5):
            _, _, _, _, info = env.step(action)
        assert info["reward_terms"]["stagnation_penalty"] == pytest.approx(-0.09)
        assert info["blocked_move_frames"] == 0
        assert all(p["stagnant"] for p in info["repeat_trace"])
        while "rl_episode" not in info:
            _, _, _, _, info = env.step(action)
        ep = info["rl_episode"]
        assert ep["stationary_frames"] == ep["physical_frames"]
        assert ep["stagnation_penalty"] == pytest.approx(
            cfg.REWARD_CONFIG["stagnation_per_frame"] * ep["stagnant_frames"])
        env.reset(seed=11)
        _, _, _, _, info = env.step(action)
        assert info["reward_terms"]["stagnation_penalty"] == 0
    finally:
        env.close()


def test_new_policy_masks_interaction_and_preserves_wait_when_all_directions_blocked(tmp_path):
    env = PPOGoalObservation(RealtimeMansionEnv(stage=2), **cfg.ACTION_MASK_CONFIG)
    try:
        obs, info = env.reset(seed=11)
        assert info["action_mask"][4:] == [False, True]
        model = PPO(MovementAwareActorCriticPolicy, env, n_steps=2, batch_size=2,
                    policy_kwargs={**cfg.ACTION_MASK_CONFIG,
                                   "features_extractor_class": WorldGoalFeaturesExtractor})
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.zero_()
            model.policy.action_net.bias[4] = 100
        obs[MOVEMENT_MASK_SLICE] = 0
        tensor = model.policy.obs_to_tensor(obs)[0]
        dist = model.policy.get_distribution(tensor)
        np.testing.assert_array_equal(dist.distribution.probs.detach().numpy(), [[0, 0, 0, 0, 0, 1]])
        _, _, forward_logprob = model.policy(tensor)
        _, checked_logprob, _ = model.policy.evaluate_actions(tensor, torch.tensor([5]))
        torch.testing.assert_close(forward_logprob, checked_logprob)
        model.save(tmp_path / "policy.zip")
        loaded = PPO.load(tmp_path / "policy.zip", device="cpu")
        assert loaded.policy.mask_ineffective_interaction
        assert int(loaded.predict(obs)[0]) == 5
    finally:
        env.close()


def test_legacy_policy_keeps_original_interaction_behavior():
    env = PPOGoalObservation(RealtimeMansionEnv(stage=2))
    try:
        obs, _ = env.reset(seed=11)
        policy = MovementAwareActorCriticPolicy(env.observation_space, env.action_space, lambda _: 0.0001)
        assert not policy.mask_ineffective_interaction
        probabilities = policy.get_distribution(policy.obs_to_tensor(obs)[0]).distribution.probs
        assert probabilities[0, 4].item() > 0
    finally:
        env.close()


def test_regression_seed_is_included_in_quick_evaluation():
    assert 11 in cfg.EVAL_CONFIG["stochastic_seeds"][:cfg.EVAL_CONFIG["quick_stochastic_episodes"]]
