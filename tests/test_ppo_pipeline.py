"""PPO와 A2C의 보상 비교, 시작 상태 구분, 평가 시 난수/가중치 보존."""
from copy import deepcopy
import random

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from rl.ppo_common import build_config, make_env, training_targets, validate_config
from rl.ppo_reward import PPOReward, REWARD_CONFIG
from rl.reward import ProposedReward
from rl.evaluate_ppo import rollout_episode


@pytest.mark.parametrize("change", [
    {"batch_size": 67}, {"n_epochs": 0}, {"total_timesteps": 600001},
])
def test_reject_invalid_ppo_rollouts(change):
    config = build_config()
    config["ppo"].update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_curriculum_is_training_only_and_targets_are_exact():
    config = build_config()
    assert training_targets(config) == [200000, 400000, 600000]
    assert training_targets(config, smoke=True) == [2000]
    train_env, eval_env = make_env(config, training=True), make_env(config)
    try:
        obs, info = train_env.reset(seed=11)
        assert train_env.observation_space.contains(obs)
        assert obs.shape == (636,)
        assert info["snapshot"]["quest"]["collected"]
        assert info["snapshot"]["carried"] == 4
        assert info["curriculum_start"] == 15
        _, reward, terminated, truncated, info = train_env.step(4)
        assert terminated and not truncated and info["rl_episode"]["success"]
        assert reward == 12.0
        obs, info = eval_env.reset(seed=11)
        assert eval_env.observation_space.contains(obs)
        assert not info["snapshot"]["quest"]["collected"]
        assert info["snapshot"]["carried"] == 0
        assert "curriculum_start" not in info
    finally:
        train_env.close()
        eval_env.close()


def test_ppo_reward_matches_a2c_for_full_warmth_and_return():
    # 14점 전부 수집하는 유효한 Stage 1 경로. 중간 온기와 별 보상도 비교한다.
    route = ([3] * 126 + [4] + [1] * 36 + [2] * 18 + [1] * 36
             + [2] * 18 + [1] * 18 + [2] * 54 + [0] * 54
             + [2] * 18 + [0] * 36 + [2] * 18 + [4])
    old = ProposedReward(RealtimeMansionEnv(stage=1, safety_coefficient=0.0))
    new = PPOReward(RealtimeMansionEnv(stage=1, safety_coefficient=0.0))
    try:
        old.reset(seed=11)
        new.reset(seed=11)
        for action in route:
            _, old_reward, old_done, old_truncated, old_info = old.step(action)
            _, new_reward, new_done, new_truncated, new_info = new.step(action)
            assert old_reward == new_reward
            assert (old_done, old_truncated) == (new_done, new_truncated)
            assert old_info["reward_terms"] == new_info["reward_terms"]
        assert new_done and new_info["rl_episode"]["success"]
        assert new_info["rl_episode"]["warmth"] == 14
        assert new_info["rl_episode"]["warmth_reward"] == pytest.approx(4.4)
        assert old_info["rl_episode"] == new_info["rl_episode"]
    finally:
        old.close()
        new.close()


def test_saved_reward_config_is_independent_of_current_defaults(monkeypatch):
    saved_config = build_config()
    monkeypatch.setitem(REWARD_CONFIG, "success", 999.0)
    env = make_env(saved_config, training=True)
    try:
        env.reset(seed=11)
        _, reward, done, _, _ = env.step(4)
        assert done and reward == 12.0
    finally:
        env.close()


def test_evaluation_preserves_rng_policy_mode_and_weights():
    torch.set_num_threads(1)
    config = build_config()
    config["environment"]["max_episode_steps"] = 8
    env = make_env(config)
    model = PPO("MlpPolicy", env, n_steps=100, batch_size=100, n_epochs=1,
                seed=11, device="cpu")
    model.policy.set_training_mode(True)
    weights = deepcopy(model.policy.state_dict())
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        first = rollout_episode(model, env, seed=10000, deterministic=False, config=config)
        second = rollout_episode(model, env, seed=10000, deterministic=False, config=config)
        assert first["actions"] == second["actions"]
        assert first["truncated"] and len(first["actions"]) == 8
        assert model.policy.training
        assert random.getstate() == python_state
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
        assert np.random.get_state()[2:] == numpy_state[2:]
        assert torch.equal(torch.get_rng_state(), torch_state)
        assert all(torch.equal(model.policy.state_dict()[key], value)
                   for key, value in weights.items())
    finally:
        env.close()
