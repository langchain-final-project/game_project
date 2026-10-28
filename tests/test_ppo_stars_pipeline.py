"""별 수집의 목표 안내, 실제 귀환 규칙, 가중치/옵티마이저 전이를 검증한다."""
from copy import deepcopy
import json

import pytest
import torch
from stable_baselines3 import PPO

from rl import ppo_common
from rl import ppo_stars_common as stars


@pytest.fixture(scope="module")
def source_model(tmp_path_factory):
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp("ppo_source")
    config = ppo_common.build_config()
    config["curriculum"] = None
    env = ppo_common.make_env(config)
    try:
        model = PPO("MlpPolicy", env, n_steps=100, batch_size=100, n_epochs=1,
                    learning_rate=0.0002, seed=11, device="cpu")
        model.learn(400)
        model.num_timesteps = 600000
        model.save(path / "final_model.zip")
        ppo_common.write_json(path / "config.json", config)
        ppo_common.write_json(path / "metadata.json", {"training_seed": 11})
        return path / "final_model.zip"
    finally:
        env.close()


def test_finetune_restores_exact_weights_and_adam_state(source_model):
    config = stars.build_config(source_model)
    assert config["curriculum"] is None
    assert config["environment"]["warmth_goal"] == 7
    assert config["ppo"]["total_timesteps"] == 200000
    old = PPO.load(source_model, device="cpu")
    env = stars.make_env(config, training=True)
    try:
        model, provenance = stars.load_initial_model(config, env, 11)
        assert provenance["source_model_timesteps"] == 600000
        assert provenance["source_cumulative_timesteps"] == 600000
        assert model.num_timesteps == 0
        assert model.lr_schedule(1.0) == 0.00005
        assert all(group["lr"] == 0.00005 for group in model.policy.optimizer.param_groups)
        assert all(torch.equal(model.policy.state_dict()[key], value)
                   for key, value in old.policy.state_dict().items())
        old_state = old.policy.optimizer.state_dict()["state"]
        new_state = model.policy.optimizer.state_dict()["state"]
        assert old_state and old_state.keys() == new_state.keys()
        for param, values in old_state.items():
            for key, value in values.items():
                assert torch.equal(value, new_state[param][key])
        model.learn(400, reset_num_timesteps=False)
        assert model.num_timesteps == 400
        assert provenance["source_cumulative_timesteps"] + model.num_timesteps == 600400
    finally:
        env.close()


def test_goal_changes_at_seven_without_forbidding_four_point_return(source_model):
    config = stars.build_config(source_model)
    env = stars.make_env(config, training=True)
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (636,) and env.observation_space.contains(obs)
        assert info["snapshot"]["carried"] == 0
        assert not info["snapshot"]["quest"]["collected"]
        assert "curriculum_start" not in info
        for action in [3] * 126 + [4]:
            obs, _, _, _, info = env.step(action)
        assert info["snapshot"]["target"] == 4
        assert info["snapshot"]["carried"] == 4
        assert obs[-6:-3].tolist() == [0.0, 1.0, 0.0]
        for action in [2] * 126:
            obs, _, _, _, info = env.step(action)
        assert obs[-1] == 1.0  # 실제 게임상 4점으로도 귀환 가능
        _, _, done, _, info = env.step(4)
        assert done and info["rl_episode"]["success"]
        assert info["rl_episode"]["score"] == 40

        env.reset(seed=11)
        for action in [3] * 126 + [4] + [1] * 36 + [2] * 18:
            obs, _, _, _, info = env.step(action)
        assert info["snapshot"]["carried"] == 7
        assert obs[-6:-3].tolist() == [0.0, 0.0, 1.0]
        for action in [0] * 36 + [2] * 108 + [4]:
            obs, _, done, _, info = env.step(action)
        assert done and info["rl_episode"]["stars"] == 1
        assert info["rl_episode"]["score"] == 58
    finally:
        env.close()


@pytest.mark.parametrize("change", ["curriculum", "goal", "version"])
def test_reject_invalid_star_configuration(source_model, change):
    config = stars.build_config(source_model)
    if change == "curriculum":
        config["curriculum"] = "stage1_route_v3"
    elif change == "goal":
        config["environment"]["warmth_goal"] = 15
    else:
        config["environment"]["observation_version"] = "incorrect"
    with pytest.raises(ValueError):
        stars.validate_config(config)


def test_missing_source_does_not_fall_back_to_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        stars.build_config(tmp_path / "missing.zip")


def test_reward_comes_from_source_run(source_model):
    source = json.loads(source_model.with_name("config.json").read_text(encoding="utf-8"))
    config = stars.build_config(source_model)
    assert config["reward"] == source["reward"]
    assert config["reward"] is not source["reward"]
