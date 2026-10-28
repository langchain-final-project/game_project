"""탐색 설정 검사. 이 파일의 테스트는 PPO.learn를 실행하지 않는다."""
from copy import deepcopy
import json
import sys

import pytest
import torch
from stable_baselines3 import PPO

from rl import ppo_common
from rl import ppo_stars_common as stars
from rl import train_ppo_stars as runner


@pytest.fixture(autouse=True)
def prohibit_training(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("이 설정 검사에서는 학습을 실행하면 안 됩니다.")
    monkeypatch.setattr(PPO, "learn", reject)


@pytest.fixture(scope="module")
def source_model(tmp_path_factory):
    torch.set_num_threads(1)
    directory = tmp_path_factory.mktemp("untrained_ppo_source")
    config = ppo_common.build_config()
    config["curriculum"] = None
    env = ppo_common.make_env(config)
    try:
        # 객체 초기화와 저장만 하며 학습은 하지 않는다.
        model = PPO("MlpPolicy", env, n_steps=100, batch_size=100, seed=11, device="cpu")
        model.num_timesteps = 600000
        model.save(directory / "final_model.zip")
        ppo_common.write_json(directory / "config.json", config)
        ppo_common.write_json(directory / "metadata.json", {"training_seed": 11})
        return directory / "final_model.zip"
    finally:
        env.close()


@pytest.mark.parametrize("preset,coefficient", [
    ("baseline", 0.01), ("entropy03", 0.03), ("entropy05", 0.05),
])
def test_named_conditions_have_separate_outputs(source_model, preset, coefficient):
    config = stars.build_config(source_model, exploration=preset)
    assert config["ppo"]["ent_coef"] == coefficient
    assert preset in config["experiment"]
    assert config["curriculum"] is None
    assert config["environment"]["warmth_goal"] == 7
    assert stars.training_targets(config) == [50000, 100000, 150000, 200000]


def test_comparison_changes_only_entropy_and_labels(source_model):
    defaults = deepcopy(stars.CONFIG)
    configs = [stars.build_config(source_model, exploration=preset)
               for preset in stars.EXPLORATION_PRESETS]
    for config in configs:
        config.pop("experiment")
        config.pop("exploration")
        config["ppo"].pop("ent_coef")
    assert configs[0] == configs[1] == configs[2]
    assert stars.CONFIG == defaults


def test_loading_applies_entropy_without_changing_source_weights(source_model):
    reference = PPO.load(source_model, device="cpu")
    for preset, coefficient in stars.EXPLORATION_PRESETS.items():
        config = stars.build_config(source_model, exploration=preset)
        env = stars.make_env(config, training=True)
        try:
            model, provenance = stars.load_initial_model(config, env, 11)
            assert model.ent_coef == coefficient
            assert model.num_timesteps == 0
            assert provenance["source_model_timesteps"] == 600000
            assert all(torch.equal(model.policy.state_dict()[key], value)
                       for key, value in reference.policy.state_dict().items())
        finally:
            env.close()


def test_show_config_does_not_call_training(source_model, monkeypatch, capsys):
    def reject(*args, **kwargs):
        pytest.fail("--show-config가 학습 함수를 호출했습니다.")
    monkeypatch.setattr(runner, "train", reject)
    monkeypatch.setattr(sys, "argv", [
        "train_ppo_stars", "--init-model", str(source_model),
        "--exploration", "entropy05", "--show-config",
    ])
    runner.main()
    config = json.loads(capsys.readouterr().out)
    assert config["ppo"]["ent_coef"] == 0.05
    assert config["exploration"] == "entropy05"


def test_reject_unknown_or_mismatched_condition(source_model):
    with pytest.raises(ValueError):
        stars.build_config(source_model, exploration="typo")
    config = stars.build_config(source_model, exploration="entropy03")
    config["ppo"]["ent_coef"] = 0.01
    with pytest.raises(ValueError):
        stars.validate_config(config)


def test_legacy_v1_saved_config_remains_usable(source_model):
    config = stars.build_config(source_model, exploration="baseline")
    config.pop("exploration")
    config["experiment"] = "RTenv_s01_ppo_star1_goal7_ft200k_v1"
    stars.validate_config(config)
    assert config["ppo"]["ent_coef"] == 0.01
