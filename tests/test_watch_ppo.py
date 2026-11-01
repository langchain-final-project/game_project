"""PPO 관찰기의 행동 유지 주기와 저장된 모델 설정 호환성을 확인합니다."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from kkamang.env import RealtimeMansionEnv
from rl import watch_ppo
from rl import ppo_config as cfg
from rl.ppo_observation import (
    PPOGoalObservation, PPO_OBSERVATION_VERSION, LEGACY_PPO_OBSERVATION_VERSION,
)


class FixedModel:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, deterministic):
        self.calls += 1
        return np.array(3), None


def test_one_prediction_is_held_for_nine_rendered_physics_frames():
    env = PPOGoalObservation(RealtimeMansionEnv(stage=2))
    model = FixedModel()
    player = watch_ppo.PPOPlayer(model, env, repeat=9)
    try:
        for _ in range(9):
            player.advance()
        assert model.calls == player.decisions == 1
        assert player.info["snapshot"]["motion"]["frame"] == 9
        player.advance()
        assert model.calls == player.decisions == 2
        assert player.info["snapshot"]["motion"]["frame"] == 10
        player.reset()
        assert player.remaining_frames == player.decisions == 0
        assert player.info["snapshot"]["motion"]["frame"] == 0
        player.advance()
        assert model.calls == 3
    finally:
        env.close()


def test_restart_repeats_seeded_stochastic_actions():
    class RandomModel:
        def predict(self, obs, deterministic):
            return np.array(np.random.randint(6)), None

    env = PPOGoalObservation(RealtimeMansionEnv(stage=2))
    player = watch_ppo.PPOPlayer(RandomModel(), env, repeat=9, stochastic=True, action_seed_base=50000)
    try:
        def play():
            actions = []
            for _ in range(45):
                player.advance()
                actions.append(player.action)
            return actions

        first = play()
        player.reset()
        assert play() == first
    finally:
        env.close()


@pytest.mark.parametrize("size,version", [
    (636, LEGACY_PPO_OBSERVATION_VERSION), (640, PPO_OBSERVATION_VERSION),
])
def test_load_uses_saved_stage_observation_and_models_directory(tmp_path, monkeypatch, size, version):
    (tmp_path / "models").mkdir()
    checkpoint = tmp_path / "models" / "best_model.zip"
    checkpoint.write_bytes(b"model placeholder")
    config = {
        "algorithm": "PPO", "environment": {"stage": 2}, "observation_version": version,
        "training": {"torch_threads": 1}, "action_repeat": 9,
        "evaluation": {"action_seed_base": 50000},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    model = FixedModel()
    model.observation_space = spaces.Box(0, 1, shape=(size,), dtype=np.float32)
    model.action_space = spaces.Discrete(6)
    model.ppo_observation_version = version
    model.ppo_action_repeat = 9
    model.ppo_stage = 2
    monkeypatch.setattr(watch_ppo.PPO, "load", lambda *args, **kwargs: model)
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 1)
    player, path = watch_ppo.load_player(tmp_path)
    try:
        assert path == checkpoint
        assert player.obs.shape == (size,)
        assert player.info["snapshot"]["stage"] == 2
        assert player.repeat == 9
    finally:
        player.env.close()


def test_checkpoint_resolution_supports_root_and_explicit_models_paths(tmp_path):
    root_model = tmp_path / "last_model.zip"
    root_model.write_bytes(b"placeholder")
    assert watch_ppo.resolve_checkpoint(tmp_path, "last_model.zip") == root_model
    (tmp_path / "models").mkdir()
    model = tmp_path / "models" / "best_model.zip"
    model.write_bytes(b"placeholder")
    assert watch_ppo.resolve_checkpoint(tmp_path, "models/best_model.zip") == model
    with pytest.raises(FileNotFoundError):
        watch_ppo.resolve_checkpoint(tmp_path, "missing.zip")
