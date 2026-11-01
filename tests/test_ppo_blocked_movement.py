"""실제 Stage 2 벽 충돌에만 벌점을 주고 프레임 단위로 집계하는지 검증합니다."""

import math
from types import SimpleNamespace

import pytest
import torch

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.evaluate_ppo import evaluate_model, make_eval_env
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_observation import PPOGoalObservation, PPO_OBSERVATION_VERSION
from rl.ppo_reward import PPOReward
from rl.train_ppo import EpisodeLogger


@pytest.fixture
def env():
    wrapped = ActionRepeat(
        PPOReward(PPOGoalObservation(RealtimeMansionEnv(stage=2))), repeat=9,
    )
    wrapped.reset(seed=11)
    try:
        yield wrapped
    finally:
        wrapped.close()


def reach_lower_wall(env):
    for _ in range(14):
        _, _, done, truncated, info = env.step(1)
        assert not done and not truncated
    assert info["snapshot"]["motion"]["position"] == pytest.approx([4.0, 6.22])


def test_repeating_blocked_direction_costs_per_physical_frame(env):
    reach_lower_wall(env)
    for _ in range(2):
        _, reward, _, _, info = env.step(1)
        assert info["blocked_move_frames"] == info["repeat_frames"] == 9
        assert all(point["blocked_move"] for point in info["repeat_trace"])
        assert info["reward_terms"]["blocked_move_penalty"] == pytest.approx(-0.09)
        assert reward == pytest.approx(sum(info["reward_terms"].values()))


def test_turning_to_open_direction_stops_the_penalty(env):
    reach_lower_wall(env)
    before = list(env.unwrapped.core.snapshot()["motion"]["position"])
    _, _, _, _, info = env.step(3)
    assert info["snapshot"]["motion"]["position"] != before
    assert info["blocked_move_frames"] == 0
    assert info["reward_terms"]["blocked_move_penalty"] == 0.0


@pytest.mark.parametrize("action", [4, 5])
def test_interaction_and_wait_are_not_mistaken_for_blocked_movement(env, action):
    reach_lower_wall(env)
    _, _, _, _, info = env.step(action)
    assert not info["snapshot"]["motion"]["moving"]
    assert info["blocked_move_frames"] == 0
    assert info["reward_terms"]["blocked_move_penalty"] == 0.0


def test_partly_moving_action_only_penalizes_its_blocked_frames(env):
    _, _, _, _, info = env.step(2)
    count = info["blocked_move_frames"]
    assert 0 < count < 9
    assert info["reward_terms"]["blocked_move_penalty"] == pytest.approx(-0.01 * count)


def test_episode_totals_and_reset_are_independent(env):
    for _ in range(160):
        _, _, done, truncated, info = env.step(1)
        if done or truncated:
            break
    episode = info["rl_episode"]
    assert episode["reason"] == "timeout"
    assert episode["blocked_move_frames"] > 1200
    assert episode["blocked_move_penalty"] == pytest.approx(
        -0.01 * episode["blocked_move_frames"],
    )
    env.reset(seed=11)
    _, _, _, _, info = env.step(1)
    assert info["blocked_move_frames"] == 0
    assert info["reward_terms"]["blocked_move_penalty"] == 0.0


def test_corner_assistance_is_not_penalized_as_a_wall_collision(env):
    core = env.unwrapped.core
    grid = [["#" if x in (0, 15) or y in (0, 8) else "."
             for x in range(16)] for y in range(9)]
    grid[4][7] = "#"
    core._state["grid"] = ["".join(row) for row in grid]
    core._state["shortcuts"] = []
    core._state["ghosts"] = []
    core._ghost_origins = {}
    core._state["chaser"] = None
    core._chaser_origin = None
    core._motion_position = [6.6, 3.22]
    core._state["position"] = [math.floor(v + 0.5) for v in core._motion_position]
    _, _, _, _, info = env.step(1)
    assert info["snapshot"]["motion"]["position"][0] < 6.6
    assert info["blocked_move_frames"] == 0
    assert info["reward_terms"]["blocked_move_penalty"] == 0.0


def test_evaluation_reports_blocked_frames_without_training(monkeypatch):
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 2)

    class Policy:
        training = True

        def set_training_mode(self, value):
            self.training = value

    class DownModel:
        ppo_observation_version = PPO_OBSERVATION_VERSION
        ppo_action_repeat = 9
        device = torch.device("cpu")
        policy = Policy()

        def predict(self, obs, deterministic):
            return 1, None

    evaluation_env = make_eval_env()
    model = DownModel()
    model.observation_space = evaluation_env.observation_space
    model.action_space = evaluation_env.action_space
    evaluation_env.close()
    report = evaluate_model(model, smoke=True)
    for result in report["results"].values():
        summary = result["summary"]
        assert summary["success_rate"] == 0.0
        assert summary["mean_blocked_move_frames"] > 1200
        assert summary["blocked_move_frame_rate"] > 0.9


def test_parallel_training_logger_counts_current_frames_without_double_counting():
    recorded = {}
    callback = EpisodeLogger(total_steps=100_000)
    callback.model = SimpleNamespace(
        logger=SimpleNamespace(record=lambda key, value: recorded.update({key: value})),
    )
    infos = [{"repeat_frames": 9, "blocked_move_frames": count} for count in (2, 3, 4, 5)]
    infos[0]["rl_episode"] = {
        "success": False,
        "quest_collected": False,
        "quest_collected_at_start": False,
        "physical_frames": 1440,
        "blocked_move_frames": 1000,
    }
    callback.locals = {"infos": infos}
    assert callback._on_step()
    assert callback.physical_frames == 36
    assert callback.blocked_move_frames == 14
    assert recorded["task/blocked_move_frame_rate_100"] == pytest.approx(1000 / 1440)
