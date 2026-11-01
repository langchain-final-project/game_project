"""수집 후 방향 전환, 귀환 보상 중복 방지, 조건부 성공률을 검증합니다."""

from collections import Counter
from io import BytesIO

import pytest
import torch
from stable_baselines3 import PPO

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.evaluate_ppo import (
    post_quest_metrics, format_post_quest, evaluation_cases, evaluate_model,
    make_eval_env,
)
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage1RouteCurriculum, Stage2RouteCurriculum
from rl.ppo_observation import PPOGoalObservation, PPO_OBSERVATION_VERSION
from rl.ppo_reward import PPOReward
from rl.ppo_features import WorldGoalFeaturesExtractor
from rl.train_ppo import training_targets, evaluation_key, make_vec, make_model


@pytest.fixture(autouse=True)
def stage1_test_settings(monkeypatch):
    """사용 중인 Stage 2 설정과 독립적으로 Stage 1 동작을 검증합니다."""
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 1)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "initial_frontier", 7)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "enabled", True)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)


def make_start(monkeypatch, start, stage=1):
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "initial_frontier", start)
    curriculum = Stage1RouteCurriculum if stage == 1 else Stage2RouteCurriculum
    base = RealtimeMansionEnv(stage=stage, safety_coefficient=0.0)
    return ActionRepeat(PPOReward(PPOGoalObservation(curriculum(base))), repeat=9)


def test_acquire_then_turn_and_no_reward_for_retracing(monkeypatch):
    env = make_start(monkeypatch, 7)
    try:
        _, info = env.reset(seed=11)
        assert not info["snapshot"]["quest"]["collected"]
        _, _, done, truncated, info = env.step(3)
        assert not done and not truncated
        assert info["snapshot"]["quest"]["collected"]
        assert info["goal_phase"] == 2
        _, _, _, _, info = env.step(2)
        progress = info["reward_terms"]["return_progress_reward"]
        # 물건 획득 프레임에서 기준 거리를 잡으므로 첫 귀환은 일부만 갱신합니다.
        assert 0.0 < progress <= 0.25
        for action in (3, 2):
            _, _, _, _, info = env.step(action)
            assert info["reward_terms"]["return_progress_reward"] == pytest.approx(0.0)
        _, _, _, _, info = env.step(2)
        assert info["reward_terms"]["return_progress_reward"] == pytest.approx(0.25)
    finally:
        env.close()


@pytest.mark.parametrize("start", [7, 8])
def test_success_metric_distinguishes_acquired_and_preset_quest(monkeypatch, start):
    env = make_start(monkeypatch, start)
    try:
        _, info = env.reset(seed=11)
        for _ in range(30):
            action = 2 if info["snapshot"]["quest"]["collected"] else 3
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        episode = info["rl_episode"]
        assert episode["success"]
        assert episode["return_progress_reward"] > 0
        metrics = post_quest_metrics([episode])
        assert metrics["post_quest_episode_count"] == (1 if start == 7 else 0)
        assert metrics["post_quest_success_rate"] == (1.0 if start == 7 else None)
    finally:
        env.close()


def test_no_return_progress_before_acquisition_or_in_stage2(monkeypatch):
    for stage, start, action in ((1, 7, 2), (2, 13, 0)):
        env = make_start(monkeypatch, start, stage=stage)
        try:
            env.reset(seed=11)
            _, _, _, _, info = env.step(action)
            assert info["reward_terms"]["return_progress_reward"] == 0.0
        finally:
            env.close()


def test_post_quest_rate_has_explicit_denominator_and_missing_value():
    episodes = [
        {"quest_collected": True, "success": True},
        {"quest_collected": True, "success": False},
        {"quest_collected": False, "success": False},
        {"quest_collected": True, "quest_collected_at_start": True, "success": True},
    ]
    assert post_quest_metrics(episodes) == {
        "post_quest_episode_count": 2,
        "post_quest_success_count": 1,
        "post_quest_success_rate": 0.5,
    }
    assert "-- (0/0)" in format_post_quest(post_quest_metrics([]))


def test_balanced_curriculum_samples_both_sides_of_acquisition():
    env = Stage1RouteCurriculum(RealtimeMansionEnv(stage=1))
    try:
        _, info = env.reset(seed=11)
        assert info["curriculum_start"] == 7
        counts = Counter(env._choose_start() for _ in range(10_000))
        assert counts[7] / 10_000 == pytest.approx(0.50, abs=0.03)
        assert sum(counts[s] for s in range(8, 16)) / 10_000 == pytest.approx(0.40, abs=0.03)
        assert counts[0] / 10_000 == pytest.approx(0.10, abs=0.03)
    finally:
        env.close()


def test_quick_run_has_five_checkpoints_and_small_evaluation(monkeypatch):
    monkeypatch.setitem(cfg.EVAL_CONFIG, "quick_stochastic_episodes", 10)
    monkeypatch.setitem(cfg.TRAIN_CONFIG, "quick_timesteps", 100_000)
    monkeypatch.setitem(cfg.TRAIN_CONFIG, "quick_checkpoint_interval", 20_000)
    assert training_targets(False, quick=True) == [20_000, 40_000, 60_000, 80_000, 100_000]
    env = RealtimeMansionEnv(stage=1)
    try:
        cases = evaluation_cases(env, smoke=False, quick=True)
        assert [len(group[2]) for group in cases] == [5, 10]
    finally:
        env.close()


def test_model_selection_detects_progress_when_success_rates_tie(monkeypatch):
    monkeypatch.setitem(cfg.EVAL_CONFIG, "primary_mode", "all")
    def report(quest_rate, return_rate):
        summary = {
            "success_rate": 0.0, "mean_score": 0.0, "mean_steps": 1440,
            "quest_rate": quest_rate, "post_quest_success_rate": return_rate,
        }
        return {"selection_objective": "success", "results": {mode: {"summary": summary} for mode in ("deterministic", "stochastic")}}

    assert evaluation_key(report(1.0, 0.0)) > evaluation_key(report(0.0, None))
    one_mode = report(0.0, None)
    one_mode["results"]["deterministic"] = report(1.0, 0.0)["results"]["deterministic"]
    assert evaluation_key(one_mode) > evaluation_key(report(0.0, None))


def test_quick_evaluation_reports_turn_success_without_formal_pass(monkeypatch):
    monkeypatch.setitem(cfg.EVAL_CONFIG, "quick_stochastic_episodes", 10)
    class Policy:
        training = True

        def set_training_mode(self, value):
            self.training = value

    class RouteModel:
        ppo_observation_version = PPO_OBSERVATION_VERSION
        ppo_action_repeat = 9
        device = torch.device("cpu")
        policy = Policy()

        def predict(self, obs, deterministic):
            return (3 if obs[-6] == 1.0 else 2), None

    env = make_eval_env()
    model = RouteModel()
    model.observation_space = env.observation_space
    model.action_space = env.action_space
    env.close()
    report = evaluate_model(model, quick=True)
    assert report["quick"] and not report["passed"]
    assert report["star_goal"] == {"enabled": False}
    assert model.policy.training
    for mode, count in (("deterministic", 5), ("stochastic", 10)):
        summary = report["results"][mode]["summary"]
        assert summary["episodes"] == summary["post_quest_episode_count"] == count
        assert summary["success_rate"] == summary["post_quest_success_rate"] == 1.0


def test_goal_branch_preserves_phase_information_and_both_branches_get_gradients():
    env = make_eval_env()
    try:
        extractor = WorldGoalFeaturesExtractor(env.observation_space)
        obs = torch.zeros((2, env.observation_space.shape[0]))
        obs[0, -6] = 1.0
        obs[1, -4] = 1.0
        features = extractor(obs)
        assert features.shape == (2, 64)
        assert torch.equal(features[0, :32], features[1, :32])
        assert not torch.equal(features[0, 32:], features[1, 32:])
        features.sum().backward()
        assert extractor.world[0].bias.grad.abs().sum() > 0
        assert extractor.goal[0].weight.grad.abs().sum() > 0
    finally:
        env.close()


def test_goal_branch_model_can_be_saved_and_loaded():
    torch.set_num_threads(1)
    env = make_vec(11)
    try:
        model = make_model(env, 11)
        obs = env.reset()
        expected, _ = model.predict(obs, deterministic=True)
        archive = BytesIO()
        model.save(archive)
        archive.seek(0)
        loaded = PPO.load(archive, device="cpu")
        actual, _ = loaded.predict(obs, deterministic=True)
        assert isinstance(loaded.policy.features_extractor, WorldGoalFeaturesExtractor)
        assert (actual == expected).all()
        assert loaded.ppo_features_extractor_version == "world_goal_branches_v1"
    finally:
        env.close()
