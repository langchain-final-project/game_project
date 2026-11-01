"""실제 물리 경로·별 3개 진급·수집/귀환 보상 충돌을 학습 없이 검증합니다."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_curriculum import Stage1ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.evaluate_ppo import make_eval_env, evaluation_cases, evaluation_passed, evaluate_model
from rl.train_ppo import make_env, make_model, training_curriculum_class, evaluation_key


@pytest.fixture(autouse=True)
def isolated_stage1_profile(monkeypatch):
    """검증은 로컬 최종 모델 파일이나 현재 미세 조정 시작 번호에 의존하지 않습니다."""
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 1)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "name", "stage1_three_star_v1")
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "initial_frontier", 29)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "quick_initial_frontier", 29)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "stage1_three_star", True)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "stage2_three_star", False)


def prepared(start, **overrides):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(initial_frontier=start, full_start_probability=0, review_probability=0)
    settings.update(overrides)
    return Stage1ThreeStarCurriculum(RealtimeMansionEnv(stage=1), settings=settings)


def learning_env(start):
    return ActionRepeat(PPOReward(PPOGoalObservation(
        prepared(start), **cfg.ACTION_MASK_CONFIG)), repeat=cfg.ACTION_REPEAT)


@pytest.mark.parametrize("start", range(30))
def test_every_prepared_state_can_finish_with_three_stars(start):
    env = prepared(start)
    try:
        _, info = env.reset(seed=11)
        assert info["snapshot"]["motion"]["frame"] == env.prefix_frames[start]
        assert info["snapshot"]["damage_count"] == 0
        for action in env.route[env.prefix_frames[start]:]:
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done and not truncated and info["curriculum_task_success"]
        assert info["outcome"]["score"]["total"] == 100
        assert info["outcome"]["score"]["stars"] == 3
        assert info["snapshot"]["carried"] == 14
        assert info["snapshot"]["motion"]["frame"] == 532
    finally:
        env.close()


def test_two_star_return_does_not_advance_curriculum():
    env = prepared(29, history_size=2, required_successes=2)
    try:
        for episode in range(3):
            env.reset(seed=11 if episode == 0 else None)
            if episode == 0:
                for _ in range(900):
                    _, _, done, truncated, _ = env.step(5)
                    assert not done and not truncated
            for _ in range(18):
                _, _, done, _, info = env.step(2)
                if done:
                    break
            assert done
            assert info["outcome"]["score"]["stars"] == (2 if episode == 0 else 3)
            assert info["curriculum_task_success"] == (episode != 0)
            assert env.frontier == (28 if episode == 2 else 29)
    finally:
        env.close()


def test_surplus_reward_is_capped_and_preparation_does_not_earn_collection_reward():
    env = learning_env(29)
    try:
        _, info = env.reset(seed=11)
        reward_env = env.env
        snapshot = deepcopy(info["snapshot"])
        assert reward_env._warmth_value(snapshot) == pytest.approx(11.4)
        snapshot["carried"] = 15
        assert reward_env._warmth_value(snapshot) == pytest.approx(11.4)
        snapshot["carried"] = 4
        assert reward_env._warmth_value(snapshot) == pytest.approx(.4)
        snapshot["carried"] = 5
        assert reward_env._warmth_value(snapshot) == pytest.approx(1.5)
        _, _, _, _, info = env.step(5)
        assert info["reward_terms"]["warmth_reward"] == 0
        assert info["reward_terms"]["quest_reward"] == 0
        assert info["reward_terms"]["time_penalty"] == pytest.approx(-.009)
    finally:
        env.close()


def test_extra_collection_is_not_penalized_for_moving_away_from_quest():
    env = learning_env(18)  # (10,1), 온기 13점. 마지막 온기는 왼쪽 (8,1).
    try:
        _, info = env.reset(seed=11)
        assert info["goal_phase"] == 1
        assert info["goal_position"] == [9, 1]
        _, _, _, _, info = env.step(2)
        assert info["reward_terms"]["warmth_progress_reward"] > 0
        assert info["reward_terms"]["quest_progress_reward"] == 0
        assert info["reward_terms"]["return_progress_reward"] == 0
    finally:
        env.close()


def test_basic_warmth_and_quest_do_not_trigger_early_return_guidance():
    env = PPOReward(PPOGoalObservation(RealtimeMansionEnv(stage=1)))
    try:
        env.reset(seed=11)
        for _ in range(126):
            _, _, done, truncated, info = env.step(3)
            assert not done and not truncated
        assert info["snapshot"]["carried"] == 4
        assert info["snapshot"]["quest"]["collected"]
        assert info["goal_phase"] == 1
        assert env._return_distance(info["snapshot"]) is None
    finally:
        env.close()


def test_return_backtracking_has_no_net_progress_reward():
    env = learning_env(25)
    try:
        _, info = env.reset(seed=11)
        assert info["goal_phase"] == 2
        progress = [env.step(a)[4]["reward_terms"]["return_progress_reward"] for a in (2, 3, 2, 3)]
        assert progress == pytest.approx([.25, -.25, .25, -.25])
        assert sum(progress) == pytest.approx(0)
    finally:
        env.close()


def test_three_star_finish_receives_bonus_and_terminal_repeat_stops():
    env = learning_env(29)
    try:
        env.reset(seed=11)
        env.step(2)
        _, reward, done, truncated, info = env.step(2)
        assert done and not truncated
        assert info["physical_frames"] == 10
        assert info["repeat_frames"] == 1
        assert info["reward_terms"]["star_completion_adjustment"] == 10
        assert info["reward_terms"]["success_reward"] == 38
        assert info["rl_episode"]["curriculum_success"]
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
    finally:
        env.close()


def test_two_star_completion_keeps_lower_bonus_after_safety_fallback():
    env = PPOReward(PPOGoalObservation(prepared(29)))
    try:
        env.reset(seed=11)
        for _ in range(900):
            env.step(5)
        for _ in range(18):
            _, _, done, _, info = env.step(2)
            if done:
                break
        assert done and info["rl_episode"]["success"]
        assert info["rl_episode"]["stars"] == 2
        assert not info["rl_episode"]["curriculum_success"]
        assert info["reward_terms"]["star_completion_adjustment"] == -3
    finally:
        env.close()


def test_eval_starts_at_original_position_and_uses_same_100_stochastic_seeds():
    env = make_eval_env(star_goal=cfg.STAR_GOAL_CONFIG, mask_ineffective_interaction=True)
    try:
        obs, info = env.reset(seed=11)
        assert env.observation_space.contains(obs) and obs.shape == (640,)
        assert info["snapshot"]["motion"]["frame"] == 0
        assert info["snapshot"]["carried"] == 0
        assert not info["snapshot"]["quest"]["collected"]
        assert "curriculum_start" not in info
        full = evaluation_cases(env, False)
        quick = evaluation_cases(env, False, quick=True)
        assert full == quick
        assert len(full[0][2]) == 20 and len(full[1][2]) == 100
    finally:
        env.close()


def test_fresh_ppo_has_no_source_or_optimizer_updates_and_masks_interact():
    torch.set_num_threads(1)
    env = make_env()
    try:
        model = make_model(env, 11)
        obs, info = env.reset(seed=11)
        assert training_curriculum_class() is Stage1ThreeStarCurriculum
        assert model.transfer_source is None
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert model.ppo_stage == 1 and model.ppo_action_repeat == 9
        assert info["curriculum_start"] == 29
        tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            probabilities = model.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0]
        assert probabilities[4] == 0
        assert np.isclose(probabilities.sum(), 1)
    finally:
        env.close()


def test_evaluation_does_not_update_weights_or_training_curriculum(monkeypatch):
    torch.set_num_threads(1)
    env = make_env()
    try:
        model = make_model(env, 11)
        env.reset(seed=11)
        curriculum = env.env.env.env
        frontier = curriculum.frontier
        weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
        random_state = torch.get_rng_state().clone()
        monkeypatch.setitem(cfg.ENV_CONFIG, "max_episode_steps", 9)
        report = evaluate_model(model, smoke=True)
        assert report["environment"]["stage"] == 1
        assert report["results"]["stochastic"]["summary"]["episodes"] == 2
        assert curriculum.frontier == frontier
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert torch.equal(torch.get_rng_state(), random_state)
        assert all(torch.equal(v, weights[k]) for k, v in model.policy.state_dict().items())
    finally:
        env.close()


def test_selection_and_pass_conditions_require_three_stars_not_just_completion():
    def report(success, three, mean):
        return {"primary_mode": "stochastic", "selection_objective": "three_stars",
                "success_rate_threshold": .9, "target_three_star_rate": .9,
                "target_mean_stars": 2.7, "quick": False, "smoke": False,
                "results": {"stochastic": {"summary": {"success_rate": success,
                    "three_star_rate": three, "mean_stars": mean, "mean_steps": 500}}}}
    good = report(.96, .93, 2.84)
    low_stars = report(1., .3, 1.7)
    unsafe = report(.89, .89, 2.67)
    assert evaluation_passed(good)
    assert not evaluation_passed(low_stars)
    assert not evaluation_passed(unsafe)
    assert evaluation_key(good) > evaluation_key(low_stars) > evaluation_key(unsafe)
