"""별 3개 준비 경로·진급·경유지·평가를 학습 없이 검증합니다."""

from copy import deepcopy

import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage3ThreeStarCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.ppo_goal import select_goal
from rl.ppo_path_distance import next_waypoint
from rl.evaluate_ppo import evaluation_passed
from rl.train_ppo import evaluation_key


def curriculum(start):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(initial_frontier=start, full_start_probability=0, review_probability=0)
    return Stage3ThreeStarCurriculum(RealtimeMansionEnv(stage=3), settings=settings)


@pytest.mark.parametrize("start", range(23))
def test_all_three_star_prefixes_can_finish_with_100_points(start):
    env = curriculum(start)
    try:
        _, info = env.reset(seed=11)
        assert info["snapshot"]["motion"]["frame"] == env.prefix_frames[start]
        assert info["snapshot"]["damage_count"] == 0
        if start >= 11:
            assert info["snapshot"]["carried"] == 14
        for a in env.route[env.prefix_frames[start]:]:
            _, _, done, truncated, info = env.step(a)
            if done or truncated:
                break
        assert done and not truncated
        assert info["outcome"]["score"]["total"] == 100
        assert info["outcome"]["score"]["stars"] == 3
        assert info["curriculum_task_success"]
        assert info["snapshot"]["motion"]["frame"] == 964
    finally:
        env.close()


def test_two_star_return_does_not_count_toward_promotion():
    env = curriculum(22)
    try:
        for episode in range(20):
            env.reset(seed=11 if episode == 0 else None)
            if episode < 4:
                for _ in range(720):
                    env.step(5)
            _, _, done, _, info = env.step(0)
            assert done
            assert info["curriculum_task_success"] == (episode >= 4)
            assert info["outcome"]["score"]["stars"] == (2 if episode < 4 else 3)
            assert env.frontier == (21 if episode == 19 else 22)
    finally:
        env.close()


def test_waypoints_guide_around_wall_without_forcing_policy_actions():
    env = curriculum(16)  # (7,6)에서 귀환 시작
    try:
        _, info = env.reset(seed=11)
        assert next_waypoint(info["snapshot"], [1, 1]) == [6, 6]
        phase, goal, enough = select_goal(info["snapshot"], cfg.STAR_GOAL_CONFIG)
        assert phase == 2 and goal == [6, 6] and enough
        old = {"enabled": False}
        assert select_goal(info["snapshot"], old)[1] == [1, 1]
    finally:
        env.close()


def test_collecting_before_quest_is_not_penalized_for_moving_away_from_quest():
    env = PPOReward(PPOGoalObservation(curriculum(5)))  # 무적 아이템 획득 직후
    try:
        _, info = env.reset(seed=11)
        assert info["goal_phase"] == 1
        info = env.step(3)[4]
        assert info["reward_terms"]["quest_progress_reward"] == 0
        assert info["reward_terms"]["return_progress_reward"] == 0
    finally:
        env.close()


def test_three_star_completion_bonus_and_task_metric_are_recorded():
    env = PPOReward(PPOGoalObservation(curriculum(22)))
    try:
        env.reset(seed=11)
        _, _, done, _, info = env.step(0)
        assert done
        assert info["reward_terms"]["star_completion_adjustment"] == 10
        assert info["rl_episode"]["curriculum_success"]
    finally:
        env.close()


def report(three=.9, success=.95):
    return {"primary_mode": "stochastic", "selection_objective": "three_stars",
            "success_rate_threshold": .9, "target_three_star_rate": .9,
            "target_mean_stars": 2.7, "quick": False, "smoke": False,
            "results": {"stochastic": {"summary": {"success_rate": success,
                "three_star_rate": three, "mean_stars": 2.8, "mean_steps": 900}}}}


def test_evaluation_requires_three_star_rate_and_selection_keeps_success_floor():
    assert evaluation_passed(report())
    assert not evaluation_passed(report(three=.89))
    assert evaluation_key(report()) > evaluation_key(report(three=.95, success=.89))
    assert evaluation_key(report(three=.95)) > evaluation_key(report(three=.9))
