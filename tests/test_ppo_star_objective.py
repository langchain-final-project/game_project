"""평균 별 목표, 추가 수집 안내, 성공률을 지키는 모델 선택을 검증합니다."""

from copy import deepcopy

import gymnasium as gym
import numpy as np
import pytest

from kkamang.env import RealtimeMansionEnv
from kkamang.scoring import score_outcome
from rl import ppo_config as cfg
from rl.ppo_goal import select_goal
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.evaluate_ppo import summarize, evaluation_passed
from rl.train_ppo import evaluation_key


GOAL = {"enabled": True, "warmth_target": 12, "min_remaining_ticks": 24, "min_courage": 2}


@pytest.fixture(autouse=True)
def stage2_star_goal_settings(monkeypatch):
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "enabled", True)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "warmth_target", 10)


def snapshot():
    return {
        "stage": 2, "motion": {"position": [6., 6.]},
        "quest": {"position": [7, 6], "collected": True},
        "target": 4, "carried": 4, "remaining_ticks": 50, "courage": 3,
        "exit": [4, 1], "items": [{"active": True, "position": [8, 6]}],
    }


def test_four_points_can_clear_but_safe_goal_requests_extra_warmth():
    s = snapshot()
    phase, target, enough = select_goal(s, GOAL)
    assert enough and s["target"] == 4
    assert phase == 1 and target == [8, 6]
    s["carried"] = 12
    assert select_goal(s, GOAL)[:2] == (2, [4, 1])


@pytest.mark.parametrize("change", [{"remaining_ticks": 23}, {"courage": 1}])
def test_low_time_or_courage_returns_after_basic_completion(change):
    s = snapshot()
    s.update(change)
    assert select_goal(s, GOAL)[:2] == (2, [4, 1])
    s["carried"] = 2
    assert select_goal(s, GOAL)[0] == 1  # 필수 조건조차 부족하면 기본 온기는 확보합니다.


def test_no_remaining_items_and_old_saved_goal_can_return():
    s = snapshot()
    assert select_goal(s, {"enabled": False})[:2] == (2, [4, 1])
    s["items"] = []
    assert select_goal(s, GOAL)[:2] == (2, [4, 1])


def test_quest_is_still_first_and_stage1_goal_is_unchanged():
    s = snapshot()
    s["quest"]["collected"] = False
    assert select_goal(s, GOAL)[:2] == (0, [7, 6])
    s["quest"]["collected"] = True
    s["stage"] = 1
    assert select_goal(s, GOAL)[:2] == (2, [4, 1])


def test_game_scoring_and_required_warmth_are_unchanged():
    env = RealtimeMansionEnv(stage=2)
    try:
        level = env.core.config
        assert level["target_value"] == 4
        def score(warmth):
            return score_outcome(level, {"status": "COMPLETE", "banked": warmth,
                "ticks_used": level["star_fast_ticks"], "remaining_courage": 3})
        assert score(9)["stars"] == 2
        assert score(12)["stars"] == 3
    finally:
        env.close()


def report(success=0.95, stars=2.1, score=70):
    return {
        "primary_mode": "stochastic", "selection_objective": "stars",
        "success_rate_threshold": 0.9, "target_mean_stars": 2.0,
        "quick": False, "smoke": False,
        "results": {"stochastic": {"summary": {
            "success_rate": success, "mean_stars": stars, "mean_score": score,
            "mean_steps": 700,
        }}},
    }


def test_pass_requires_both_success_floor_and_two_stars():
    assert evaluation_passed(report())
    assert not evaluation_passed(report(success=0.89, stars=2.5))
    assert not evaluation_passed(report(success=0.99, stars=1.9))
    assert evaluation_passed(report(success=0.9, stars=2.0))


def test_ranking_protects_success_floor_then_optimizes_stars():
    assert evaluation_key(report(success=0.9, stars=2.1)) > evaluation_key(report(success=0.89, stars=2.5))
    assert evaluation_key(report(success=0.92, stars=2.4)) > evaluation_key(report(success=0.99, stars=1.5))
    assert evaluation_key(report(success=0.85, stars=1.0)) > evaluation_key(report(success=0.8, stars=2.3))


def test_average_stars_includes_failed_attempts():
    episodes = [{
        "success": i < 9, "stars": 2 if i < 9 else 0,
        "score": 70 if i < 9 else 0, "return": 10, "steps": 100,
        "decision_steps": 12, "physical_frames": 100, "damage_count": 0,
        "blocked_move_frames": 0, "quest_collected": True,
        "warmth_met": True, "stalled": False,
    } for i in range(10)]
    summary = summarize(episodes)
    assert summary["mean_stars"] == 1.8
    assert summary["mean_stars_on_success"] == 2.0
    r = report()
    r["results"]["stochastic"]["summary"] = summary
    assert not evaluation_passed(r)


def test_switching_warmth_target_does_not_create_a_distance_jump():
    base = RealtimeMansionEnv(stage=2)
    _, info = base.reset(seed=11)
    initial = deepcopy(info["snapshot"])
    base.close()
    initial["quest"]["collected"] = True
    initial["carried"] = 4
    initial["motion"]["position"] = [5., 1.]
    current = deepcopy(initial)
    current["motion"].update(position=[6., 1.], moving=True)

    class ScriptedEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(640,), dtype=np.float32)
        action_space = gym.spaces.Discrete(6)

        def reset(self, *, seed=None, options=None):
            return np.zeros(640, dtype=np.float32), {
                "snapshot": initial, "goal_phase": 1, "goal_position": [7, 1],
            }

        def step(self, action):
            return np.zeros(640, dtype=np.float32), 0., False, False, {
                "snapshot": current, "goal_phase": 1, "goal_position": [1, 1],
                "events": [], "outcome": None,
            }

    env = PPOReward(ScriptedEnv())
    env.reset()
    _, _, _, _, info = env.step(3)
    assert info["reward_terms"]["warmth_progress_reward"] == pytest.approx(0.25)
    env.close()


def test_saved_goal_settings_are_copied_and_validated():
    settings = deepcopy(GOAL)
    env = PPOGoalObservation(RealtimeMansionEnv(stage=2), star_goal=settings)
    try:
        settings["warmth_target"] = 99
        assert env.star_goal["warmth_target"] == 12
    finally:
        env.close()
    base = RealtimeMansionEnv(stage=2)
    try:
        with pytest.raises(ValueError, match="warmth_target"):
            PPOGoalObservation(base, star_goal=settings)
    finally:
        base.close()


def test_current_goal_collects_nine_then_returns_at_ten():
    s = snapshot()
    s["carried"] = 9
    assert select_goal(s, cfg.STAR_GOAL_CONFIG)[0] == 1
    s["carried"] = 10
    assert select_goal(s, cfg.STAR_GOAL_CONFIG)[:2] == (2, s["exit"])


@pytest.mark.parametrize(
    "stars,total,adjustment", [(0, 40, -8), (1, 52, -6), (2, 70, 6), (3, 94, 10)]
)
@pytest.mark.parametrize("escape", [{}, {"remaining_ticks": 23}, {"courage": 1}])
def test_terminal_star_gap_and_safe_escape(stars, total, adjustment, escape):
    s = snapshot()
    s["motion"]["moving"] = False
    s["damage_count"] = 0
    s.update(escape)

    class TerminalEnv(gym.Env):
        observation_space = gym.spaces.Box(0, 1, shape=(640,), dtype=np.float32)
        action_space = gym.spaces.Discrete(6)
        star_goal = deepcopy(cfg.STAR_GOAL_CONFIG)

        def reset(self, *, seed=None, options=None):
            return np.zeros(640, dtype=np.float32), {"snapshot": deepcopy(s)}

        def step(self, action):
            return np.zeros(640, dtype=np.float32), 0., True, False, {
                "snapshot": deepcopy(s), "events": [],
                "outcome": {"status": "COMPLETE", "reason": "returned",
                            "score": {"total": total, "stars": stars}},
            }

    env = PPOReward(TerminalEnv())
    try:
        env.reset()
        _, reward, _, _, info = env.step(5)
        expected = 0 if escape and adjustment < 0 else adjustment
        terms = info["reward_terms"]
        assert terms["star_completion_adjustment"] == expected
        assert terms["success_reward"] == pytest.approx(10 + total / 10 + 6 * stars)
        assert reward == pytest.approx(sum(terms.values()))
        assert info["rl_episode"]["star_completion_adjustment"] == expected
    finally:
        env.close()
