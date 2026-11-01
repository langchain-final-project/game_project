"""자석 사용 후 10점 추가 수집 구간을 실제 이동으로 준비하는지 확인합니다."""

from copy import deepcopy
from collections import Counter

import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage2MagnetCollectionCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.train_ppo import make_env
from rl.evaluate_ppo import make_eval_env


@pytest.fixture(autouse=True)
def stage2_curriculum_settings(monkeypatch):
    """현재 Stage 3 실행 설정과 별도로 Stage 2 구간 연습을 검증합니다."""
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 2)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "enabled", True)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "name", "stage2_magnet_ten_v1")
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "initial_frontier", 1)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "quick_initial_frontier", 1)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, "focus_start_weights", [.50, .25, .12, .08, .03, .02])
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, "enabled", True)


def focus_env(start):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings["initial_frontier"] = start
    settings["focus_start_weights"] = [float(i == start) for i in range(6)]
    return Stage2MagnetCollectionCurriculum(RealtimeMansionEnv(stage=2), settings=settings)


@pytest.mark.parametrize("start,warmth,quest", [(0, 0, False), (1, 4, False), (2, 4, True),
                                              (3, 6, True), (4, 9, True), (5, 10, True)])
def test_prepared_states_are_legal_and_cached_without_state_leakage(start, warmth, quest):
    env = focus_env(start)
    try:
        _, info = env.reset(seed=11)
        original = deepcopy(info["snapshot"])
        assert original["carried"] == warmth
        assert original["quest"]["collected"] == quest
        assert original["damage_count"] == 0
        assert env.env._steps == 0
        if start:
            assert original["magnet"]["collected"]
            assert info["curriculum_magnet_remote_warmth_at_start"] >= 2
            assert original["motion"]["frame"] == len(env.prefixes[start])
        for _ in range(9):
            env.step(5)
        _, reset_info = env.reset(seed=11)
        assert reset_info["snapshot"] == original
    finally:
        env.close()


def test_focus_mixture_retains_full_start_and_concentrates_on_low_warmth_collection():
    env = Stage2MagnetCollectionCurriculum(RealtimeMansionEnv(stage=2))
    try:
        env.reset(seed=11)
        counts = Counter(env.reset()[1]["curriculum_start"] for _ in range(2000))
        assert .45 < counts[0] / 2000 < .55
        assert .70 < (counts[0] + counts[1]) / 2000 < .80
        assert .15 < (counts[2] + counts[3]) / 2000 < .25
    finally:
        env.close()


def test_focused_reset_does_not_pay_for_preparation_or_change_eval_start():
    env = PPOReward(PPOGoalObservation(focus_env(2)))
    try:
        _, info = env.reset(seed=11)
        assert info["goal_phase"] == 1  # 물건 획득 후에도 수집을 안내합니다.
        info = env.step(5)[4]
        assert info["reward_terms"]["quest_reward"] == 0
        assert info["reward_terms"]["warmth_reward"] == 0
        assert info["reward_terms"]["magnet_warmth_reward"] == 0
    finally:
        env.close()
    evaluation = make_eval_env()
    try:
        _, info = evaluation.reset(seed=11)
        assert info["snapshot"]["carried"] == 0
        assert not info["snapshot"]["magnet"]["collected"]
        assert not info["snapshot"]["quest"]["collected"]
        assert "curriculum_start" not in info
    finally:
        evaluation.close()


@pytest.mark.parametrize("start,expected", [(2, False), (5, True)])
def test_early_return_is_game_success_but_only_ten_points_earns_task_bonus(start, expected):
    env = PPOReward(PPOGoalObservation(focus_env(start)))
    try:
        env.reset(seed=11)
        # 물건을 가진 두 준비 상태에서 합법적으로 출구까지 이동합니다.
        route = ((2,) * 18 + (0,) * 90 + (2,) * 36 if start == 2
                 else (2,) * 72)
        for action in route:
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert info["rl_episode"]["success"]
        assert info["rl_episode"]["magnet_ten_return_success"] is expected
        assert info["reward_terms"]["magnet_ten_success_reward"] == (3 if expected else 0)
        assert info["rl_episode"]["magnet_remote_warmth_at_start"] > 0
    finally:
        env.close()


def test_training_uses_focused_curriculum():
    env = make_env()
    try:
        _, info = env.reset(seed=11)
        assert info["curriculum_version"] == "stage2_magnet_ten_v1"
        assert info["curriculum_start"] == 1
        assert info["snapshot"]["carried"] == 4
    finally:
        env.close()
