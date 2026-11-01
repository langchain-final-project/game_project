"""연속 경로 거리와 Stage 3의 잘못된 방향·왕복 보상을 검증합니다."""

from copy import deepcopy

import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage3RouteCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_path_distance import path_distance


def env_at(start):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(initial_frontier=start, full_start_probability=0, review_probability=0)
    return ActionRepeat(PPOReward(PPOGoalObservation(
        Stage3RouteCurriculum(RealtimeMansionEnv(stage=3), settings=settings), star_goal={"enabled": False})), repeat=9)


def test_correct_corridor_direction_is_rewarded_and_wrong_upward_move_penalized():
    for action, expected in [(2, .25), (0, -.25)]:
        env = env_at(11)
        try:
            _, info = env.reset(seed=11)
            assert path_distance(info["snapshot"], [1, 1]) == pytest.approx(11)
            _, reward, _, _, info = env.step(action)
            assert info["reward_terms"]["return_progress_reward"] == pytest.approx(expected)
            assert reward == pytest.approx(sum(info["reward_terms"].values()))
        finally:
            env.close()


def test_small_return_trip_cannot_accumulate_distance_reward():
    env = env_at(11)
    try:
        env.reset(seed=11)
        rewards = [env.step(a)[4]["reward_terms"]["return_progress_reward"] for a in (2, 3, 2, 3)]
        assert rewards == pytest.approx([.25, -.25, .25, -.25])
        assert sum(rewards) == pytest.approx(0)
    finally:
        env.close()


def test_required_detour_away_from_exit_gets_positive_reward():
    env = env_at(10)  # (7,2)는 출구에서 멀어지는 아래쪽 우회가 필요합니다.
    try:
        env.reset(seed=11)
        assert env.step(1)[4]["reward_terms"]["return_progress_reward"] == pytest.approx(.25)
    finally:
        env.close()


def test_wall_free_fractional_distance_is_continuous_and_doors_change_routes():
    s = {"grid": ["#####", "#...#", "#...#", "#...#", "#####"],
         "motion": {"position": [3., 1.]}, "shortcuts": []}
    assert path_distance(s, [1, 1]) == 2
    s["motion"]["position"] = [2.5, 1.]
    assert path_distance(s, [1, 1]) == 1.5
    s["shortcuts"] = [{"gate": [2, 1], "opened": False}]
    s["motion"]["position"] = [3., 1.]
    assert path_distance(s, [1, 1]) == 4
    s["shortcuts"][0]["opened"] = True
    assert path_distance(s, [1, 1]) == 2


def test_disconnected_goal_and_diagonal_corner_cannot_be_shortcut():
    s = {"grid": ["#####", "#.#.#", "##..#", "#...#", "#####"],
         "motion": {"position": [1.2, 1.2]}, "shortcuts": []}
    assert path_distance(s, [3, 3]) is None
    assert path_distance(s, [2, 1]) is None  # 벽에 있는 목표에도 도달할 수 없습니다.


def test_added_corridor_states_and_restart_checkpoint_are_consistent():
    assert len(Stage3RouteCurriculum.prefixes) == 18
    assert cfg.CURRICULUM_CONFIG["initial_frontier"] == 18
    assert cfg.TRANSFER_CONFIG["model_path"].is_file()
    for start, x in [(11, 7), (12, 6), (13, 5), (14, 4)]:
        env = env_at(start)
        try:
            _, info = env.reset(seed=11)
            assert info["snapshot"]["motion"]["position"] == pytest.approx([x, 6])
        finally:
            env.close()
