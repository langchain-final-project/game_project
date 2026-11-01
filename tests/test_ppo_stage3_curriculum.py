"""Stage 3 실제 경로·역순 진급·전체 시작점 평가를 학습 없이 검증합니다."""

from copy import deepcopy
from collections import Counter

import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import Stage3RouteCurriculum
from rl.ppo_observation import PPOGoalObservation
from rl.ppo_reward import PPOReward
from rl.evaluate_ppo import make_eval_env


def curriculum(start, **overrides):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(initial_frontier=start, full_start_probability=0, review_probability=0)
    settings.update(overrides)
    return Stage3RouteCurriculum(RealtimeMansionEnv(stage=3), settings=settings)


@pytest.mark.parametrize("start", range(len(Stage3RouteCurriculum.prefixes)))
def test_every_prefix_is_legal_and_has_a_successful_remaining_route(start):
    env = curriculum(start)
    try:
        _, info = env.reset(seed=11)
        s = info["snapshot"]
        assert s["damage_count"] == 0
        assert s["motion"]["frame"] == env.prefix_frames[start]
        assert s["remaining_ticks"] == 100 - env.prefix_frames[start] // 18
        assert s["quest"]["collected"] == (start >= 8)
        assert (s["carried"] >= 4) == (start >= 4)
        assert env.env._steps == 0
        for action in env.route[env.prefix_frames[start]:]:
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done and not truncated
        assert info["outcome"]["status"] == "COMPLETE"
        assert info["snapshot"]["damage_count"] == 0
        assert info["snapshot"]["motion"]["frame"] == 856
    finally:
        env.close()


def test_progression_requires_twenty_current_frontier_games_with_sixteen_successes():
    last = len(Stage3RouteCurriculum.prefixes) - 1
    env = curriculum(last)
    try:
        for episode in range(20):
            env.reset(seed=11 if episode == 0 else None)
            if episode < 4:
                # 출구 직전에 머물러 시간 초과로 실패한 판도 기록합니다.
                while True:
                    _, _, done, truncated, info = env.step(5)
                    if done or truncated:
                        break
                assert info["outcome"]["status"] == "FAILED"
            else:
                _, _, done, _, info = env.step(0)
                assert done
            assert env.frontier == (last - 1 if episode == 19 else last)
        assert not env.history
        env.reset()
        assert env.start == last - 1
    finally:
        env.close()


def test_review_games_do_not_promote_and_final_stage_prefers_full_start():
    last = len(Stage3RouteCurriculum.prefixes) - 1
    env = curriculum(last - 1, review_probability=1)
    try:
        env._first = False
        for _ in range(20):
            _, info = env.reset()
            assert info["curriculum_start"] == last
            env.step(0)
        assert env.frontier == last - 1 and not env.history
        env.frontier = 0
        counts = Counter(env.reset()[1]["curriculum_start"] for _ in range(1500))
        assert .75 < counts[0] / 1500 < .85
    finally:
        env.close()


def test_prefix_reset_does_not_pay_collection_rewards_and_cache_is_independent():
    raw = curriculum(len(Stage3RouteCurriculum.prefixes) - 1)
    env = PPOReward(PPOGoalObservation(raw, **cfg.ACTION_MASK_CONFIG))
    try:
        _, info = env.reset(seed=11)
        original = deepcopy(info["snapshot"])
        _, _, _, _, info = env.step(5)
        assert info["reward_terms"]["quest_reward"] == 0
        assert info["reward_terms"]["warmth_reward"] == 0
        _, info = env.reset(seed=11)
        assert info["snapshot"] == original
    finally:
        env.close()


def test_evaluation_still_starts_from_original_unprepared_state():
    env = make_eval_env()
    try:
        _, info = env.reset(seed=11)
        s = info["snapshot"]
        assert s["motion"]["frame"] == 0
        assert s["position"] == [1, 1]
        assert s["carried"] == 0 and not s["quest"]["collected"]
        assert not s["invincibility"]["collected"]
        assert "curriculum_start" not in info
    finally:
        env.close()
