"""초반 복습 비중과 추가 학습 종료 조건을 최적화 없이 검증합니다."""

from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_connection_task import ConnectionStarts, connection_probabilities
from rl.train_ppo_connection import connection_report, selection_key
from rl.train_ppo_quest import partial_task_ready


def test_weighted_sampler_focuses_on_early_starts_without_losing_review():
    settings = cfg.CONNECTION_TASK_CONFIG
    starts = settings["starts"]
    probabilities = connection_probabilities(starts, settings)
    env = ConnectionStarts(RealtimeMansionEnv(stage=2), starts, probabilities)
    try:
        env.reset(seed=11)
        counts = Counter(env._choose_start() for _ in range(20000))
        for start, expected in zip(starts, [.4, .4, .05, .05, .05, .05]):
            assert counts[start] / 20000 == pytest.approx(expected, abs=.015)
        # 평가는 고정 구간만 사용하고 학습용 혼합 비중을 적용하지 않습니다.
        np.testing.assert_allclose(connection_probabilities([1], settings), [1.])
        assert connection_probabilities([6], settings) is None
        assert connection_probabilities([7], settings) is None
    finally:
        env.close()


@pytest.mark.parametrize("weights", [[], [.5, .5], [-.1, .5, .2, .2, .1, .1],
                                     [float('nan'), .4, .05, .05, .05, .05], [1]*6])
def test_invalid_weights_are_rejected(weights):
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG)
    settings["start_weights"] = weights
    with pytest.raises(ValueError, match="start_weights"):
        connection_probabilities(settings["starts"], settings)


def test_old_profiles_keep_uniform_sampling():
    assert connection_probabilities([0, 1, 2], {"starts": [0, 1, 2]}) is None


def test_passed_baseline_does_not_skip_minimum_additional_training():
    settings = {"minimum_training_frames": 30000, "required_consecutive_passes": 2}
    assert not partial_task_ready({"passed": True}, 0, 2, settings)
    assert not partial_task_ready({"passed": True}, 29999, 2, settings)
    assert not partial_task_ready({"passed": True}, 30000, 1, settings)
    assert partial_task_ready({"passed": True}, 30000, 2, settings)
    assert not partial_task_ready({"passed": False}, 30000, 2, settings)
    assert not partial_task_ready({"passed": True}, 45000, 0, settings)


def test_old_partial_task_can_still_finish_at_passed_baseline():
    assert partial_task_ready({"passed": True}, 0, 1, {})
    assert not partial_task_ready({"passed": False}, 0, 0, {})


def make_report(early=.9, other=.8, quest=.9, settings=None):
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG if settings is None else settings)
    results = {str(n): {"summary": {"success_rate": early if n < 2 else other}}
               for n in range(6)}
    q = {str(n): {"summary": {"success_rate": quest}} for n in (6, 7)}
    return connection_report(results, q, settings, [20000])


def test_each_early_start_requires_ninety_percent():
    assert make_report()["passed"]
    assert not make_report(early=.89)["passed"]
    assert not make_report(other=.79)["passed"]
    assert not make_report(quest=.89)["passed"]
    result = make_report()
    assert result["connection_thresholds"] == {"0": .9, "1": .9, "2": .8,
                                                "3": .8, "4": .8, "5": .8}


def test_selection_rewards_target_progress_without_forgetting_quest():
    assert selection_key(make_report(early=.88)) > selection_key(make_report(early=.82))
    assert selection_key(make_report(early=.8, quest=.9)) > selection_key(make_report(early=1., quest=.7))


def test_old_saved_profiles_keep_their_original_threshold():
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG)
    settings.pop("focus_starts")
    settings.pop("focus_success_rate_threshold")
    assert make_report(early=.8, settings=settings)["passed"]


def test_validation_and_final_test_seeds_are_disjoint():
    settings = cfg.CONNECTION_TASK_CONFIG
    assert len(settings["evaluation_seeds"]) == 50
    assert set(settings["evaluation_seeds"]).isdisjoint(settings["heldout_seeds"])
    assert set(settings["evaluation_seeds"]).isdisjoint(range(10000, 10020))
    assert set(settings["heldout_seeds"]).isdisjoint(range(30000, 30050))
