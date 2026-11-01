"""공동 복습의 시작 상태·기술 보존·모델 선택을 학습 없이 검증합니다."""

from collections import Counter
from copy import deepcopy

import pytest
import torch

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl import ppo_skill_review as review
from rl.evaluate_ppo import evaluation_passed
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.train_ppo import make_env, make_model, evaluation_key


@pytest.fixture(autouse=True)
def original_joint_profile(monkeypatch):
    # 이 파일은 이전 공동 복습 설정의 호환성을 검증합니다.
    monkeypatch.setitem(cfg.ENV_CONFIG, 'stage', 2)
    monkeypatch.setitem(cfg.CURRICULUM_CONFIG, 'initial_frontier', 9)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, 'stage2_three_star', True)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, 'stage3_three_star', False)
    monkeypatch.setitem(cfg.STAR_GOAL_CONFIG, 'warmth_target', 14)
    values = {"name": "stage2_joint_review_v3", "full_start_probability": .2,
              "magnet_focus_probability": .25, "quest_focus_probability": .2,
              "ghost_focus_probability": .15, "review_probability": .1,
              "ghost_focus_starts": [11, 12, 13, 14], "ghost_wait_frames": [0, 9, 18],
              "ghost_max_wait_frames": 18, "ghost_safe_preparation": False,
              "keep_focus_at_frontier_zero": False}
    for key, value in values.items():
        monkeypatch.setitem(cfg.CURRICULUM_CONFIG, key, value)


def test_sampler_restores_early_collection_while_retaining_quest_practice():
    env = Stage2ThreeStarCurriculum(RealtimeMansionEnv(stage=2))
    try:
        env.reset(seed=123)
        counts = Counter()
        for _ in range(20000):
            start = env._choose_start()
            counts[env._phase] += 1
            if env._phase == "magnet_focus":
                assert start in (1, 2, 3, 4, 5)
            if env._phase == "quest_focus":
                assert start in (6, 7)
        expected = {"full_start": .2, "magnet_focus": .25, "quest_focus": .2,
                    "ghost_focus": .15, "review": .1, "frontier": .1}
        for phase, probability in expected.items():
            assert counts[phase] / 20000 == pytest.approx(probability, abs=.015)
    finally:
        env.close()


@pytest.mark.parametrize("start", [1, 2, 3, 4, 5])
def test_collection_practice_starts_before_magnet_and_quest(start):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings["initial_frontier"] = start
    env = Stage2ThreeStarCurriculum(RealtimeMansionEnv(stage=2), settings=settings)
    try:
        _, info = env.reset(seed=11)
        assert not info["snapshot"]["magnet"]["collected"]
        assert not info["snapshot"]["quest"]["collected"]
        assert info["snapshot"]["carried"] <= 2
        assert info["snapshot"]["damage_count"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("change", [
    {"magnet_focus_probability": .9}, {"magnet_focus_starts": []},
    {"magnet_focus_starts": [0]}, {"magnet_focus_starts": [99]},
])
def test_invalid_collection_mix_is_rejected(change):
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    settings.update(change)
    raw = RealtimeMansionEnv(stage=2)
    try:
        with pytest.raises(ValueError):
            Stage2ThreeStarCurriculum(raw, settings=settings)
    finally:
        raw.close()


def skill(quest, magnet):
    return {"quest_minimum_rate": quest, "magnet_success_rate": magnet,
            "quest_retained": quest >= .8, "passed": quest >= .9 and magnet >= .8}


def report(quest=.95, magnet=.85, success=.95, three=.9):
    return {"primary_mode": "stochastic", "selection_objective": "three_stars",
            "success_rate_threshold": .9, "target_mean_stars": 2.7,
            "target_three_star_rate": .9, "skill_review": skill(quest, magnet),
            "results": {"stochastic": {"summary": {
                "success_rate": success, "three_star_rate": three,
                "mean_stars": 2.8, "mean_steps": 600}}}}


def test_selection_preserves_quest_then_restores_weaker_skill():
    assert evaluation_key(report(.85, .4, .1, .1)) > evaluation_key(report(.75, 1, 1, 1))
    assert evaluation_key(report(.9, .5, .1, .1)) > evaluation_key(report(.95, .3, .2, .2))
    assert evaluation_key(report(.9, .8, .1, .1)) > evaluation_key(report(.95, .79, .9, .9))


def test_selection_returns_to_full_game_stars_after_both_skills_pass():
    assert evaluation_key(report(.9, .8, .95, .94)) > evaluation_key(report(1, 1, 1, .90))
    assert evaluation_key(report(.95, .8, .9, .9)) > evaluation_key(report(1, 1, .89, .89))


def test_two_consecutive_quest_regressions_stop_and_recovery_resets_count():
    assert review.quest_regression_status(0, skill(.7, .8), 2) == (1, False)
    assert review.quest_regression_status(1, skill(.7, .8), 2) == (2, True)
    assert review.quest_regression_status(1, skill(.8, .8), 2) == (0, False)


def test_formal_pass_requires_both_skills_and_full_game_criteria():
    assert evaluation_passed(report())
    assert not evaluation_passed(report(.85, .9))
    assert not evaluation_passed(report(.95, .75))
    assert not evaluation_passed(report(success=.89))
    previous = report()
    previous.pop("skill_review")
    assert evaluation_passed(previous)


def test_new_full_model_diagnostics_preserve_weights_rng_and_optimizer(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    env = make_env()
    try:
        model = make_model(env, 11)
        assert model.ppo_training_task == "full_game"
        assert model.ppo_joint_review_config["enabled"]
        settings = deepcopy(model.ppo_joint_review_config)
        settings["seeds"] = [11]
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        result = review.evaluate_skill_review(model, settings)
        assert not result["full_game_evaluation"]
        assert set(result["quest_by_start"]) == {"6", "7"}
        episodes = result["magnet"]["episodes"]
        assert len(episodes) == 5
        assert all(e["success"] == (e["magnet_collected"] and e["remote_warmth"] > 0)
                   for e in episodes)
        assert all(e["frames"] <= 180 for e in episodes)
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(v, before[k]) for k, v in model.policy.state_dict().items())
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
    finally:
        env.close()


def test_joint_resume_checks_heldout_quest_skill(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    env = make_env()
    try:
        model = make_model(env, 11)
        path = tmp_path / "joint.zip"
        model.save(path)
    finally:
        env.close()
    calls = []
    def evaluate(model, *, seeds, task_settings):
        calls.append((seeds, task_settings))
        return {"passed": False, "results": {"stochastic": {
            str(start): {"summary": {"success_rate": .5, "episodes": 50}}
            for start in (6, 7)}}}
    monkeypatch.setattr(review, "evaluate_quest", evaluate)
    with pytest.raises(ValueError, match="6.*7"):
        review.require_joint_source_ready(path)
    assert calls[0][0] == cfg.QUEST_TASK_CONFIG["heldout_seeds"]

