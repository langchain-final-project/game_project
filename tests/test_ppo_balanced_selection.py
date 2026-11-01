"""성공률·별 하한을 지키며 자석 활용과 피해를 비교하는지 검증합니다."""

from copy import deepcopy
from types import SimpleNamespace

from rl import ppo_config as cfg
from rl import train_ppo
from rl.train_ppo import evaluation_key, training_targets
from rl.evaluate_ppo import evaluation_cases, make_eval_env


def report(success=.98, stars=1.73, magnet=.27, remote=1.31, damage=.75, stagnant=.05255):
    return {
        "primary_mode": "stochastic", "selection_objective": "magnet_ghost",
        "selection_settings": deepcopy(cfg.EVAL_CONFIG["selection_settings"]),
        "success_rate_threshold": .9,
        "results": {"stochastic": {"episodes": [{"seed": 11, "success": True}], "summary": {
            "success_rate": success, "mean_stars": stars,
            "magnet_use_rate": magnet, "mean_magnet_remote_warmth": remote,
            "mean_damage": damage, "stagnation_frame_rate": stagnant,
            "mean_score": 68.5, "mean_steps": 400,
        }}},
    }


def test_actual_magnet_checkpoint_beats_previous_star_only_best():
    source = report()
    previous_best = report(success=.99, stars=1.76, magnet=.02, remote=.08, damage=.9, stagnant=.0075)
    assert evaluation_key(source) > evaluation_key(previous_best)


def test_magnet_utilization_cannot_cover_failure_of_success_or_star_floor():
    assert evaluation_key(report()) > evaluation_key(report(success=.89, stars=3, magnet=1, remote=10, damage=0))
    assert evaluation_key(report()) > evaluation_key(report(stars=1.59, magnet=1, remote=10, damage=0))
    assert evaluation_key(report(success=.97, stars=1.6))[0] == 2


def test_damage_and_stagnation_costs_change_model_rank():
    assert evaluation_key(report(damage=.2)) > evaluation_key(report(damage=.8))
    assert evaluation_key(report(stagnant=0)) > evaluation_key(report(stagnant=.5))


def test_remote_collection_and_stars_both_improve_selection_score():
    assert evaluation_key(report(remote=2)) > evaluation_key(report(remote=1))
    assert evaluation_key(report(stars=2)) > evaluation_key(report(stars=1.7))
    assert evaluation_key(report(magnet=.5)) > evaluation_key(report(magnet=.1))


def test_report_settings_are_frozen_for_historical_comparisons(monkeypatch):
    saved = report()
    key = evaluation_key(saved)
    monkeypatch.setitem(cfg.EVAL_CONFIG["selection_settings"], "damage_weight", 99)
    assert evaluation_key(saved) == key


def test_all_101_stochastic_cases_are_identical_in_quick_and_full_evaluation():
    env = make_eval_env()
    try:
        full = {name: cases for name, _, cases in evaluation_cases(env, False)}
        quick = {name: cases for name, _, cases in evaluation_cases(env, False, quick=True)}
        assert quick["stochastic"] == full["stochastic"]
        assert [c["seed"] for c in quick["stochastic"]] == [11, *range(10000, 10100)]
    finally:
        env.close()


def test_source_and_training_budget_are_correct():
    assert cfg.TRANSFER_CONFIG["model_path"].name == "ppo_140649.zip"
    assert training_targets(False, quick=True)[-1] == 300000
    assert training_targets(False)[-1] == 300000
    assert cfg.EVAL_CONFIG["evaluate_transfer_baseline"]


def test_baseline_is_evaluated_and_kept_without_learning(tmp_path, monkeypatch):
    saved = []
    recorded = []
    baseline_report = report()
    model = SimpleNamespace(save=lambda path: saved.append(path))
    monkeypatch.setattr(train_ppo, "evaluate_model", lambda m, **kwargs: baseline_report)
    monkeypatch.setattr(train_ppo, "save_evaluation", lambda r, path: recorded.append((r, path)))
    result = train_ppo.save_transfer_baseline(model, tmp_path, tmp_path / "models", 11, quick=True)
    assert result["training_timesteps"] == result["decision_timesteps"] == 0
    assert result["training_seed"] == 11
    assert saved == [tmp_path / "models/baseline_model.zip", tmp_path / "models/best_model.zip"]
    assert recorded[0][1] == tmp_path / "evaluation/step_0"
    # 동일하거나 나빠진 후보로는 초기 best_model을 교체하지 않습니다.
    assert not evaluation_key(report(damage=1.0)) > evaluation_key(result)


def test_selection_rejects_success_drop_and_stagnation_despite_large_collection_score():
    baseline = report()
    assert evaluation_key(baseline) > evaluation_key(report(success=.96, stars=2.5, magnet=1, remote=8))
    assert evaluation_key(baseline) > evaluation_key(report(stars=2.5, magnet=1, remote=8, stagnant=.061))
    assert evaluation_key(report(success=.97, stagnant=.06))[0] == 2


def test_regression_seed_must_be_present_and_successful():
    baseline = report()
    candidate = report(stars=2.5, magnet=1, remote=8)
    candidate["results"]["stochastic"]["episodes"][0]["success"] = False
    assert evaluation_key(baseline) > evaluation_key(candidate)
    candidate["results"]["stochastic"]["episodes"] = []
    assert evaluation_key(baseline) > evaluation_key(candidate)


def test_historical_selection_settings_keep_previous_rules():
    historical = report(success=.92, stagnant=.18)
    for key in ("minimum_success_rate", "maximum_stagnation_frame_rate",
                "require_regression_seed_success", "regression_seed"):
        historical["selection_settings"].pop(key)
    historical["results"]["stochastic"].pop("episodes")
    assert evaluation_key(historical)[0] == 2
