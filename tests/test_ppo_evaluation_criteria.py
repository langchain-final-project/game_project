"""확률적 플레이 기준의 통과·모델 선택·성능 하락 감지를 확인합니다."""

import pytest

from rl import ppo_config as cfg
from rl.evaluate_ppo import (
    evaluation_passed, primary_evaluation_mode, primary_success_rate,
    evaluation_cases, make_eval_env,
)
from rl.train_ppo import evaluation_key, update_stability


def make_report(stochastic=0.95, deterministic=0.0, phase=0.0, score=40.0, mode="stochastic"):
    def summary(rate, mean_score):
        return {
            "success_rate": rate, "mean_score": mean_score, "mean_steps": 700,
            "post_quest_success_rate": rate, "quest_rate": 1.0,
        }

    return {
        "primary_mode": mode,
        "selection_objective": "success",
        "target_mean_stars": 0.0,
        "success_rate_threshold": 0.9,
        "smoke": False, "quick": False,
        "results": {
            "deterministic": {"summary": summary(deterministic, 99.0)},
            "stochastic": {"summary": summary(stochastic, score)},
            "patrol_phases": {"summary": summary(phase, 99.0)},
        },
    }


def test_stochastic_success_passes_despite_failed_diagnostics():
    report = make_report()
    assert primary_success_rate(report) == 0.95
    assert evaluation_passed(report)


def test_strong_diagnostics_cannot_cover_low_stochastic_success():
    report = make_report(stochastic=0.89, deterministic=1.0, phase=1.0)
    assert not evaluation_passed(report)


def test_threshold_boundary_and_report_settings_are_respected(monkeypatch):
    report = make_report(stochastic=0.95)
    report["success_rate_threshold"] = 0.95
    monkeypatch.setitem(cfg.EVAL_CONFIG, "primary_mode", "all")
    assert primary_evaluation_mode(report) == "stochastic"
    assert evaluation_passed(report)
    report["success_rate_threshold"] = 0.96
    assert not evaluation_passed(report)


@pytest.mark.parametrize("mode", ["quick", "smoke"])
def test_small_diagnostic_runs_are_not_formal_passes(mode):
    report = make_report(stochastic=1.0)
    report[mode] = True
    assert not evaluation_passed(report)


def test_best_model_prioritizes_stochastic_success_then_score():
    safer = make_report(stochastic=0.95, score=40)
    higher_score = make_report(stochastic=0.94, score=90, deterministic=1.0, phase=1.0)
    assert evaluation_key(safer) > evaluation_key(higher_score)
    assert evaluation_key(make_report(score=60)) > evaluation_key(make_report(score=40, deterministic=1.0))


def test_diagnostic_changes_do_not_change_stochastic_model_rank():
    assert evaluation_key(make_report()) == evaluation_key(make_report(deterministic=1.0, phase=1.0))


def test_deterministic_and_all_modes_remain_selectable():
    assert evaluation_passed(make_report(stochastic=0.0, deterministic=0.95, mode="deterministic"))
    assert not evaluation_passed(make_report(stochastic=1.0, deterministic=1.0, phase=0.0, mode="all"))
    assert evaluation_passed(make_report(stochastic=1.0, deterministic=1.0, phase=1.0, mode="all"))


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match="primary_mode"):
        primary_evaluation_mode({"primary_mode": "invalid"})


def test_stability_tracks_primary_rate_instead_of_failed_diagnostics():
    peak, bad_count, stop = update_stability(0.0, 0, primary_success_rate(make_report()))
    assert peak == 0.95 and bad_count == 0 and not stop
    for _ in range(cfg.STABILITY_CONFIG["patience"]):
        peak, bad_count, stop = update_stability(
            peak, bad_count, primary_success_rate(make_report(stochastic=0.7)),
        )
    assert stop


def test_deterministic_and_phase_diagnostics_are_still_run(monkeypatch):
    monkeypatch.setitem(cfg.ENV_CONFIG, "stage", 2)
    monkeypatch.setitem(cfg.EVAL_CONFIG, "check_patrol_phases", True)
    env = make_eval_env()
    try:
        cases = evaluation_cases(env, smoke=False, quick=True)
        assert [(name, deterministic) for name, deterministic, _ in cases] == [
            ("deterministic", True), ("stochastic", False), ("patrol_phases", True),
        ]
    finally:
        env.close()
