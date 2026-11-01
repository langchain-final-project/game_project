"""성공률을 보존하며 안전한 별 3개 귀환 모델을 고르고 최종 검증합니다."""

from copy import deepcopy

from rl import ppo_config as cfg


def performance_ready(report):
    from rl.evaluate_ppo import evaluation_passed
    # 짧은 실행의 정식 통과 표시와 조기 종료를 위한 성능 기준을 구분합니다.
    return evaluation_passed({**report, "quick": False, "smoke": False})


def retention_ok(summary, settings):
    return (summary["success_rate"] >= settings["minimum_success_rate"]
            and summary["three_star_rate"] >= settings["minimum_three_star_rate"]
            and summary["mean_stars"] >= settings["minimum_mean_stars"]
            and summary["magnet_use_rate"] >= settings["minimum_magnet_use_rate"])


def safe_selection_key(report, summary):
    settings = report.get("safe_return_selection", cfg.EVAL_CONFIG["safe_return_selection"])
    eligible = retention_ok(summary, settings)
    score = (summary["three_star_rate"] - settings["damage_weight"] * summary["mean_damage"]
             + settings["damage_free_weight"] * summary["damage_free_success_rate"])
    progress = min(summary["success_rate"] / settings["minimum_success_rate"],
                   summary["three_star_rate"] / settings["minimum_three_star_rate"],
                   summary["mean_stars"] / settings["minimum_mean_stars"],
                   summary["magnet_use_rate"] / settings["minimum_magnet_use_rate"])
    key = (int(performance_ready(report)), int(eligible), score if eligible else progress,
           summary["success_rate"], summary["three_star_rate"], summary["mean_stars"],
           -summary["mean_damage"], -summary["mean_steps"])
    if report.get("skill_review"):
        from rl.ppo_skill_review import review_selection_key
        return (*review_selection_key(report["skill_review"]), *key)
    return key


def final_training_status(report, frames, previous_passes, previous_regressions, settings):
    passes = previous_passes + 1 if performance_ready(report) else 0
    summary = report["results"]["stochastic"]["summary"]
    floors = report.get("safe_return_selection", cfg.EVAL_CONFIG["safe_return_selection"])
    regressions = 0 if retention_ok(summary, floors) else previous_regressions + 1
    ready = (frames >= settings["minimum_training_frames"]
             and passes >= settings["required_consecutive_passes"])
    return passes, regressions, ready, regressions >= settings["regression_patience"]


def verify_final_model(path, output_dir):
    """최적 모델 하나만 새 시드로 검증하며 이 결과로 체크포인트를 다시 고르지 않습니다."""
    return verify_stage_model(path,output_dir,2)


def verify_stage_model(path, output_dir, stage):
    """저장된 스테이지별 설정으로 최종 평가하고 현재 작업 설정은 복원합니다."""
    from stable_baselines3 import PPO
    from rl.evaluate_ppo import evaluate_model, save_evaluation
    model = PPO.load(path, device="cpu")
    settings = getattr(model, f'ppo_final_stage{stage}_config',{})
    if not settings.get("enabled", False) or model.ppo_stage != stage:
        raise ValueError(f'최종 평가 설정을 저장한 Stage {stage} 모델이 필요합니다.')
    original = deepcopy(cfg.EVAL_CONFIG)
    original_environment = deepcopy(cfg.ENV_CONFIG)
    try:
        cfg.EVAL_CONFIG.update(deepcopy(settings.get("evaluation", {})))
        cfg.ENV_CONFIG.update(deepcopy(settings.get("environment", {})))
        base, count = settings["heldout_seed_base"], settings["heldout_episodes"]
        cfg.EVAL_CONFIG["stochastic_seeds"] = list(range(base, base + count))
        cfg.EVAL_CONFIG["deterministic_seeds"] = list(range(base, base + 20))
        report = evaluate_model(model)
        report[f'final_stage{stage}_verification'] = True
        report["model"] = str(path)
        save_evaluation(report, output_dir)
        return report
    finally:
        cfg.EVAL_CONFIG.clear()
        cfg.EVAL_CONFIG.update(original)
        cfg.ENV_CONFIG.clear()
        cfg.ENV_CONFIG.update(original_environment)
