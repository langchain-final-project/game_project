"""전체 PPO 정책의 초반 자석 수집과 물건 획득 능력을 별도로 점검합니다."""

from copy import deepcopy

import torch
from stable_baselines3 import PPO

from rl import ppo_config as cfg
from rl.probe_stage2 import probe_model
from rl.train_ppo_quest import evaluate_quest, print_report, require_full_return_ready, TASK_VERSION


def evaluate_skill_review(model, settings=None, *, smoke=False):
    settings = deepcopy(model.ppo_joint_review_config if settings is None else settings)
    seeds = list(settings["seeds"][:2] if smoke else settings["seeds"])
    task = deepcopy(settings.get("quest_settings", cfg.QUEST_TASK_CONFIG))
    task["goals"] = deepcopy(model.ppo_star_goal_config)
    quest = evaluate_quest(model, seeds=seeds, task_settings=task)
    magnet = probe_model(model, seeds=seeds, frames=settings["frame_limit"], tasks=("magnet",))
    quest_by_start = quest["results"]["stochastic"]
    minimum = min(quest_by_start[str(start)]["summary"]["success_rate"] for start in (6, 7))
    magnet_result = magnet["results"]["magnet"]
    magnet_rate = magnet_result["summary"]["success_rate"]
    return {
        "version": "stage2_joint_review_v1", "full_game_evaluation": False,
        "seeds": seeds, "frame_limit": settings["frame_limit"],
        "quest_by_start": quest_by_start, "quest_minimum_rate": minimum,
        "magnet": magnet_result, "magnet_success_rate": magnet_rate,
        "quest_safety_floor": settings["quest_safety_floor"],
        "quest_target": settings["quest_target"], "magnet_target": settings["magnet_target"],
        "quest_retained": minimum >= settings["quest_safety_floor"],
        "passed": minimum >= settings["quest_target"] and magnet_rate >= settings["magnet_target"],
    }


def review_selection_key(review):
    # 양쪽 기술을 회복하기 전에는 더 약한 기술을 개선하는 후보를 우선합니다.
    # 회복 후에는 전체 게임 성공률과 별 획득률로 비교합니다.
    progress = 1.0 if review["passed"] else min(
        review["quest_minimum_rate"], review["magnet_success_rate"]
    )
    return int(review["quest_retained"]), int(review["passed"]), progress


def quest_regression_status(previous_count, review, patience):
    count = 0 if review["quest_retained"] else previous_count + 1
    return count, count >= patience


def format_skill_review(review):
    rates = review["quest_by_start"]
    return (f"복습: 6번 {rates['6']['summary']['success_rate']:.1%}, "
            f"7번 {rates['7']['summary']['success_rate']:.1%}, "
            f"초반 자석 원격 수집 {review['magnet_success_rate']:.1%}")


def require_joint_source_ready(path):
    """부분 모델 또는 공동 복습 모델의 물건 획득 능력을 새 시드로 확인합니다."""
    torch.set_num_threads(cfg.TRAIN_CONFIG["torch_threads"])
    model = PPO.load(path, device="cpu")
    if getattr(model, "ppo_training_task", None) == "stage2_magnet_quest_connection_v1":
        from rl.train_ppo_connection import require_connection_ready
        return require_connection_ready(path)
    if getattr(model, "ppo_training_task", None) == TASK_VERSION:
        if cfg.JOINT_REVIEW_CONFIG.get("require_connection_source_ready", False):
            raise ValueError("먼저 --connection-only --quick으로 자석·물건 연결을 연습하고, "
                             "통과한 모델을 --from-connection 경로로 지정하세요.")
        return require_full_return_ready(path)
    profile = getattr(model, "ppo_joint_review_config", {})
    if not profile.get("enabled", False):
        raise ValueError("물건 연습을 통과한 모델 또는 공동 복습 체크포인트가 필요합니다.")
    task = deepcopy(profile["quest_settings"])
    task["goals"] = deepcopy(model.ppo_star_goal_config)
    report = evaluate_quest(model, seeds=task["heldout_seeds"], task_settings=task)
    print_report(report, "공동 복습 재시작 전 물건 확인")
    if not report["passed"]:
        raise ValueError("6·7번 새 시드 평가가 각각 기준을 통과해야 합니다.")
    return report
