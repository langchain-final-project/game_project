import argparse
from collections import deque
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from rl.ppo_action_repeat import ActionRepeat

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl.ppo_curriculum import (
    Stage1RouteCurriculum,
    Stage1ThreeStarCurriculum,
    Stage2RouteCurriculum,
    Stage2MagnetCollectionCurriculum,
    Stage3RouteCurriculum,
    Stage3ThreeStarCurriculum,
    Stage3EfficientCurriculum,
    Stage2ThreeStarCurriculum,
    Stage2SafeReturnCurriculum,
)
from rl.ppo_observation import (
    PPOGoalObservation,
    PPO_OBSERVATION_VERSION,
    LEGACY_PPO_OBSERVATION_VERSION,
    ACTION_MASK_VERSION,
)
from rl.ppo_reward import PPOReward
from rl.ppo_features import WorldGoalFeaturesExtractor
from rl.ppo_policy import MovementAwareActorCriticPolicy
from rl.evaluate_ppo import (
    evaluate_model,
    save_evaluation,
    isolated_rng,
    post_quest_metrics,
    format_post_quest,
    primary_evaluation_mode,
    primary_success_rate,
)


CURRICULUMS = {
    1: Stage1RouteCurriculum,
    2: Stage2RouteCurriculum,
    3: Stage3RouteCurriculum,
}


def training_targets(smoke, quick=False):
    t, p = cfg.TRAIN_CONFIG, cfg.PPO_CONFIG
    primary_evaluation_mode()

    positive = (
        t["n_envs"],
        t["torch_threads"],
        p["n_steps"],
        p["batch_size"],
        p["n_epochs"],
    )

    if any(type(v) is not int or v <= 0 for v in positive):
        raise ValueError(
            "환경·스레드·수집·배치·반복 횟수를 확인하세요."
        )

    if p["device"] != "cpu":
        raise ValueError("현재 코드는 CPU 설정을 사용합니다.")

    rollout = t["n_envs"] * p["n_steps"]

    if p["batch_size"] < 2 or rollout % p["batch_size"]:
        raise ValueError(
            "batch_size는 2 이상이며 rollout을 나누어야 합니다."
        )

    if smoke and quick:
        raise ValueError("smoke와 quick은 함께 사용할 수 없습니다.")

    if smoke:
        total = t["smoke_timesteps"]
        targets = [total]
    elif quick:
        total = t["quick_timesteps"]
        interval = t["quick_checkpoint_interval"]
        if type(interval) is not int or interval <= 0:
            raise ValueError("짧은 실험의 평가 간격은 양의 정수여야 합니다.")
        targets = sorted(set([*range(interval, total + 1, interval), total]))
    else:
        total = t["total_timesteps"]
        targets = sorted(set([*t["checkpoint_steps"], total]))

    if any(
        type(n) is not int
        or n <= 0
        or n > total
        # or n % rollout
        for n in targets
    ):
        raise ValueError(
            "학습량과 저장 시점은 양의 물리 프레임 수여야 합니다."
        )

    if cfg.TRANSFER_CONFIG["enabled"]:
        path = cfg.TRANSFER_CONFIG["model_path"]

        if not path.is_file():
            raise FileNotFoundError(path)

        if not 0 < cfg.TRANSFER_CONFIG["action_head_scale"] <= 1:
            raise ValueError(
                "action_head_scale은 0 초과 1 이하여야 합니다."
            )

    s = cfg.STABILITY_CONFIG

    if (
        not 0 <= s["activation_success_rate"] <= 1
        or not 0 < s["max_success_drop"] <= 1
        or type(s["patience"]) is not int
        or s["patience"] < 1
        or s["ent_coef_after_success"] < 0
    ):
        raise ValueError("성능 하락 감시 설정을 확인하세요.")

    return targets


def effective_curriculum_settings(quick=False):
    """실행 모드별 시작 구간을 결정하고 원본 설정은 보존합니다."""
    settings = deepcopy(cfg.CURRICULUM_CONFIG)
    if quick and "quick_initial_frontier" in settings:
        settings["initial_frontier"] = settings["quick_initial_frontier"]
    return settings


def make_env(*, curriculum_settings=None):
    if cfg.ENV_NAME != "RealtimeMansionEnv":
        raise ValueError("RealtimeMansionEnv가 필요합니다.")

    env = RealtimeMansionEnv(**cfg.ENV_CONFIG)

    if cfg.CURRICULUM_CONFIG["enabled"]:
        curriculum_class = training_curriculum_class()
        env = curriculum_class(env, settings=curriculum_settings)

    env = PPOReward(PPOGoalObservation(env, **cfg.ACTION_MASK_CONFIG))
    return ActionRepeat(env, repeat=cfg.ACTION_REPEAT)


def training_curriculum_class():
    if cfg.ENV_CONFIG['stage']==3 and cfg.CURRICULUM_CONFIG['name']=='stage3_efficient_three_star_v2':
        return Stage3EfficientCurriculum
    if cfg.ENV_CONFIG["stage"] == 2 and cfg.CURRICULUM_CONFIG["name"] == "stage2_safe_return_v1":
        return Stage2SafeReturnCurriculum
    if (
        cfg.ENV_CONFIG["stage"] == 2
        and cfg.CURRICULUM_CONFIG["name"] in (
            "stage2_three_star_magnet_v1", "stage2_quest_first_mixed_v2", "stage2_joint_review_v3"
        )
    ):
        return Stage2ThreeStarCurriculum
    if cfg.ENV_CONFIG["stage"] == 1 and cfg.CURRICULUM_CONFIG["name"] == "stage1_three_star_v1":
        return Stage1ThreeStarCurriculum
    if cfg.ENV_CONFIG["stage"] == 3 and cfg.CURRICULUM_CONFIG["name"] == "stage3_three_star_v1":
        return Stage3ThreeStarCurriculum
    if cfg.ENV_CONFIG["stage"] == 2 and cfg.CURRICULUM_CONFIG["name"] == "stage2_magnet_ten_v1":
        return Stage2MagnetCollectionCurriculum
    return CURRICULUMS[cfg.ENV_CONFIG["stage"]]


def make_model(env, seed, *, transfer_settings=None, ppo_settings=None, star_goal_settings=None):
    """같은 신경망의 부분 과제도 전역 설정을 바꾸지 않고 초기화합니다."""
    transfer = cfg.TRANSFER_CONFIG if transfer_settings is None else transfer_settings
    source = None

    if transfer["enabled"]:
        with isolated_rng():
            source = PPO.load(
                transfer["model_path"],
                device="cpu",
                verbose=0,
            )

        source_version = getattr(source, "ppo_observation_version", None)
        if source_version not in (PPO_OBSERVATION_VERSION, LEGACY_PPO_OBSERVATION_VERSION):
            raise ValueError("전이 모델의 관측 버전이 다릅니다.")

        compatible_observation = (
            source_version == PPO_OBSERVATION_VERSION
            and source.observation_space == env.observation_space
        )
        if source_version == LEGACY_PPO_OBSERVATION_VERSION:
            # 마스크 4개는 학습 가중치를 늘리지 않으므로 기존 분기를 그대로 이식합니다.
            space = env.observation_space
            legacy_low = np.concatenate((space.low[:630], space.low[-6:]))
            legacy_high = np.concatenate((space.high[:630], space.high[-6:]))
            compatible_observation = (
                source.observation_space.shape == (636,)
                and space.shape == (640,)
                and source.observation_space.dtype == space.dtype
                and np.array_equal(source.observation_space.low, legacy_low)
                and np.array_equal(source.observation_space.high, legacy_high)
            )
        if not compatible_observation or source.action_space != env.action_space:
            raise ValueError(
                "전이 모델과 환경의 관측·행동 공간이 다릅니다."
            )

    activation = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
    }[cfg.NETWORK_CONFIG["activation"]]

    extractor = cfg.NETWORK_CONFIG["features_extractor"]
    if extractor != "world_goal_branches_v1":
        raise ValueError(f"지원하지 않는 특징 추출기입니다: {extractor}")

    ppo_options = deepcopy(cfg.PPO_CONFIG if ppo_settings is None else ppo_settings)
    if ppo_options.pop("policy") != "MovementAwareActorCriticPolicy":
        raise ValueError("이동 가능 방향을 적용하는 PPO 정책이 필요합니다.")

    # 새 최적화기와 0스텝 카운터로 시작합니다.
    model = PPO(
        policy=MovementAwareActorCriticPolicy,
        env=env,
        seed=seed,
        policy_kwargs={
            **cfg.ACTION_MASK_CONFIG,
            "net_arch": deepcopy(
                cfg.NETWORK_CONFIG["net_arch"]
            ),
            "activation_fn": activation,
            "features_extractor_class": WorldGoalFeaturesExtractor,
            "features_extractor_kwargs": {
                "branch_size": cfg.NETWORK_CONFIG["feature_branch_size"],
            },
        },
        **ppo_options,
    )

    if source is not None:
        if not isinstance(source.policy.features_extractor, WorldGoalFeaturesExtractor):
            raise ValueError(
                "전이 모델의 특징 추출 구조가 다릅니다. "
                "목표 정보 분리 구조로 학습한 Stage 1 모델이 필요합니다."
            )
        if source.policy.activation_fn != activation:
            raise ValueError(
                "전이 모델과 활성화 함수가 다릅니다."
            )

        # 정책망·가치망 가중치만 가져옵니다.
        model.policy.load_state_dict(
            source.policy.state_dict()
        )

        # 설정이 1이면 기존 행동 가중치를 그대로 유지합니다.
        with torch.no_grad():
            scale = transfer["action_head_scale"]
            model.policy.action_net.weight.mul_(scale)
            model.policy.action_net.bias.mul_(scale)

    model.ppo_observation_version = PPO_OBSERVATION_VERSION
    model.ppo_action_mask_version = ACTION_MASK_VERSION
    model.ppo_features_extractor_version = extractor
    model.ppo_reward_version = cfg.REWARD_VERSION
    model.ppo_star_goal_config = deepcopy(cfg.STAR_GOAL_CONFIG if star_goal_settings is None else star_goal_settings)
    model.ppo_stage = cfg.ENV_CONFIG["stage"]
    model.ppo_action_repeat = cfg.ACTION_REPEAT
    model.ppo_physical_frames = 0
    model.ppo_training_task = "full_game"
    if (ppo_settings is None and cfg.ENV_CONFIG["stage"] == 2
            and cfg.CURRICULUM_CONFIG["name"] in ("stage2_joint_review_v3", "stage2_safe_return_v1")
            and cfg.JOINT_REVIEW_CONFIG["enabled"]):
        model.ppo_joint_review_config = deepcopy(cfg.JOINT_REVIEW_CONFIG)
        model.ppo_joint_review_config["quest_settings"] = deepcopy(cfg.QUEST_TASK_CONFIG)
        if cfg.CURRICULUM_CONFIG["name"] == "stage2_safe_return_v1":
            model.ppo_final_stage2_config = deepcopy(cfg.FINAL_STAGE2_CONFIG)
            model.ppo_final_stage2_config["evaluation"] = deepcopy(cfg.EVAL_CONFIG)
            model.ppo_final_stage2_config["environment"] = deepcopy(cfg.ENV_CONFIG)
    if (ppo_settings is None and cfg.ENV_CONFIG['stage']==3
            and cfg.CURRICULUM_CONFIG['name']=='stage3_efficient_three_star_v2'):
        model.ppo_final_stage3_config = deepcopy(cfg.STAGE3_TRAINING_CONFIG)
        model.ppo_final_stage3_config['evaluation'] = deepcopy(cfg.EVAL_CONFIG)
        model.ppo_final_stage3_config['environment'] = deepcopy(cfg.ENV_CONFIG)

    model.transfer_source = (
        str(transfer["model_path"])
        if source else None
    )
    model.transfer_source_timesteps = (
        int(source.num_timesteps) if source else 0
    )
    model.transfer_source_observation_version = (
        getattr(source, "ppo_observation_version", None) if source else None
    )

    return model


def write_json(path, data):
    path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def evaluation_key(report):
    """평가 보고서에 저장된 목표와 설정에 따라 모델을 비교합니다."""
    results = report["results"]
    mode = primary_evaluation_mode(report)
    if mode != "all":
        summary = results[mode]["summary"]
        objective = report.get("selection_objective", cfg.EVAL_CONFIG.get("selection_objective", "success"))
        if objective == "safe_three_stars":
            from rl.ppo_safe_return import safe_selection_key
            return safe_selection_key(report, summary)
        if objective == "three_stars":
            success = summary["success_rate"]
            three = summary["three_star_rate"]
            safe = success >= report.get("success_rate_threshold", cfg.EVAL_CONFIG["success_rate_threshold"])
            key = (int(safe), three if safe else success, success if safe else three,
                    summary["mean_stars"], -summary.get("stagnation_frame_rate", 0),
                    -summary.get("mean_damage", 0), -summary["mean_steps"])
            if report.get("skill_review"):
                from rl.ppo_skill_review import review_selection_key
                return (*review_selection_key(report["skill_review"]), *key)
            return key
        if objective == "magnet_ghost":
            settings = report.get("selection_settings", cfg.EVAL_CONFIG["selection_settings"])
            success = summary["success_rate"]
            stars = summary["mean_stars"]
            threshold = report.get("success_rate_threshold", cfg.EVAL_CONFIG["success_rate_threshold"])
            threshold = max(threshold, settings.get("minimum_success_rate", threshold))
            stagnation_ok = summary["stagnation_frame_rate"] <= settings.get("maximum_stagnation_frame_rate", float("inf"))
            regression_ok = True
            if settings.get("require_regression_seed_success", False):
                cases = [e for e in results[mode].get("episodes", [])
                         if e["seed"] == settings.get("regression_seed", 11)]
                regression_ok = bool(cases) and all(e["success"] for e in cases)
            score = (
                stars
                + settings["magnet_use_weight"] * summary["magnet_use_rate"]
                + settings["remote_warmth_weight"] * summary["mean_magnet_remote_warmth"]
                - settings["damage_weight"] * summary["mean_damage"]
                - settings["stagnation_weight"] * summary["stagnation_frame_rate"]
            )
            # 하한 미달이면 성공률을 먼저, 다음으로 별 개수를 회복시킵니다.
            tier = 0 if success < threshold else (
                1 if stars < settings["minimum_mean_stars"] or not stagnation_ok or not regression_ok else 2
            )
            value = success if tier == 0 else (stars if tier == 1 else score)
            return (tier, value, success, stars, -summary["mean_damage"],
                    summary.get("mean_score", 0.0), -summary["mean_steps"])
        if objective == "stars":
            threshold = report.get("success_rate_threshold", cfg.EVAL_CONFIG["success_rate_threshold"])
            safe = summary["success_rate"] >= threshold
            stars = summary.get("mean_stars", 0.0)
            return (
                int(safe),
                stars if safe else summary["success_rate"],
                summary["success_rate"] if safe else stars,
                summary.get("mean_score", 0.0),
                -summary["mean_steps"],
            )
        return (
            summary["success_rate"],
            summary.get("mean_score", 0.0),
            summary.get("post_quest_success_rate") or 0.0,
            summary.get("quest_rate", 0.0),
            -summary["mean_steps"],
        )

    # all은 기존처럼 모든 평가의 최저 성공률을 우선합니다.
    d = results["deterministic"]["summary"]
    s = results["stochastic"]["summary"]
    phase = results.get("patrol_phases", {}).get(
        "summary", d
    )

    rates = (
        d["success_rate"],
        s["success_rate"],
        phase["success_rate"],
    )

    return (
        min(rates),
        *rates,
        # 성공률 동점일 때 수집·귀환의 진행 상황도 비교합니다.
        min(
            d.get("post_quest_success_rate") or 0.0,
            s.get("post_quest_success_rate") or 0.0,
        ),
        d.get("post_quest_success_rate") or 0.0,
        s.get("post_quest_success_rate") or 0.0,
        min(d.get("quest_rate", 0.0), s.get("quest_rate", 0.0)),
        d.get("quest_rate", 0.0) + s.get("quest_rate", 0.0),
        d["mean_score"],
        -s["mean_steps"],
    )


def update_stability(peak, bad_count, rate):
    settings = cfg.STABILITY_CONFIG
    peak = max(peak, rate)

    dropped = (
        peak >= settings["activation_success_rate"]
        and peak - rate
        > settings["max_success_drop"] + 1e-9
    )

    bad_count = bad_count + 1 if dropped else 0

    return (
        peak,
        bad_count,
        bad_count >= settings["patience"],
    )


def save_transfer_baseline(model, run_dir, model_dir, seed, *, smoke=False, quick=False):
    """학습 전 정책을 같은 조건으로 평가하고 복원 가능한 기준 모델로 보관합니다."""
    report = evaluate_model(model, smoke=smoke, quick=quick)
    report.update(training_timesteps=0, training_seed=seed,
                  decision_timesteps=0, timestep_unit="physical_frame")
    save_evaluation(report, run_dir / "evaluation" / "step_0")
    model.save(model_dir / "baseline_model.zip")
    model.save(model_dir / "best_model.zip")
    return report


class EpisodeLogger(BaseCallback):
    def __init__(self, total_steps, csv_path=None):
        super().__init__()

        self.total_steps = total_steps
        self.last_print = 0
        self.physical_frames = 0
        self.blocked_move_frames = 0

        self.recent = deque(maxlen=100)
        self.full_recent = deque(maxlen=100)
        self.quest_recent = deque(maxlen=100)

        self.episodes = self.successes = 0
        self.full_episodes = self.full_successes = 0
        self.post_quest_episodes = self.post_quest_successes = 0
        self.total_stars = 0

        self.handle = (
            csv_path.open(
                "w",
                newline="",
                encoding="utf-8-sig",
            )
            if csv_path is not None else None
        )
        self.writer = None

    def _on_step(self):
        self.physical_frames += sum(
            info["repeat_frames"]
            for info in self.locals["infos"]
        )
        self.blocked_move_frames += sum(
            info["blocked_move_frames"]
            for info in self.locals["infos"]
        )

        for rank, info in enumerate(self.locals["infos"]):
            episode = info.get("rl_episode")

            if episode is None:
                continue

            success = bool(episode["success"])
            self.episodes += 1
            self.successes += int(success)
            self.total_stars += episode.get("stars", 0)
            self.recent.append(success)
            self.quest_recent.append({
                "quest_collected": episode["quest_collected"],
                "quest_collected_at_start": episode["quest_collected_at_start"],
                "success": success,
                "blocked_move_frames": episode["blocked_move_frames"],
                "physical_frames": episode["physical_frames"],
                "stars": episode.get("stars", 0),
                "stagnant_frames": episode.get("stagnant_frames", 0),
                "magnet_acquired": episode.get("magnet_acquired", False),
                "magnet_used": episode.get("magnet_used", False),
                "magnet_remote_warmth": episode.get("magnet_remote_warmth", 0),
                "damage_free_success": episode.get("damage_free_success", False),
                "magnet_ten_return_success": episode.get("magnet_ten_return_success", False),
                "curriculum_success": episode.get("curriculum_success", success),
            })
            if (
                episode["quest_collected"]
                and not episode["quest_collected_at_start"]
            ):
                self.post_quest_episodes += 1
                self.post_quest_successes += int(success)

            start = info.get("curriculum_start", 0)

            if start == 0:
                self.full_episodes += 1
                self.full_successes += int(success)
                self.full_recent.append(success)

            row = {
                "total_timesteps": self.physical_frames,
                "decision_timesteps": self.num_timesteps,
                "env_rank": rank,
                "curriculum_start": start,
                "curriculum_frontier": info.get(
                    "curriculum_frontier", 0
                ),
                "curriculum_next_frontier": info.get(
                    "curriculum_next_frontier", 0
                ),
                **episode,
            }

            if self.handle is not None:
                if self.writer is None:
                    self.writer = csv.DictWriter(
                        self.handle,
                        fieldnames=list(row),
                    )
                    self.writer.writeheader()

                self.writer.writerow(row)
                self.handle.flush()

            self.logger.record(
                "task/train_success_rate_100",
                float(np.mean(self.recent)),
            )
            self.logger.record("task/mean_stars_100", float(np.mean([e["stars"] for e in self.quest_recent])))
            self.logger.record("task/curriculum_success_rate_100", float(np.mean([e["curriculum_success"] for e in self.quest_recent])))
            for metric in ("magnet_acquired", "magnet_used", "magnet_remote_warmth", "damage_free_success",
                           "magnet_ten_return_success"):
                self.logger.record(f"task/{metric}_100", float(np.mean([e[metric] for e in self.quest_recent])))

            if self.full_recent:
                self.logger.record(
                    "task/full_start_success_rate_100",
                    float(np.mean(self.full_recent)),
                )

            for name, value in post_quest_metrics(self.quest_recent).items():
                if value is not None:
                    self.logger.record(f"task/{name}_100", value)
            self.logger.record(
                "task/blocked_move_frame_rate_100",
                sum(e["blocked_move_frames"] for e in self.quest_recent)
                / sum(e["physical_frames"] for e in self.quest_recent),
            )
            self.logger.record(
                "task/stagnation_frame_rate_100",
                sum(e["stagnant_frames"] for e in self.quest_recent)
                / sum(e["physical_frames"] for e in self.quest_recent),
            )

            self.logger.record(
                f"curriculum/env_{rank}",
                row["curriculum_next_frontier"],
            )

        if (
            self.physical_frames - self.last_print >= 10_000
            or (
                self.physical_frames >= self.total_steps
                and self.last_print < self.total_steps
            )
        ):
            train_rate = (
                f"{np.mean(self.recent):.1%}"
                if self.recent else "--"
            )
            full_rate = (
                f"{np.mean(self.full_recent):.1%}"
                if self.full_recent else "--"
            )
            recent_blocked = sum(e["blocked_move_frames"] for e in self.quest_recent)
            recent_frames = sum(e["physical_frames"] for e in self.quest_recent)
            blocked_rate = f"{recent_blocked / recent_frames:.1%}" if recent_frames else "--"
            recent_stagnant = sum(e["stagnant_frames"] for e in self.quest_recent)
            stagnant_rate = f"{recent_stagnant / recent_frames:.1%}" if recent_frames else "--"
            stars_label = f"{np.mean([e['stars'] for e in self.quest_recent]):.2f}" if self.quest_recent else "--"
            task_label = f"{np.mean([e['curriculum_success'] for e in self.quest_recent]):.1%}" if self.quest_recent else "--"
            levels = "/".join(
                str(info.get("curriculum_next_frontier", "-"))
                for info in self.locals["infos"]
            )

            print(
                f"[{self.physical_frames:,}/{self.total_steps:,}F] "
                f"학습 {train_rate} | "
                f"평균 별 {stars_label} | "
                f"과제 성공 {task_label} | "
                f"{format_post_quest(post_quest_metrics(self.quest_recent))} | "
                f"처음부터 {full_rate} | 벽막힘 {blocked_rate} | 정체 {stagnant_rate} | 단계 {levels}",
                flush=True,
            )

            self.last_print = self.physical_frames

        return True

    def close(self):
        if self.handle is not None:
            self.handle.close()


def make_vec(seed, quick=False):
    return make_vec_env(
        make_env,
        n_envs=cfg.TRAIN_CONFIG["n_envs"],
        seed=seed,
        vec_env_cls=DummyVecEnv,
        env_kwargs={"curriculum_settings": effective_curriculum_settings(quick)},
    )


def check_setup(seed, quick=False):
    training_targets(False, quick=quick)
    torch.set_num_threads(
        cfg.TRAIN_CONFIG["torch_threads"]
    )

    env = make_vec(seed, quick=quick)

    try:
        model = make_model(env, seed)
        obs = env.reset()

        actions, _ = model.predict(
            obs,
            deterministic=True,
        )
        next_obs, _, _, _ = env.step(actions)

        assert obs.shape == (
            cfg.TRAIN_CONFIG["n_envs"],
            cfg.OBSERVATION_CONFIG["final_size"],
        )
        assert next_obs.shape == obs.shape
        assert (
            env.action_space.n
            == cfg.OBSERVATION_CONFIG["action_count"]
        )
        assert model.num_timesteps == 0
        assert not model.policy.optimizer.state

        print(
            f"연결 정상 | Stage {model.ppo_stage} | "
            f"관측 {obs.shape} | "
            f"전이 원본 {model.transfer_source_timesteps:,}스텝 | "
            f"새 학습 0스텝"
        )
        print("학습·파일 저장 없음")
        print(f"주 평가 기준: {primary_evaluation_mode()}")
        print(f"초기화: {'전이 ' + str(cfg.TRANSFER_CONFIG['model_path']) if cfg.TRANSFER_CONFIG['enabled'] else '새 무작위 가중치'}")
        print(f"선택 기준: {cfg.EVAL_CONFIG['selection_objective']} | "
              f"짧은 확률적 평가 {cfg.EVAL_CONFIG['quick_stochastic_episodes']}판")
        if cfg.STAR_GOAL_CONFIG["enabled"]:
            print(f"추가 온기 목표: {cfg.STAR_GOAL_CONFIG['warmth_target']}점 | 평균 별 목표: {cfg.EVAL_CONFIG.get('target_mean_stars', 0)}")
        if cfg.CURRICULUM_CONFIG["enabled"]:
            print(f"시작 구간: {effective_curriculum_settings(quick)['initial_frontier']}")

    finally:
        env.close()


def train(seed, smoke=False, quick=False):
    targets = training_targets(smoke, quick=quick)
    torch.set_num_threads(
        cfg.TRAIN_CONFIG["torch_threads"]
    )

    stamp = datetime.now(
        timezone(timedelta(hours=9))
    ).strftime("%Y%m%d_%H%M%S_%f")

    run_dir = (
        cfg.RUN_ROOT
        / ("smoke" if smoke else "quick" if quick else "train")
        / f"seed_{seed}_{stamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    model_dir = run_dir / "models"
    model_dir.mkdir()

    env = model = logger = callback = None
    status = "failed"

    best_key = None
    best_step = None
    best_passed = False
    peak = 0.0
    peak_deterministic = 0.0
    bad_count = 0
    quest_bad_count = 0
    best_review = None
    final_passes = final_regressions = 0
    final_verification = None
    stage3_passes = 0

    try:
        env = make_vec(seed, quick=quick)
        model = make_model(env, seed)

        if cfg.LOG_CONFIG["save_config"]:
            write_json(run_dir / "config.json", {
                "algorithm": "PPO",
                "seed": seed,
                "smoke": smoke,
                "quick": quick,
                "planned_timesteps": targets[-1],
                "action_repeat": cfg.ACTION_REPEAT,
                "timestep_unit": "physical_frame",
                "environment_name": cfg.ENV_NAME,
                "environment": cfg.ENV_CONFIG,
                "observation": cfg.OBSERVATION_CONFIG,
                "observation_version": PPO_OBSERVATION_VERSION,
                "action_mask_version": ACTION_MASK_VERSION,
                "curriculum": effective_curriculum_settings(quick),
                "curriculum_version": (
                    training_curriculum_class().version if cfg.CURRICULUM_CONFIG["enabled"] else None
                ),
                "network": cfg.NETWORK_CONFIG,
                "ppo": cfg.PPO_CONFIG,
                "training": cfg.TRAIN_CONFIG,
                "evaluation": cfg.EVAL_CONFIG,
                "joint_review": getattr(model, "ppo_joint_review_config", None),
                "final_stage2": getattr(model, "ppo_final_stage2_config", None),
                "final_stage3": getattr(model, "ppo_final_stage3_config", None),
                "reward": cfg.REWARD_CONFIG,
                "reward_version": cfg.REWARD_VERSION,
                "star_goal": deepcopy(cfg.STAR_GOAL_CONFIG),
                "transfer": cfg.TRANSFER_CONFIG,
                "action_mask": cfg.ACTION_MASK_CONFIG,
                "stagnation": cfg.STAGNATION_CONFIG,
                "transfer_source_timesteps": (
                    model.transfer_source_timesteps
                ),
                "transfer_source_observation_version": model.transfer_source_observation_version,
                "stability": cfg.STABILITY_CONFIG,
                "logging": cfg.LOG_CONFIG,
                "python_executable": sys.executable,
                "versions": {
                    name: version(name)
                    for name in (
                        "torch",
                        "stable-baselines3",
                        "gymnasium",
                        "numpy",
                    )
                },
            })

        formats = ["csv"]

        if cfg.LOG_CONFIG["tensorboard"]:
            formats.append("tensorboard")

        logger = configure(
            str(run_dir / "logs"),
            formats,
        )
        model.set_logger(logger)

        callback = EpisodeLogger(
            targets[-1],
            run_dir / "train_episodes.csv"
            if cfg.LOG_CONFIG["save_episode_csv"] else None,
        )

        print(f"저장 위치: {run_dir}", flush=True)
        print(
            f"주 평가 기준: {primary_evaluation_mode()} | "
            f"통과 기준 {cfg.EVAL_CONFIG['success_rate_threshold']:.1%} | "
            f"평균 별 목표 {cfg.EVAL_CONFIG.get('target_mean_stars', 0):.2f}",
            flush=True,
        )

        if cfg.TRANSFER_CONFIG["enabled"] and cfg.EVAL_CONFIG.get("evaluate_transfer_baseline", False):
            print("[기준 모델 평가] 추가 학습 전 원본을 같은 시드로 평가합니다.", flush=True)
            baseline = save_transfer_baseline(model, run_dir, model_dir, seed, smoke=smoke, quick=quick)
            best_key, best_step, best_passed = evaluation_key(baseline), 0, baseline["passed"]
            best_review = baseline.get("skill_review")
            peak = primary_success_rate(baseline)
            peak_deterministic = baseline["results"]["deterministic"]["summary"]["success_rate"]
            summary = baseline["results"][primary_evaluation_mode(baseline)]["summary"] if primary_evaluation_mode(baseline) != "all" else None
            if summary is not None:
                equipment = (f"무적 획득 {summary.get('invincibility_acquire_rate',0):.1%}" if cfg.ENV_CONFIG['stage']==3
                             else f"자석 활용 {summary['magnet_use_rate']:.1%}")
                print(f"[기준 모델] 성공 {summary['success_rate']:.1%} | 평균 별 {summary['mean_stars']:.2f} | "
                      f"{equipment} | 피해 {summary['mean_damage']:.2f}", flush=True)
            if best_review:
                from rl.ppo_skill_review import format_skill_review
                print(format_skill_review(best_review), flush=True)

        rollout = (
            cfg.TRAIN_CONFIG["n_envs"]
            * cfg.PPO_CONFIG["n_steps"]
        )
        eval_index = 0

        while callback.physical_frames < targets[-1]:
            previous_frames = callback.physical_frames

            model.learn(
                total_timesteps=rollout,
                reset_num_timesteps=False,
                callback=callback,
                log_interval=None,
            )

            frames = callback.physical_frames

            if frames <= previous_frames:
                raise RuntimeError(
                    "물리 프레임이 진행되지 않았습니다."
                )

            model.ppo_physical_frames = frames

            logger.record("time/physical_frames", frames)
            logger.record(
                "time/decision_timesteps",
                model.num_timesteps,
            )

            if frames < targets[eval_index]:
                logger.dump(step=frames)
                continue

            while (
                eval_index < len(targets)
                and targets[eval_index] <= frames
            ):
                eval_index += 1

            target = frames
            model.save(model_dir / f"ppo_{target}.zip")

            print(
                f"[평가 {target:,}] 진행 중...",
                end="\r",
                flush=True,
            )

            report = evaluate_model(model, smoke=smoke, quick=quick)
            report["training_timesteps"] = target
            report["training_seed"] = seed
            report["decision_timesteps"] = int(model.num_timesteps)
            report["timestep_unit"] = "physical_frame"

            save_evaluation(
                report,
                run_dir / "evaluation" / f"step_{target}",
            )

            labels = {
                "deterministic": "결정적",
                "stochastic": "확률적",
                "patrol_phases": "유령 상태",
            }
            parts = []

            for mode, result in report["results"].items():
                summary = result["summary"]
                equipment = (f"무적 획득 {summary.get('invincibility_acquire_rate',0):.1%}, "
                             if cfg.ENV_CONFIG['stage']==3 else f"자석 활용 {summary['magnet_use_rate']:.1%}, ")

                parts.append(
                    f"{labels.get(mode, mode)}"
                    f"({'주' if primary_evaluation_mode(report) in (mode, 'all') else '참고'}) "
                    f"{summary['success_rate']:.1%}, "
                    f"평균 별 {summary['mean_stars']:.2f}, "
                    f"별 3개 {summary['three_star_rate']:.1%}, "
                    f"{format_post_quest(summary)}, "
                    f"벽막힘 {summary['blocked_move_frame_rate']:.1%}, "
                    f"정체 {summary['stagnation_frame_rate']:.1%}, "
                    f"{equipment}"
                    f"무피해 귀환 {summary['damage_free_success_rate']:.1%}"
                )

                for name, value in summary.items():
                    if value is None:
                        continue
                    logger.record(
                        f"eval/{mode}/{name}",
                        value,
                    )

            logger.dump(step=target)

            if report.get("skill_review"):
                from rl.ppo_skill_review import format_skill_review
                parts.append(format_skill_review(report["skill_review"]))
            print(
                f"[평가 {target:,}] " + " | ".join(parts),
                flush=True,
            )

            # 평가 기준 변경이 자동으로 탐색 강도 변경까지 일으키지 않도록 구분합니다.
            if (
                not smoke and not quick and report["passed"]
                and cfg.STABILITY_CONFIG.get("reduce_entropy_after_success", False)
            ):
                model.ent_coef = cfg.STABILITY_CONFIG[
                    "ent_coef_after_success"
                ]

            key = evaluation_key(report)

            if best_key is None or key > best_key:
                best_key = key
                best_step = target
                best_passed = report["passed"]
                best_review = report.get("skill_review")
                model.save(model_dir / "best_model.zip")

            if report.get("skill_review") and not smoke:
                from rl.ppo_skill_review import quest_regression_status
                quest_bad_count, quest_stop = quest_regression_status(
                    quest_bad_count, report["skill_review"],
                    model.ppo_joint_review_config["stop_after_quest_regressions"],
                )
                if quest_stop:
                    status = "quest_retention_stopped"
                    print("[중단] 물건 획득 복습 성능이 연속 하락해 최적 모델을 보관합니다.", flush=True)
                    break

            final_settings = getattr(model, "ppo_final_stage2_config", {})
            if not smoke and final_settings.get("enabled", False):
                from rl.ppo_safe_return import final_training_status
                final_passes, final_regressions, ready, regressed = final_training_status(
                    report, frames, final_passes, final_regressions, final_settings)
                if ready or regressed:
                    status = "performance_ready" if ready else "performance_retention_stopped"
                    print(f"[중단] {status} | 통과 연속 {final_passes}회 | 하한 미달 연속 {final_regressions}회", flush=True)
                    break

            stage3_settings = getattr(model,'ppo_final_stage3_config',{})
            if not smoke and stage3_settings.get('enabled',False):
                from rl.ppo_safe_return import performance_ready
                stage3_passes = stage3_passes + 1 if performance_ready(report) else 0
                if (frames >= stage3_settings['minimum_training_frames']
                        and stage3_passes >= stage3_settings['required_consecutive_passes']):
                    status = 'stage3_performance_ready'
                    print('[중단] Stage 3 실제 시작점의 별 3개 기준을 두 번 연속 통과했습니다.',flush=True)
                    break

            if not smoke:
                rate = primary_success_rate(report)
                peak_deterministic = max(
                    peak_deterministic,
                    report["results"]["deterministic"]["summary"]["success_rate"],
                )

                peak, bad_count, stop = update_stability(
                    peak,
                    bad_count,
                    rate,
                )

                if stop and not quick:
                    status = "early_stopped"
                    print(
                        "[중단] 성공률 하락이 연속 발생해 "
                        "최적 모델을 선택합니다.",
                        flush=True,
                    )
                    break

        else:
            status = "complete"

        # 마지막 시점의 모델과 평가상 최적 모델을 구분합니다.
        model.save(model_dir / "last_model.zip")

        shutil.copyfile(
            model_dir / "best_model.zip",
            model_dir / "final_model.zip",
        )
        if not smoke and (getattr(model, "ppo_final_stage2_config", {}).get("enabled", False)
                          or getattr(model,'ppo_final_stage3_config',{}).get('enabled',False)):
            from rl.ppo_safe_return import verify_stage_model
            print("[최종 검증] 선택한 모델 하나를 새 시드 200판으로 확인합니다. 학습 없음.", flush=True)
            final_verification = verify_stage_model(model_dir / "best_model.zip", run_dir / "final_verification", model.ppo_stage)
            result = final_verification["results"]["stochastic"]["summary"]
            print(f"[최종 검증] 통과 {final_verification['passed']} | 성공 {result['success_rate']:.1%} | "
                  f"별 3개 {result['three_star_rate']:.1%} | 평균 별 {result['mean_stars']:.2f} | "
                  f"평균 피해 {result['mean_damage']:.2f}", flush=True)

    except KeyboardInterrupt:
        status = "interrupted"

        if model is not None:
            model.save(model_dir / "interrupted_model.zip")

        raise

    except Exception:
        status = "failed"
        raise

    finally:
        try:
            write_json(run_dir / "training_summary.json", {
                "status": status,
                "smoke": smoke,
                "quick": quick,
                "total_timesteps": (
                    callback.physical_frames if callback else 0
                ),
                "decision_timesteps": (
                    int(model.num_timesteps) if model else 0
                ),
                "action_repeat": cfg.ACTION_REPEAT,
                "selected_model_timesteps": best_step,
                "blocked_move_frames": (
                    callback.blocked_move_frames if callback else 0
                ),
                "blocked_move_frame_rate": (
                    callback.blocked_move_frames / callback.physical_frames
                    if callback and callback.physical_frames else None
                ),
                "best_passed": best_passed,
                "selected_skill_review": best_review,
                "quest_regression_count": quest_bad_count,
                "final_stage2_passed": final_verification['passed'] if final_verification and model.ppo_stage==2 else None,
                "final_stage3_passed": final_verification['passed'] if final_verification and model.ppo_stage==3 else None,
                "final_verification_file": "final_verification/evaluation.json" if final_verification else None,
                "final_consecutive_passes": final_passes,
                "stage3_consecutive_passes": stage3_passes,
                "final_regression_count": final_regressions,
                "primary_evaluation_mode": primary_evaluation_mode(),
                "selection_objective": cfg.EVAL_CONFIG.get("selection_objective", "success"),
                "target_mean_stars": cfg.EVAL_CONFIG.get("target_mean_stars", 0.0),
                "training_mean_stars": callback.total_stars / callback.episodes if callback and callback.episodes else None,
                "peak_primary_success_rate": peak,
                "peak_deterministic_success_rate": peak_deterministic,
                "consecutive_drops": bad_count,
                "episodes": (
                    callback.episodes if callback else 0
                ),
                "successes": (
                    callback.successes if callback else 0
                ),
                "full_start_episodes": (
                    callback.full_episodes if callback else 0
                ),
                "full_start_successes": (
                    callback.full_successes if callback else 0
                ),
                "post_quest_episode_count": (
                    callback.post_quest_episodes if callback else 0
                ),
                "post_quest_success_count": (
                    callback.post_quest_successes if callback else 0
                ),
                "post_quest_success_rate": (
                    callback.post_quest_successes / callback.post_quest_episodes
                    if callback and callback.post_quest_episodes else None
                ),
            })

        finally:
            if callback is not None:
                callback.close()
            if env is not None:
                env.close()
            if logger is not None:
                logger.close()

    print(
        f"[완료] 실행 {callback.physical_frames:,}프레임 | "
        f"선택 {best_step:,}프레임 | {status}"
    )

    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="RealtimeMansionEnv PPO 학습"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=cfg.TRAIN_CONFIG["default_seed"],
    )
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--quest-only", action="store_true", help="Stage 2 물건 획득 부분 과제만 연습")
    parser.add_argument("--connection-only", action="store_true", help="초반 자석·물건 연결 과제와 6·7번 복습")
    parser.add_argument("--from-quest", type=Path, help="구간별 새 시드 평가 통과 후 물건 연습 모델로 전체 과제 복귀")
    parser.add_argument("--from-connection", type=Path, help="새 시드 연결 검증 후 전체 별 3개 과제로 복귀")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument(
        "--quick", action="store_true",
        help="quick_timesteps에 지정한 물리 프레임 예산으로 실험합니다.",
    )
    args = parser.parse_args()

    if not 0 <= args.seed < 2**31:
        parser.error("seed는 0 이상 2**31 미만이어야 합니다.")

    if args.connection_only:
        if args.quest_only or args.from_quest or args.from_connection or args.smoke:
            parser.error("--connection-only는 다른 부분 과제·복귀 옵션·--smoke와 함께 사용할 수 없습니다.")
        from rl.train_ppo_connection import check_setup as check_connection, train as train_connection
        if args.check:
            check_connection(args.seed)
        else:
            train_connection(args.seed, quick=args.quick)
        return

    if args.quest_only:
        if args.from_quest or args.from_connection or args.smoke:
            parser.error("--quest-only는 --from-quest 또는 --smoke와 함께 사용할 수 없습니다.")
        from rl.train_ppo_quest import check_setup as check_quest, train as train_quest
        if args.check:
            check_quest(args.seed)
        else:
            train_quest(args.seed, quick=args.quick)
        return

    if args.from_quest and args.from_connection:
        parser.error("복귀 모델 옵션은 하나만 지정하세요.")
    if args.from_connection:
        from rl.train_ppo_connection import require_connection_ready
        require_connection_ready(args.from_connection)
        cfg.TRANSFER_CONFIG.update(enabled=True, model_path=args.from_connection, action_head_scale=1.0)
    elif args.from_quest:
        from rl.train_ppo_quest import require_full_return_ready
        require_full_return_ready(args.from_quest)
        cfg.TRANSFER_CONFIG.update(enabled=True, model_path=args.from_quest, action_head_scale=1.0)
    elif (cfg.ENV_CONFIG["stage"] == 2
          and cfg.CURRICULUM_CONFIG["name"] in ("stage2_joint_review_v3", "stage2_safe_return_v1")
          and cfg.JOINT_REVIEW_CONFIG.get("require_quest_source_ready", False)):
        from rl.ppo_skill_review import require_joint_source_ready
        require_joint_source_ready(cfg.TRANSFER_CONFIG["model_path"])

    if args.check:
        check_setup(args.seed, quick=args.quick)
    else:
        train(args.seed, smoke=args.smoke, quick=args.quick)


if __name__ == "__main__":
    main()
