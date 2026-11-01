### reset()에서 기준 상태 초기화, step()에서 보상 계산

from copy import deepcopy
import math

import gymnasium as gym

from rl.ppo_config import REWARD_CONFIG, REWARD_VERSION, STAR_GOAL_CONFIG
from rl.ppo_goal import bonus_collection_safe, select_goal
from rl.ppo_stagnation import StagnationTracker
from rl.ppo_path_distance import path_distance


# 매 스텝의 보상과 에피소드 누적 보상을 기록할 항목입니다.
REWARD_TERM_NAMES = (
    "quest_reward",
    "quest_progress_reward",
    "warmth_reward",
    "magnet_warmth_reward",
    "magnet_pickup_reward",
    "invincibility_pickup_reward",
    "warmth_progress_reward",
    "return_progress_reward",
    "success_reward",
    "star_completion_adjustment",
    "failure_penalty",
    "damage_penalty",
    "ghost_risk_penalty",
    "damage_free_success_reward",
    "magnet_ten_success_reward",
    "blocked_move_penalty",
    "time_penalty",
    "stagnation_penalty",
)


class PPOReward(gym.Wrapper):
    """기본 환경 보상을 PPO 전용 보상으로 대체합니다."""

    def __init__(self, env):
        super().__init__(env)

        # 원본 설정과 분리된 복사본을 사용합니다.
        self.reward_config = deepcopy(REWARD_CONFIG)
        self.reward_version = REWARD_VERSION
        self.star_goal = deepcopy(getattr(env, "star_goal", STAR_GOAL_CONFIG))
        stage = env.unwrapped.core.config["stage"]
        self._three_star_goal = bool(
            self.star_goal.get("enabled", False)
            and self.star_goal.get(f"stage{stage}_three_star", False)
        )

    def _warmth_value(self, snapshot):
        """현재 보유 온기에 해당하는 보상 기준값을 계산합니다."""
        cfg = self.reward_config

        warmth = min(snapshot["carried"], cfg["warmth_cap"])
        surplus = max(0, warmth - snapshot["target"])

        value = (
            cfg["warmth_per_point"] * warmth
            + cfg["surplus_warmth_per_point"] * surplus
        )
        if snapshot["stage"] in (2,3):
            # 기본 4점까지 보강하고 이후 점당 보상은 기존 척도를 유지합니다.
            value += (cfg.get(f"stage{snapshot['stage']}_basic_warmth_per_point", cfg["warmth_per_point"])
                      - cfg["warmth_per_point"]) * min(warmth, snapshot["target"])
        return value

    def _set_collection_target(self, snapshot, info):
        self._previous_warmth_target = None
        self._previous_collection_distance = None
        if info.get("goal_phase") == 1:
            if self.star_goal.get("collection_path_reward", False):
                settings = {**self.star_goal, "path_waypoints": False}
                self._previous_warmth_target = list(select_goal(snapshot, settings)[1])
                self._previous_collection_distance = path_distance(snapshot, self._previous_warmth_target)
            else:
                self._previous_warmth_target = list(info["goal_position"])

    def _quest_distance(self, snapshot):
        """물건의 고정 위치와 현재 연속 좌표 사이의 맨해튼 거리입니다."""
        position = snapshot["motion"]["position"]
        target = snapshot["quest"]["position"]
        if self._three_star_goal:
            return path_distance(snapshot, target)
        return sum(abs(position[i] - target[i]) for i in (0, 1))

    def _return_distance(self, snapshot):
        """귀환 준비 후 출구까지의 거리. Stage 2는 별 3개 목표일 때 적용합니다."""
        if (
            (snapshot["stage"] not in (1, 3) and not self._three_star_goal)
            or not snapshot["quest"]["collected"]
            or snapshot["carried"] < snapshot["target"]
        ):
            return None

        safe = bonus_collection_safe(snapshot,self.star_goal)
        desired = self.star_goal.get('warmth_target', snapshot['target'])
        if snapshot['stage']==3 and self.star_goal.get('stage3_route_plan',False):
            from rl.ppo_stage3_plan import stage3_warmth_target
            desired = stage3_warmth_target(snapshot,self.star_goal,safe)
        if (self._three_star_goal and safe and snapshot['carried'] < desired):
            return None

        position = snapshot["motion"]["position"]
        if snapshot["stage"] == 3 or self._three_star_goal:
            return path_distance(snapshot, snapshot["exit"])
        return sum(abs(position[i] - snapshot["exit"][i]) for i in (0, 1))

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        snapshot = info["snapshot"]

        # 시작 상태를 기준으로 새로 얻은 것만 계산합니다.
        self._previous_quest = snapshot["quest"]["collected"]
        self._previous_quest_distance = (
            None if self._previous_quest else self._quest_distance(snapshot)
        )
        self._quest_collected_at_start = self._previous_quest
        self._previous_goal_phase = info.get("goal_phase")
        self._previous_warmth_value = self._warmth_value(snapshot)
        self._best_return_distance = self._return_distance(snapshot)
        self._previous_return_distance = self._best_return_distance
        self._previous_position = list(snapshot["motion"]["position"])
        self._stagnation = StagnationTracker(self._previous_position)
        self._previous_carried = snapshot["carried"]
        magnet = snapshot.get("magnet") or {}
        self._magnet_available = bool(magnet)
        self._magnet_collected_at_start = bool(magnet.get("collected", False))
        self._magnet_acquired = False
        self._invincibility_acquired = False
        self._invincibility_at_start = bool((snapshot.get('invincibility') or {}).get('collected',False))
        self._magnet_remote_warmth = 0
        self._magnet_remote_warmth_at_start = info.get("curriculum_magnet_remote_warmth_at_start", 0)
        self._set_collection_target(snapshot, info)

        self._steps = 0
        self._blocked_move_frames = 0
        self._totals = dict.fromkeys(REWARD_TERM_NAMES, 0.0)

        return obs, info

    def step(self, action):
        # 기본 환경 보상은 사용하지 않습니다.
        obs, _original_reward, terminated, truncated, info = (
            self.env.step(action)
        )

        snapshot = info["snapshot"]
        events = info["events"]
        outcome = info["outcome"] or {}
        cfg = self.reward_config

        self._steps += 1
        terms = dict.fromkeys(REWARD_TERM_NAMES, 0.0)

        # 프레임당 시간 비용은 정상 이동·대기·상호작용·마지막 프레임에 모두 적용합니다.
        terms["time_penalty"] = cfg.get("time_per_frame", 0.0)
        if snapshot['stage'] == 3:
            terms['time_penalty'] = cfg.get('stage3_time_per_frame',terms['time_penalty'])

        # 획득 프레임까지 기존 물건 위치로 거리를 비교합니다.
        # 획득 후 출구로 목표가 바뀌어도 이 항목으로 귀환 이동을 감점하지 않습니다.
        if (not self._previous_quest
                and (not self._three_star_goal or self._previous_goal_phase == 0)):
            current_distance = self._quest_distance(snapshot)
            progress = (self._previous_quest_distance - current_distance
                        if self._previous_quest_distance is not None and current_distance is not None else 0.0)
            scale = cfg.get("quest_progress_per_cell", 0.0)
            if snapshot["stage"] in (2,3):
                scale = cfg.get(f"stage{snapshot['stage']}_quest_progress_per_cell", scale)
            terms["quest_progress_reward"] = scale * progress

        # 1. 필수 물건을 이번 행동에서 새로 얻었을 때만 지급합니다.
        if (
            snapshot["quest"]["collected"]
            and not self._previous_quest
        ):
            terms["quest_reward"] = cfg.get(f"stage{snapshot['stage']}_quest",cfg['quest'])
        if (snapshot['stage'] == 3 and not self._invincibility_at_start and not self._invincibility_acquired
                and any(event['type']=='invincibility_collect' for event in events)):
            self._invincibility_acquired = True
            terms['invincibility_pickup_reward'] = cfg.get('stage3_invincibility_pickup',0.)

        # 2. 보유 온기의 변화에 해당하는 보상을 계산합니다.
        warmth_value = self._warmth_value(snapshot)

        terms["warmth_reward"] = (
            warmth_value - self._previous_warmth_value
        )

        # magnetic=True는 자석 활성 중 같은 칸 수집에도 붙으므로 위치도 비교합니다.
        remote_warmth = sum(
            event["value"] for event in events
            if event["type"] == "collect" and event.get("magnetic", False)
            and event["source_position"] != event["position"]
        )
        self._magnet_remote_warmth += remote_warmth
        acquired_now = any(event["type"] == "magnet_collect" for event in events)
        if acquired_now and not self._magnet_acquired and not self._magnet_collected_at_start:
            terms["magnet_pickup_reward"] = cfg.get("stage2_magnet_pickup", 0.0) if snapshot["stage"] == 2 else 0.0
        self._magnet_acquired |= acquired_now
        gain = snapshot["carried"] - self._previous_carried
        if snapshot["stage"] == 2 and gain > 0 and remote_warmth > 0:
            # 같은 온기에 기본 보상과 자석 보상을 중복 지급하지 않습니다.
            share = max(0.0, terms["warmth_reward"]) * min(remote_warmth, gain) / gain
            terms["warmth_reward"] -= share
            terms["magnet_warmth_reward"] = (
                share * cfg.get("magnet_remote_warmth_multiplier", 1.0)
            )

        # 이전 프레임의 같은 온기 목표로 비교해, 아이템 획득 후 목표 교체로
        # 거리 보상이 갑자기 튀는 것을 방지합니다.
        if self._previous_warmth_target is not None:
            target = self._previous_warmth_target
            position = snapshot["motion"]["position"]
            before = sum(abs(self._previous_position[i] - target[i]) for i in (0, 1))
            after = sum(abs(position[i] - target[i]) for i in (0, 1))
            if self.star_goal.get("collection_path_reward", False):
                before = self._previous_collection_distance
                after = path_distance(snapshot, target)
            terms["warmth_progress_reward"] = (
                cfg.get("warmth_progress_per_cell", 0.0) * (before - after)
                if before is not None and after is not None else 0.0
            )

        # 별 3개 목표는 경로 거리의 증감 모두 반영해 잘못된 방향을 구분합니다.
        # 이전 기본 귀환 실험은 최소 거리 갱신 방식을 유지합니다.
        distance = self._return_distance(snapshot)
        if snapshot["stage"] == 3 or self._three_star_goal:
            if distance is not None and self._previous_return_distance is not None:
                coefficient = cfg.get("return_progress_per_cell", 0.0)
                if (snapshot["stage"] == 3
                        and self.star_goal.get('stage3_waypoint_mode') == 'corridor_lookahead_v1'):
                    coefficient = cfg.get("stage3_return_progress_per_cell", coefficient)
                terms["return_progress_reward"] = coefficient * (
                    self._previous_return_distance - distance
                )
            self._previous_return_distance = distance
        elif distance is not None:
            if self._best_return_distance is not None:
                progress = max(0.0, self._best_return_distance - distance)
                terms["return_progress_reward"] = (
                    cfg.get("return_progress_per_cell", 0.0) * progress
                )
                distance = min(self._best_return_distance, distance)
            self._best_return_distance = distance

        # 3. 게임 코어의 최종 성공 결과를 확인합니다.
        if terminated and outcome.get("status") == "COMPLETE":
            score = outcome["score"]

            terms["success_reward"] = (
                cfg["success"]
                + score["total"] / cfg["score_divisor"]
                + cfg["star_bonus"] * score["stars"]
            )
            if (
                snapshot["stage"] == 2
                and self.star_goal.get("enabled", False)
                and not self._three_star_goal
            ):
                adjustment = cfg.get("star_completion_adjustments", [0.0] * 4)[score["stars"]]
                # 무리한 수집 때문에 생존을 포기하도록 만들지 않습니다.
                if adjustment >= 0 or bonus_collection_safe(snapshot, self.star_goal):
                    terms["star_completion_adjustment"] = adjustment
            elif self._three_star_goal:
                terms["star_completion_adjustment"] = cfg.get(
                    f"stage{snapshot['stage']}_star_completion_adjustments", [0.] * 4
                )[score["stars"]]

        # 4. 게임 자체의 시간 초과·용기 소진에만 벌점을 줍니다.
        elif (
            terminated
            and outcome.get("status") == "FAILED"
            and outcome.get("reason") in ("timeout", "courage")
        ):
            terms["failure_penalty"] = cfg["failure"]

        # 5. 단순 접촉이 아닌 실제 피해 이벤트를 셉니다.
        damage_count = sum(
            event["type"] == "damage"
            for event in events
        )

        damage_cost = cfg.get("stage2_damage", cfg["damage"]) if snapshot["stage"] == 2 else cfg["damage"]
        if snapshot["stage"] == 3:
            damage_cost = cfg.get("stage3_damage", damage_cost)
        terms["damage_penalty"] = damage_cost * damage_count
        if snapshot["stage"] == 2 and snapshot.get("protection", 0) == 0:
            positions = snapshot.get("motion", {}).get("ghost_positions", {})
            radius = cfg.get("stage2_ghost_risk_distance", 0.0)
            if positions and radius > 0:
                player = snapshot["motion"]["position"]
                distance = min(math.dist(player, point) for point in positions.values())
                terms["ghost_risk_penalty"] = cfg.get("stage2_ghost_risk_per_frame", 0.0) * max(0., 1. - distance / radius)
        if (terminated and outcome.get("status") == "COMPLETE"
                and snapshot["stage"] == 2 and snapshot["damage_count"] == 0
                and (not cfg.get("stage2_damage_free_requires_three_stars", False)
                     or outcome["score"]["stars"] == 3)):
            terms["damage_free_success_reward"] = cfg.get("stage2_damage_free_success", 0.0)
        if (terminated and outcome.get('status')=='COMPLETE' and snapshot['stage']==3
                and snapshot['damage_count']==0 and outcome['score']['stars']==3):
            terms['damage_free_success_reward'] = cfg.get('stage3_damage_free_success',0.)
        magnet_ten_success = bool(
            terminated and outcome.get("status") == "COMPLETE" and snapshot["stage"] == 2
            and snapshot["carried"] >= 10
            and self._magnet_remote_warmth_at_start + self._magnet_remote_warmth > 0
        )
        if magnet_ten_success:
            terms["magnet_ten_success_reward"] = cfg.get("magnet_ten_success", 0.0)

        # 실제 물리 이동 여부를 사용하므로 코너 보정 이동도 정상 이동입니다.
        # 상호작용(4)·대기(5)는 이동 명령이 아니므로 이 벌점을 주지 않습니다.
        blocked_move = 0 <= int(action) <= 3 and not snapshot["motion"]["moving"]
        if blocked_move:
            self._blocked_move_frames += 1
            terms["blocked_move_penalty"] = cfg.get("blocked_move_per_frame", 0.0)

        stagnant, local_loop = self._stagnation.update(
            snapshot["motion"]["position"],
            made_progress=(snapshot["carried"] > self._previous_carried
                           or (snapshot["quest"]["collected"] and not self._previous_quest)),
        )
        if stagnant:
            terms["stagnation_penalty"] = cfg.get("stagnation_per_frame", 0.0)
        self._previous_carried = snapshot["carried"]

        reward = float(sum(terms.values()))

        # 에피소드 전체의 항목별 보상을 누적합니다.
        for name, value in terms.items():
            self._totals[name] += value

        # 다음 행동에서 비교할 기준을 갱신합니다.
        self._previous_quest = snapshot["quest"]["collected"]
        self._previous_goal_phase = info.get("goal_phase")
        self._previous_quest_distance = (
            None if self._previous_quest else self._quest_distance(snapshot)
        )
        self._previous_warmth_value = warmth_value
        self._previous_position = list(snapshot["motion"]["position"])
        self._set_collection_target(snapshot, info)

        info = {
            **info,
            "reward_terms": terms,
            "reward_version": self.reward_version,
            "blocked_move": blocked_move,
            "stagnant": stagnant,
            "local_loop": local_loop,
        }

        # 종료 또는 외부 중단 시 최종 통계를 남깁니다.
        if terminated or truncated:
            score = outcome.get("score", {})

            info["rl_episode"] = {
                "success": (
                    terminated
                    and outcome.get("status") == "COMPLETE"
                ),
                "curriculum_success": info.get(
                    "curriculum_task_success",
                    terminated and outcome.get("status") == "COMPLETE"
                    and (not self._three_star_goal or score.get("stars", 0) == 3),
                ),
                "status": outcome.get(
                    "status",
                    "TRUNCATED" if truncated else "UNKNOWN",
                ),
                "reason": outcome.get(
                    "reason",
                    info.get("truncation_reason", "unknown"),
                ),
                "score": score.get("total", 0),
                "stars": score.get("stars", 0),
                "quest_collected": snapshot["quest"]["collected"],
                "quest_collected_at_start": self._quest_collected_at_start,
                "warmth": snapshot["carried"],
                "conditions_met": bool(
                    snapshot["quest"]["collected"]
                    and snapshot["carried"] >= snapshot["target"]
                ),
                "damage_count": snapshot["damage_count"],
                "damage_free_success": bool(terminated and outcome.get("status") == "COMPLETE"
                                            and snapshot["damage_count"] == 0),
                "magnet_available": self._magnet_available,
                "magnet_collected_at_start": self._magnet_collected_at_start,
                "magnet_acquired": self._magnet_acquired,
                "magnet_remote_warmth": self._magnet_remote_warmth,
                "magnet_remote_warmth_at_start": self._magnet_remote_warmth_at_start,
                "magnet_ten_return_success": magnet_ten_success,
                "curriculum_phase": info.get("curriculum_phase", "full_start"),
                "curriculum_ghost_wait_frames": info.get("curriculum_ghost_wait_frames", 0),
                "magnet_used": self._magnet_remote_warmth > 0,
                "invincibility_acquired": self._invincibility_acquired,
                "invincibility_collected_at_start": self._invincibility_at_start,
                "remaining_ticks": snapshot["remaining_ticks"],
                "courage": snapshot["courage"],
                "steps": self._steps,
                "blocked_move_frames": self._blocked_move_frames,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "return": float(sum(self._totals.values())),
                **self._stagnation.summary(),
                **self._totals,
            }

        return obs, reward, terminated, truncated, info

"""
| 기록 이름 | 발생 조건 | 보상 |
|---|---|---:|
| `quest_reward` | 필수 물건을 새로 획득 | `+1.0` |
| `quest_progress_reward` | 획득 전 물건까지의 거리 감소/증가 | 거리 감소량 × 설정 계수, 멀어지면 음수 |
| `warmth_reward` | 목표 온기까지 수집 | 온기 1점당 `+0.1` |
| `warmth_reward` | 기본 온기 4점을 초과해 수집 | 현재 초과 온기 1점당 합계 `+1.1`, 총 14점 상한 |
| `warmth_progress_reward` | 이전 온기 목표까지의 거리 감소/증가 | 거리 감소량 × `0.25` |
| `return_progress_reward` | 별 3개 수집 후 귀환 | 경로 거리 감소/증가 × 설정 계수 |
| `success_reward` | 최종 결과가 `COMPLETE` | `10 + 게임 점수 / 10 + 별 개수 × 6` |
| `star_completion_adjustment` | 이전 Stage 2 기본 목표의 별 0/1/2/3개 성공 | `-8 / -6 / +6 / +10`, 위험 시 음수 조정 면제 |
| `star_completion_adjustment` | 별 3개 목표의 Stage 1·2·3 성공 | 각 Stage 보상표 사용, 현재 `-8 / -6 / -3 / +10` |
| `failure_penalty` | 시간 초과 또는 용기 소진 | `-2.0` |
| `damage_penalty` | 실제 피해 발생 | 각 Stage 설정 사용, 현재 Stage 2는 회당 `-1.0` |
| `blocked_move_penalty` | 이동 명령에도 실제 움직임이 없는 물리 프레임 | 설정 계수, 기본 `-0.01` |
| `time_penalty` | 모든 물리 프레임 | 기본 `-0.001` |

"""
