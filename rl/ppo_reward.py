### reset()에서 기준 상태 초기화, step()에서 보상 계산

from copy import deepcopy

import gymnasium as gym

from rl.ppo_config import REWARD_CONFIG, REWARD_VERSION


# 매 스텝의 보상과 에피소드 누적 보상을 기록할 항목입니다.
REWARD_TERM_NAMES = (
    "quest_reward",
    "warmth_reward",
    "success_reward",
    "failure_penalty",
    "damage_penalty",
)


class PPOReward(gym.Wrapper):
    """기본 환경 보상을 PPO 전용 보상으로 대체합니다."""

    def __init__(self, env):
        super().__init__(env)

        # 원본 설정과 분리된 복사본을 사용합니다.
        self.reward_config = deepcopy(REWARD_CONFIG)
        self.reward_version = REWARD_VERSION

    def _warmth_value(self, snapshot):
        """현재 보유 온기에 해당하는 보상 기준값을 계산합니다."""
        cfg = self.reward_config

        warmth = min(snapshot["carried"], cfg["warmth_cap"])
        surplus = max(0, warmth - snapshot["target"])

        return (
            cfg["warmth_per_point"] * warmth
            + cfg["surplus_warmth_per_point"] * surplus
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        snapshot = info["snapshot"]

        # 시작 상태를 기준으로 새로 얻은 것만 계산합니다.
        self._previous_quest = snapshot["quest"]["collected"]
        self._previous_warmth_value = self._warmth_value(snapshot)

        self._steps = 0
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

        # 1. 필수 물건을 이번 행동에서 새로 얻었을 때만 지급합니다.
        if (
            snapshot["quest"]["collected"]
            and not self._previous_quest
        ):
            terms["quest_reward"] = cfg["quest"]

        # 2. 보유 온기의 변화에 해당하는 보상을 계산합니다.
        warmth_value = self._warmth_value(snapshot)

        terms["warmth_reward"] = (
            warmth_value - self._previous_warmth_value
        )

        # 3. 게임 코어의 최종 성공 결과를 확인합니다.
        if terminated and outcome.get("status") == "COMPLETE":
            score = outcome["score"]

            terms["success_reward"] = (
                cfg["success"]
                + score["total"] / cfg["score_divisor"]
                + cfg["star_bonus"] * score["stars"]
            )

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

        terms["damage_penalty"] = cfg["damage"] * damage_count

        reward = float(sum(terms.values()))

        # 에피소드 전체의 항목별 보상을 누적합니다.
        for name, value in terms.items():
            self._totals[name] += value

        # 다음 행동에서 비교할 기준을 갱신합니다.
        self._previous_quest = snapshot["quest"]["collected"]
        self._previous_warmth_value = warmth_value

        info = {
            **info,
            "reward_terms": terms,
            "reward_version": self.reward_version,
        }

        # 종료 또는 외부 중단 시 최종 통계를 남깁니다.
        if terminated or truncated:
            score = outcome.get("score", {})

            info["rl_episode"] = {
                "success": (
                    terminated
                    and outcome.get("status") == "COMPLETE"
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
                "warmth": snapshot["carried"],
                "conditions_met": bool(
                    snapshot["quest"]["collected"]
                    and snapshot["carried"] >= snapshot["target"]
                ),
                "damage_count": snapshot["damage_count"],
                "remaining_ticks": snapshot["remaining_ticks"],
                "courage": snapshot["courage"],
                "steps": self._steps,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "return": float(sum(self._totals.values())),
                **self._totals,
            }

        return obs, reward, terminated, truncated, info

"""
| 기록 이름 | 발생 조건 | 보상 |
|---|---|---:|
| `quest_reward` | 필수 물건을 새로 획득 | `+1.0` |
| `warmth_reward` | 목표 온기까지 수집 | 온기 1점당 `+0.1` |
| `warmth_reward` | 목표 온기를 초과해 수집 | 초과 온기 1점당 합계 `+0.4` |
| `success_reward` | 최종 결과가 `COMPLETE` | `10 + 게임 점수 / 20 + 별 개수 × 2` |
| `failure_penalty` | 시간 초과 또는 용기 소진 | `-2.0` |
| `damage_penalty` | 실제 피해 발생 | 피해 1회당 `-0.1` |

"""