# MansionEnv의 보상을 첫 수집·성공 귀환 중심의 공통 보상
from __future__ import annotations

import gymnasium as gym

REWARD_VERSION = "stage1_collection_return_score20_v2"
REWARD_CONFIG = {
    "quest": 1.0, "warmth_per_point": 0.1, "warmth_cap": 4,
    "success": 10.0, "score_divisor": 20.0,
    "failure": -2.0, "damage": -0.1,
}
TERM_NAMES = ("quest_reward", "warmth_reward", "success_reward",
              "failure_penalty", "damage_penalty")


class ProposedReward(gym.Wrapper):
    """게임 판정은 원본에 맡기고, 학습에 반환하는 보상만 팀원 협의 기준"""

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.previous = info["snapshot"]
        self.totals = dict.fromkeys(TERM_NAMES, 0.0)
        self.steps = self.blocked = self.ineffective_interact = self.waits = 0
        self.quest_step = self.warmth_step = self.conditions_met_step = None
        return obs, info

    def step(self, action):
        # 원래 보상을 더하면 성공 보상이 중복되므로 의도적으로 사용하지 않음
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        now, events = info["snapshot"], info["events"]
        self.steps += 1
        terms = dict.fromkeys(TERM_NAMES, 0.0)
        cfg = REWARD_CONFIG
        if now["quest"]["collected"] and not self.previous["quest"]["collected"]:
            terms["quest_reward"] = cfg["quest"]
            self.quest_step = self.steps
        # 3→6점 획득도 처음 4점 중 남은 1점만 보상, 총 중간 온기 보상은 +0.4
        increase = min(now["carried"], cfg["warmth_cap"]) - min(
            self.previous["carried"], cfg["warmth_cap"])
        terms["warmth_reward"] = cfg["warmth_per_point"] * increase
        if now["carried"] >= now["target"] and self.warmth_step is None:
            self.warmth_step = self.steps
        conditions = now["quest"]["collected"] and now["carried"] >= now["target"]
        if conditions and self.conditions_met_step is None:
            self.conditions_met_step = self.steps

        outcome = info["outcome"] or {}
        # 귀환 요청·문 도착이 아닌, 피해와 시간 처리를 마친 최종 게임 결과를 사용
        if terminated and outcome.get("status") == "COMPLETE":
            terms["success_reward"] = cfg["success"] + outcome["score"]["total"] / cfg["score_divisor"]
        elif terminated and outcome.get("status") == "FAILED" and outcome.get("reason") in ("timeout", "courage"):
            terms["failure_penalty"] = cfg["failure"]
        terms["damage_penalty"] = cfg["damage"] * sum(e["type"] == "damage" for e in events)

        self.blocked += sum(e["type"] == "blocked" for e in events)
        effective_e = any(e["type"] in ("quest_collect", "open", "return") for e in events)
        self.ineffective_interact += int(int(action) == 4 and not effective_e)
        self.waits += int(int(action) == 5)
        for name, value in terms.items():
            self.totals[name] += value
        info["reward_terms"] = terms
        info["reward_version"] = REWARD_VERSION
        self.previous = now

        if terminated or truncated:
            score = outcome.get("score", {})
            success = outcome.get("status") == "COMPLETE"
            info["rl_episode"] = {
                "success": success, "score": score.get("total", 0),
                "stars": score.get("stars", 0),
                "perfect": success and score.get("total", 0) == 100,
                "quest_collected": now["quest"]["collected"], "warmth": now["carried"],
                "steps": self.steps, "reason": outcome.get("reason", info.get("truncation_reason", "unknown")),
                "status": outcome.get("status", "TRUNCATED"),
                "quest_step": self.quest_step, "warmth_step": self.warmth_step,
                "conditions_met": bool(conditions), "conditions_met_step": self.conditions_met_step,
                "blocked": self.blocked, "ineffective_interact": self.ineffective_interact,
                "waits": self.waits, "damage_count": now["damage_count"],
                "remaining_ticks": now["remaining_ticks"], "courage": now["courage"],
                "return": sum(self.totals.values()), **self.totals,
            }
        return obs, float(sum(terms.values())), terminated, truncated, info
