from collections import defaultdict
from numbers import Integral

import gymnasium as gym

from kkamang.env import RealtimeMansionEnv


class ActionRepeat(gym.Wrapper):
    """같은 행동을 최대 repeat개 물리 프레임 동안 유지합니다."""

    def __init__(self, env, repeat=9):
        super().__init__(env)

        if not isinstance(env.unwrapped, RealtimeMansionEnv):
            raise ValueError("RealtimeMansionEnv가 필요합니다.")

        if (
            isinstance(repeat, bool)
            or not isinstance(repeat, Integral)
            or repeat < 1
        ):
            raise ValueError("repeat는 양의 정수여야 합니다.")

        self.repeat = int(repeat)
        self._finished = True

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        self.decisions = 0
        self.physical_frames = 0
        self._finished = False

        return obs, {
            **info,
            "repeat_frames": 0,
            "physical_frames": 0,
            "decision_steps": 0,
            "repeat_trace": [],
        }

    def step(self, action):
        if self._finished:
            raise RuntimeError(
                "새 에피소드를 reset()한 뒤 실행하세요."
            )

        reward_sum = 0.0
        terms = defaultdict(float)
        events = []
        trace = []

        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = (
                self.env.step(action)
            )

            reward_sum += float(reward)
            self.physical_frames += 1

            for name, value in info.get("reward_terms", {}).items():
                terms[name] += float(value)

            events.extend(info.get("events", []))

            snapshot = info["snapshot"]
            x, y = snapshot["motion"]["position"]

            trace.append({
                "frame": self.physical_frames,
                "action": int(action),
                "x": float(x),
                "y": float(y),
                "goal_phase": info.get("goal_phase"),
                "warmth": snapshot["carried"],
                "quest_collected": snapshot["quest"]["collected"],
            })

            if terminated or truncated:
                break

        self.decisions += 1
        self._finished = bool(terminated or truncated)

        info = {
            **info,
            "events": events,
            "reward_terms": dict(terms),
            "repeat_frames": len(trace),
            "physical_frames": self.physical_frames,
            "decision_steps": self.decisions,
            "repeat_trace": trace,
        }

        if "rl_episode" in info:
            info["rl_episode"] = {
                **info["rl_episode"],
                "physical_frames": self.physical_frames,
                "decision_steps": self.decisions,
            }

        return obs, reward_sum, terminated, truncated, info