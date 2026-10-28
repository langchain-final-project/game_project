import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kkamang.env import (
    REALTIME_OBSERVATION_SIZE,
    REALTIME_OBSERVATION_VERSION,
)
from rl.observation import interaction_ready


TASK_GOAL_SIZE = REALTIME_OBSERVATION_SIZE + 6
TASK_GOAL_VERSION = f"{REALTIME_OBSERVATION_VERSION}_task_goal_v1"


class TaskGoalObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        if env.observation_space.shape != (REALTIME_OBSERVATION_SIZE,):
            raise ValueError("630차원 실시간 관측이 필요합니다.")

        self.observation_space = spaces.Box(
            0.0, 1.0, (TASK_GOAL_SIZE,), dtype=np.float32
        )
        self.observation_version = TASK_GOAL_VERSION

    def _augment(self, obs, info):
        snapshot = info["snapshot"]
        x, y = snapshot["motion"]["position"]

        if not snapshot["quest"]["collected"]:
            phase = 0
            target = snapshot["quest"]["position"]

        elif snapshot["carried"] < snapshot["target"]:
            phase = 1
            active_items = [
                item for item in snapshot["items"] if item["active"]
            ]

            if active_items:
                nearest = min(
                    active_items,
                    key=lambda item: (
                        abs(item["position"][0] - x)
                        + abs(item["position"][1] - y)
                    ),
                )
                target = nearest["position"]
            else:
                target = snapshot["exit"]

        else:
            phase = 2
            target = snapshot["exit"]

        # 목표 종류를 세 개의 0/1 값으로 표현합니다.
        phase_flags = [float(phase == i) for i in range(3)]

        # 상대 거리는 0~1 범위로 변환합니다.
        # 0.5가 거리 0, 0.5보다 작으면 왼쪽/위쪽입니다.
        extra = np.array(
            [
                *phase_flags,
                0.5 + (target[0] - x) / 30,
                0.5 + (target[1] - y) / 16,
                float(interaction_ready(snapshot)),
            ],
            dtype=np.float32,
        )

        obs = np.concatenate((obs, extra))
        info = {
            **info,
            "observation_version": self.observation_version,
        }
        return obs, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self._augment(obs, info)
        return obs, reward, terminated, truncated, info