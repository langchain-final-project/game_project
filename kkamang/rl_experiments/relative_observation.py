# 세번째 시도

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kkamang.env import (
    MansionEnv,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
)


RELATIVE_OBSERVATION_VERSION = (
    OBSERVATION_VERSION + "_relative_v1"
)


class RelativeGoalObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        if env.observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError("620차원 환경에만 적용할 수 있습니다.")

        self.observation_version = RELATIVE_OBSERVATION_VERSION

        # 상대 좌표 4개는 -1~1, 도착 여부 2개는 0~1
        extra_low = np.array(
            [-1, -1, -1, -1, 0, 0],
            dtype=np.float32,
        )
        extra_high = np.ones(6, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate((
                env.observation_space.low,
                extra_low,
            )),
            high=np.concatenate((
                env.observation_space.high,
                extra_high,
            )),
            dtype=np.float32,
        )

    def observation(self, obs):
        # 기존 관측에서 까망이와 수건 좌표 복원
        x = int(round(float(obs[432]) * 15))
        y = int(round(float(obs[433]) * 8))
        qx = int(round(float(obs[599]) * 15))
        qy = int(round(float(obs[600]) * 8))

        # 귀환 문은 144칸짜리 지도에 표시되어 있다.
        exit_index = int(np.argmax(obs[288:432]))
        ey, ex = divmod(exit_index, 16)

        extra = np.array([
            (qx - x) / 15,              # 수건까지 가로 차이
            (qy - y) / 8,               # 수건까지 세로 차이
            (ex - x) / 15,              # 문까지 가로 차이
            (ey - y) / 8,               # 문까지 세로 차이
            float((x, y) == (qx, qy)),  # 수건 칸인가?
            float((x, y) == (ex, ey)),  # 문 칸인가?
        ], dtype=np.float32)

        return np.concatenate((obs, extra))

    def _version_info(self, info):
        return {
            **info,
            "base_observation_version": info["observation_version"],
            "observation_version": self.observation_version,
        }

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, self._version_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        return (
            obs,
            reward,
            terminated,
            truncated,
            self._version_info(info),
        )


if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = RelativeGoalObservation(MansionEnv(stage=1))

    try:
        check_env(env)
        obs, info = env.reset(seed=22)

        print("관측 모양:", obs.shape)
        print("추가된 관측:", obs[-6:])
        print("관측 버전:", info["observation_version"])
    finally:
        env.close()