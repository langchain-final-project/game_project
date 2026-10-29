### 실시간 630차원 관측에 목표 안내를 추가하는 래퍼

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kkamang.env import (
    RealtimeMansionEnv,
    REALTIME_OBSERVATION_SIZE,
    REALTIME_OBSERVATION_VERSION,
)
from rl.ppo_config import OBSERVATION_CONFIG


PPO_OBSERVATION_SIZE = OBSERVATION_CONFIG["final_size"]

# 관측 구성 변경을 구분하기 위한 이름입니다.
PPO_OBSERVATION_VERSION = (
    f"{REALTIME_OBSERVATION_VERSION}_ppo_goal_v1"
)


class PPOGoalObservation(gym.Wrapper):
    """실시간 관측 630개 뒤에 목표 정보 6개를 추가합니다."""

    def __init__(self, env):
        super().__init__(env)

        # 격자 환경이 실수로 연결되는 것을 방지합니다.
        if not isinstance(env.unwrapped, RealtimeMansionEnv):
            raise ValueError(
                "PPOGoalObservation은 RealtimeMansionEnv가 필요합니다."
            )

        base_size = OBSERVATION_CONFIG["base_size"]
        extra_size = OBSERVATION_CONFIG["goal_feature_size"]

        if (
            base_size != REALTIME_OBSERVATION_SIZE
            or extra_size != 6
            or PPO_OBSERVATION_SIZE != base_size + extra_size
        ):
            raise ValueError(
                "관측 설정은 630 + 6 = 636이어야 합니다."
            )

        if env.observation_space.shape != (base_size,):
            raise ValueError(
                f"기본 관측 크기가 잘못됐습니다: "
                f"{env.observation_space.shape}"
            )

        # PPO에 전달할 새로운 관측 공간입니다.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(PPO_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

        self.observation_version = PPO_OBSERVATION_VERSION

    def _augment(self, obs, info):
        snapshot = info["snapshot"]

        # 격자 칸 번호가 아닌 실제 연속 좌표를 사용합니다.
        x, y = snapshot["motion"]["position"]

        enough_warmth = (
            snapshot["carried"] >= snapshot["target"]
        )

        # 0단계: 필수 물건을 먼저 목표로 합니다.
        if not snapshot["quest"]["collected"]:
            phase = 0
            target = snapshot["quest"]["position"]

        # 1단계: 온기가 부족하면 남아 있는 온기를 목표로 합니다.
        elif not enough_warmth:
            phase = 1

            active_items = [
                item
                for item in snapshot["items"]
                if item["active"]
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
                # 남은 온기가 없는 예외 상태에서는 출구를 가리킵니다.
                # 실제 귀환 가능 여부는 게임 코어가 판단합니다.
                target = snapshot["exit"]

        # 2단계: 필수 물건과 온기 조건을 갖추면 출구가 목표입니다.
        else:
            phase = 2
            target = snapshot["exit"]

        # 기존 관측 뒤에 붙일 6개 값입니다.
        extra = np.array(
            [
                float(phase == 0),
                float(phase == 1),
                float(phase == 2),
                0.5 + (target[0] - x) / 30.0,
                0.5 + (target[1] - y) / 16.0,
                float(enough_warmth),
            ],
            dtype=np.float32,
        )

        augmented = np.concatenate((obs, extra))

        # 학습 입력과 별도로 확인할 수 있는 설명 정보입니다.
        info = {
            **info,
            "observation_version": self.observation_version,
            "goal_phase": phase,
            "goal_position": list(target),
        }

        return augmented, info

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(
            seed=seed,
            options=options,
        )

        return self._augment(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        obs, info = self._augment(obs, info)

        return obs, reward, terminated, truncated, info