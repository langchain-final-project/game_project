### 실시간 관측에 실제 이동 가능 방향과 목표 안내를 추가하는 래퍼

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy

from kkamang.env import (
    RealtimeMansionEnv,
    REALTIME_OBSERVATION_SIZE,
    REALTIME_OBSERVATION_VERSION,
)
from rl.ppo_config import OBSERVATION_CONFIG, STAR_GOAL_CONFIG
from rl.ppo_goal import select_goal


PPO_OBSERVATION_SIZE = OBSERVATION_CONFIG["final_size"]

# 관측 구성 변경을 구분하기 위한 이름입니다.
LEGACY_PPO_OBSERVATION_VERSION = (
    f"{REALTIME_OBSERVATION_VERSION}_ppo_goal_v1"
)
PPO_OBSERVATION_VERSION = f"{REALTIME_OBSERVATION_VERSION}_ppo_goal_moves_v2"
MOVEMENT_MASK_SLICE = slice(REALTIME_OBSERVATION_SIZE, REALTIME_OBSERVATION_SIZE + 4)
ACTION_MASK_VERSION = "physical_can_move_no_interact_v2"


class PPOGoalObservation(gym.Wrapper):
    """환경 630개 + 이동 가능 여부 4개 + 목표 정보 6개를 전달합니다."""

    def __init__(self, env, *, include_movement=True, star_goal=None,
                 mask_ineffective_interaction=False):
        super().__init__(env)
        self.mask_ineffective_interaction = mask_ineffective_interaction
        self.star_goal = deepcopy(STAR_GOAL_CONFIG if star_goal is None else star_goal)
        if self.star_goal.get("enabled", False):
            for name, low, high in (
                ("warmth_target", 1, 14),
                ("min_remaining_ticks", 0, 512),
                ("min_courage", 0, 3),
            ):
                value = self.star_goal.get(name)
                if type(value) is not int or not low <= value <= high:
                    raise ValueError(f"{name}는 {low}~{high} 범위의 정수여야 합니다.")

        # 격자 환경이 실수로 연결되는 것을 방지합니다.
        if not isinstance(env.unwrapped, RealtimeMansionEnv):
            raise ValueError(
                "PPOGoalObservation은 RealtimeMansionEnv가 필요합니다."
            )

        base_size = OBSERVATION_CONFIG["base_size"]
        movement_size = OBSERVATION_CONFIG["movement_feature_size"]
        extra_size = OBSERVATION_CONFIG["goal_feature_size"]

        if (
            base_size != REALTIME_OBSERVATION_SIZE
            or movement_size != 4
            or extra_size != 6
            or PPO_OBSERVATION_SIZE != base_size + movement_size + extra_size
        ):
            raise ValueError(
                "관측 설정은 630 + 4 + 6 = 640이어야 합니다."
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
            shape=(PPO_OBSERVATION_SIZE if include_movement else base_size + extra_size,),
            dtype=np.float32,
        )

        self.include_movement = include_movement
        self.observation_version = (
            PPO_OBSERVATION_VERSION if include_movement else LEGACY_PPO_OBSERVATION_VERSION
        )

    def _augment(self, obs, info):
        snapshot = info["snapshot"]

        # 격자 칸 번호가 아닌 실제 연속 좌표를 사용합니다.
        x, y = snapshot["motion"]["position"]

        phase, target, enough_warmth = select_goal(snapshot, self.star_goal)

        dx = target[0] - x
        dy = target[1] - y

        if self.star_goal.get("direction_encoding") == "local_clipped_v1":
            # 한 칸 이내의 상대 위치를 0~1 범위로 크게 표현합니다.
            direction_x = 0.5 + 0.5 * float(np.clip(dx, -1.0, 1.0))
            direction_y = 0.5 + 0.5 * float(np.clip(dy, -1.0, 1.0))
        else:
            # 기존 모델은 저장 당시의 관측 계산을 유지합니다.
            direction_x = 0.5 + dx / 30.0
            direction_y = 0.5 + dy / 16.0

        # 기존 관측 뒤에 붙일 6개 값입니다.
        extra = np.array(
            [
                float(phase == 0),
                float(phase == 1),
                float(phase == 2),
                direction_x,
                direction_y,
                float(enough_warmth),
            ],
            dtype=np.float32,
        )

        # 논리 격자 대신 실제 몸체 충돌·닫힌 문·코너 보정까지 반영합니다.
        movement = np.array(
            [self.env.unwrapped.core.can_move(action) for action in range(4)],
            dtype=np.float32,
        )
        augmented = np.concatenate((obs, movement, extra) if self.include_movement else (obs, extra))

        # 학습 입력과 별도로 확인할 수 있는 설명 정보입니다.
        info = {
            **info,
            "observation_version": self.observation_version,
            "goal_phase": phase,
            "goal_position": list(target),
            "movement_available": movement.astype(bool).tolist(),
            "action_mask": [*movement.astype(bool).tolist(),
                            not self.mask_ineffective_interaction, True],
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
