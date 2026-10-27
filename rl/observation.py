# 기존 격자 관측 뒤에 현재 상태의 상호작용 가능 여부 한 값만 추가합니다.
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kkamang.env import OBSERVATION_SIZE, OBSERVATION_VERSION

INTERACTION_READY_SIZE = OBSERVATION_SIZE + 1
INTERACTION_READY_VERSION = f"{OBSERVATION_VERSION}_interaction_ready_v1"


def interaction_ready(snapshot: dict) -> bool:
    """지금 E로 수집·개방·귀환을 요청할 수 있는가. 이후 시간·피해 판정의 성공 보장은 아닙니다."""
    if snapshot['terminal']:
        return False
    position, quest = snapshot['position'], snapshot['quest']
    # 코어와 같은 우선순위: 문 위에서는 귀환 조건만 검사합니다.
    if position == snapshot['exit']:
        return bool(quest['collected'] and snapshot['carried'] >= snapshot['target'])
    return bool(
        (not quest['collected'] and position == quest['position'])
        or any(not shortcut['opened'] and position == shortcut['terminal']
               for shortcut in snapshot['shortcuts'])
    )


class InteractionReadyObservation(gym.Wrapper):
    """기존 620개 관측·게임 전이는 유지하고 파생 이진값 하나를 덧붙입니다."""

    def __init__(self, env):
        super().__init__(env)
        if env.observation_space.shape != (OBSERVATION_SIZE,):
            raise ValueError('interaction_ready는 620차원 격자 관측에서만 지원합니다.')
        self.observation_space = spaces.Box(
            0.0, 1.0, (INTERACTION_READY_SIZE,), dtype=np.float32,
        )
        self.observation_version = INTERACTION_READY_VERSION

    def _augment(self, observation, info):
        ready = interaction_ready(info['snapshot'])
        observation = np.concatenate((observation, np.array([ready], dtype=np.float32)))
        info = {**info, 'observation_version': self.observation_version,
                'interaction_ready': ready}
        return observation, info

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self._augment(observation, info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation, info = self._augment(observation, info)
        return observation, reward, terminated, truncated, info
