"""환경 정보와 목표 정보를 별도로 처리하는 PPO 특징 추출기."""

import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class WorldGoalFeaturesExtractor(BaseFeaturesExtractor):
    """마지막 목표 정보 6개를 환경 정보와 별도 분기에서 인코딩합니다."""

    def __init__(self, observation_space, branch_size=32):
        if observation_space.shape not in ((636,), (640,)):
            raise ValueError("기존 636차원 또는 이동 정보가 포함된 640차원 관측이 필요합니다.")
        if type(branch_size) is not int or branch_size <= 0:
            raise ValueError("branch_size는 양의 정수여야 합니다.")

        super().__init__(observation_space, features_dim=branch_size * 2)
        self.world = nn.Sequential(
            # 이동 정보 4개는 정책의 행동 마스크에서 사용합니다.
            # 630개 환경·마지막 6개 목표 분기의 가중치 구조는 보존합니다.
            nn.Linear(630, branch_size),
            nn.Tanh(),
        )
        self.goal = nn.Sequential(nn.Linear(6, branch_size), nn.Tanh())

    def forward(self, observations):
        return torch.cat((
            self.world(observations[:, :630]),
            self.goal(observations[:, -6:]),
        ), dim=1)
