"""지도 CNN과 상태 MLP를 결합합니다. A2C 손실과 갱신은 SB3가 담당합니다."""
import math
from gymnasium import spaces
from torch import nn
import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

FEATURES_VERSION = 'small_map_layernorm_tanh_v1'


class BoundedMapExtractor(BaseFeaturesExtractor):
    """지도 64특징과 상태 64특징을 합쳐 정책·가치망에 전달합니다.

    LayerNorm·Tanh는 특징의 크기를 조절하는 설계입니다. 탐색의 최솟값을
    보장하거나 정책이 한 행동에 고착되지 않는다고 보장하지는 않습니다.
    """
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=128)
        channels, height, width = observation_space['map'].shape
        self.map_network = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(32 * height * width, 64),
            nn.LayerNorm(64), nn.Tanh())
        self.state_network = nn.Sequential(
            nn.Linear(observation_space['state'].shape[0], 64),
            nn.LayerNorm(64), nn.Tanh())

    def forward(self, observations):
        # 공간 배치와 나머지 상태를 따로 처리한 뒤 128개 공통 특징으로 연결합니다.
        return torch.cat((self.map_network(observations['map'].float()),
                          self.state_network(observations['state'].float())), dim=1)


class UnifiedA2CPolicy(MultiInputActorCriticPolicy):
    """5개 행동의 확률을 만드는 이산 A2C 정책이며 로짓의 범위를 제한합니다.

    경험 수집·학습·관람에 같은 분포를 씁니다. 임의 행동 삽입, 행동 마스크,
    정답 행동이나 과거 경험 버퍼는 추가하지 않습니다. SB3 A2C 손실은 유지하며
    특징·로짓 제한의 효과는 아직 별도 대조 실험으로 확정되지 않았습니다.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args,
                 logit_bound=4., **kwargs):
        if (type(logit_bound) not in (int, float) or not math.isfinite(logit_bound)
                or logit_bound <= 0):
            raise ValueError('logit_bound must be finite and positive.')
        if not isinstance(action_space, spaces.Discrete):
            raise ValueError('UnifiedA2CPolicy supports a categorical Discrete action space only.')
        self.logit_bound = float(logit_bound)
        kwargs.setdefault('features_extractor_class', BoundedMapExtractor)
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise ValueError('The unified policy requires a categorical action distribution.')
        raw_logits = self.action_net(latent_pi)
        # 로짓을 Tanh로 제한한 뒤 확률 분포로 바꿉니다. 행동 자체를 강제하지 않습니다.
        logits = self.logit_bound * torch.tanh(raw_logits / self.logit_bound)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_constructor_parameters(self):
        # 저장한 모델을 다시 읽을 때 사용자 정의 로짓 범위도 동일하게 복원합니다.
        data = super()._get_constructor_parameters()
        data['logit_bound'] = self.logit_bound
        return data
