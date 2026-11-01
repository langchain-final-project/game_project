"""실제 이동 가능 방향을 학습·평가·예측에 동일하게 적용하는 PPO 정책."""

import torch
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from rl.ppo_observation import MOVEMENT_MASK_SLICE


class MovementAwareActorCriticPolicy(ActorCriticPolicy):
    """이동 불가 방향의 확률을 0으로 만들고 나머지 행동을 PPO로 학습합니다."""

    def __init__(self, *args, mask_ineffective_interaction=False, **kwargs):
        # 이전 저장 모델에는 인자가 없으므로 기존 동작을 유지합니다.
        self.mask_ineffective_interaction = mask_ineffective_interaction
        super().__init__(*args, **kwargs)
        if self.observation_space.shape != (640,) or not isinstance(
            self.action_dist, CategoricalDistribution,
        ) or self.action_space.n != 6:
            raise ValueError("640차원 관측과 이산 행동 6개가 필요합니다.")

    def get_distribution(self, obs):
        distribution = super().get_distribution(obs)
        movement = obs[:, MOVEMENT_MASK_SLICE] > 0.5
        # 네 방향이 모두 막혀도 대기는 항상 허용합니다.
        other_actions = torch.ones((obs.shape[0], 2), dtype=torch.bool, device=obs.device)
        if self.mask_ineffective_interaction:
            # 현재 게임의 수집·귀환은 이동으로 자동 실행되므로 대기만 남깁니다.
            other_actions[:, 0] = False
        allowed = torch.cat((movement, other_actions), dim=1)
        logits = distribution.distribution.logits.masked_fill(~allowed, -1e9)
        return CategoricalDistribution(6).proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.predict_values(obs)
        log_prob = distribution.log_prob(actions)
        return actions.reshape((-1, *self.action_space.shape)), values, log_prob

    def evaluate_actions(self, obs, actions):
        # 경험을 수집할 때와 같은 마스크로 학습 시 대수 확률을 계산합니다.
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()
