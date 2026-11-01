"""학습 없이 실제 충돌 정보, PPO 마스크 일관성, 기존 모델 전이를 검증합니다."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy

from kkamang.env import RealtimeMansionEnv
from rl import ppo_config as cfg
from rl import train_ppo
from rl.ppo_action_repeat import ActionRepeat
from rl.ppo_features import WorldGoalFeaturesExtractor
from rl.ppo_observation import (
    PPOGoalObservation, MOVEMENT_MASK_SLICE, LEGACY_PPO_OBSERVATION_VERSION,
    ACTION_MASK_VERSION,
)
from rl.ppo_policy import MovementAwareActorCriticPolicy
from rl.ppo_reward import PPOReward


@pytest.fixture
def env():
    value = ActionRepeat(PPOReward(PPOGoalObservation(RealtimeMansionEnv(stage=2))), repeat=9)
    try:
        yield value
    finally:
        value.close()


def at_wall(env):
    env.reset(seed=11)
    for _ in range(14):
        obs, _, done, truncated, info = env.step(1)
        assert not done and not truncated
    assert info["snapshot"]["motion"]["position"] == pytest.approx([4, 6.22])
    return obs, info


def make_policy(env):
    return MovementAwareActorCriticPolicy(
        env.observation_space, env.action_space, lambda _: 0.0003,
        net_arch={"pi": [16], "vf": [16]},
        features_extractor_class=WorldGoalFeaturesExtractor,
        features_extractor_kwargs={"branch_size": 8},
    )


def test_observation_exposes_actual_movement_options(env):
    obs, info = at_wall(env)
    actual = [env.unwrapped.core.can_move(a) for a in range(4)]
    assert actual == [True, False, True, True]
    assert obs.shape == (640,) and env.observation_space.contains(obs)
    np.testing.assert_array_equal(obs[MOVEMENT_MASK_SLICE], actual)
    assert info["movement_available"] == actual
    assert info["action_mask"] == [*actual, True, True]


def test_blocked_highest_logit_cannot_be_sampled_or_selected(env):
    obs, _ = at_wall(env)
    policy = make_policy(env)
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.copy_(torch.tensor([0., 50., 0., 10., -20., -20.]))
    tensor = policy.obs_to_tensor(obs)[0]
    distribution = policy.get_distribution(tensor)
    assert distribution.distribution.probs[0, 1].item() == 0.0
    assert not (distribution.distribution.sample((500,)) == 1).any()
    action, _ = policy.predict(obs, deterministic=True)
    assert int(action) == 3
    action, _, _ = policy(tensor, deterministic=True)
    assert action.item() == 3


def test_forward_and_training_log_probabilities_use_identical_masks(env):
    obs, _ = at_wall(env)
    policy = make_policy(env)
    batch = policy.obs_to_tensor(np.repeat(obs[None, :], 4, axis=0))[0]
    # 배치 내 관측별로 서로 다른 이동 가능 방향을 적용합니다.
    batch[:, MOVEMENT_MASK_SLICE] = torch.eye(4)
    actions, values, log_probs = policy(batch)
    checked_values, checked_log_probs, entropy = policy.evaluate_actions(batch, actions)
    torch.testing.assert_close(values, checked_values)
    torch.testing.assert_close(log_probs, checked_log_probs)
    for index, action in enumerate(actions.tolist()):
        assert action >= 4 or action == index
    loss = checked_values.sum() + checked_log_probs.sum() + entropy.sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_no_open_directions_still_has_finite_wait_and_interaction(env):
    obs, _ = at_wall(env)
    obs[MOVEMENT_MASK_SLICE] = 0
    policy = make_policy(env)
    tensor = policy.obs_to_tensor(obs)[0]
    distribution = policy.get_distribution(tensor)
    probabilities = distribution.distribution.probs[0]
    assert torch.equal(probabilities[:4], torch.zeros(4))
    torch.testing.assert_close(probabilities[4:].sum(), torch.tensor(1.0))
    assert torch.isfinite(distribution.entropy()).all()


def test_legacy_world_and_goal_weights_have_identical_outputs():
    legacy = WorldGoalFeaturesExtractor(spaces.Box(0, 1, shape=(636,), dtype=np.float32))
    current = WorldGoalFeaturesExtractor(spaces.Box(0, 1, shape=(640,), dtype=np.float32))
    current.load_state_dict(legacy.state_dict())
    before = torch.rand((3, 636))
    after = torch.cat((before[:, :630], torch.zeros((3, 4)), before[:, -6:]), dim=1)
    torch.testing.assert_close(legacy(before), current(after))


def test_transfer_imports_legacy_636_weights_into_masked_policy(monkeypatch, env):
    torch.set_num_threads(1)
    legacy_space = spaces.Box(0, 1, shape=(636,), dtype=np.float32)
    legacy_policy = ActorCriticPolicy(
        legacy_space, env.action_space, lambda _: 0.0003,
        net_arch=cfg.NETWORK_CONFIG["net_arch"],
        features_extractor_class=WorldGoalFeaturesExtractor,
        features_extractor_kwargs={"branch_size": cfg.NETWORK_CONFIG["feature_branch_size"]},
    )
    source = SimpleNamespace(
        policy=legacy_policy, observation_space=legacy_space, action_space=env.action_space,
        ppo_observation_version=LEGACY_PPO_OBSERVATION_VERSION, num_timesteps=48400,
    )
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", True)
    monkeypatch.setattr(train_ppo.PPO, "load", lambda *args, **kwargs: source)
    model = train_ppo.make_model(env, 11)
    assert isinstance(model.policy, MovementAwareActorCriticPolicy)
    assert model.observation_space.shape == (640,)
    assert model.ppo_action_mask_version == ACTION_MASK_VERSION
    assert model.transfer_source_observation_version == LEGACY_PPO_OBSERVATION_VERSION
    torch.testing.assert_close(
        model.policy.features_extractor.world[0].weight,
        legacy_policy.features_extractor.world[0].weight,
    )
    torch.testing.assert_close(
        model.policy.action_net.weight,
        legacy_policy.action_net.weight * cfg.TRANSFER_CONFIG["action_head_scale"],
    )
