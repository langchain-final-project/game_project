"""물건 획득 부분 과제를 최적화 단계 없이 검증합니다."""

from copy import deepcopy
import hashlib

import numpy as np
import pytest
import torch

from rl import ppo_config as cfg
from rl.ppo_quest_task import make_quest_env
from rl.train_ppo import make_env, make_model
from rl import train_ppo_quest as runner


@pytest.mark.parametrize("start,down_steps", [(6, 4), (7, 2)])
def test_pickup_ends_immediately_without_reporting_game_clear(start, down_steps):
    env = make_quest_env(start)
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (640,) and info["goal_phase"] == 0
        assert info["snapshot"]["magnet"]["collected"]
        assert not info["snapshot"]["quest"]["collected"]
        for _ in range(down_steps):
            _, _, done, truncated, _ = env.step(1)
            assert not done and not truncated
        _, reward, done, truncated, info = env.step(3)
        assert done and not truncated
        assert info["quest_task_episode"]["success"]
        assert info["quest_task_episode"]["game_status"] == "RUNNING"
        assert info["outcome"] is None
        assert "rl_episode" not in info
        assert "stars" not in info["quest_task_episode"]
        assert info["reward_terms"]["quest_task_success"] == 10
        assert reward == pytest.approx(sum(info["reward_terms"].values()))
        with pytest.raises(RuntimeError):
            env.step(3)
    finally:
        env.close()


def test_deadline_truncates_partial_task_without_failing_game():
    settings = deepcopy(cfg.QUEST_TASK_CONFIG)
    settings["frame_limit"] = 9
    env = make_quest_env(6, settings)
    try:
        env.reset(seed=11)
        _, reward, done, truncated, info = env.step(5)
        assert truncated and not done
        assert info["physical_frames"] == 9
        assert not info["quest_task_episode"]["success"]
        assert info["quest_task_episode"]["reason"] == "local_timeout"
        assert info["outcome"] is None
        assert reward == pytest.approx(-2.045)
    finally:
        env.close()


def test_backtracking_does_not_accumulate_progress_reward():
    env = make_quest_env(7)
    try:
        env.reset(seed=11)
        progress = [env.step(a)[4]["reward_terms"]["quest_task_progress"] for a in (1, 0)]
        assert progress == pytest.approx([.5, -.5])
        assert sum(progress) == pytest.approx(0)
    finally:
        env.close()


@pytest.fixture
def source_model(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    env = make_env()
    try:
        model = make_model(env, 11)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(-4)
            model.policy.action_net.bias[3] = 4
        path = tmp_path / "source.zip"
        model.save(path)
        return path, model
    finally:
        env.close()


def local_settings(source_model):
    settings = deepcopy(cfg.QUEST_TASK_CONFIG)
    settings["source_model"] = source_model[0]
    return settings


def test_initialization_softens_actions_balances_workers_and_preserves_source(source_model):
    path, source = source_model
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    settings = local_settings(source_model)
    env, model, settings = runner.setup(11, settings)
    try:
        obs = env.reset()
        assert [info["quest_task_start"] for info in env.reset_infos] == [6, 7, 6, 7]
        tensor, _ = model.policy.obs_to_tensor(obs[:1])
        with torch.no_grad():
            new_prob = model.policy.get_distribution(tensor).distribution.probs[0]
            old_prob = source.policy.get_distribution(tensor).distribution.probs[0]
        assert new_prob[1] > old_prob[1]
        assert new_prob[3] < old_prob[3]
        assert torch.allclose(model.policy.action_net.bias,
                              source.policy.action_net.bias * settings["action_head_scale"])
        assert torch.count_nonzero(model.policy.value_net.weight) == 0
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert model.ppo_training_task == runner.TASK_VERSION
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    finally:
        env.close()


def test_evaluation_reports_each_start_and_preserves_rng_and_weights(source_model):
    env, model, _ = runner.setup(11, local_settings(source_model))
    try:
        # 검증용 규칙 행동이며 PPO 학습이나 실제 모델 성능 주장이 아닙니다.
        def follow_goal(obs, deterministic=False):
            dx, dy = obs[-3] - .5, obs[-2] - .5
            action = (3 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 0)
            return np.asarray(action), None
        model.predict = follow_goal
        weights = {k: v.clone() for k, v in model.policy.state_dict().items()}
        rng = torch.get_rng_state().clone()
        report = runner.evaluate_quest(model, seeds=[11, 12])
        assert report["passed"] and not report["full_game_evaluation"]
        assert set(report["results"]["stochastic"]) == {"6", "7"}
        for result in report["results"]["stochastic"].values():
            assert result["summary"]["success_rate"] == 1
            assert result["summary"]["episodes"] == 2
            assert all(e["game_status"] == "RUNNING" for e in result["episodes"])
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(v, weights[k]) for k, v in model.policy.state_dict().items())
    finally:
        env.close()


def test_model_selection_protects_weaker_start():
    def report(a, b):
        return {"results": {"stochastic": {
            "6": {"summary": {"success_rate": a, "mean_frames": 50}},
            "7": {"summary": {"success_rate": b, "mean_frames": 50}},
        }}}
    assert runner.selection_key(report(.5, .5)) > runner.selection_key(report(1., .1))


def test_full_return_gate_rejects_one_failed_start(source_model, tmp_path, monkeypatch):
    env, model, _ = runner.setup(11, local_settings(source_model))
    path = tmp_path / "partial.zip"
    model.save(path)
    env.close()
    called_seeds = []
    def evaluation(model, *, seeds):
        called_seeds.extend(seeds)
        return {"passed": False, "results": {"stochastic": {
            "6": {"summary": {"success_rate": .8, "episodes": 50}},
            "7": {"summary": {"success_rate": 1., "episodes": 50}},
        }}}
    monkeypatch.setattr(runner, "evaluate_quest", evaluation)
    with pytest.raises(ValueError, match="6.*7"):
        runner.require_full_return_ready(path)
    assert called_seeds == cfg.QUEST_TASK_CONFIG["heldout_seeds"]


def test_full_return_gate_rejects_full_game_checkpoint(source_model):
    with pytest.raises(ValueError, match="부분 과제"):
        runner.require_full_return_ready(source_model[0])
