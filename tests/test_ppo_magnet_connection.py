"""자석 범위 안내·보상·연결 종료·전이 호환성을 최적화 없이 검증합니다."""

from copy import deepcopy

import pytest
import torch

from rl import ppo_config as cfg
from rl.ppo_connection_task import make_connection_env
from rl.ppo_curriculum import Stage2ThreeStarCurriculum
from rl.ppo_goal import collection_target
from rl import train_ppo_connection as runner


@pytest.mark.parametrize("start", range(8))
def test_real_route_connects_pickup_without_claiming_game_clear(start):
    env = make_connection_env([start])
    try:
        obs, info = env.reset(seed=11)
        assert obs.shape == (640,)
        route = Stage2ThreeStarCurriculum.route
        for index in range(Stage2ThreeStarCurriculum.prefix_frames[start], len(route), 9):
            _, reward, done, truncated, info = env.step(route[index])
            assert reward == pytest.approx(sum(info["reward_terms"].values()))
            if done or truncated:
                break
        episode = info["connection_episode"]
        assert episode["success"] and done and not truncated
        assert episode["quest_collected"] and episode["magnet_collected"]
        if start < 6:
            assert episode["remote_warmth"] > 0
        assert episode["game_status"] == "RUNNING"
        assert info["outcome"] is None and "rl_episode" not in info
        assert "stars" not in episode
        with pytest.raises(RuntimeError):
            env.step(5)
    finally:
        env.close()


def test_pickup_bonus_is_once_and_not_connection_success():
    env = make_connection_env([5])
    try:
        env.reset(seed=11)
        _, _, done, truncated, info = env.step(1)
        assert info["reward_terms"]["magnet_pickup_reward"] == 1
        assert info["snapshot"]["magnet"]["collected"]
        assert not done and not truncated and "connection_episode" not in info
        assert info["reward_terms"]["connection_success"] == 0
        assert env.step(5)[4]["reward_terms"]["magnet_pickup_reward"] == 0
    finally:
        env.close()


def test_local_timeout_truncates_and_reset_restores_task():
    settings = deepcopy(cfg.CONNECTION_TASK_CONFIG)
    settings["frame_limit"] = 9
    env = make_connection_env([0], settings)
    try:
        env.reset(seed=11)
        _, _, done, truncated, info = env.step(5)
        assert truncated and not done
        assert info["connection_episode"]["frames"] == 9
        assert not info["connection_episode"]["success"]
        assert info["outcome"] is None
        assert info["reward_terms"]["connection_timeout"] == -2
        env.reset(seed=11)
        assert not env.env._finished
    finally:
        env.close()


def test_magnet_guide_handles_range_expiry_large_warmth_and_old_models():
    env = make_connection_env([6])
    try:
        _, info = env.reset(seed=11)
        snapshot = info["snapshot"]
        item = next(i for i in snapshot["items"] if i["position"] == [9, 6])
        distance, target = collection_target(snapshot, item, cfg.STAR_GOAL_CONFIG)
        direct, original = collection_target(snapshot, item, {})
        assert distance < direct and target != original
        assert max(abs(target[i] - item["position"][i]) for i in (0, 1)) <= 1
        snapshot["magnet"]["remaining_ticks"] = 1
        assert collection_target(snapshot, item, cfg.STAR_GOAL_CONFIG) == (direct, original)
        snapshot["magnet"]["remaining_ticks"] = 12
        large = next(i for i in snapshot["items"] if i["value"] == 3)
        assert collection_target(snapshot, large, cfg.STAR_GOAL_CONFIG) == collection_target(snapshot, large, {})
    finally:
        env.close()


def test_initial_warmth_and_quest_rewards_match_full_task():
    env = make_connection_env([0])
    try:
        env.reset(seed=11)
        heat, quest = 0., 0.
        for index in range(0, len(Stage2ThreeStarCurriculum.route), 9):
            _, _, done, truncated, info = env.step(Stage2ThreeStarCurriculum.route[index])
            terms = info["reward_terms"]
            heat += terms["warmth_reward"] + terms["magnet_warmth_reward"]
            quest += terms["quest_reward"]
            if done or truncated:
                break
        assert quest == 5
        # 초반 두 점은 각각 0.5, 자석 원격 수집 두 점은 각각 0.75입니다.
        assert heat == pytest.approx(2.5)
    finally:
        env.close()


def test_collection_progress_uses_fixed_destination_and_cancels_backtracking():
    env = make_connection_env([3])
    try:
        _, info = env.reset(seed=11)
        reward_wrapper = env.env.env
        assert info["goal_position"] == [6, 2]
        assert reward_wrapper._previous_warmth_target == [6, 4]
        approach = env.step(1)[4]["reward_terms"]["warmth_progress_reward"]
        retreat = env.step(0)[4]["reward_terms"]["warmth_progress_reward"]
        assert approach > 0 and retreat < 0
        assert approach + retreat == pytest.approx(0)
    finally:
        env.close()


def test_full_return_gate_rejects_failed_connection(monkeypatch):
    from types import SimpleNamespace
    model = SimpleNamespace(ppo_training_task=runner.TASK_VERSION,
                            ppo_connection_settings={"heldout_seeds": [30000, 30001]})
    monkeypatch.setattr(runner.PPO, "load", lambda *args, **kwargs: model)
    called = []
    def evaluate(model, *, seeds):
        called.extend(seeds)
        return {"passed": False, "quest_minimum_rate": .95, "results": {"stochastic": {
            "0": {"summary": {"success_rate": .7}}}}}
    monkeypatch.setattr(runner, "evaluate_connection", evaluate)
    with pytest.raises(ValueError, match="0~5"):
        runner.require_connection_ready("unused.zip")
    assert called == [30000, 30001]


def test_evaluation_requires_connection_from_every_start_and_retains_quest():
    good = {"quest_retained": True, "passed": True,
            "connection_minimum_rate": .8, "quest_minimum_rate": .9,
            "results": {"stochastic": {str(n): {"summary": {"success_rate": .8}} for n in range(6)}}}
    forgot = {**good, "quest_retained": False, "passed": False,
              "connection_minimum_rate": 1., "quest_minimum_rate": .5}
    assert runner.selection_key(good) > runner.selection_key(forgot)


def test_source_transfer_and_diagnostics_do_not_learn(tmp_path, monkeypatch):
    from rl.train_ppo import make_env, make_model
    torch.set_num_threads(1)
    monkeypatch.setitem(cfg.TRANSFER_CONFIG, "enabled", False)
    source_env = make_env()
    try:
        source = make_model(source_env, 11)
        source_path = tmp_path / "source.zip"
        source.save(source_path)
    finally:
        source_env.close()
    monkeypatch.setitem(cfg.CONNECTION_TASK_CONFIG, "source_model", source_path)
    env, model, settings = runner.setup(11)
    try:
        obs = env.reset()
        assert obs.shape == (4, 640)
        assert [i["curriculum_start"] for i in env.reset_infos][2:] == [6, 7]
        assert all(i["curriculum_start"] < 6 for i in env.reset_infos[:2])
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        assert all(torch.equal(v, source.policy.state_dict()[k]) for k, v in before.items())
        rng = torch.get_rng_state().clone()
        result = runner.evaluate_connection(model, seeds=[11])
        assert set(result["results"]["stochastic"]) == {str(n) for n in range(6)}
        assert set(result["quest_review"]) == {"6", "7"}
        assert not result["full_game_evaluation"]
        assert model.num_timesteps == 0 and not model.policy.optimizer.state
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(torch.equal(v, before[k]) for k, v in model.policy.state_dict().items())
        assert model.ppo_star_goal_config["magnet_pickup_targets"]
        assert model.ppo_training_task == runner.TASK_VERSION
    finally:
        env.close()
