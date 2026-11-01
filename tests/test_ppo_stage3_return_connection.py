"""9프레임 제어에서 중심 정렬 왕복과 긴 귀환 연결을 회귀 검증합니다."""

from copy import deepcopy

import pytest

from kkamang.env import RealtimeMansionEnv
from kkamang.scoring import score_outcome
from rl import ppo_config as cfg
from rl.ppo_path_distance import next_waypoint, path_distance
from rl.ppo_stage3_plan import stage3_warmth_target
from rl.train_ppo import make_env


@pytest.mark.parametrize('damage, expected', [(0, 12), (1, 13), (2, 14)])
def test_warmth_requirement_uses_actual_score_and_remaining_courage(damage, expected):
    env = RealtimeMansionEnv(stage=3)
    try:
        snapshot = env.reset(seed=11)[1]['snapshot']
        snapshot.update(damage_count=damage, courage=3-damage)
        assert stage3_warmth_target(snapshot, cfg.STAR_GOAL_CONFIG, True) == expected
        assert stage3_warmth_target(snapshot, cfg.STAR_GOAL_CONFIG, False) == 4
        result = {'status': 'COMPLETE', 'banked': expected, 'ticks_used': 70,
                  'remaining_courage': 3-damage}
        assert score_outcome(env.core.config, result)['stars'] == 3
        result['banked'] -= 1
        assert score_outcome(env.core.config, result)['stars'] < 3
    finally:
        env.close()


def test_late_return_raises_collection_target_and_legacy_recipe_is_preserved():
    env = RealtimeMansionEnv(stage=3)
    try:
        snapshot = env.reset(seed=11)[1]['snapshot']
        snapshot['tick'] = 74
        assert stage3_warmth_target(snapshot, cfg.STAR_GOAL_CONFIG, True) == 13
        legacy = {**cfg.STAR_GOAL_CONFIG, 'warmth_target': 13}
        legacy.pop('stage3_score_aware_warmth')
        assert stage3_warmth_target(snapshot, legacy, True) == 13
        snapshot.update(damage_count=2, courage=1)
        assert stage3_warmth_target(snapshot, legacy, True) == 14
    finally:
        env.close()


def test_wall_margin_does_not_force_small_vertical_alignment_in_open_corridor():
    env = RealtimeMansionEnv(stage=3)
    try:
        snapshot = env.reset(seed=11)[1]['snapshot']
        snapshot['motion']['position'] = [7.0, .78]
        assert next_waypoint(snapshot, [8, 1]) == [7, 1]
        assert next_waypoint(snapshot, [8, 1], lookahead=True) == [8, .78]
        snapshot['motion']['position'] = [6.28, 2.22]
        goal = next_waypoint(snapshot, [1, 1], lookahead=True)
        assert goal[0] == 6.28 and goal[1] > 2.22
        after = deepcopy(snapshot)
        after['motion']['position'][1] += .5
        assert path_distance(after, [1, 1]) < path_distance(snapshot, [1, 1])
    finally:
        env.close()


@pytest.mark.parametrize('start', [0, 13, 14, 15, 16, 17, 18])
def test_goal_guide_finishes_long_return_with_nine_frame_actions(start):
    settings = {**deepcopy(cfg.CURRICULUM_CONFIG), 'initial_frontier': start}
    env = make_env(curriculum_settings=settings)
    try:
        _, info = env.reset(seed=11)
        for _ in range(200):
            snapshot = info['snapshot']
            position, goal = snapshot['motion']['position'], info['goal_position']
            dx, dy = goal[0]-position[0], goal[1]-position[1]
            moves = [(abs(dx), 3 if dx > 0 else 2), (abs(dy), 1 if dy > 0 else 0)]
            action = max(((size, a) for size, a in moves
                          if size > 1e-8 and env.unwrapped.core.can_move(a)), default=(0, 5))[1]
            _, _, done, truncated, info = env.step(action)
            if done or truncated:
                break
        assert done and not truncated
        assert info['outcome']['score']['stars'] == 3
        assert info['snapshot']['quest']['collected']
        assert info['snapshot']['motion']['frame'] < 1000
        if start == 0:
            assert info['snapshot']['carried'] == 12
            assert info['snapshot']['damage_count'] == 0
        # 목표 안내 경로의 검증이며 PPO가 학습한 성능을 측정하는 테스트는 아닙니다.
    finally:
        env.close()


def test_return_distance_reward_is_symmetric_after_enabling_new_guide():
    settings = {**deepcopy(cfg.CURRICULUM_CONFIG), 'initial_frontier': 16}
    env = make_env(curriculum_settings=settings)
    try:
        env.reset(seed=11)
        forward = env.step(2)[4]['reward_terms']['return_progress_reward']
        backward = env.step(3)[4]['reward_terms']['return_progress_reward']
        assert forward == pytest.approx(.5)
        assert backward == pytest.approx(-.5)
        assert forward + backward == pytest.approx(0)
    finally:
        env.close()
