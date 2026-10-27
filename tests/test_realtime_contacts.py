"""Regressions for physical contact timing shared by the desktop and RL core."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import pygame
import pytest

from kkamang.app import GameApp
from kkamang.core import Action
from kkamang.env import RealtimeMansionEnv, encode_realtime_observation
from kkamang.levels import load_level
from kkamang.realtime import (FRAME_SECONDS, FRAMES_PER_TICK, GHOST_B_CONTACT_RADIUS,
                             GHOST_CONTACT_RADIUS,
                             PLAYER_CONTACT_RADIUS, RealtimeGameCore)


def patrol_fixture(game, player=(12., 1.)):
    """Place a patrol one tile away, scheduled to approach at the next tick."""
    game._state['position'] = [math.floor(v + .5) for v in player]
    game._motion_position = list(player)
    game._state['chaser'] = None
    game._chaser_origin = None
    game._state['ghosts'] = [dict(id='A', position=[13, 1], route=[[13, 1], [12, 1]],
                                 route_index=0, direction=1, countdown=1,
                                 ticks_per_move=1, next_position=[12, 1])]
    game._ghost_origins = {'A': [13, 1]}


def chaser_fixture(game, *, position=(13, 1), player=(12., 1.)):
    game._state['ghosts'] = []
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=list(position), countdown=1,
                                 ticks_per_move=1)
    game._chaser_origin = list(position)
    game._chaser_stunned = False
    game._motion_position = list(player)
    game._state['position'] = [math.floor(v + .5) for v in player]


def test_target_cell_entry_does_not_damage_until_the_patrol_body_arrives():
    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game)
    for frame in range(1, 25):
        result = game.step(Action.WAIT)
        assert not any(event['type'] == 'contact' for event in result.events)
        if frame == FRAMES_PER_TICK:
            assert result.snapshot['ghosts'][0]['position'] == [12, 1]
            assert result.snapshot['motion']['ghost_positions']['A'] == [13., 1.]
    result = game.step(Action.WAIT)
    assert result.snapshot['courage'] == 2
    assert sum(event['type'] == 'damage' for event in result.events) == 1
    assert result.snapshot['motion']['ghost_positions']['A'][0] < 12.64


def test_same_cell_is_safe_when_bodies_are_apart_but_adjacent_cells_can_touch():
    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game, player=(12.1, 1.))
    game.advance(Action.WAIT, .3)
    assert game.snapshot()['courage'] == 3
    assert game.snapshot()['position'] == game.snapshot()['ghosts'][0]['position']
    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game, player=(12.49, 1.))
    result = game.step(Action.WAIT)
    assert any(event['type'] == 'damage' for event in result.events)
    assert result.snapshot['position'] != result.snapshot['ghosts'][0]['position']


def test_patrol_origins_update_at_every_tick_without_backtracking_or_diagonal_cuts():
    game = RealtimeGameCore(load_level(3))
    previous = game.snapshot()['motion']['ghost_positions']['A']
    moved_axes = set()
    for _ in range(180):
        state = game.step(Action.WAIT).snapshot
        current = state['motion']['ghost_positions']['A']
        dx, dy = [current[i] - previous[i] for i in (0, 1)]
        assert abs(dx) < 1e-10 or abs(dy) < 1e-10
        assert math.dist(previous, current) <= 1 / FRAMES_PER_TICK + 1e-10
        if abs(dx) > 1e-10: moved_axes.add(0)
        if abs(dy) > 1e-10: moved_axes.add(1)
        if state['motion']['frame'] == 54:
            assert current == [7., 3.]
        previous = current
    assert moved_axes == {0, 1}


@pytest.mark.parametrize('gap,expected', [(0., True), (.639, True), (.64, True), (.641, False)])
def test_contact_circle_boundary_and_tangency(gap, expected):
    assert PLAYER_CONTACT_RADIUS + GHOST_CONTACT_RADIUS == pytest.approx(.64)
    assert RealtimeGameCore._bodies_contact([0, 0], [0, 0], [gap, 0], [gap, 0]) is expected


def test_relative_sweep_catches_crossing_and_rejects_a_near_miss():
    assert RealtimeGameCore._bodies_contact([0, 0], [1, 0], [1, 0], [0, 0])
    assert not RealtimeGameCore._bodies_contact([0, 0], [1, 0], [1, .65], [0, .65])


def test_larger_visible_B_body_contacts_when_diagonally_overlapping_cat():
    player = (12.4, 1.35)
    assert math.dist(player, (13., 1.)) == pytest.approx(.6946221995)
    assert PLAYER_CONTACT_RADIUS + GHOST_CONTACT_RADIUS < math.dist(player, (13., 1.))
    assert PLAYER_CONTACT_RADIUS + GHOST_B_CONTACT_RADIUS > math.dist(player, (13., 1.))

    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game, player=player)
    game._state['ghosts'][0]['id'] = 'B'
    game._ghost_origins = {'B': [13, 1]}
    result = game.step(Action.WAIT)
    assert result.snapshot['damage_count'] == 1
    assert next(event for event in result.events if event['type'] == 'contact')['ghost_ids'] == ['B']

    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game, player=(12.3, 1.35))
    game._state['ghosts'][0]['id'] = 'B'
    game._ghost_origins = {'B': [13, 1]}
    assert game.step(Action.WAIT).snapshot['damage_count'] == 0


def test_chaser_freezes_at_contact_and_resumes_the_interrupted_segment():
    game = RealtimeGameCore(load_level(11))
    chaser_fixture(game)
    result = game.advance(Action.WAIT, FRAME_SECONDS * 29)
    frozen = result.snapshot['motion']['chaser_position']
    assert result.snapshot['motion']['chaser_stunned']
    assert 12 < frozen[0] < 13
    assert result.snapshot['chaser']['countdown'] == 4
    # Move the cat away. Three following world ticks must leave the ghost still.
    while game.snapshot()['motion']['frame'] < 90:
        action = Action.LEFT if game.snapshot()['motion']['position'][0] > 11 else Action.WAIT
        state = game.step(action).snapshot
        assert state['motion']['chaser_position'] == frozen
        if state['motion']['frame'] in (36, 54, 72):
            assert state['chaser']['countdown'] == {36: 3, 54: 2, 72: 1}[state['motion']['frame']]
        if state['motion']['frame'] < 90:
            assert state['motion']['chaser_stunned']
    assert not game.snapshot()['motion']['chaser_stunned']
    previous = frozen
    for _ in range(36):
        state = game.step(Action.WAIT).snapshot
        current = state['motion']['chaser_position']
        assert current[1] == frozen[1]
        assert math.dist(previous, current) <= 1 / FRAMES_PER_TICK + 1e-10
        previous = current
    assert previous[0] < frozen[0]


def test_contact_and_stun_do_not_depend_on_render_cadence():
    states = []
    for fps in (10, 30, 60, 120):
        game = RealtimeGameCore(load_level(11))
        chaser_fixture(game)
        for _ in range(fps):
            game.advance(Action.WAIT, 1 / fps)
        states.append(game.snapshot())
    assert states[0]['damage_count'] == 1
    assert states[0]['motion']['chaser_stunned']
    assert all(state == states[0] for state in states)


def test_reset_clears_frozen_motion_and_realigns_randomized_patrol_origins():
    game = RealtimeGameCore(load_level(11))
    chaser_fixture(game)
    assert game.advance(Action.WAIT, FRAME_SECONDS * 29).snapshot['motion']['chaser_stunned']
    state = game.reset(phase_overrides={'A': {'index': 2, 'countdown': 1}})
    assert state['motion']['frame'] == 0
    assert not state['motion']['chaser_stunned']
    assert state['motion']['chaser_position'] is None
    for ghost in state['ghosts']:
        assert state['motion']['ghost_positions'][ghost['id']] == ghost['position']
    assert encode_realtime_observation(state).shape == (630,)


def test_item_invincibility_still_stuns_and_does_not_grant_extra_protection():
    game = RealtimeGameCore(load_level(11))
    chaser_fixture(game, position=(14, 6), player=(14., 6.))
    game._chaser_origin = [13.35, 6.]  # Enter contact while finishing a segment.
    result = game.step(Action.WAIT)
    contact = next(event for event in result.events if event['type'] == 'contact')
    assert contact['item_invincible'] and contact['protected']
    assert result.snapshot['motion']['chaser_stunned']
    assert result.snapshot['courage'] == 3 and result.snapshot['protection'] == 0
    game.advance(Action.WAIT, FRAME_SECONDS * 179)
    assert game.snapshot()['invincibility']['remaining_ticks'] == 0
    assert game.snapshot()['courage'] == 3
    result = game.step(Action.WAIT)
    assert result.snapshot['courage'] == 2 and result.snapshot['protection'] == 3


def test_two_bodies_touching_in_one_frame_only_remove_one_life():
    game = RealtimeGameCore(load_level(11))
    patrol_fixture(game, player=(12.6, 1.))
    second = deepcopy(game._state['ghosts'][0])
    second['id'] = 'B'
    game._state['ghosts'].append(second)
    game._ghost_origins['B'] = [13, 1]
    result = game.step(Action.WAIT)
    contact = next(event for event in result.events if event['type'] == 'contact')
    assert contact['ghost_ids'] == ['A', 'B']
    assert sum(event['type'] == 'damage' for event in result.events) == 1


def test_desktop_rendering_contact_and_rl_transitions_use_identical_centers(tmp_path):
    app = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    env = RealtimeMansionEnv(11)
    try:
        app.campaign.data['completed'] = list(range(1, 11))
        app.stage = 11
        app.start_run()
        env.reset(seed=0)
        patrol_fixture(app.core)
        patrol_fixture(env.core)
        for frame in range(1, 41):
            app.advance(FRAME_SECONDS)
            observation, _, terminated, truncated, info = env.step(Action.WAIT)
            state = app.core.snapshot()
            assert state == info['snapshot']
            np.testing.assert_array_equal(observation, encode_realtime_observation(state))
            assert not terminated and not truncated
            if frame in (17, 18, 19, 28, 29):
                with patch('kkamang.ui.field.ghost') as draw:
                    app.draw()
                assert app.core.snapshot() == state
                x, y = state['motion']['ghost_positions']['A']
                tile = min(978 / 16, 550 / 9)
                expected = (31 + (978 - tile * 16) / 2 + (x + .5) * tile,
                            106 + (550 - tile * 9) / 2 + (y + .95) * tile)
                assert draw.call_args.args[1] == pytest.approx(expected)
        app.transition('pause')
        frozen = app.core.snapshot()
        app.advance(5)
        assert app.core.snapshot() == frozen
        app.transition('game')
        assert app.core.snapshot() == frozen
    finally:
        env.close()
        app.close()


def test_observation_exposes_all_physical_centers_and_stun_and_rejects_v6():
    game = RealtimeGameCore(load_level(11))
    chaser_fixture(game)
    state = game.advance(Action.WAIT, FRAME_SECONDS * 29).snapshot
    observation = encode_realtime_observation(state)
    assert observation.shape == (630,)
    assert not observation[623:627].any()
    np.testing.assert_allclose(observation[627:629],
                               np.array(state['motion']['chaser_position']) / [15, 8])
    assert observation[629] == 1
    with pytest.raises(ValueError, match='version mismatch'):
        RealtimeMansionEnv(expected_observation_version='vector_v6_realtime')
    with pytest.raises(ValueError, match='version mismatch'):
        RealtimeMansionEnv(expected_observation_version='vector_v7_realtime_contact')
    with pytest.raises(ValueError, match='version mismatch'):
        RealtimeMansionEnv(expected_observation_version='vector_v8_realtime_body')
    state['motion']['chaser_position'][0] = float('nan')
    with pytest.raises(ValueError):
        encode_realtime_observation(state)


def test_replay_reproduces_actual_contact_stun_and_resumption():
    from tools.replay_validation import replay_game_record
    routes = json.loads((Path(__file__).resolve().parents[1] /
                         'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']
    route = next(r for r in routes if r['level_id'] == 'stage_11' and r['target'] == 14)
    game = RealtimeGameCore(load_level(11))
    for action in route['actions']:
        game.advance(action, .3)
        if game.snapshot()['quest']['collected']:
            break
    for _ in range(600):
        result = game.step(Action.WAIT)
        if any(event['type'] == 'damage' for event in result.events):
            break
        assert not result.terminated
    assert game.snapshot()['damage_count'] > 0
    game.advance(Action.WAIT, .3)
    record = game.replay_record()
    assert replay_game_record(record) == game.snapshot()
    record['observation_version'] = 'vector_v6_realtime'
    with pytest.raises(ValueError, match='incompatible'):
        replay_game_record(record)
