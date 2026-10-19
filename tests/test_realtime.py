"""Free movement, shared human/RL physics and independent world-clock checks."""
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
from gymnasium.utils.env_checker import check_env

from kkamang.app import GameApp
from kkamang.core import Action
from kkamang.env import RealtimeMansionEnv, encode_realtime_observation
from kkamang.levels import load_level
from kkamang.realtime import (FRAME_SECONDS, FRAMES_PER_TICK, PLAYER_RADIUS,
                              RealtimeGameCore, REALTIME_OBSERVATION_VERSION)

ROOT = Path(__file__).resolve().parents[1]
ROUTES = json.loads((ROOT / 'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']


def place(game, position):
    """Isolated test placement, not a gameplay teleport API."""
    game._motion_position = list(position)
    game._state['position'] = [math.floor(value + .5) for value in position]


@pytest.fixture
def app(tmp_path):
    game = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    game.start_run()
    yield game
    game.close()


def key(app, event_type, value):
    app.handle_event(pygame.event.Event(event_type, key=value))


def test_mid_cell_direction_changes_on_next_frame_without_waiting_for_tick(app):
    start_x, start_y = app.core.config['start']
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.advance(.1)
    first = app.core.snapshot()
    x, y = first['motion']['position']
    assert start_x < x < start_x + .5 and y == start_y
    assert first['tick'] == 0
    key(app, pygame.KEYDOWN, pygame.K_s)
    app.advance(FRAME_SECONDS)
    second = app.core.snapshot()
    assert second['motion']['position'][0] == x
    assert second['motion']['position'][1] > y
    assert second['tick'] == 0
    assert app.facing == 'down'
    with patch('kkamang.ui.field.cat') as draw_cat:
        app.draw()
        first_foot = draw_cat.call_args.args[1]
    app.advance(FRAME_SECONDS)
    with patch('kkamang.ui.field.cat') as draw_cat:
        app.draw()
        second_foot = draw_cat.call_args.args[1]
    assert second_foot[0] == first_foot[0]
    assert second_foot[1] > first_foot[1]


def test_start_after_idle_and_release_both_take_one_frame(app):
    app.advance(.41)
    before = app.core.snapshot()
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.advance(FRAME_SECONDS)
    moving = app.core.snapshot()
    assert moving['motion']['position'][0] > before['motion']['position'][0]
    assert moving['tick'] == before['tick']
    key(app, pygame.KEYUP, pygame.K_d)
    app.advance(FRAME_SECONDS)
    assert app.core.snapshot()['motion']['position'] == moving['motion']['position']
    assert not app.core.snapshot()['motion']['moving']


def test_reversing_mid_cell_does_not_snap_to_a_cell_center(app):
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.advance(.1)
    before = app.core.snapshot()['motion']['position']
    key(app, pygame.KEYDOWN, pygame.K_a)
    app.advance(FRAME_SECONDS)
    after = app.core.snapshot()['motion']['position']
    assert after[0] == pytest.approx(before[0] - 1 / FRAMES_PER_TICK)
    assert after[1] == before[1]


def test_rapid_input_and_interaction_do_not_accelerate_world(app):
    for _ in range(40):
        for value in (pygame.K_w, pygame.K_d, pygame.K_e):
            key(app, pygame.KEYDOWN, value)
            key(app, pygame.KEYUP, value)
    assert app.core.snapshot()['tick'] == 0
    app.advance(.29)
    assert app.core.snapshot()['tick'] == 0
    app.advance(.01)
    assert app.core.snapshot()['tick'] == 1
    assert app.core.snapshot()['remaining_ticks'] == 79


def test_short_tap_survives_one_interaction_frame_then_stops(app):
    for value in (pygame.K_d, pygame.K_e):
        key(app, pygame.KEYDOWN, value)
        key(app, pygame.KEYUP, value)
    app.advance(FRAME_SECONDS * 3)
    assert app.core.actions == [Action.INTERACT, Action.RIGHT, Action.WAIT]
    assert app.core.snapshot()['motion']['position'][0] == pytest.approx(
        app.core.config['start'][0] + 1 / 18)


def test_pause_preserves_fractional_position_and_clock_without_held_input(app):
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.advance(.115)
    before = app.core.snapshot()
    remainder = app.motion_accumulator
    key(app, pygame.KEYDOWN, pygame.K_ESCAPE)
    app.advance(10)
    assert app.core.snapshot() == before
    app.dispatch('resume')
    assert app.motion_accumulator == remainder
    app.advance(FRAME_SECONDS)
    assert app.core.snapshot()['motion']['position'] == before['motion']['position']
    assert app.core.actions[-1] == Action.WAIT
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.handle_event(pygame.event.Event(pygame.WINDOWFOCUSLOST))
    paused = app.core.snapshot()
    app.advance(10)
    assert app.scene == 'pause' and app.core.snapshot() == paused


def test_restart_discards_fractional_position_and_input(app):
    key(app, pygame.KEYDOWN, pygame.K_d)
    app.advance(.115)
    app.start_run()
    assert app.motion_accumulator == 0
    assert app.core.snapshot()['motion']['position'] == app.core.config['start']
    app.advance(FRAME_SECONDS)
    assert app.core.actions == [Action.WAIT]


def test_wall_blocks_body_without_preventing_immediate_parallel_turn():
    game = RealtimeGameCore(load_level(1))
    place(game, [6, 2])
    blocked = game.advance(Action.RIGHT, .6).snapshot
    assert blocked['motion']['position'] == pytest.approx([6.5 - PLAYER_RADIUS, 2])
    assert blocked['position'] == [6, 2]
    game.step(Action.UP)
    assert game.snapshot()['motion']['position'][1] < 2
    assert game.snapshot()['motion']['position'][0] == blocked['motion']['position'][0]


def test_closed_gate_blocks_body_and_open_gate_allows_passage():
    game = RealtimeGameCore(load_level(5))
    gate = game._state['shortcuts'][0]
    x, y = gate['gate']
    for action, dx, dy in ((3, -1, 0), (2, 1, 0), (1, 0, -1), (0, 0, 1)):
        if not game._blocked(x + dx, y + dy):
            break
    place(game, [x + dx, y + dy])
    game.advance(action, .6)
    assert game.snapshot()['position'] != [x, y]
    gate['opened'] = True
    game.advance(action, .3)
    assert game.snapshot()['position'] == [x, y]


@pytest.mark.parametrize('stage', [1, 7, 11])
def test_fixed_step_results_do_not_depend_on_render_cadence(stage):
    results = []
    for fps in (10, 30, 60, 120):
        game = RealtimeGameCore(load_level(stage))
        for _ in range(fps):
            game.advance(Action.RIGHT, 1 / fps)
        results.append(game.snapshot())
    assert all(result == results[0] for result in results)
    assert results[0]['tick'] == 3
    assert results[0]['motion']['frame'] == 60


def test_warmth_is_collected_between_world_ticks_once():
    game = RealtimeGameCore(load_level(1))
    result = game.advance(Action.RIGHT, .15)
    assert result.snapshot['tick'] == 0
    assert result.snapshot['position'] == [5, 1]
    assert result.snapshot['carried'] == 1
    assert sum(event['type'] == 'collect' for event in result.events) == 1
    assert game.step(Action.WAIT).snapshot['carried'] == 1


def test_contact_between_ticks_damages_once_and_stuns_chaser():
    game = RealtimeGameCore(load_level(11))
    game._state['ghosts'] = []
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=[13, 1], countdown=1)
    place(game, [12, 1])
    result = game.advance(Action.RIGHT, .15)
    assert result.snapshot['courage'] == 2
    assert result.snapshot['protection'] == 3
    assert result.snapshot['chaser']['countdown'] == 4
    game.advance(Action.WAIT, .1)
    assert game.snapshot()['damage_count'] == 1
    assert game.snapshot()['chaser']['countdown'] == 4
    game.advance(Action.WAIT, .05)
    assert game.snapshot()['chaser']['countdown'] == 3
    assert game.snapshot()['protection'] == 2


def test_patrol_swap_on_world_boundary_is_still_a_contact():
    game = RealtimeGameCore(load_level(11))
    game._state['ghosts'] = [dict(id='A', position=[13, 1], route=[[13, 1], [12, 1]],
                                  route_index=0, direction=1, countdown=1,
                                  ticks_per_move=1, next_position=[12, 1])]
    place(game, [12.49, 1])
    game.advance(Action.WAIT, FRAME_SECONDS * 17)
    result = game.step(Action.RIGHT)
    contact = next(event for event in result.events if event['type'] == 'contact')
    assert contact['contacts'][0]['kind'] == 'swap'
    assert result.snapshot['courage'] == 2


def test_invincibility_pickup_protects_same_frame_and_keeps_chaser_stun():
    game = RealtimeGameCore(load_level(11))
    game._state['ghosts'] = []
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=[14, 6], countdown=1)
    place(game, [13, 6])
    result = game.advance(Action.RIGHT, .15)
    assert result.snapshot['invincibility']['collected']
    assert result.snapshot['invincibility']['remaining_ticks'] == 10
    assert result.snapshot['courage'] == 3
    assert result.snapshot['protection'] == 0
    assert result.snapshot['chaser']['countdown'] == 4
    assert any(event['type'] == 'contact' and event['item_invincible']
               for event in result.events)
    game.advance(Action.WAIT, .15)
    assert game.snapshot()['invincibility']['remaining_ticks'] == 9
    game.advance(Action.WAIT, 2.7)
    assert game.snapshot()['invincibility']['remaining_ticks'] == 0
    assert game.snapshot()['damage_count'] == 0


@pytest.mark.parametrize('value', [1, 3])
def test_magnet_collects_only_one_point_heat_across_wall_without_advancing_timer_per_frame(value):
    game = RealtimeGameCore(load_level(2))
    place(game, game._state['magnet']['position'])
    game._state['items'] = [dict(id='heat_00', position=[7, 4], value=value, active=True)]
    assert game._state['grid'][4][7] == '#'
    result = game.step(Action.WAIT)
    assert result.snapshot['carried'] == (1 if value == 1 else 0)
    assert result.snapshot['items'][0]['active'] == (value == 3)
    assert result.snapshot['magnet']['remaining_ticks'] == 13
    collected = [event for event in result.events if event['type'] == 'collect']
    assert len(collected) == (1 if value == 1 else 0)
    assert all(event['magnetic'] and event['value'] == 1 for event in collected)
    game.advance(Action.WAIT, FRAME_SECONDS * 17)
    assert game.snapshot()['magnet']['remaining_ticks'] == 12


def test_active_magnet_still_allows_direct_rich_heat_collection_in_realtime():
    config = load_level(1)
    config['magnet'] = {'position': [6, 3], 'duration_ticks': 13}
    game = RealtimeGameCore(config)
    place(game, [6, 2])
    game.advance(Action.DOWN, .3)
    nearby = game.advance(Action.LEFT, .3)
    assert nearby.snapshot['items'][8]['active']
    entered = game.advance(Action.UP, .3)
    collected = [event for event in entered.events if event['type'] == 'collect' and event['value'] == 3]
    assert len(collected) == 1 and not collected[0]['magnetic']
    assert not entered.snapshot['items'][8]['active']
    assert entered.snapshot['magnet']['remaining_ticks'] > 0


@pytest.mark.parametrize('stage', range(1, 13))
def test_realtime_env_observation_contract_and_gym_checker(stage):
    env = RealtimeMansionEnv(stage)
    obs, info = env.reset(seed=42)
    assert obs.shape == (623,)
    assert obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert info['observation_version'] == REALTIME_OBSERVATION_VERSION
    check_env(env, skip_render_check=True)
    env.close()


def test_desktop_and_realtime_learning_environment_use_identical_transitions(app):
    env = RealtimeMansionEnv(1)
    env.reset(seed=0)
    for key_value, action, frames in ((pygame.K_d, 3, 5), (pygame.K_s, 1, 4),
                                       (pygame.K_a, 2, 3), (pygame.K_e, 4, 1)):
        key(app, pygame.KEYDOWN, key_value)
        for _ in range(frames):
            app.advance(FRAME_SECONDS)
            obs, _, terminated, truncated, info = env.step(action)
            assert app.core.snapshot() == info['snapshot']
            np.testing.assert_array_equal(obs, encode_realtime_observation(app.core.snapshot()))
            assert not terminated and not truncated
        key(app, pygame.KEYUP, key_value)


def test_fractional_position_and_clock_phase_are_observed_and_old_models_rejected():
    env = RealtimeMansionEnv()
    obs, _ = env.reset(seed=0)
    moved, *_ = env.step(Action.RIGHT)
    assert moved[620] > obs[620]
    assert moved[622] == pytest.approx(1 / 18)
    assert moved[432] == obs[432]  # Still in the same occupied cell.
    with pytest.raises(ValueError, match='version mismatch'):
        RealtimeMansionEnv(expected_observation_version='vector_v5_invincibility')
    snapshot = env.core.snapshot()
    snapshot['motion']['position'][0] = float('nan')
    with pytest.raises(ValueError):
        encode_realtime_observation(snapshot)


def test_realtime_record_reproduces_changes_inside_one_world_tick():
    game = RealtimeGameCore(load_level(7), seed=42, randomize_phase=True)
    for action in [3] * 5 + [1] * 3 + [2] * 4 + [4, 5] * 12:
        game.step(action)
    record = game.replay_record()
    assert record['replay_version'] == '2.0' and record['physics_hz'] == 60
    other = RealtimeGameCore(load_level(7), seed=record['seed'])
    other.reset(phase_overrides=record['initial_phase'])
    for action in record['actions']:
        other.step(action)
    assert other.snapshot() == game.snapshot()
    from tools.replay_validation import replay_game_record
    assert replay_game_record(record) == game.snapshot()
    for field, value in (('physics_hz', 30), ('config_hash', 'wrong'),
                         ('observation_version', 'vector_v5_invincibility')):
        invalid = deepcopy(record)
        invalid[field] = value
        with pytest.raises(ValueError):
            replay_game_record(invalid)


def test_realtime_completion_still_settles_and_persists_once(app):
    # Stage 1 has no patrol. Walk the existing geometric route, allowing .3s for
    # each cardinal segment. E is a one-frame command in the realtime controller.
    from kkamang.campaign import SaveStore
    route = next(route for route in ROUTES
                 if route['level_id'] == 'stage_01' and route['target'] == 4)
    mapping = {0: pygame.K_w, 1: pygame.K_s, 2: pygame.K_a, 3: pygame.K_d, 4: pygame.K_e}
    for action in route['actions']:
        value = mapping.get(action, pygame.K_SPACE)
        key(app, pygame.KEYDOWN, value)
        app.advance(.3 if action != Action.INTERACT else FRAME_SECONDS)
        key(app, pygame.KEYUP, value)
    assert app.last_outcome['status'] == 'COMPLETE'
    assert app.last_outcome['banked'] == route['banked']
    assert SaveStore(app.store.path).load().data['completed'] == [1]
    assert len(app.campaign.data['settled_ids']) == 1
    record_file = next((app.store.path.parent / 'replays').glob('*.json'))
    from tools.replay_validation import replay_game_record
    record = json.loads(record_file.read_text(encoding='utf-8'))
    assert replay_game_record(record)['outcome'] == app.last_outcome
    assert app.display_elapsed_seconds == pytest.approx(
        app.core.snapshot()['motion']['frame'] * FRAME_SECONDS
    )
    context = app.context()
    assert context['display_elapsed_seconds'] == app.display_elapsed_seconds
    before = deepcopy(app.last_outcome)
    with patch('kkamang.ui.menus.text') as rendered_text:
        app.draw()
    labels = [call.args[2] for call in rendered_text.call_args_list]
    assert f'{app.display_elapsed_seconds:.1f}초' in labels
    assert app.last_outcome == before
    app.start_run()
    assert app.display_elapsed_seconds is None
    assert app.context()['display_elapsed_seconds'] is None


@pytest.mark.parametrize('movement_mode', ['grid', 'realtime'])
def test_countdown_display_is_smooth_and_read_only_across_pause(tmp_path, movement_mode):
    game = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True,
                   movement_mode=movement_mode)
    try:
        game.start_run()
        tick_seconds = game.core.config['tick_seconds']
        initial_seconds = game.core.snapshot()['remaining_ticks'] * tick_seconds

        def displayed_seconds():
            before = game.core.snapshot()
            with patch('kkamang.app.draw_game') as draw:
                game.draw()
            assert game.core.snapshot() == before
            return draw.call_args.kwargs['remaining_seconds']

        assert displayed_seconds() == pytest.approx(initial_seconds)
        game.advance(.1)
        assert game.core.snapshot()['tick'] == 0
        assert displayed_seconds() == pytest.approx(initial_seconds - .1)
        game.advance(.2)
        assert game.core.snapshot()['tick'] == 1
        assert displayed_seconds() == pytest.approx(initial_seconds - .3)

        before_pause = game.core.snapshot()
        game.transition('pause')
        game.advance(2.)
        assert game.core.snapshot() == before_pause
        game.transition('game')
        assert displayed_seconds() == pytest.approx(initial_seconds - .3)
    finally:
        game.close()


@pytest.mark.parametrize('route', ROUTES,
                         ids=lambda route: f"{route['level_id']}-target{route['target']}")
def test_existing_geometric_routes_still_allow_completion_in_realtime(route):
    # These are completion witnesses only. Continuous contact timing can cause
    # damage, so the old grid route's no-contact claim is deliberately not reused.
    game = RealtimeGameCore(load_level(route['level_id']))
    for action in route['actions']:
        result = game.advance(action, .3)
        if result.terminated:
            break
    assert result.terminated
    assert result.outcome['status'] == 'COMPLETE'
    assert result.outcome['banked'] >= route['target']
    assert result.outcome['quest_collected']


def test_default_controller_completes_and_saves_all_twelve_stages(app):
    from kkamang.campaign import SaveStore
    mapping = {0: pygame.K_w, 1: pygame.K_s, 2: pygame.K_a, 3: pygame.K_d, 4: pygame.K_e}
    for stage in range(1, 13):
        app.stage = stage
        app.start_run()
        route = next(route for route in ROUTES
                     if route['level_id'] == f'stage_{stage:02}' and route['target'] == 14)
        for action in route['actions']:
            value = mapping.get(action, pygame.K_SPACE)
            key(app, pygame.KEYDOWN, value)
            app.advance(.3)
            key(app, pygame.KEYUP, value)
        assert app.last_outcome['status'] == 'COMPLETE'
        assert app.last_outcome['banked'] == 14
    restored = SaveStore(app.store.path).load()
    assert restored.data['completed'] == list(range(1, 13))
    assert restored.data['photos'] == list(range(1, 13))
    assert len(restored.data['settled_ids']) == 12


@pytest.mark.parametrize('kind', ['patrol', 'chaser'])
def test_ghosts_still_animate_turns_along_cardinal_segments(app, kind):
    app.campaign.data['completed'] = list(range(1, 11))
    app.stage = 2 if kind == 'patrol' else 11
    app.start_run()
    state = app.core._state
    if kind == 'patrol':
        state['ghosts'][0].update(countdown=1, ticks_per_move=1)
    else:
        state['ghosts'] = []
        place(app.core, [11, 1])
        state['quest']['collected'] = True
        state['chaser'].update(active=True, position=[12, 2], countdown=1)

    def foot():
        with patch('kkamang.ui.field.ghost') as draw:
            app.draw()
            assert draw.call_count == 1
            return draw.call_args.args[1]

    app.advance(.3)
    start = foot()
    app.advance(.15)
    middle = foot()
    app.advance(.15)
    corner = foot()
    app.advance(.15)
    turned = foot()
    if kind == 'patrol':
        assert start[1] == middle[1] == corner[1]
        assert start[0] < middle[0] < corner[0]
        assert turned[0] == corner[0] and turned[1] > corner[1]
    else:
        assert start[0] == middle[0] == corner[0]
        assert start[1] > middle[1] > corner[1]
        assert turned[1] == corner[1] and turned[0] < corner[0]


def test_hurt_walking_and_shield_follow_continuous_position(app):
    from kkamang.ui import field
    app.core._state['protection'] = 2
    key(app, pygame.KEYDOWN, pygame.K_d)
    shields = []
    for _ in range(2):
        app.advance(.05)
        with patch.object(field, '_draw_png', wraps=field._draw_png) as png, \
                patch.object(field, 'cat') as cat:
            app.draw()
            assert cat.call_args.args[5] == 'surprised'
            assert cat.call_args.kwargs['walking']
            shields.append(next(call.args[3].copy() for call in png.call_args_list
                                if call.args[2] == 'protection'))
    assert shields[1].centerx > shields[0].centerx
    assert shields[1].centery == shields[0].centery


def test_app_cadence_does_not_change_movement_or_ghost_clock(app):
    states = []
    for fps in (10, 30, 60, 120):
        app.start_run()
        key(app, pygame.KEYDOWN, pygame.K_d)
        for _ in range(fps):
            app.advance(1 / fps)
        states.append(app.core.snapshot())
    assert all(state == states[0] for state in states)


def test_contract_file_matches_encoder_and_grid_planners_reject_free_movement():
    from tools.evaluate import TimedGreedyPolicy, find_path
    from tools.plan_routes import plan
    contract = json.loads((ROOT / 'data/levels/realtime_observation_contract.json')
                          .read_text(encoding='utf-8'))
    env = RealtimeMansionEnv()
    obs, _ = env.reset(seed=42)
    assert contract['observation_version'] == env.observation_version
    assert contract['shape'] == list(obs.shape)
    assert contract['frames_per_world_tick'] == FRAMES_PER_TICK
    assert contract['player_half_extent_tiles'] == PLAYER_RADIUS
    for operation in (lambda: TimedGreedyPolicy(env.core),
                      lambda: find_path(env.core, (2, 1)), lambda: plan(env.core)):
        with pytest.raises(ValueError, match='grid'):
            operation()


@pytest.mark.parametrize('seconds', [-1, float('nan'), float('inf'), True, '0.1'])
def test_invalid_elapsed_time_does_not_mutate_core(seconds):
    game = RealtimeGameCore(load_level(1))
    before = game.snapshot()
    with pytest.raises(ValueError):
        game.advance(Action.RIGHT, seconds)
    assert game.snapshot() == before and game.actions == []
