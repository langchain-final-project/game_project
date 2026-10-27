"""Corner assistance must be escapable, bounded and identical in desktop/RL."""
from copy import deepcopy
import math
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np
import pygame
import pytest

from kkamang.app import GameApp
from kkamang.core import Action
from kkamang.env import RealtimeMansionEnv
from kkamang.levels import load_level
from kkamang.realtime import (CORNER_ASSIST_DISTANCE, FRAME_SECONDS,
                             FRAMES_PER_TICK, RealtimeGameCore)


def place(core, position):
    core._motion_position = list(position)
    core._state['position'] = [math.floor(v + .5) for v in position]


def corner_fixture(core, position, *, gate=False):
    grid = [['#' if x in (0, 15) or y in (0, 8) else '.'
             for x in range(16)] for y in range(9)]
    if not gate:
        grid[4][7] = '#'
    core._state['grid'] = [''.join(row) for row in grid]
    core._state['ghosts'] = []
    core._ghost_origins = {}
    core._state['chaser'] = None
    core._chaser_origin = None
    if gate:
        core._state['shortcuts'][0].update(gate=[7, 4], opened=False)
    else:
        core._state['shortcuts'] = []
    place(core, position)


@pytest.mark.parametrize('position,action,side_axis,alignment', [
    ((6.6, 3.22), Action.DOWN, 0, 6.22),
    ((7.4, 3.22), Action.DOWN, 0, 7.78),
    ((6.6, 4.78), Action.UP, 0, 6.22),
    ((7.4, 4.78), Action.UP, 0, 7.78),
    ((6.22, 3.6), Action.RIGHT, 1, 3.22),
    ((6.22, 4.4), Action.RIGHT, 1, 4.78),
    ((7.78, 3.6), Action.LEFT, 1, 3.22),
    ((7.78, 4.4), Action.LEFT, 1, 4.78),
])
def test_eight_corner_approaches_slide_then_follow_input_without_tunneling(
        position, action, side_axis, alignment):
    core = RealtimeGameCore(load_level(11))
    corner_fixture(core, position)
    before = core.snapshot()
    assert core.can_move(action)
    assert core.snapshot() == before and not core.actions
    previous = list(position)
    for _ in range(12):
        current = core.step(action).snapshot['motion']['position']
        changes = [abs(current[i] - previous[i]) for i in (0, 1)]
        assert min(changes) < 1e-10  # No diagonal movement or corner cutting.
        assert math.dist(previous, current) <= 1 / FRAMES_PER_TICK + 1e-10
        assert all(not core._blocked(*cell) for cell in core._body_cells(current))
        previous = current
    assert current[side_axis] == pytest.approx(alignment)
    assert current[1 - side_axis] != position[1 - side_axis]
    assert core.snapshot()['motion']['distance'] <= 12 / FRAMES_PER_TICK + 1e-10


def test_flat_wall_and_large_offsets_do_not_trigger_sideways_drift():
    core = RealtimeGameCore(load_level(11))
    for x in (7., 6.22 + CORNER_ASSIST_DISTANCE + .001):
        corner_fixture(core, (x, 3.22))
        assert not core.can_move(Action.DOWN)
        assert core.advance(Action.DOWN, .3).snapshot['motion']['position'] == [x, 3.22]
    corner_fixture(core, (6.6, 3.22))
    row = list(core._state['grid'][4])
    row[6] = '#'
    core._state['grid'][4] = ''.join(row)
    assert not core.can_move(Action.DOWN)
    assert core.advance(Action.DOWN, .3).snapshot['motion']['position'] == [6.6, 3.22]


def test_assist_stops_on_release_and_does_not_change_free_movement():
    core = RealtimeGameCore(load_level(11))
    corner_fixture(core, (6.6, 3.22))
    after = core.step(Action.DOWN).snapshot['motion']['position']
    assert after == pytest.approx([6.6 - 1 / FRAMES_PER_TICK, 3.22])
    assert core.step(Action.WAIT).snapshot['motion']['position'] == after
    assert core.step(Action.INTERACT).snapshot['motion']['position'] == after
    assert core.step(Action.UP).snapshot['motion']['position'] == pytest.approx(
        [after[0], after[1] - 1 / FRAMES_PER_TICK])


def test_closed_gate_corner_is_assisted_but_the_gate_body_still_blocks():
    core = RealtimeGameCore(load_level(11))
    corner_fixture(core, (6.6, 3.22), gate=True)
    core.advance(Action.DOWN, .2)
    x, y = core.snapshot()['motion']['position']
    assert x == pytest.approx(6.22) and y > 3.22
    assert all(not core._blocked(*cell) for cell in core._body_cells((x, y)))
    place(core, (7., 3.22))
    assert not core.can_move(Action.DOWN)
    core._state['shortcuts'][0]['opened'] = True
    assert core.step(Action.DOWN).snapshot['motion']['position'] == pytest.approx(
        [7., 3.22 + 1 / FRAMES_PER_TICK])


def test_corner_motion_is_identical_at_different_render_rates():
    results = []
    for fps in (10, 30, 60, 120):
        core = RealtimeGameCore(load_level(11))
        corner_fixture(core, (6.6, 3.22))
        for _ in range(fps):
            core.advance(Action.DOWN, 1 / fps)
        results.append(core.snapshot())
    assert all(state == results[0] for state in results)


def test_desktop_and_rl_share_assistance_pause_and_display(tmp_path):
    app = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    env = RealtimeMansionEnv(1)
    try:
        app.start_run()
        env.reset(seed=0)
        for core in (app.core, env.core):
            corner_fixture(core, (6.6, 3.22))
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_s))
        for _ in range(12):
            app.advance(FRAME_SECONDS)
            observation, _, terminated, truncated, info = env.step(Action.DOWN)
            assert not terminated and not truncated
            assert app.core.snapshot() == info['snapshot']
            np.testing.assert_allclose(observation[620:622],
                np.array(app.core.snapshot()['motion']['position']) / [15, 8])
        app.transition('pause')
        before = deepcopy(app.core.snapshot())
        app.advance(1.)
        assert app.core.snapshot() == before
    finally:
        env.close()
        app.close()


def test_assisted_motion_replays_from_real_stage_start():
    from tools.replay_validation import replay_game_record
    core = RealtimeGameCore(load_level(11))
    core.advance(Action.UP, 54 * FRAME_SECONDS)
    core.advance(Action.RIGHT, 65 * FRAME_SECONDS)
    before = core.snapshot()['motion']['position']
    core.advance(Action.DOWN, 18 * FRAME_SECONDS)
    after = core.snapshot()['motion']['position']
    assert after[0] < before[0] and after[1] > before[1]
    assert replay_game_record(core.replay_record()) == core.snapshot()
