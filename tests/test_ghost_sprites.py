"""유령 식별·방향별 PNG·게임 판정과 분리된 시선 처리를 검증한다."""
from copy import deepcopy
import math
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.app import GameApp
from kkamang.ui import ghost_sprites, illustration
from kkamang.ui.likes import VARIANTS


@pytest.fixture
def app(tmp_path):
    game = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    game.campaign.data['completed'] = list(range(1, 11))
    game.stage = 11
    game.start_run()
    yield game
    game.close()


@pytest.mark.parametrize('ghost_id', ['A', 'B', 'C'])
def test_eight_direction_poses_are_distinct_transparent_and_ground_aligned(ghost_id):
    frames = [ghost_sprites.ghost_frame(256, ghost_id, direction)
              for direction in ghost_sprites.DIRECTION_INDEX]
    assert len({pygame.image.tobytes(frame, 'RGBA') for frame in frames}) == 8
    for frame in frames:
        assert frame.get_size() == ghost_sprites.CANVAS_SIZE
        bounds = frame.get_bounding_rect(min_alpha=10)
        assert 0 < bounds.width < frame.get_width()
        assert 0 < bounds.height < frame.get_height()
        assert abs(bounds.bottom - ghost_sprites.GROUND_ANCHOR[1]) <= 1
        assert frame.get_at((0, 0)).a == frame.get_at((255, 281)).a == 0


def test_ghost_ids_select_different_files_and_scaled_frames_are_cached():
    assert ghost_sprites.GHOST_FILES == {'A': 'a.png', 'B': 'b.png', 'C': 'c.png'}
    frames = [ghost_sprites.ghost_frame(75, ident, 'down') for ident in ('A', 'B', 'C')]
    assert len({pygame.image.tobytes(frame, 'RGBA') for frame in frames}) == 3
    assert ghost_sprites.ghost_frame(75, 'A', 'down') is frames[0]


@pytest.mark.parametrize('name,expected', [('bosong', 'A'), ('ipari', 'B'), ('dore', 'C')])
def test_existing_resident_variants_keep_their_character_identity(name, expected):
    surface = pygame.Surface((200, 200), pygame.SRCALPHA)
    with patch.object(illustration, 'ghost_frame', wraps=ghost_sprites.ghost_frame) as draw:
        illustration.ghost(surface, (100, 180), 100, resident=True, variant=VARIANTS[name])
        assert draw.call_args.args == (100, expected, 'down')


def test_bobbing_moves_only_the_rendered_image():
    first = pygame.Surface((200, 200), pygame.SRCALPHA)
    second = first.copy()
    illustration.ghost(first, (100, 180), 100, time=0, ghost_id='C', direction='left')
    illustration.ghost(second, (100, 180), 100, time=math.pi / 4,
                        ghost_id='C', direction='left')
    a, b = first.get_bounding_rect(), second.get_bounding_rect()
    assert b.x == a.x and b.y == a.y + 3
    assert b.size == a.size


def test_field_keeps_all_three_ids_and_directions_separate_when_draws_are_deferred(app):
    state = app.core._state
    state['ghosts'][0]['position'] = [8, 2]
    state['ghosts'][1]['position'] = [13, 6]
    state['chaser'].update(active=True, position=[12, 1])
    previous = app.core.snapshot()
    previous['ghosts'][0]['position'] = [7, 2]
    previous['ghosts'][1]['position'] = [13, 5]
    previous['ghosts'].reverse()
    previous['chaser']['position'] = [12, 2]
    app.previous = previous
    before = app.core.snapshot()
    with patch('kkamang.ui.field.ghost') as draw:
        app.draw()
        assert draw.call_count == 3
        actual = {call.kwargs['ghost_id']: call.kwargs['direction']
                  for call in draw.call_args_list}
    assert actual == {'A': 'right', 'B': 'down', 'C': 'up'}
    assert app.core.snapshot() == before


def test_chaser_retains_last_direction_during_contact_stun_and_pause(app):
    state = app.core._state
    state['ghosts'] = []
    state['position'] = [11, 1]
    app.core._motion_position = [11, 1]
    state['quest']['collected'] = True
    state['chaser'].update(active=True, position=[12, 2], countdown=1)
    app.advance(.3)
    assert app.ghost_facings['C'] == 'up'
    app.advance(.3)
    assert app.ghost_facings['C'] == 'left'
    assert state['chaser']['countdown'] == 4
    app.advance(.3)
    assert state['chaser']['countdown'] == 3
    assert app.ghost_facings['C'] == 'left'
    before = app.core.snapshot()
    app.transition('pause')
    app.advance(10)
    app.transition('game')
    with patch('kkamang.ui.field.ghost') as draw:
        app.draw()
        assert draw.call_args.kwargs['direction'] == 'left'
    assert app.core.snapshot() == before
    app.start_run()
    assert app.ghost_facings == {}


def test_initial_patrol_heading_does_not_change_during_its_countdown(app):
    app.core._state['ghosts'][0]['countdown'] = 3
    def direction():
        with patch('kkamang.ui.field.ghost') as draw:
            app.draw()
        return next(call.kwargs['direction'] for call in draw.call_args_list
                    if call.kwargs['ghost_id'] == 'A')
    initial = direction()
    app.advance(.3)
    assert direction() == initial


def test_heading_updates_do_not_mutate_simulation_snapshots(app):
    before = app.core.snapshot()
    after = deepcopy(before)
    after['ghosts'][0]['position'][0] += 1
    app._remember_ghost_facings(before, after)
    assert app.core.snapshot() == before
    assert before == app.core.snapshot()
    assert app.ghost_facings['A'] == 'right'


@pytest.mark.parametrize('identity,direction', [('D', 'down'), ('a', 'down'), ('A', 'unknown')])
def test_invalid_identity_or_direction_is_rejected(identity, direction):
    with pytest.raises(ValueError):
        ghost_sprites.ghost_frame(50, identity, direction)
