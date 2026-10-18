"""PNG atlas boundaries, fixed pivots and distance-based cat animation."""
from unittest.mock import patch

import pygame
import pytest

from kkamang.ui import cat_sprites
from kkamang.ui.illustration import cat, cat_frame


@pytest.mark.parametrize('direction', ['up', 'down', 'left', 'right'])
def test_six_walk_frames_are_distinct_and_fit_the_same_transparent_canvas(direction):
    frames = [cat_sprites.sprite_frame(256, direction, True, index) for index in range(6)]
    assert len({pygame.image.tobytes(frame, 'RGBA') for frame in frames}) == 6
    for frame in frames:
        assert frame.get_size() == (256, 288)
        bounds = frame.get_bounding_rect(min_alpha=10)
        assert 0 < bounds.width < 256 and 0 < bounds.height < 288
        assert frame.get_at((0, 0)).a == 0
        assert frame.get_at((255, 287)).a == 0
        # The atlas foot anchors align to a common ground line, independent of
        # the changing tail width. Resampling may leave one transparent pixel.
        assert abs(bounds.bottom - cat_sprites.GROUND_ANCHOR[1]) <= 1


@pytest.mark.parametrize('direction', ['up', 'down', 'left', 'right'])
def test_idle_is_stable_and_independent_of_unimplemented_expression_and_scarf(direction):
    reference = pygame.image.tobytes(cat_frame(128, direction), 'RGBA')
    for phase, mood, scarf in ((0, 'happy', 'peach'), (4, 'surprised', 'lavender'),
                                (7, 'collect', 'default')):
        assert pygame.image.tobytes(cat_frame(128, direction, phase, mood, scarf), 'RGBA') == reference


def test_original_sheet_pixels_supply_the_idle_frame_and_all_rectangles_are_in_bounds():
    sheet = pygame.image.load(str(cat_sprites.SHEET_PATH))
    assert sheet.get_size() == cat_sprites.SHEET_SIZE
    for direction in cat_sprites.IDLE_FRAMES:
        regions = [cat_sprites.IDLE_FRAMES[direction], *cat_sprites.WALK_ATLAS[direction]]
        for rect, anchor in regions:
            assert sheet.get_rect().contains(pygame.Rect(rect))
            assert rect[0] <= anchor[0] <= rect[0] + rect[2]
    rect, anchor = cat_sprites.IDLE_FRAMES['down']
    scale = cat_sprites.IDLE_SCALE['down']
    source = pygame.transform.smoothscale(sheet.subsurface(rect),
                                         (round(rect[2] * scale), round(rect[3] * scale)))
    expected = pygame.Surface(cat_sprites.CANVAS_SIZE, pygame.SRCALPHA)
    expected.blit(source, (cat_sprites.GROUND_ANCHOR[0] - round((anchor[0] - rect[0]) * scale),
                           cat_sprites.GROUND_ANCHOR[1] - round((anchor[1] - rect[1]) * scale)))
    actual = cat_sprites.sprite_frame(256, 'down')
    assert pygame.image.tobytes(actual, 'RGBA') == pygame.image.tobytes(expected, 'RGBA')


def test_walk_progress_uses_all_six_frames_loops_and_freezes_when_stopped():
    surface = pygame.Surface((300, 300), pygame.SRCALPHA)
    phases = []
    with patch('kkamang.ui.illustration.cat_frame', wraps=cat_frame) as frame:
        for index in range(7):
            cat(surface, (150, 250), walking=True, walk_progress=index / 6)
            phases.append(frame.call_args.args[2])
        assert len(set(phases[:6])) == 6
        assert phases[0] == phases[6]
    first = pygame.Surface((300, 300), pygame.SRCALPHA)
    second = first.copy()
    cat(first, (150, 250), time=1, walking=False, walk_progress=.2)
    cat(second, (150, 250), time=100, walking=False, walk_progress=.8)
    assert pygame.image.tobytes(first, 'RGBA') == pygame.image.tobytes(second, 'RGBA')


def test_frame_cache_reuses_scaled_surface_and_does_not_require_a_display():
    frame = cat_sprites.sprite_frame(71, 'left', True, 3)
    assert cat_sprites.sprite_frame(71, 'left', True, 3) is frame
    assert frame.get_size() == (71, 79)
