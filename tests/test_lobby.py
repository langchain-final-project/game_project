"""완료 기록에 따른 쉼터 표시와 재시도·그리기의 독립성을 검증한다."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.ui import lobby


def render(completed, **kwargs):
    surface = pygame.Surface((1280, 720))
    lobby.draw_lobby(surface, completed, **kwargs)
    return pygame.image.tobytes(surface, 'RGB')


@pytest.mark.parametrize('stage,expected', [
    (1, {'A', '수건 선반'}),
    (2, {'안내 지도'}),
    (3, {'발자국 사진'}),
    (4, {'정원 안내서', 'C'}),
    (5, {'스카프 고리'}),
    (6, {'씨앗 주머니'}),
    (7, {'잃어버린 악보', 'B'}),
    (8, {'찻잔 받침'}),
    (9, {'커튼 끈'}),
    (10, {'담요'}),
    (11, {'마지막 악보'}),
])
def test_each_clear_adds_only_the_requested_objects_and_keeps_previous_ones(stage, expected):
    previous = {spec.name for _, spec in lobby.lobby_layers(range(1, stage))}
    current = {spec.name for _, spec in lobby.lobby_layers(range(1, stage + 1))}
    assert previous <= current
    assert current - previous == expected


def test_empty_lobby_contains_only_cat_and_nonconsecutive_ids_do_not_unlock_other_ghosts():
    assert [spec.name for _, spec in lobby.lobby_layers([])] == ['까망이']
    assert {spec.name for _, spec in lobby.lobby_layers([7])} == {'까망이', 'B', '잃어버린 악보'}


def test_repeated_or_reordered_completion_ids_render_the_same_room_without_mutating_input():
    completed = [7, 1, 4, 2, 7, 4]
    original = deepcopy(completed)
    assert render(completed) == render([1, 2, 4, 7])
    assert completed == original


@pytest.mark.parametrize('with_display', [False, True])
def test_final_lobby_is_exactly_the_supplied_after_art_without_duplicate_characters(with_display):
    path = Path(__file__).resolve().parents[1] / 'assets/art/lobby/after.png'
    lobby._image.cache_clear()
    lobby._scene.cache_clear()
    if with_display:
        pygame.display.init()
        pygame.display.set_mode((1, 1), pygame.HIDDEN)
    else:
        pygame.display.quit()
    try:
        expected = pygame.transform.smoothscale(pygame.image.load(str(path)), (1280, 720))
        expected_pixels = pygame.image.tobytes(expected, 'RGB')
        for background in ((0, 0, 0), (255, 255, 255)):
            surface = pygame.Surface((1280, 720))
            surface.fill(background)
            lobby.draw_lobby(surface, range(1, 13))
            assert pygame.image.tobytes(surface, 'RGB') == expected_pixels
        assert render([12]) != expected_pixels
    finally:
        lobby._image.cache_clear()
        lobby._scene.cache_clear()
        pygame.display.quit()


def test_before_view_shows_the_initial_room_without_erasing_progress():
    completed = list(range(1, 13))
    original = completed.copy()
    assert render(completed, before=True) == render([])
    assert completed == original
    assert render(completed) != render([])


def test_dialogue_background_hides_characters_even_after_all_stages_are_complete():
    lobby._scene.cache_clear()
    with patch.object(lobby, '_sprite', wraps=lobby._sprite) as sprite:
        render(range(1, 13), show_actors=False)
    assert len(sprite.call_args_list) == 11
    assert all(call.args[0] == 'item.png' for call in sprite.call_args_list)


def test_home_and_before_toggle_only_adds_collection_discovery(tmp_path):
    from kkamang.app import GameApp
    from kkamang.ui import menus

    game = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    try:
        game.campaign.data['completed'] = [1, 2, 3, 4]
        assert game.store.save(game.campaign)
        original = deepcopy(game.campaign.data)
        saved_bytes = game.store.path.read_bytes()
        game.transition('home')
        with patch.object(menus, 'draw_lobby', wraps=lobby.draw_lobby) as draw:
            game.draw()
            assert draw.call_args.args[1] == [1, 2, 3, 4]
            assert draw.call_args.kwargs['before'] is False
            game.dispatch('toggle_home_view')
            game.draw()
            assert draw.call_args.kwargs['before'] is True
            game.dispatch('toggle_home_view')
            game.draw()
            assert draw.call_args.kwargs['before'] is False
        original['collection']['unlocked'] = [1, 2, 7]
        original['collection']['discoveries'] = ['shelter_before', 'shelter_return']
        assert game.campaign.data == original
        previous_save = json.loads(saved_bytes)
        current_save = json.loads(game.store.path.read_bytes())
        previous_save['collection'] = original['collection']
        assert current_save == previous_save
    finally:
        game.close()


@pytest.fixture
def story_game(tmp_path):
    from kkamang.app import GameApp

    game = GameApp(save_path=tmp_path / 'story.json', headless=True, no_audio=True)
    yield game
    game.close()


@pytest.mark.parametrize('stage', range(1, 13))
def test_twelve_room_clears_replay_three_silent_pictures_without_dialogue_portraits(
        story_game, stage):
    from kkamang.ui import menus

    game = story_game
    game.campaign.data['completed'] = list(range(1, stage + 1))

    game.campaign.data['photos'] = list(range(1, 13))
    original = deepcopy(game.campaign.data)
    game.start_memory(stage, replay=True)
    memory_id = (stage - 1) // 4 + 1
    expected_regions = [number <= stage for number in
                        range((memory_id - 1) * 4 + 1, memory_id * 4 + 1)]
    with patch.object(menus, '_memory_art', wraps=menus._memory_art) as picture, \
            patch.object(menus, 'draw_story_character') as portrait, \
            patch.object(menus, 'draw_story_item') as item:
        game.draw()
    assert picture.called
    assert all(call.args[1] == memory_id and call.args[2] == expected_regions
               for call in picture.call_args_list)
    portrait.assert_not_called()
    item.assert_not_called()
    assert game.context()['silent'] and game.context()['dialogue'] is None
    assert game.dialogues == []
    assert game.campaign.data == original
    assert {'story_next', 'story_skip'} <= {control['id'] for control in game.controls}


def test_prologue_uses_its_png_without_procedural_or_dialogue_portraits(story_game):
    from kkamang.ui import menus

    game = story_game
    original = deepcopy(game.campaign.data)
    game.start_prologue()
    with patch.object(menus, 'draw_cinematic', wraps=menus.draw_cinematic) as cinematic, \
            patch.object(menus, '_prologue_picture', wraps=menus._prologue_picture) as picture, \
            patch.object(menus, 'draw_story_character') as portrait, \
            patch.object(menus, 'draw_story_item') as item:
        game.draw()
    cinematic.assert_not_called()
    assert picture.called
    assert all(call.args[0] is game.canvas and call.args[1] is game.assets
               for call in picture.call_args_list)
    portrait.assert_not_called()
    item.assert_not_called()
    assert game.dialogues == [] and game.context()['dialogue'] is None
    assert game.campaign.data == original


@pytest.mark.parametrize('dialogue,expected', [
    (None, None),
    (('', '설명'), None),
    (('까망이', '...'), None),
    (('보송', '...'), 'A'),
    (('이파리', '...'), 'B'),
    (('도레', '...'), 'C'),
    ({'speaker': '보송', 'line': '...'}, 'A'),
    ({'speaker': '이파리', 'line': '...'}, 'B'),
    ({'speaker': '도레', 'line': '...'}, 'C'),
])
def test_optional_speaker_helper_selects_only_the_actual_speaker(dialogue, expected):
    from kkamang.ui import menus

    assert menus._story_ghost_id({'dialogue': dialogue}) == expected


@pytest.mark.parametrize('character_id', ['cat', 'A', 'B', 'C'])
def test_lobby_character_crop_and_optional_portrait_helper_render_independently(character_id):
    spec = lobby.CHARACTERS[character_id]
    source = lobby._image('character.png')
    assert source.get_rect().contains(pygame.Rect(spec.crop))
    surface = pygame.Surface((440, 360), pygame.SRCALPHA)
    target = pygame.Rect(83, 41, 250, 280)
    with patch.object(lobby, '_sprite', wraps=lobby._sprite) as sprite:
        lobby.draw_story_character(surface, character_id, target)
    assert sprite.call_args.args[0] == 'character.png'
    assert sprite.call_args.args[1].crop == spec.crop
    assert sprite.call_args.args[1].size == target.size
    painted = surface.get_bounding_rect(min_alpha=10)
    assert painted.w > 0 and painted.h > 0
    assert target.contains(painted)


@pytest.mark.parametrize('stage', range(1, 13))
def test_twelve_png_item_crops_and_optional_item_helper_render_independently(stage):
    spec = lobby.STORY_ITEMS[stage]
    source = lobby._image('item.png')
    assert source.get_rect().contains(pygame.Rect(spec.crop))
    surface = pygame.Surface((440, 360), pygame.SRCALPHA)
    target = pygame.Rect(83, 41, 250, 280)
    with patch.object(lobby, '_sprite', wraps=lobby._sprite) as sprite:
        lobby.draw_story_item(surface, stage, target)
    assert sprite.call_args.args[0] == 'item.png'
    assert sprite.call_args.args[1].crop == spec.crop
    assert sprite.call_args.args[1].size == target.size
    painted = surface.get_bounding_rect(min_alpha=10)
    assert painted.w > 0 and painted.h > 0
    assert target.contains(painted)


@pytest.mark.parametrize('scene,home_view,active,released,expected_before,show_residents', [
    ('home', 'after', None, (), False, True),
    ('home', 'before', 'happy_all', (0, 1, 2), True, True),
    ('home', 'after', 'ghost_a', (0,), False, True),
    ('home', 'after', 'sad', (), False, False),
    ('album', 'before', 'sad', (), False, False),
])
def test_menu_background_forwards_current_world_state_to_main_lobby_art(
        scene, home_view, active, released, expected_before, show_residents):
    from kkamang.ui import menus

    surface = pygame.Surface((1280, 720))
    completed = [1, 4, 7]
    context = {'campaign': {'completed': completed}, 'scene': scene,
               'home_view': home_view, 'active_ending': active,
               'released_variants': released}
    original = deepcopy(context)
    with patch.object(menus, 'draw_lobby') as draw:
        menus._background(surface, None, ctx=context, actors=False)
    assert draw.call_args.args == (surface, completed)
    assert draw.call_args.kwargs == {
        'before': expected_before, 'show_actors': False,
        'released_variants': released, 'show_residents': show_residents,
    }
    assert context == original


@pytest.mark.parametrize('screen', ['title', 'result'])
def test_title_and_result_use_the_relocated_sitting_cat_png(story_game, screen):
    game = story_game
    game.transition(screen)
    with patch.object(game.assets, 'icon', wraps=game.assets.icon) as icon:
        game.draw()
    sitting_paths = [call.args[0] for call in icon.call_args_list
                     if call.args[0].endswith('kkamang_wait.png')]
    assert sitting_paths and set(sitting_paths) == {'art/character/kkamang_wait.png'}
