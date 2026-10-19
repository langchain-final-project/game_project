"""Saved completion IDs control the lobby, independently of retries and draw calls."""
from copy import deepcopy
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


def test_home_and_before_toggle_use_saved_progress_without_changing_the_save(tmp_path):
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
        assert game.campaign.data == original
        assert game.store.path.read_bytes() == saved_bytes
    finally:
        game.close()


@pytest.fixture
def story_game(tmp_path):
    from kkamang.app import GameApp

    game = GameApp(save_path=tmp_path / 'story.json', headless=True, no_audio=True)
    game.campaign.data['completed'] = [1]
    game.campaign.data['photos'] = [1]
    yield game
    game.close()


@pytest.mark.parametrize('stage,speaker,ghost_id', [
    (1, '보송', 'A'),
    (6, '이파리', 'B'),
    (7, '도레', 'C'),
])
def test_story_uses_lobby_portraits_without_changing_dialogue_or_progress(
        story_game, stage, speaker, ghost_id):
    from kkamang.ui import menus

    game = story_game
    game.campaign.data['completed'] = list(range(1, stage + 1))
    game.campaign.data['photos'] = list(range(1, stage + 1))
    game.start_memory(stage)
    game.dialogue_index = next(i for i, line in enumerate(game.dialogues) if line[0] == speaker)
    game.dialogue_time = 99
    dialogue = deepcopy(game.dialogues)
    campaign = deepcopy(game.campaign.data)
    with patch.object(menus, 'draw_story_character', wraps=lobby.draw_story_character) as portrait:
        with patch.object(lobby, '_sprite', wraps=lobby._sprite) as sprite:
            game.draw()
    # The app may paint again after choosing the initial keyboard focus.
    assert {call.args[1] for call in portrait.call_args_list} == {'cat', ghost_id}
    assert sum(call.args[1] == 'cat' for call in portrait.call_args_list) == sum(
        call.args[1] == ghost_id for call in portrait.call_args_list)
    painted_characters = [call.args[1].name for call in sprite.call_args_list
                          if call.args[0] == 'character.png']
    assert set(painted_characters) == {'까망이', ghost_id}
    assert game.dialogues == dialogue
    assert game.campaign.data == campaign
    assert {'story_next', 'story_skip'} <= {control['id'] for control in game.controls}


def test_all_twelve_memory_items_render_from_png_but_prologue_has_no_item(story_game):
    from kkamang.ui import menus

    game = story_game
    game.campaign.data['completed'] = list(range(1, 13))
    game.campaign.data['photos'] = list(range(1, 13))
    original = deepcopy(game.campaign.data)
    with patch.object(menus, 'draw_story_item', wraps=lobby.draw_story_item) as item:
        for stage in range(1, 13):
            item.reset_mock()
            game.start_memory(stage)
            game.dialogue_time = 99
            game.draw()
            assert item.called
            assert all(call.args[1] == stage for call in item.call_args_list)
            assert item.call_args.args[1] == stage
            assert pygame.Rect(0, 0, 1280, 488).contains(item.call_args.args[2])
        item.reset_mock()
        game.start_prologue()
        game.draw()
        item.assert_not_called()
    assert game.campaign.data == original


@pytest.mark.parametrize('stage,expected', [
    (1, '..A.A'),
    (2, '..A.A'),
    (3, '..A.A'),
    (4, '..A.A.'),
    (5, '..A.A'),
    (6, '..B.B'),
    (7, '..C.C'),
    (8, '..CA.'),
    (9, '..B.B'),
    (10, '...A.'),
    (11, '..C..'),
    (12, '...BC.A'),
])
def test_every_story_line_hides_ghosts_for_narration_and_shows_only_the_speaker(
        story_game, stage, expected):
    from kkamang.ui import menus

    game = story_game
    game.campaign.data['completed'] = list(range(1, stage + 1))
    game.campaign.data['photos'] = list(range(1, stage + 1))
    game.start_memory(stage)
    dialogue = deepcopy(game.dialogues)
    campaign = deepcopy(game.campaign.data)
    assert len(game.dialogues) == len(expected)
    for index, ghost_id in enumerate(expected):
        game.dialogue_index = index
        game.dialogue_time = 99
        with patch.object(menus, 'draw_story_character', wraps=lobby.draw_story_character) as portrait:
            game.draw()
        expected_portraits = {'cat'} if ghost_id == '.' else {'cat', ghost_id}
        assert {call.args[1] for call in portrait.call_args_list} == expected_portraits
    assert game.dialogues == dialogue
    assert game.campaign.data == campaign


def test_prologue_also_hides_ghosts_during_narration(story_game):
    from kkamang.ui import menus

    game = story_game
    game.start_prologue()
    expected = '.A.AAA.'
    assert len(game.dialogues) == len(expected)
    for index, ghost_id in enumerate(expected):
        game.dialogue_index = index
        game.dialogue_time = 99
        with patch.object(menus, 'draw_story_character', wraps=lobby.draw_story_character) as portrait:
            game.draw()
        expected_portraits = {'cat'} if ghost_id == '.' else {'cat', ghost_id}
        assert {call.args[1] for call in portrait.call_args_list} == expected_portraits
