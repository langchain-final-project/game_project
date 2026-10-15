"""Human input arbitration; these tests need no display or audio device."""
import pygame
import pytest

from kkamang.input import InputAdapter


def keydown(key, timestamp=None):
    data = {"key": key}
    if timestamp is not None:
        data["timestamp"] = timestamp
    return pygame.event.Event(pygame.KEYDOWN, data)


def keyup(key):
    return pygame.event.Event(pygame.KEYUP, {"key": key})


def test_no_input_is_wait():
    assert InputAdapter().consume() == 5


@pytest.mark.parametrize("key,action", [
    (pygame.K_UP, 0), (pygame.K_w, 0),
    (pygame.K_DOWN, 1), (pygame.K_s, 1),
    (pygame.K_LEFT, 2), (pygame.K_a, 2),
    (pygame.K_RIGHT, 3), (pygame.K_d, 3),
])
def test_direction_is_held_and_released(key, action):
    adapter = InputAdapter()
    adapter.handle(keydown(key))
    assert adapter.consume() == action
    assert adapter.consume() == action
    adapter.handle(keyup(key))
    assert adapter.consume() == 5


def test_interaction_is_a_new_press_not_a_held_action():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_e))
    assert adapter.consume() == 4
    assert adapter.consume() == 5
    adapter.handle(keydown(pygame.K_e))  # An OS repeat is still the same press.
    assert adapter.consume() == 5
    adapter.handle(keyup(pygame.K_e))
    adapter.handle(keydown(pygame.K_e))
    assert adapter.consume() == 4


def test_short_interaction_press_is_retained_until_tick():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_e))
    adapter.handle(keyup(pygame.K_e))
    assert adapter.consume() == 4
    assert adapter.consume() == 5


@pytest.mark.parametrize('key,action',[(pygame.K_UP,0),(pygame.K_DOWN,1),(pygame.K_LEFT,2),(pygame.K_RIGHT,3)])
def test_short_movement_tap_is_retained_once_until_tick(key, action):
    adapter=InputAdapter()
    adapter.handle(keydown(key))
    adapter.handle(keyup(key))
    assert adapter.consume()==action
    assert adapter.consume()==5


def test_most_recent_direction_wins_then_release_restores_older():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_UP, 10))
    adapter.handle(keydown(pygame.K_LEFT, 11))
    assert adapter.consume() == 2
    adapter.handle(keyup(pygame.K_LEFT))
    assert adapter.consume() == 0


@pytest.mark.parametrize("keys,expected", [
    ((pygame.K_UP, pygame.K_e), 4),
    ((pygame.K_e, pygame.K_UP), 4),
    ((pygame.K_RIGHT, pygame.K_UP), 0),
    ((pygame.K_DOWN, pygame.K_RIGHT), 3),
    ((pygame.K_LEFT, pygame.K_DOWN), 1),
    ((pygame.K_SPACE, pygame.K_LEFT), 2),
])
def test_equal_timestamps_follow_declared_priority(keys, expected):
    adapter = InputAdapter()
    for key in keys:
        adapter.handle(keydown(key, 50))
    assert adapter.consume() == expected


def test_pending_interaction_wins_over_later_direction():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_e, 50))
    adapter.handle(keydown(pygame.K_RIGHT, 51))

    # 방향키가 나중에 들어와도 E를 먼저 처리합니다.
    assert adapter.consume() == 4

    # E를 누르고 있는 것만으로 상호작용을 반복하지 않습니다.
    adapter.handle(keyup(pygame.K_RIGHT))
    assert adapter.consume() == 5


def test_same_action_aliases_release_independently():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_UP))
    adapter.handle(keydown(pygame.K_w))
    adapter.handle(keyup(pygame.K_w))
    assert adapter.consume() == 0


def test_transition_clear_discards_held_and_pending_input():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_UP))
    adapter.handle(keydown(pygame.K_e))
    adapter.clear()
    assert adapter.consume() == 5
    adapter.handle(keyup(pygame.K_UP))
    adapter.handle(keyup(pygame.K_e))
    assert adapter.consume() == 5


def test_non_game_keys_are_ignored():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_ESCAPE))
    adapter.handle(pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1}))
    assert adapter.consume() == 5

def test_separate_interaction_preserves_held_movement():
    adapter = InputAdapter()
    adapter.handle(keydown(pygame.K_RIGHT))
    adapter.handle(keydown(pygame.K_e))

    # E를 처리해도 누르고 있는 방향키는 유지됩니다.
    assert adapter.consume_interaction() is True
    assert adapter.consume_movement() == 3
    assert adapter.consume_movement() == 3

    # E를 누른 채로 있어도 자동 반복하지 않습니다.
    assert adapter.consume_interaction() is False
    adapter.handle(keydown(pygame.K_e))
    assert adapter.consume_interaction() is False


def test_short_movement_and_interaction_are_preserved_separately():
    adapter = InputAdapter()

    adapter.handle(keydown(pygame.K_RIGHT))
    adapter.handle(keyup(pygame.K_RIGHT))
    adapter.handle(keydown(pygame.K_e))
    adapter.handle(keyup(pygame.K_e))

    # 짧게 누른 입력도 각각 한 번씩 가져올 수 있습니다.
    assert adapter.consume_movement() == 3
    assert adapter.consume_interaction() is True

    # 처리한 입력은 다시 실행되지 않습니다.
    assert adapter.consume_movement() == 5
    assert adapter.consume_interaction() is False