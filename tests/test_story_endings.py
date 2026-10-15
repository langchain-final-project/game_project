"""Ending eligibility, old saves, and a replayed return through the app."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.app import GameApp
from kkamang.campaign import Campaign, SaveStore
from kkamang.levels import load_level
from kkamang import story


ROOT = Path(__file__).resolve().parents[1]
ROUTES = {(route['level_id'], route['target']): route
          for route in json.loads((ROOT / 'validation/reference_routes.json').read_text(
              encoding='utf-8'))['routes'] if 'trace' in route}
ACTION_KEYS = {0: pygame.K_UP, 1: pygame.K_DOWN, 2: pygame.K_LEFT,
               3: pygame.K_RIGHT, 4: pygame.K_e}


def settle(campaign, stage, banked=4, *, settlement_id=None):
    level = load_level(stage)
    outcome = {'status': 'COMPLETE', 'reason': 'returned',
               'banked': banked, 'carried': banked, 'damage_count': 1,
               'ticks_used': 1, 'remaining_ticks': level['time_budget_ticks'] - 1,
               'quest_collected': True}
    return campaign.settle(level, outcome, 'ending-test', 'basic', settlement_id or
                           f'{stage}-{banked}-{len(campaign.data["settled_ids"])}')


def completed_campaign(missing_rich=None):
    campaign = Campaign()
    for stage in range(1, 13):
        settle(campaign, stage, 4 if stage == missing_rich else 10)
    return campaign


@pytest.fixture
def app(tmp_path):
    application = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    application.campaign.data['intro_seen'] = True
    application.campaign.data['tutorial_seen'] = True
    application.transition('home')
    yield application
    application.close()


def play_reference(app, stage, target):
    app.prepare(stage)
    app.dispatch('start')
    assert app.scene == 'game'
    route = ROUTES[(f'stage_{stage:02d}', target)]
    for action in route['actions']:
        before_tick = app.core.snapshot()['tick']
        key = ACTION_KEYS.get(action)
        if key is not None:
            app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key, mod=0))
        if app.core.snapshot()['tick'] == before_tick:
            app.advance(max(0., app.core.config['tick_seconds'] - app.accumulator))
        if key is not None:
            app.handle_event(pygame.event.Event(pygame.KEYUP, key=key, mod=0))
        assert app.core.snapshot()['tick'] == before_tick + 1
    assert app.scene == 'result'
    assert app.last_outcome['status'] == 'COMPLETE'
    assert app.last_outcome['banked'] == route['banked']
    assert app.last_outcome['ticks_used'] == route['ticks']


def controls(app):
    app.draw()
    return {control['id']: control for control in app.controls}


def click(app, ident):
    control = controls(app)[ident]
    assert not control['disabled']
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                     pos=control['rect'].center, button=1))


def test_quiet_badges_do_not_restore_body_warmth():
    campaign = Campaign({'completed': list(range(1, 13)),
                         'badges': [f'stage_{stage:02d}:quiet' for stage in range(1, 13)]})
    assert campaign.optional_badges() == 12
    assert campaign.warmth_memories() == []
    assert not campaign.extended_ending_unlocked()


def test_warmth_progress_counts_distinct_completed_rooms_only():
    campaign = Campaign({'completed': list(range(1, 12)),
                         'badges': [f'stage_{stage:02d}:rich' for stage in range(1, 13)]})
    # A repeated in-memory badge must not stand in for a different room.
    campaign.data['badges'].extend(['stage_01:rich'] * 12)
    assert campaign.warmth_memories() == list(range(1, 12))
    assert not campaign.extended_ending_unlocked()
    campaign.data['completed'].append(12)
    campaign.data['badges'].remove('stage_12:rich')
    assert campaign.warmth_memories() == list(range(1, 12))
    assert not campaign.extended_ending_unlocked()


def test_ten_point_threshold_unlocks_once_when_an_earlier_room_is_replayed():
    campaign = Campaign()
    for stage in range(1, 13):
        reward = settle(campaign, stage, 9 if stage == 7 else 10)
        assert not reward['new_ending']
    assert campaign.warmth_memories() == [stage for stage in range(1, 13) if stage != 7]
    reward = settle(campaign, 7, 10, settlement_id='last-warmth')
    assert reward['new_ending']
    assert not reward['new_photo']
    assert campaign.extended_ending_unlocked()
    assert campaign.warmth_memories() == list(range(1, 13))
    before = deepcopy(campaign.data)
    repeated = settle(campaign, 7, 10, settlement_id='last-warmth')
    assert repeated['already_settled'] and not repeated['new_ending']
    assert campaign.data == before
    assert not settle(campaign, 7, 14)['new_ending']
    assert campaign.warmth_memories() == list(range(1, 13))


@pytest.mark.parametrize('legacy', [False, True], ids=['existing-v1', 'legacy-v0'])
def test_existing_save_restores_warmth_progress_and_unlock_after_reload(tmp_path, legacy):
    campaign = completed_campaign(missing_rich=12)
    payload = deepcopy(campaign.data)
    if legacy:
        payload['version'] = 0
        payload.pop('save_version')
        payload['first_complete'] = {f'stage_{stage:02d}': True for stage in payload.pop('completed')}
        payload['unique_badges'] = payload.pop('badges') + ['stage_01:rich']
    else:
        payload['save_version'] = 1
    path = tmp_path / 'existing-save.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    store = SaveStore(path)
    loaded = store.load()
    assert store.error is None
    assert loaded.warmth_memories() == list(range(1, 12))
    assert not loaded.extended_ending_unlocked()
    assert settle(loaded, 12, 10, settlement_id='old-save-last-warmth')['new_ending']
    assert store.save(loaded)
    reloaded = SaveStore(path).load()
    assert reloaded.warmth_memories() == list(range(1, 13))
    assert reloaded.extended_ending_unlocked()
    before = deepcopy(reloaded.data)
    assert not settle(reloaded, 12, 10, settlement_id='old-save-last-warmth')['new_ending']
    assert reloaded.data == before


def test_ending_cannot_open_before_all_rooms_complete(app):
    app.campaign = Campaign({'completed': list(range(1, 12))})
    before = deepcopy(app.campaign.data)
    app.start_ending()
    assert app.scene == 'likes'
    app.dispatch('ending')
    assert app.scene == 'likes'
    assert app.campaign.data == before
    assert 'ending' not in controls(app)


def test_minimum_warmth_completions_open_basic_ending_without_changing_rewards(app):
    for stage in range(1, 13):
        assert not settle(app.campaign, stage, 4)['new_ending']
    assert not app.campaign.extended_ending_unlocked()
    before = deepcopy(app.campaign.data)
    app.start_ending()
    assert app.scene == 'ending'
    assert app.ending_variant == 'basic'
    assert app.dialogues == list(story.ENDING)
    click(app, 'story_skip')
    assert app.scene == 'home'
    assert app.campaign.data == before
    click(app, 'ending')
    assert app.ending_variant == 'basic'
    app.go_back()
    assert app.campaign.data == before


def test_last_warmth_in_earlier_room_opens_extended_ending_from_result_and_home(app):
    app.campaign = completed_campaign(missing_rich=1)
    app.campaign.data['tutorial_seen'] = True
    assert not app.campaign.extended_ending_unlocked()
    play_reference(app, 1, 14)
    assert app.last_reward['new_ending'] and not app.last_reward['new_photo']
    assert app.campaign.extended_ending_unlocked()
    rendered = controls(app)
    assert '후일담' in rendered['ending']['label']
    assert app.controls[app.focus]['id'] == 'ending'
    before = deepcopy(app.campaign.data)
    click(app, 'ending')
    assert app.scene == 'ending'
    assert app.ending_variant == 'extended'
    assert app.dialogues == list(story.EXTENDED_ENDING)
    click(app, 'story_skip')
    assert app.scene == 'home'
    assert app.campaign.data == before
    click(app, 'ending')
    assert app.ending_variant == 'extended'
    app.go_back()
    assert app.scene == 'home'
    assert app.campaign.data == before
    reloaded = SaveStore(app.store.path).load()
    assert reloaded.extended_ending_unlocked()


def test_new_final_room_memory_has_priority_over_new_ending(app):
    for stage in range(1, 12):
        settle(app.campaign, stage, 10)
    play_reference(app, 12, 14)
    assert app.last_reward['new_photo'] and app.last_reward['new_ending']
    controls(app)
    assert app.controls[app.focus]['id'] == 'view_memory'
    before = deepcopy(app.campaign.data)
    click(app, 'view_memory')
    assert app.scene == 'story'
    click(app, 'story_skip')
    assert app.scene == 'ending'
    assert app.ending_variant == 'extended'
    assert app.campaign.data == before


def test_old_home_comparison_does_not_hide_the_restored_ending(app):
    from kkamang.ui.illustration import shelter

    app.campaign = completed_campaign()
    app.dispatch('toggle_home_view')
    with patch('kkamang.ui.menus.shelter', wraps=shelter) as draw_shelter:
        controls(app)
        assert draw_shelter.call_args.kwargs['before'] is True
        app.dispatch('ending')
        controls(app)
        assert app.ending_variant == 'extended'
        assert draw_shelter.call_args.kwargs['before'] is False
        assert draw_shelter.call_args.kwargs['released'] is True
