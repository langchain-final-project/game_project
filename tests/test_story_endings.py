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
from kkamang import likes
from kkamang.ending_content import ENDING_CONTENT, released_variants
from kkamang.likes_preview import configure_preview, sample_record, SAMPLES
from kkamang.ui.memories import memory_original


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
    application = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True,
                          movement_mode='grid')
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


def gameplay_progress(app):
    data = deepcopy(app.campaign.data)
    data.pop('narrative')
    data['likes'].pop('seen_endings')
    return data


def finish_resolution(app):
    assert app.scene == 'story' and app.story_kind == 'garden'
    assert app.dialogues == []
    app.advance(3.2)
    if app.ending_released_variants:
        assert app.scene == 'story' and app.story_kind == 'liberation'
        assert app.context()['ending_released_variants'] == app.ending_released_variants
        app.advance(4.)
    assert app.scene == 'ending' and app.ending_art_only


@pytest.mark.parametrize('ending_id', likes.ENDING_IDS)
def test_six_main_endings_are_playable_after_the_shared_three_memories(app, ending_id):
    configure_preview(app, ending_id)
    before = gameplay_progress(app)
    original = pygame.image.tobytes(memory_original(3), 'RGB')
    finish_resolution(app)
    assert app.ending_id == ending_id
    assert app.context()['ending_content'] == ENDING_CONTENT[ending_id]
    assert app.campaign.active_ending() == ending_id
    assert app.context()['released_variants'] == released_variants(ending_id)
    assert app.campaign.data['likes']['seen_endings'] == [ending_id]
    assert app.campaign.memory_progress() == [4, 4, 4]
    rendered = controls(app)
    assert 'ending_continue' in rendered
    click(app, 'ending_continue')
    if ending_id == 'sad':
        assert app.scene == 'home' and app.dialogues == []
    else:
        assert app.scene == 'ending' and not app.ending_art_only
        assert app.dialogues == ENDING_CONTENT[ending_id]['lines']
        while app.scene == 'ending':
            app.dialogue_time = 99.
            app.next_dialogue()
        assert app.scene == 'home'
    assert gameplay_progress(app) == before
    assert pygame.image.tobytes(memory_original(3), 'RGB') == original
    assert SaveStore(app.store.path).load().active_ending() == ending_id
    with patch.object(app.audio, 'configure', wraps=app.audio.configure) as configure:
        app.draw()
    assert configure.call_args.kwargs['released'] == (ending_id == 'happy_all')


def test_last_room_reveals_the_last_picture_before_choosing_any_ending(app):
    for stage in range(1, 12):
        settle(app.campaign, stage, 4)
        likes.offer_memory(app.campaign.data['likes'], sample_record(stage, SAMPLES['normal']))
    play_reference(app, 12, 14)
    assert app.last_reward['first_complete'] and app.last_reward['new_ending']
    assert app.campaign.selected_ending() == 'normal'
    assert app.campaign.active_ending() is None
    assert controls(app)[app.controls[app.focus]['id']]['id'] == 'view_memory'
    before = gameplay_progress(app)
    click(app, 'view_memory')
    assert app.story_kind == 'memory' and app.memory_selected == 3
    app.advance(2.)
    assert app.scene == 'story' and app.campaign.memory_progress() == [4, 4, 4]
    assert app.campaign.data['likes']['seen_endings'] == []
    click(app, 'story_next')
    finish_resolution(app)
    assert app.ending_id == 'normal'
    assert gameplay_progress(app) == before


def test_replacing_pending_representative_refreshes_the_result_ending_action(app):
    state = likes.empty_state()
    for stage in range(1, 13):
        likes.offer_memory(state, sample_record(stage, (30, 17, 10) if stage == 1 else (42, 17, 10)))
    app.campaign.data['completed'] = list(range(1, 13))
    app.campaign.data['likes'] = state
    assert app.campaign.selected_ending() == 'normal'
    likes.offer_memory(state, sample_record(1, (60, 10, 5), 'replacement'))
    app.last_outcome = {'status': 'COMPLETE', 'banked': 14, 'carried': 14, 'ticks_used': 30}
    app.last_reward = {'first_complete': False, 'new_ending': False}
    app.stage = 1
    app.transition('memory_choice')
    app.dispatch('memory_replace')
    assert app.scene == 'result' and app.last_reward['new_ending']
    assert app.campaign.selected_ending() == 'ghost_a'
    assert app.campaign.active_ending() is None
    assert not controls(app)['ending']['disabled']
    click(app, 'ending')
    finish_resolution(app)
    assert app.ending_id == 'ghost_a' and app.campaign.released_variants() == (0,)


def test_replay_changes_no_active_world_or_gameplay_progress(app):
    configure_preview(app, 'happy_all')
    finish_resolution(app)
    app.finish_story()
    assert app.campaign.active_ending() == 'happy_all'
    normal_state = likes.empty_state()
    for stage in range(1, 13):
        likes.offer_memory(normal_state, sample_record(stage, SAMPLES['normal']))
    normal_state['seen_endings'] = ['happy_all', 'normal']
    app.campaign.data['likes'] = normal_state
    before = deepcopy(app.campaign.data)
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_id == 'normal' and app.ending_art_only
    assert app.campaign.active_ending() == 'happy_all'
    app.next_dialogue()
    app.finish_story()
    assert app.campaign.data == before
    app.transition('home')
    assert app.context()['released_variants'] == (0, 1, 2)
    assert app.context()['released']


def test_ending_cannot_open_before_all_rooms_or_before_legacy_scores_are_known(app):
    app.campaign = Campaign({'completed': list(range(1, 12))})
    before = deepcopy(app.campaign.data)
    app.start_ending()
    assert app.scene == 'likes' and app.campaign.data == before
    configure_preview(app, 'normal')
    app.campaign.data['likes']['representatives']['1']['scores'] = None
    app.campaign.data['badges'] = [f'stage_{n:02d}:rich' for n in range(1, 13)]
    before = deepcopy(app.campaign.data)
    app.start_ending()
    assert app.scene == 'likes' and app.campaign.data == before
    assert controls(app)['show_ending']['disabled']
    assert app.campaign.active_ending() is None
