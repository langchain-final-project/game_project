"""앱에서 대표 기록 교체·호감도 반영·엔딩 연결을 검증한다."""
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame
import pytest

from kkamang.app import GameApp, CINEMATIC_DURATIONS
from kkamang.campaign import SaveStore
from kkamang.likes_preview import configure_preview, sample_record

ROOT = Path(__file__).resolve().parents[1]
ROUTES = json.loads((ROOT / 'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True, movement_mode='grid')
    yield instance
    instance.close()


def complete(app, stage=1):
    app.stage = stage
    app.start_run()
    route = next(r for r in ROUTES if r['level_id'] == f'stage_{stage:02d}' and r['target'] == 4)
    for action in route['actions']:
        result = app.core.step(action)
    assert result.outcome['status'] == 'COMPLETE'
    app.finish_run(result)


def show_ending_art(app):
    for _ in range(3):
        if app.scene == 'ending':
            break
        assert app.scene == 'story' and app.story_kind == 'ending_intro'
        app.finish_story()
    assert app.scene == 'ending' and app.ending_art_only


def test_real_completion_provider_choice_escape_and_no_star_loss(app):
    calls = []
    def provider(level, outcome, snapshot, mode):
        calls.append((level['stage'], outcome['banked'], snapshot['courage'], mode))
        return sample_record(1, (60, 10, 15) if len(calls) == 1 else (36, 25, 15))['scores']
    app.score_provider = provider
    complete(app)
    assert app.scene == 'result'
    original = deepcopy(app.campaign.data['likes']['representatives']['1'])
    complete(app)
    assert len(calls) == 2
    assert app.scene == 'memory_choice'
    app.draw()
    assert [c['id'] for c in app.controls] == ['memory_keep', 'memory_replace']
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert app.scene == 'result'
    assert app.campaign.data['likes']['representatives']['1'] == original
    complete(app)
    before_badges = deepcopy(app.campaign.data['badges'])
    app.dispatch('memory_replace')
    assert app.campaign.data['likes']['representatives']['1']['scores']['total'] == 76
    assert app.campaign.data['likes']['best_stars']['1'] == 3
    assert app.campaign.data['badges'] == before_badges
    assert SaveStore(app.store.path).load().data == app.campaign.data


def test_real_scores_persist_and_reload_resumes_choice(app):
    complete(app)
    score = app.campaign.data['likes']['representatives']['1']['scores']
    assert score['total'] == app.last_outcome['score']['total']
    assert score['life'] == app.last_outcome['score']['courage']
    assert app.campaign.data['likes']['best_stars']['1'] == app.campaign.stars_for_stage(1)
    complete(app)
    restored = GameApp(save_path=app.store.path, headless=True, no_audio=True, movement_mode='grid')
    try:
        restored.dispatch('continue')
        assert restored.scene == 'memory_choice'
        assert restored.last_outcome is None
        restored.dispatch('memory_keep')
        assert restored.scene == 'likes'
        restored.draw()
        assert restored.campaign.data['likes']['pending'] is None
    finally:
        restored.close()


def test_real_replay_keeps_star_best_and_syncs_memory(app):
    complete(app)
    first = deepcopy(app.campaign.data['likes']['representatives']['1'])
    app.start_run()
    route = next(r for r in ROUTES if r['level_id'] == 'stage_01' and r['target'] == 14)
    for action in route['actions']:
        result = app.core.step(action)
    app.finish_run(result)
    best = app.campaign.star_record(1)
    assert app.scene == 'memory_choice'
    app.dispatch('memory_keep')
    assert app.campaign.data['likes']['representatives']['1'] == first
    assert app.campaign.star_record(1) == best
    complete(app)
    app.dispatch('memory_replace')
    selected = app.campaign.data['likes']['representatives']['1']['scores']
    legacy = app.campaign.data['friend_memories']['stage_01']
    assert legacy['courage'] == selected['life']
    assert legacy['total'] == selected['total']
    assert app.campaign.star_record(1) == best
    app.draw()
    assert not {'remember_friend', 'keep_friend'} & {c['id'] for c in app.controls}


def test_new_memory_story_can_continue_directly_to_the_next_room(app):
    app.campaign.data['completed'] = [1, 2, 3, 4]
    app.campaign.data['photos'] = [1, 2, 3, 4]
    app.start_memory(4)
    assert app.dialogues == []
    app.advance(2.)
    assert app.scene == 'story'
    app.draw()
    assert {'story_finish_home', 'story_finish_next'} <= {c['id'] for c in app.controls}
    app.dispatch('story_finish_next')
    assert app.stage == 5
    assert app.scene == 'briefing'


@pytest.mark.parametrize('input_method', ['enter', 'click'])
def test_partial_picture_reveal_waits_for_next_room_input_without_dialogue(app, input_method):
    app.campaign.data['completed'] = [1]
    app.campaign.data['photos'] = [1]
    before = deepcopy(app.campaign.data)
    app.start_memory(1)
    assert app.dialogues == []
    app.draw()
    buttons = {c['id']: c for c in app.controls}
    assert buttons['story_finish_next']['label'] == '다음 방으로'
    assert buttons['story_finish_next']['disabled']
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert app.scene == 'story'
    app.advance(1.9)
    assert app.scene == 'story'
    app.draw()
    app.advance(.1)
    app.draw()
    buttons = {c['id']: c for c in app.controls}
    assert len(buttons) == len(app.controls)
    assert app.scene == 'story'
    assert not buttons['story_finish_next']['disabled']
    assert buttons['story_finish_home']['rect'].x < buttons['story_finish_next']['rect'].x
    assert app.controls[app.focus]['id'] == 'story_finish_next'
    assert 'story_next' not in buttons
    app.advance(10.)
    assert app.scene == 'story' and app.campaign.data == before
    if input_method == 'enter':
        app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    else:
        app.handle_event(pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, pos=buttons['story_finish_next']['rect'].center, button=1))
    assert app.scene == 'briefing' and app.stage == 2
    assert app.campaign.data == before
    assert app.campaign.memory_progress() == [1, 0, 0]


@pytest.mark.parametrize('stage', range(1, 13))
@pytest.mark.parametrize('new_ending', [False, True])
def test_recleared_result_goes_to_next_room_and_keeps_last_room_ending(app, stage, new_ending):
    configure_preview(app, 'normal')
    app.campaign.record_ending('normal', activate=True)
    app.stage = stage
    app.last_outcome = {'status': 'COMPLETE', 'banked': 4, 'carried': 4, 'ticks_used': 10}
    app.last_reward = {'first_complete': False, 'new_ending': new_ending}
    app.transition('result')
    before = deepcopy(app.campaign.data)
    app.draw()
    buttons = {c['id']: c for c in app.controls}
    target = 'next_stage' if stage < 12 else 'ending'
    assert buttons[target]['label'] == ('다음 방으로' if stage < 12 else '엔딩 보기')
    assert ('ending' if stage < 12 else 'next_stage') not in buttons
    assert 'view_memory' not in buttons
    assert app.controls[app.focus]['id'] == target
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    if stage < 12:
        assert app.scene == 'briefing' and app.stage == stage + 1
    else:
        assert app.scene == 'story' and app.story_kind == 'ending_intro'
    assert app.campaign.data == before


def test_result_retry_returns_to_the_same_room_briefing(app):
    complete(app)

    assert app.scene == 'result'
    app.dispatch('retry')

    assert app.scene == 'briefing'
    assert app.stage == 1


def test_relaxed_completion_is_practice_only_and_has_no_progress_actions(app):
    app.mode = 'relaxed'
    app.campaign.data['settings']['mode'] = 'relaxed'
    before = deepcopy(app.campaign.data)

    complete(app)

    assert app.scene == 'result'
    assert app.last_reward['practice_mode']
    before['collection']['practice_rooms'] = [1]
    before['collection']['unlocked'] = [1]
    assert app.campaign.data == before
    app.draw()
    assert {'retry', 'home', 'rooms'} <= {c['id'] for c in app.controls}
    assert not {'next_stage', 'view_memory'} & {c['id'] for c in app.controls}


def test_practice_can_run_with_a_pending_basic_memory_without_resolving_it(app):
    complete(app)
    complete(app)
    assert app.campaign.data['likes']['pending'] is not None
    app.mode = 'relaxed'
    app.campaign.data['settings']['mode'] = 'relaxed'
    before = deepcopy(app.campaign.data)
    with patch.object(app,'score_provider',side_effect=AssertionError('practice relationship score')):
        complete(app)
    assert app.scene == 'result' and app.last_reward['practice_mode']
    before['collection']['practice_rooms'] = [1]
    assert app.campaign.data == before
    assert SaveStore(app.store.path).load().data == before
    app.dispatch('next_stage')
    assert app.scene == 'rooms'
    assert app.campaign.data == before


def test_save_failure_retains_choice_then_can_retry(app):
    complete(app)
    complete(app)
    with patch.object(app.store, '_atomic_write', side_effect=OSError('disk full')):
        app.dispatch('memory_replace')
    assert app.scene == 'save_error'
    selected = deepcopy(app.campaign.data['likes']['representatives'])
    app.dispatch('save_retry')
    assert app.scene == 'result'
    assert app.campaign.data['likes']['representatives'] == selected
    assert SaveStore(app.store.path).load().data['likes']['pending'] is None


def test_ending_waits_for_choice_and_preserves_seen_history(app):
    configure_preview(app, 'choice')
    app.start_ending()
    assert app.scene == 'memory_choice'
    assert app.campaign.data['likes']['seen_endings'] == []
    app.dispatch('memory_keep')
    app.dispatch('show_ending')
    assert app.scene == 'story' and app.story_kind == 'ending_intro'
    show_ending_art(app)
    assert app.ending_id == 'normal'
    assert app.campaign.active_ending() == 'normal'
    app.finish_story()
    assert app.campaign.data['likes']['seen_endings'] == ['normal']
    before = deepcopy(app.campaign.data)
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_art_only
    assert app.campaign.data == before
    app.campaign.data['likes']['representatives']['1']['scores'] = None
    app.start_ending(replay=True)
    assert app.scene == 'likes'
    assert app.campaign.data['likes']['seen_endings'] == ['normal']


@pytest.mark.parametrize('preview', ['overview', 'choice', 'happy_all', 'sad', 'ghost_a', 'ghost_b', 'ghost_c', 'normal'])
def test_new_screens_read_only_and_controls_on_canvas(app, preview):
    configure_preview(app, preview)
    before = deepcopy(app.campaign.data)
    app.draw()
    app.draw()
    assert app.campaign.data == before
    assert len({c['id'] for c in app.controls}) == len(app.controls)
    for c in app.controls:
        assert pygame.Rect(0, 0, 1280, 720).contains(c['rect'])


@pytest.mark.parametrize('ending', ['happy_all', 'sad', 'ghost_a', 'ghost_b', 'ghost_c', 'normal'])
def test_six_ending_art_epilogue_keyboard_and_save_preservation(app, ending):
    from kkamang.ending_content import ENDING_CONTENT
    from kkamang import likes
    configure_preview(app, ending)
    assert likes.select_ending(app.campaign.data['likes'], app.campaign.data['completed']) == ending
    assert (ROOT / 'assets' / 'art' / ENDING_CONTENT[ending]['art']).is_file()
    representatives = deepcopy(app.campaign.data['likes']['representatives'])
    stars = deepcopy(app.campaign.data['likes']['best_stars'])
    assert app.ending_id == ending
    assert app.scene == 'story' and app.story_kind == 'ending_intro'
    assert app.context()['dialogue'] is None
    show_ending_art(app)
    assert app.dialogues == ENDING_CONTENT[ending]['lines']
    assert app.context()['ending_content'] == ENDING_CONTENT[ending]
    assert app.context()['ending_art_only']
    with patch('kkamang.ui.menus.wrapped') as dialogue_box:
        app.draw()
    dialogue_box.assert_not_called()
    continue_button = next(c for c in app.controls if c['id']=='ending_continue')
    assert continue_button['label'] == '후일담 보기'
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert not app.ending_art_only and app.dialogue_index == 0
    assert app.dialogue_time == 0
    assert len(app.dialogues) == 1 and app.dialogues[0][0] == ''
    with patch('kkamang.ui.menus._ending_intro_picture') as cinematic, \
         patch('kkamang.ui.menus._ending_picture') as art:
        app.draw()
    cinematic.assert_not_called()
    assert art.call_args.args[2] == ENDING_CONTENT[ending]['art']
    app.dialogue_time = 99
    app.draw()
    app.dispatch('ending_continue')
    assert app.scene == 'home'
    assert app.campaign.active_ending() == ending
    assert app.campaign.data['likes']['seen_endings'] == [ending]
    assert app.campaign.data['likes']['representatives'] == representatives
    assert app.campaign.data['likes']['best_stars'] == stars
    saved = SaveStore(app.store.path).load()
    assert saved.data['likes'] == app.campaign.data['likes']
    assert saved.active_ending() == ending
    assert saved.released_variants() == app.campaign.released_variants()
    before = deepcopy(app.campaign.data)
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_art_only
    app.go_back()
    assert app.scene == 'home'
    assert app.campaign.data == before
