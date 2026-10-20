"""현재 자유 이동 앱에서 기억 공개와 호감도 엔딩 흐름을 검증한다."""
from copy import deepcopy
import os
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang.app import GameApp, CINEMATIC_DURATIONS
from kkamang.campaign import Campaign, SaveStore
from kkamang.levels import load_level
from kkamang.likes_preview import sample_record, SAMPLES
from kkamang import likes, story
from kkamang.ending_content import ENDING_CONTENT
from kkamang.realtime import FRAME_SECONDS, FRAMES_PER_TICK, RealtimeGameCore


@pytest.fixture
def app(tmp_path):
    application = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    application.campaign.data['intro_seen'] = True
    application.campaign.data['tutorial_seen'] = True
    application.transition('home')
    yield application
    application.close()


def settle(campaign, stage, banked=4):
    level = load_level(stage)
    outcome = {'status': 'COMPLETE', 'reason': 'returned', 'banked': banked,
               'carried': banked, 'damage_count': 1, 'ticks_used': 1,
               'remaining_ticks': level['time_budget_ticks'] - 1,
               'quest_collected': True}
    reward = campaign.settle(level, outcome, 'visual-story-test', 'basic',
                             f'{stage}-{banked}-{len(campaign.data["settled_ids"])}')
    return outcome, reward


def complete_all(campaign, missing_rich=None, rich=False, ending="normal"):
    for stage in range(1, 13):
        settle(campaign, stage, 10 if rich and stage != missing_rich else 4)
    state = likes.empty_state()
    for stage in range(1, 13):
        likes.offer_memory(state, sample_record(stage, SAMPLES[ending]))
    campaign.data['likes'] = state


def test_silent_intro_never_unlocks_a_memory(app):
    before = deepcopy(app.campaign.data)
    app.start_prologue()
    assert app.scene == 'story' and app.story_kind == 'prologue'
    assert app.dialogues == [] and app.context()['dialogue'] is None
    assert app.context()['silent']
    app.advance(CINEMATIC_DURATIONS['prologue'])
    assert app.scene == 'home' and app.scene_elapsed == 0
    assert app.campaign.memory_progress() == [0, 0, 0]
    assert app.campaign.data['records'] == before['records']


def test_legacy_photo_does_not_suppress_first_completed_region(app):
    app.campaign = Campaign({'photos': [1, 4, 12]})
    app.last_outcome, app.last_reward = settle(app.campaign, 1)
    assert app.last_reward['first_complete'] and not app.last_reward['new_photo']
    app.stage = 1
    app.transition('result')
    app.dispatch('next_stage')
    context = app.context()
    assert app.story_kind == 'memory' and app.dialogues == []
    assert context['regions'] == [True, False, False, False]
    assert context['previous_regions'] == [False] * 4
    assert context['reveal_region'] == 0
    assert context['memory_progress'] == [1, 0, 0]


def test_partial_reveal_waits_for_enter_then_opens_next_room_without_awarding_again(app):
    settle(app.campaign, 1)
    before = deepcopy(app.campaign.data)
    app.start_memory(1)
    app.draw()
    app.advance(1.99)
    assert app.scene == 'story'
    assert app.context()['reveal_progress'] == pytest.approx(.995)
    app.advance(.02)
    app.draw()
    assert app.scene == 'story' and app.context()['reveal_progress'] == 1.
    assert app.controls[app.focus]['id'] == 'story_finish_next'
    app.advance(10.)
    assert app.scene == 'story' and app.campaign.data == before
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN, mod=0))
    assert app.scene == 'briefing' and app.stage == 2
    assert app.campaign.data == before


def test_completed_picture_and_album_replay_wait_for_user(app):
    for stage in range(1, 5):
        settle(app.campaign, stage)
    before = deepcopy(app.campaign.data)
    app.start_memory(4)
    with patch.object(app.audio, 'play') as play:
        app.advance(2.)
        app.advance(10.)
    assert app.scene == 'story'
    assert [call.args for call in play.call_args_list].count(('memory_complete',)) == 1
    assert app.context()['regions'] == [True] * 4
    assert app.context()['previous_regions'] == [True, True, True, False]
    app.next_dialogue()
    assert app.scene == 'briefing' and app.stage == 5
    app.start_memory(3, replay=True)
    app.advance(10.)
    assert app.scene == 'story' and app.context()['replay']
    assert app.context()['previous_regions'] == [True] * 4
    assert app.context()['reveal_region'] is None
    app.go_back()
    assert app.scene == 'album' and app.campaign.data == before


def test_replayed_completion_does_not_force_another_reveal(app):
    settle(app.campaign, 1)
    app.last_outcome, app.last_reward = settle(app.campaign, 1)
    assert not app.last_reward['first_complete']
    app.stage = 1
    app.transition('result')
    app.dispatch('next_stage')
    assert app.scene == 'briefing' and app.stage == 2


def test_album_memory_selection_is_separate_from_legacy_photo_selection(app):
    complete_all(app.campaign)
    app.photo_selected = 11
    app.dispatch('memory:2')
    assert app.scene == 'album' and app.memory_selected == 2
    assert app.photo_selected == 11
    app.dispatch('replay_memory')
    assert app.story_kind == 'memory' and app.stage == 8
    assert app.context()['memory_id'] == 2 and app.context()['replay']
    app.finish_story()
    app.dispatch('visit_memory')
    assert app.scene == 'briefing' and app.stage == 8
    app.dispatch('visit_photo')
    assert app.stage == 11


def test_normal_ending_uses_png_and_one_narration_without_altering_records(app):
    complete_all(app.campaign)
    app.campaign.data['likes']['seen_endings'] = ['normal', 'ghost_a']
    original_likes = deepcopy(app.campaign.data['likes'])
    original_records = deepcopy(app.campaign.data['records'])
    app.start_ending()
    assert app.scene == 'story' and app.story_kind == 'ending_intro'
    assert app.ending_id == 'normal' and app.context()['dialogue'] is None
    app.finish_story()
    assert app.scene == 'ending' and app.ending_art_only
    assert app.dialogues == ENDING_CONTENT['normal']['lines']
    app.next_dialogue()
    assert not app.ending_art_only and len(app.dialogues) == 1
    assert app.dialogues[0][0] == ''
    app.finish_story()
    assert app.scene == 'home' and app.campaign.active_ending() == 'normal'
    assert app.campaign.data['likes'] == original_likes
    assert app.campaign.data['records'] == original_records
    app.dispatch('ending')
    assert app.scene == 'ending' and app.ending_art_only
    assert app.dialogues == story.ENDING


def test_best_affection_on_first_journey_releases_all_without_rich_badges(app):
    complete_all(app.campaign, ending='happy_all')
    assert app.campaign.warmth_memories() == []
    app.start_memory(12)
    app.finish_story()
    assert app.story_kind == 'ending_intro' and app.ending_id == 'happy_all'
    app.advance(8.)
    assert app.story_kind == 'ending_intro' and app.dialogues == []
    assert app.campaign.active_ending() is None
    app.finish_story()
    assert app.scene == 'ending' and app.dialogues == story.EXTENDED_ENDING
    assert app.ending_art_only and app.context()['released']
    assert app.context()['released_variants'] == (0, 1, 2)
    app.finish_story()
    restored = SaveStore(app.store.path).load()
    assert restored.active_ending() == 'happy_all'
    assert restored.data['likes']['seen_endings'] == ['happy_all']


def test_rich_badges_and_legacy_flags_do_not_auto_release_on_home_entry(app):
    complete_all(app.campaign, missing_rich=3, rich=True)
    app.campaign.data['narrative'].update(basic_seen=True, true_seen=True, liberation_seen=True)
    app.last_outcome, app.last_reward = settle(app.campaign, 3, 10)
    assert app.campaign.warmth_memories() == list(range(1, 13))
    app.transition('home')
    assert app.scene == 'home' and not app.context()['released']
    assert app.context()['released_variants'] == ()
    assert app.campaign.selected_ending() == 'normal'
    app.dispatch('show_ending')
    assert app.story_kind == 'ending_intro' and app.ending_id == 'normal'
    app.finish_story()
    assert app.scene == 'ending' and app.campaign.active_ending() == 'normal'
    app.finish_story()
    app.start_memory(11, replay=True)
    assert not app.context()['released'] and app.context()['regions'] == [True] * 4


def test_pending_representative_choice_still_blocks_ending(app):
    complete_all(app.campaign, rich=True)
    state = app.campaign.data['likes']
    likes.offer_memory(state, sample_record(1, (60, 25, 15), run_id='new'))
    before = deepcopy(state)
    app.start_ending()
    assert app.scene == 'memory_choice'
    assert state == before
    assert not any(app.campaign.data['narrative'].values())


def test_save_failure_during_release_keeps_progress_and_retry_target(app):
    complete_all(app.campaign, ending='happy_all')
    app.start_ending()
    assert app.story_kind == 'ending_intro'
    before_completed = list(app.campaign.data['completed'])
    with patch.object(app.store, '_atomic_write', side_effect=OSError('disk full')):
        app.finish_story()
    assert app.scene == 'save_error' and app.save_return == 'ending'
    assert app.campaign.active_ending() == 'happy_all'
    assert app.campaign.data['completed'] == before_completed
    app.dispatch('save_retry')
    assert app.scene == 'ending' and app.dialogues == story.EXTENDED_ENDING
    assert SaveStore(app.store.path).load().active_ending() == 'happy_all'


def test_default_realtime_motion_and_automatic_warmth_pickup_remain_active(app):
    app.prepare(1)
    app.start_run()
    assert isinstance(app.core, RealtimeGameCore)
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0))
    app.advance(FRAME_SECONDS * FRAMES_PER_TICK)
    snapshot = app.core.snapshot()
    assert snapshot['tick'] == 1 and snapshot['motion']['frame'] == FRAMES_PER_TICK
    assert snapshot['position'] == [app.levels[0]['start'][0] + 1, app.levels[0]['start'][1]]
    assert snapshot['carried'] >= 1
    app.handle_event(pygame.event.Event(pygame.KEYUP, key=pygame.K_RIGHT, mod=0))
    position = list(snapshot['motion']['position'])
    app.advance(FRAME_SECONDS)
    assert app.core.snapshot()['motion']['position'] == position


def test_realtime_clock_and_result_precision_stay_separate_from_official_outcome(app):
    app.prepare(1)
    app.start_run()
    app.advance(FRAME_SECONDS * 3.5)
    snapshot = app.core.snapshot()
    assert snapshot['motion']['frame'] == 3 and snapshot['tick'] == 0
    with patch('kkamang.app.draw_game') as draw_game:
        app.draw()
    assert draw_game.call_args.kwargs['remaining_seconds'] == pytest.approx(
        snapshot['remaining_ticks'] * app.core.config['tick_seconds'] - app.accumulator)
    result = app.core.abort()
    outcome = deepcopy(result.outcome)
    app.finish_run(result)
    assert app.display_elapsed_seconds == pytest.approx(3 * FRAME_SECONDS)
    assert app.context()['display_elapsed_seconds'] == pytest.approx(3 * FRAME_SECONDS)
    assert app.last_outcome == outcome and result.outcome == outcome
    assert 'display_elapsed_seconds' not in app.last_outcome
    before = deepcopy(app.campaign.data)
    app.finish_run(type('Result', (), {'outcome': outcome})())
    assert app.display_elapsed_seconds == pytest.approx(3 * FRAME_SECONDS)
    assert app.campaign.data == before


def test_real_menu_controls_follow_silent_resolution_then_main_art_and_afterword(app):
    complete_all(app.campaign, ending='happy_all')
    app.start_memory(12)

    def click(ident):
        app.draw()
        control = next(c for c in app.controls if c['id'] == ident)
        app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN,
                                           pos=control['rect'].center, button=1))

    app.advance(2.)
    click('story_next')
    assert app.story_kind == 'ending_intro' and app.context()['silent']
    click('story_skip')
    assert app.scene == 'ending' and app.ending_art_only and len(app.dialogues) == 1
    click('ending_continue')
    assert not app.ending_art_only
    app.dialogue_time = 99.
    click('ending_continue')
    app.draw()
    assert app.scene == 'home' and app.context()['released']
    assert {'memory:1', 'memory:2', 'memory:3'} <= {c['id'] for c in app.controls}


def test_third_memory_new_region_uses_cold_motif_without_forcing_it_on_replay(app):
    for stage in range(1, 10):
        settle(app.campaign, stage)
    with patch.object(app.audio, 'play') as play:
        app.start_memory(9)
    play.assert_called_once_with('memory_cold')
    app.finish_story()
    with patch.object(app.audio, 'play') as play:
        app.start_memory(9, replay=True)
        app.advance(3.)
    play.assert_not_called()
    assert app.scene == 'story' and app.context()['replay']
