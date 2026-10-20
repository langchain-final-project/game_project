"""세 기억의 진행과 여섯 엔딩 선택이 기존 캠페인 규칙과 맞는지 검증한다."""
from copy import deepcopy
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pytest

from kkamang import likes
from kkamang.app import GameApp
from kkamang.campaign import Campaign, SaveStore, SAVE_VERSION
from kkamang.ending_content import ENDING_CONTENT, released_variants
from kkamang.likes_preview import configure_preview


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True,
                       movement_mode='grid')
    yield instance
    instance.close()


def recover_warmth(campaign):
    campaign.data['badges'] = sorted(set(campaign.data['badges']) |
                                    {f'stage_{stage:02d}:rich' for stage in range(1, 13)})


def finish_silent_resolution(app, ending_id):
    assert app.scene == 'story' and app.story_kind == 'ending_intro'
    assert app.dialogues == []
    previous_active = app.campaign.active_ending()
    app.advance(8.)
    assert app.scene == 'story' and app.campaign.active_ending() == previous_active
    app.finish_story()
    assert app.scene == 'ending'
    assert app.ending_art_only
    assert app.ending_id == ending_id
    assert app.campaign.active_ending() == ending_id
    assert app.campaign.released_variants() == released_variants(ending_id)


def finish_art_and_afterword(app, ending_id):
    assert app.ending_art_only
    assert app.dialogues == list(ENDING_CONTENT[ending_id]['lines'])
    app.next_dialogue()
    if app.dialogues:
        assert app.scene == 'ending' and not app.ending_art_only
        for _ in app.dialogues:
            app.dialogue_time = 99.
            app.next_dialogue()
    assert app.scene == 'home'


@pytest.mark.parametrize('ending_id', likes.ENDING_IDS)
def test_three_common_memories_reach_each_main_ending_without_warmth_badge_override(app, ending_id):
    configure_preview(app, ending_id)
    state = app.campaign.data['likes']
    representatives = deepcopy(state['representatives'])
    stars = deepcopy(state['best_stars'])
    affection = likes.totals(state)
    assert app.campaign.memory_progress() == [4, 4, 4]
    assert app.campaign.selected_ending() == likes.select_ending(
        state, app.campaign.data['completed']) == ending_id
    assert app.campaign.active_ending() is None
    assert state['seen_endings'] == []

    app.start_memory(12)
    app.advance(2.)
    assert app.story_kind == 'memory' and app.dialogues == []
    assert app.context()['regions'] == [True] * 4
    app.finish_story()
    finish_silent_resolution(app, ending_id)
    assert app.dialogues == list(ENDING_CONTENT[ending_id]['lines'])
    assert state['seen_endings'] == [ending_id]
    app.draw()
    finish_art_and_afterword(app, ending_id)

    recover_warmth(app.campaign)
    assert app.campaign.warmth_memories() == list(range(1, 13))
    assert app.campaign.selected_ending() == ending_id
    assert app.campaign.extended_ending_unlocked() == (ending_id == 'happy_all')
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_art_only
    assert app.ending_id == ending_id
    assert app.dialogues == list(ENDING_CONTENT[ending_id]['lines'])
    finish_art_and_afterword(app, ending_id)
    assert state['seen_endings'] == [ending_id]
    assert state['representatives'] == representatives
    assert state['best_stars'] == stars
    assert likes.totals(state) == affection
    assert app.campaign.active_ending() == ending_id


@pytest.mark.parametrize('choice', ['memory_keep', 'memory_replace'])
def test_pending_friend_memory_blocks_resolution_until_choice_then_uses_main_selection(app, choice):
    configure_preview(app, 'choice')
    recover_warmth(app.campaign)
    state = app.campaign.data['likes']
    original = deepcopy(state['representatives']['1'])
    pending = deepcopy(state['pending'])
    stars = deepcopy(state['best_stars'])
    assert app.campaign.selected_ending() is None
    assert not app.campaign.extended_ending_unlocked()
    app.start_ending()
    assert app.scene == 'memory_choice'
    assert state['seen_endings'] == [] and state['pending'] == pending
    assert app.campaign.active_ending() is None

    app.dispatch(choice)
    assert app.scene == 'likes'
    assert state['pending'] is None and state['seen_endings'] == []
    assert state['representatives']['1'] == (pending if choice == 'memory_replace' else original)
    assert state['best_stars'] == stars
    expected = likes.select_ending(state, app.campaign.data['completed'])
    assert expected is not None
    app.start_ending()
    finish_silent_resolution(app, expected)
    finish_art_and_afterword(app, expected)
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_art_only
    assert app.ending_id == expected
    assert state['seen_endings'] == [expected]


@pytest.mark.parametrize('rich_badges', [False, True], ids=['no-rich-records', 'twelve-rich-records'])
def test_unknown_scores_open_affection_screen_without_inventing_an_ending(app, rich_badges):
    configure_preview(app, 'normal')
    state = app.campaign.data['likes']
    state['representatives']['1']['scores'] = None
    if rich_badges:
        recover_warmth(app.campaign)
    before = deepcopy(app.campaign.data)
    assert app.campaign.selected_ending() is None
    assert all(value is None for value in likes.totals(state).values())
    app.start_ending()
    assert app.scene == 'likes'
    assert app.campaign.data == before
    assert state['seen_endings'] == []
    assert app.campaign.active_ending() is None
    assert app.campaign.released_variants() == ()
    assert state['representatives']['1']['scores'] is None


def test_v4_save_preserves_active_ending_history_stars_affection_and_old_bool_flags(app):
    configure_preview(app, 'happy_all')
    recover_warmth(app.campaign)
    for stage, record in app.campaign.data['likes']['representatives'].items():
        score = record['scores']
        key = f'stage_{int(stage):02d}'
        components = {'warmth': score['warmth'], 'speed': score['speed'],
                      'courage': score['life']}
        app.campaign.data['star_records'][key] = {
            'score': score['total'], 'stars': score['stars'], **components}
        app.campaign.data['friend_memories'][key] = {
            'total': score['total'], 'stars': score['stars'], **components}
    app.campaign = Campaign(app.campaign.data)
    assert app.campaign.set_cosmetic('frame', 'stars')
    app.campaign.data['narrative'].update(
        basic_seen=True, true_seen=True, liberation_seen=True)
    assert app.campaign.active_ending() is None
    assert app.campaign.released_variants() == ()
    assert app.campaign.data['likes']['seen_endings'] == []
    finish_silent_resolution(app, 'happy_all')
    finish_art_and_afterword(app, 'happy_all')
    before = deepcopy(app.campaign.data)
    stars = app.campaign.total_stars()
    affection = likes.totals(app.campaign.data['likes'])
    friendship = app.campaign.friendship_scores()
    assert app.store.save(app.campaign)
    loaded = SaveStore(app.store.path).load()
    assert loaded.data['save_version'] == SAVE_VERSION
    assert loaded.data == before
    assert loaded.total_stars() == stars
    assert likes.totals(loaded.data['likes']) == affection
    assert loaded.friendship_scores() == friendship
    assert loaded.warmth_memories() == list(range(1, 13))
    assert loaded.selected_ending() == 'happy_all'
    assert loaded.active_ending() == 'happy_all'
    assert loaded.released_variants() == (0, 1, 2)
    assert loaded.data['likes']['seen_endings'] == ['happy_all']
    assert all(loaded.data['narrative'][flag]
               for flag in ('basic_seen', 'true_seen', 'liberation_seen'))


def test_replay_records_history_without_replacing_last_activated_world(app):
    configure_preview(app, 'happy_all')
    app.campaign.record_ending('ghost_a', activate=True)
    previous_world = app.campaign.active_ending()
    representatives = deepcopy(app.campaign.data['likes']['representatives'])
    app.start_ending(replay=True)
    assert app.scene == 'ending' and app.ending_art_only
    assert app.ending_id == 'happy_all'
    assert app.campaign.active_ending() == previous_world
    assert app.campaign.released_variants() == (0,)
    finish_art_and_afterword(app, 'happy_all')
    assert app.campaign.active_ending() == previous_world
    assert app.campaign.data['likes']['seen_endings'] == ['ghost_a', 'happy_all']
    assert app.campaign.data['likes']['representatives'] == representatives
    restored = SaveStore(app.store.path).load()
    assert restored.active_ending() == 'ghost_a'
    assert restored.released_variants() == (0,)
    app.start_ending()
    finish_silent_resolution(app, 'happy_all')
    assert app.campaign.active_ending() == 'happy_all'
    assert app.campaign.data['likes']['seen_endings'] == ['ghost_a', 'happy_all']
