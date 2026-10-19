"""Existing affection records survive the two-ending visual story."""
from copy import deepcopy
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pytest

from kkamang import likes
from kkamang.app import GameApp
from kkamang.campaign import Campaign, SaveStore, SAVE_VERSION
from kkamang.likes_preview import configure_preview
from kkamang.story import ENDING, EXTENDED_ENDING


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True,
                       movement_mode='grid')
    yield instance
    instance.close()


def recover_warmth(campaign):
    campaign.data['badges'] = sorted(set(campaign.data['badges']) |
                                    {f'stage_{stage:02d}:rich' for stage in range(1, 13)})


def finish_silent_resolution(app, recovered=False):
    assert app.scene == 'story' and app.story_kind == 'garden'
    assert app.dialogues == []
    app.advance(3.2)
    if recovered:
        assert app.scene == 'story' and app.story_kind == 'liberation'
        app.advance(4.)
    assert app.scene == 'ending'


@pytest.mark.parametrize('ending_id', likes.ENDING_IDS)
def test_six_affection_classifications_survive_independently_of_two_story_endings(app, ending_id):
    configure_preview(app, ending_id)
    state = app.campaign.data['likes']
    representatives = deepcopy(state['representatives'])
    stars = deepcopy(state['best_stars'])
    affection = likes.totals(state)
    assert likes.select_ending(state, app.campaign.data['completed']) == ending_id
    finish_silent_resolution(app)
    assert app.ending_id == 'basic'
    assert app.ending_variant == 'basic'
    assert app.dialogues == list(ENDING)
    assert state['seen_endings'] == []
    app.dispatch('story_skip')
    assert app.campaign.data['narrative']['basic_seen']

    recover_warmth(app.campaign)
    assert app.campaign.extended_ending_unlocked()
    app.start_ending()
    finish_silent_resolution(app, recovered=True)
    assert app.ending_id == 'true'
    assert app.ending_variant == 'extended'
    assert app.dialogues == list(EXTENDED_ENDING)
    app.dispatch('story_skip')
    app.dispatch('show_ending')
    assert app.scene == 'ending'
    assert app.ending_id == 'true'
    assert state['seen_endings'] == []
    assert app.campaign.data['narrative'] == {
        'basic_seen': True, 'true_seen': True, 'liberation_seen': True}
    assert state['representatives'] == representatives
    assert state['best_stars'] == stars
    assert likes.totals(state) == affection
    assert likes.select_ending(state, app.campaign.data['completed']) == ending_id


@pytest.mark.parametrize('choice', ['memory_keep', 'memory_replace'])
def test_pending_friend_memory_blocks_warmth_epilogue_and_ending_history(app, choice):
    configure_preview(app, 'choice')
    recover_warmth(app.campaign)
    state = app.campaign.data['likes']
    original = deepcopy(state['representatives']['1'])
    pending = deepcopy(state['pending'])
    assert app.campaign.extended_ending_unlocked()
    assert likes.select_ending(state, app.campaign.data['completed']) is None
    app.dispatch('show_ending')
    assert app.scene == 'memory_choice'
    assert state['seen_endings'] == []
    assert state['pending'] == pending

    app.dispatch(choice)
    assert app.scene == 'likes'
    assert state['pending'] is None
    assert state['seen_endings'] == []
    assert state['representatives']['1'] == (pending if choice == 'memory_replace' else original)
    expected = likes.select_ending(state, app.campaign.data['completed'])
    assert expected is not None
    app.dispatch('show_ending')
    finish_silent_resolution(app, recovered=True)
    assert app.ending_id == 'true'
    assert app.ending_variant == 'extended'
    assert app.dialogues == list(EXTENDED_ENDING)
    app.finish_story()
    app.start_ending(replay=True)
    assert app.scene == 'ending'
    assert likes.select_ending(state, app.campaign.data['completed']) == expected
    assert state['seen_endings'] == []


@pytest.mark.parametrize('recovered', [False, True], ids=['basic', 'warmth-recovered'])
def test_unknown_affection_scores_stay_unknown_while_warmth_alone_selects_story_ending(app, recovered):
    configure_preview(app, 'normal')
    state = app.campaign.data['likes']
    state['representatives']['1']['scores'] = None
    seen = list(state['seen_endings'])
    if recovered:
        recover_warmth(app.campaign)
    assert likes.select_ending(state, app.campaign.data['completed']) is None
    assert all(value is None for value in likes.totals(state).values())
    app.start_ending()
    finish_silent_resolution(app, recovered=recovered)
    assert app.ending_id == ('true' if recovered else 'basic')
    assert app.ending_variant == ('extended' if recovered else 'basic')
    assert app.dialogues == list(EXTENDED_ENDING if recovered else ENDING)
    assert state['seen_endings'] == seen
    assert state['representatives']['1']['scores'] is None


def test_v4_save_roundtrip_preserves_stars_affection_and_warmth_progress(app):
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
    app.campaign.data['narrative'] = {'basic_seen': True, 'true_seen': True, 'liberation_seen': True}
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
    assert loaded.extended_ending_unlocked()
    assert likes.select_ending(loaded.data['likes'], loaded.data['completed']) == 'happy_all'
    assert loaded.data['likes']['seen_endings'] == []
    assert all(loaded.data['narrative'].values())
