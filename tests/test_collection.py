"""수집 보상이 기존 진행을 건드리지 않는지와 화면·저장 연결을 검증한다."""
from copy import deepcopy
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import pygame
import pytest

from kkamang import collection
from kkamang.app import GameApp
from kkamang.campaign import Campaign, SaveStore, SaveFormatError
from kkamang.levels import load_levels
from kkamang.paths import ASSET_DIR


def finish(campaign, stage=1, mode='basic', controller='human', status='COMPLETE', sid=None):
    level = load_levels()[stage-1]
    outcome = dict(status=status, banked=14 if status == 'COMPLETE' else 0,
                   carried=14, damage_count=0, ticks_used=level['star_fast_ticks'],
                   remaining_ticks=level['time_budget_ticks']-level['star_fast_ticks'],
                   remaining_courage=level['initial_courage'], quest_collected=True)
    return campaign.settle(level, outcome, 'collection-test', mode,
                           sid or f'{stage}-{mode}-{len(campaign.data["settled_ids"])}', controller)


def complete_all(campaign):
    for stage in range(1, 13):
        finish(campaign, stage)


def other_progress(data):
    return {key: value for key, value in data.items() if key != 'collection'}


def test_catalog_has_exactly_24_existing_portraits_and_agreed_rules():
    entries = collection.catalog()
    assert [e['id'] for e in entries] == list(range(1, 25))
    root = ASSET_DIR / 'art' / 'collection'
    assert {p.name for p in root.glob('*.png')} == {e['file'] for e in entries}
    assert all(e['name'].endswith('까망이') for e in entries[:4])
    assert sorted(e['goal'] for e in entries if e['rule'] == 'stars') == list(range(3, 37, 3))
    for rule in ('rich', 'quiet'):
        assert sorted(e['goal'] for e in entries if e['rule'] == rule) == [3, 6, 9, 12]
    for entry in entries:
        assert entry['file'].startswith(f'{entry["id"]:02d}_')
        assert pygame.image.load(str(root / entry['file'])).get_size() == (1254, 1254)


def test_practice_only_grants_collection_first_step_once():
    campaign = Campaign()
    before = deepcopy(campaign.data)
    reward = finish(campaign, mode='relaxed')
    assert reward['new_collection'] == [1]
    assert campaign.data['collection']['practice_rooms'] == [1]
    assert other_progress(campaign.data) == other_progress(before)
    assert finish(campaign, mode='relaxed')['new_collection'] == []
    assert campaign.data['completed'] == []


@pytest.mark.parametrize('status', ['FAILED', 'ABORTED'])
def test_practice_failure_and_abort_never_grant_cards(status):
    campaign = Campaign()
    before = deepcopy(campaign.data)
    finish(campaign, mode='relaxed', status=status)
    assert campaign.data == before


@pytest.mark.parametrize('stage,controller', [(1, 'ai'), (2, 'human')])
def test_ai_and_locked_rooms_cannot_grant_practice_cards(stage, controller):
    campaign = Campaign()
    finish(campaign, stage, mode='relaxed', controller=controller)
    assert collection.available(campaign.data) == set()


def test_three_distinct_rooms_combines_modes_without_counting_repeats():
    campaign = Campaign()
    finish(campaign)
    finish(campaign, 2, mode='relaxed')
    finish(campaign, 2, mode='relaxed')
    assert 7 not in collection.available(campaign.data)
    finish(campaign, 2)
    assert 7 not in collection.available(campaign.data)
    reward = finish(campaign, 3, mode='relaxed')
    assert reward['new_collection'] == [7]
    assert campaign.data['completed'] == [1, 2]


def test_all_thresholds_and_easter_eggs_unlock_exactly_24_cards():
    campaign = Campaign()
    for stage in range(1, 13):
        finish(campaign, stage)
        for entry in collection.catalog():
            if entry['rule'] in ('stars', 'rich', 'quiet'):
                amount = stage * 3 if entry['rule'] == 'stars' else stage
                assert (entry['id'] in collection.available(campaign.data)) == (amount >= entry['goal'])
    assert len(collection.available(campaign.data)) == 22
    before = deepcopy(other_progress(campaign.data))
    assert collection.discover(campaign.data, 'shelter_return') == []
    collection.discover(campaign.data, 'shelter_before')
    assert collection.discover(campaign.data, 'shelter_return') == [2]
    assert collection.discover(campaign.data, 'memory_paw') == [23]
    assert collection.discover(campaign.data, 'memory_paw') == []
    assert collection.available(campaign.data) == set(range(1, 25))
    assert other_progress(campaign.data) == before


def test_incomplete_memories_cannot_grant_paw_card():
    campaign = Campaign()
    finish(campaign)
    assert collection.discover(campaign.data, 'memory_paw') == []


def test_unlocked_card_and_favorite_survive_save_reload_and_lower_records(tmp_path):
    campaign = Campaign()
    finish(campaign)
    assert not collection.set_favorite(campaign.data, 4)
    assert collection.set_favorite(campaign.data, 1)
    store = SaveStore(tmp_path / 'collection.json')
    assert store.save(campaign)
    loaded = SaveStore(store.path).load()
    assert loaded.data == campaign.data
    loaded.data['star_records'].clear()
    collection.sync(loaded.data)
    assert 5 in collection.available(loaded.data)
    assert loaded.data['collection']['favorite'] == 1


def test_v4_migration_grants_existing_rewards_without_touching_likes_or_legacy_cosmetics():
    campaign = Campaign()
    complete_all(campaign)
    raw = deepcopy(campaign.data)
    raw['save_version'] = 4
    del raw['collection']
    raw['cosmetics']['frame'] = 'stars'
    before = deepcopy(raw)
    migrated = Campaign(raw)
    assert len(collection.available(migrated.data)) == 22
    assert migrated.data['collection']['unlocked'] == sorted(collection.available(migrated.data))
    for key in before:
        if key != 'save_version':
            assert migrated.data[key] == before[key]


@pytest.mark.parametrize('field,value', [('unlocked', [0]), ('seen', [True]),
                                         ('practice_rooms', [13]), ('favorite', 25),
                                         ('discoveries', ['unknown'])])
def test_invalid_collection_save_is_rejected(field, value):
    raw = Campaign().data
    raw['collection'][field] = value
    with pytest.raises(SaveFormatError):
        Campaign(raw)


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'collection.json', headless=True, no_audio=True)
    yield instance
    instance.close()


def test_collection_ui_navigation_favorite_and_item_cards(app):
    finish(app.campaign)
    app.dispatch('appearance')
    app.draw()
    cards = [c for c in app.controls if c['id'].startswith('collection_card:')]
    assert len(cards) == 24
    assert len({c['rect'].x for c in cards}) == 6
    assert len({c['rect'].y for c in cards}) == 4
    assert len({c['rect'].x for c in cards[:4]}) == 1
    first = next(i for i, c in enumerate(app.controls) if c['id'] == 'collection_card:looks:1')
    app.focus = first
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT, mod=0))
    assert app.controls[app.focus]['id'] == 'collection_card:looks:5'
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_DOWN, mod=0))
    assert app.controls[app.focus]['id'] == 'collection_card:looks:6'
    app.dispatch('collection_card:looks:1')
    app.draw()
    assert any(c['id'] == 'collection_favorite' for c in app.controls)
    app.dispatch('collection_favorite')
    assert app.campaign.data['collection']['favorite'] == 1
    app.go_back()
    assert app.scene == 'appearance' and app.collection_detail is None
    app.draw()
    assert app.controls[app.focus]['id'] == 'collection_card:looks:1'
    app.go_back()
    app.dispatch('appearance')
    assert app.collection_cover
    app.draw()
    assert any(c['id'] == 'collection_open:items' for c in app.controls)
    app.dispatch('collection_open:items')
    before = deepcopy(app.campaign.data)
    app.draw()
    assert len([c for c in app.controls if c['id'].startswith('collection_card:')]) == 12
    app.dispatch('collection_card:items:1')
    app.draw()
    assert app.campaign.data == before
    assert not any(c['id'] == 'collection_favorite' for c in app.controls)
    assert SaveStore(app.store.path).load().data == app.campaign.data


def test_locked_card_can_be_inspected_but_cannot_be_favorited(app):
    app.dispatch('appearance')
    app.dispatch('collection_card:looks:4')
    before = deepcopy(app.campaign.data)
    app.draw()
    assert not any(c['id'] == 'collection_favorite' for c in app.controls)
    app.dispatch('collection_favorite')
    assert app.campaign.data == before


def test_failed_collection_save_keeps_reward_and_retries(app, monkeypatch):
    finish(app.campaign)
    app.dispatch('appearance')
    app.dispatch('collection_card:looks:1')
    write = app.store._atomic_write
    def fail(*args):
        raise OSError('test disk full')
    monkeypatch.setattr(app.store, '_atomic_write', fail)
    app.dispatch('collection_favorite')
    assert app.scene == 'save_error'
    assert app.campaign.data['collection']['favorite'] == 1
    monkeypatch.setattr(app.store, '_atomic_write', write)
    app.dispatch('save_retry')
    assert app.scene == 'appearance'
    assert SaveStore(app.store.path).load().data == app.campaign.data


def test_easter_eggs_are_saved_and_require_completed_memory(app):
    app.dispatch('album')
    app.dispatch('collection_memory_paw')
    assert 23 not in collection.available(app.campaign.data)
    for stage in range(1, 5):
        finish(app.campaign, stage)
    app.dispatch('home')
    app.dispatch('toggle_home_view')
    app.dispatch('toggle_home_view')
    app.dispatch('album')
    app.draw()
    paw = next(c for c in app.controls if c['id'] == 'collection_memory_paw')
    app.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=paw['rect'].center))
    assert {2, 23} <= collection.available(app.campaign.data)
    assert SaveStore(app.store.path).load().data == app.campaign.data
    app.draw()
    assert not any(c['id'] == 'collection_memory_paw' for c in app.controls)


def test_collection_controls_stay_inside_canvas_and_do_not_overlap(app):
    complete_all(app.campaign)
    for tab in ('looks', 'items'):
        app.dispatch(f'collection_tab:{tab}')
        before = deepcopy(app.campaign.data)
        app.draw()
        assert app.campaign.data == before
        for index, control in enumerate(app.controls):
            assert app.canvas.get_rect().contains(control['rect'])
            assert not any(control['rect'].colliderect(other['rect']) for other in app.controls[index+1:])
