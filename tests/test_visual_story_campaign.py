"""Silent picture progression, old saves, and independent warmth recovery."""
from copy import deepcopy
import json

import pytest

from kkamang.campaign import Campaign, SaveFormatError, SaveStore, SAVE_VERSION
from kkamang.levels import load_level, load_levels
from kkamang.story import ENDING, EXTENDED_ENDING, MEMORIES, PROLOGUE, STAGES, memory_for_stage


def settle(campaign, stage=1, banked=4, *, status='COMPLETE', controller='human', sid=None, mode='basic'):
    level = load_level(stage)
    outcome = {
        'status': status, 'reason': 'returned' if status == 'COMPLETE' else 'timeout',
        'quest_collected': status == 'COMPLETE', 'banked': banked, 'carried': banked,
        'damage_count': 2, 'remaining_courage': 1,
        'ticks_used': level['time_budget_ticks'], 'remaining_ticks': 0,
    }
    return campaign.settle(level, outcome, 'visual-story-test', mode,
                           sid or f'run-{len(campaign.data["settled_ids"])}', controller)


def test_first_success_reveals_one_permanent_region_per_room():
    campaign = Campaign()
    assert campaign.memory_progress() == [0, 0, 0]
    for stage in range(1, 13):
        memory_id, region = memory_for_stage(stage)
        before = [campaign.memory_regions(number) for number in range(1, 4)]
        result = settle(campaign, stage)
        assert result['first_complete']
        expected = deepcopy(before)
        expected[memory_id - 1][region] = True
        assert [campaign.memory_regions(number) for number in range(1, 4)] == expected
        assert campaign.memory_complete(memory_id) == (stage % 4 == 0)
        assert sum(campaign.memory_progress()) == stage
    assert campaign.memory_progress() == [4, 4, 4]
    assert campaign.total_stars() == 0
    assert not campaign.extended_ending_unlocked()
    assert campaign.data['narrative'] == {
        'basic_seen': False, 'true_seen': False, 'liberation_seen': False}


def test_replaying_a_room_or_settlement_does_not_add_picture_regions():
    campaign = Campaign()
    settle(campaign)
    regions = [campaign.memory_regions(number) for number in range(1, 4)]
    result = settle(campaign, banked=14, sid='replay')
    assert not result['first_complete'] and not result['new_photo']
    assert campaign.memory_progress() == [1, 0, 0]
    assert [campaign.memory_regions(number) for number in range(1, 4)] == regions
    before = deepcopy(campaign.data)
    assert settle(campaign, banked=14, sid='replay')['already_settled']
    assert campaign.data == before
    detached = campaign.memory_regions(1)
    detached[1] = True
    assert campaign.memory_regions(1) == [True, False, False, False]


def test_practice_first_success_never_reveals_a_region_or_saves_rewards():
    campaign = Campaign()
    before = deepcopy(campaign.data)
    reward = settle(campaign, banked=14, mode='relaxed')
    assert reward['practice_mode'] and not reward['eligible']
    assert not reward['first_complete'] and not reward['new_photo'] and not reward['new_ending']
    assert campaign.memory_progress() == [0, 0, 0]
    assert campaign.data == before


def test_practice_cannot_fill_the_last_warmth_or_change_seen_endings(tmp_path):
    campaign = Campaign()
    for stage in range(1, 13):
        settle(campaign, stage, 4 if stage == 5 else 10)
    campaign.data['narrative']['basic_seen'] = True
    campaign.data['likes']['seen_endings'] = ['normal']
    before = deepcopy(campaign.data)
    reward = settle(campaign, 5, 14, mode='relaxed')
    assert reward['practice_mode'] and not reward['new_ending']
    assert campaign.memory_progress() == [4, 4, 4]
    assert not campaign.extended_ending_unlocked()
    assert campaign.data == before
    store = SaveStore(tmp_path / 'practice-save.json')
    assert store.save(campaign)
    assert SaveStore(store.path).load().data == before


@pytest.mark.parametrize('status,controller,stage,banked', [
    ('FAILED', 'human', 1, 0), ('ABORTED', 'human', 1, 0),
    ('COMPLETE', 'ai', 1, 14), ('COMPLETE', 'human', 2, 14),
])
def test_unsuccessful_ineligible_and_locked_runs_never_reveal_a_region(status, controller, stage, banked):
    campaign = Campaign()
    settle(campaign, stage, banked, status=status, controller=controller)
    assert campaign.memory_progress() == [0, 0, 0]
    assert campaign.data['photos'] == []
    assert campaign.warmth_memories() == []


@pytest.mark.parametrize('memory_id', [0, 4, -1, True, 1.0, '1', None])
def test_memory_queries_reject_invalid_identifiers(memory_id):
    campaign = Campaign()
    with pytest.raises(ValueError):
        campaign.memory_regions(memory_id)
    with pytest.raises(ValueError):
        campaign.memory_complete(memory_id)


@pytest.mark.parametrize('version', [0, 1, 2, 3, 4])
def test_existing_saves_derive_regions_from_completed_without_inventing_warmth(tmp_path, version):
    payload = {'save_version': version, 'completed': [10, 1, 3, 6, 3],
               'photos': [1, 8, 12], 'favorite_photo': 12,
               'badges': ['stage_01:quiet']}
    original = deepcopy(payload)
    path = tmp_path / 'old-save.json'
    path.write_text(json.dumps(payload), encoding='utf-8')
    store = SaveStore(path)
    campaign = store.load()
    assert store.error is None
    assert payload == original
    assert campaign.data['save_version'] == SAVE_VERSION
    assert campaign.data['photos'] == [1, 3, 6, 8, 10, 12]
    assert campaign.data['favorite_photo'] == 12
    assert campaign.memory_regions(1) == [True, False, True, False]
    assert campaign.memory_regions(2) == [False, True, False, False]
    assert campaign.memory_regions(3) == [False, True, False, False]
    assert campaign.memory_progress() == [2, 1, 1]
    assert campaign.warmth_memories() == []
    assert not campaign.extended_ending_unlocked()
    assert not any(campaign.data['narrative'].values())
    before = deepcopy(campaign.data)
    assert store.save(campaign)
    assert SaveStore(path).load().data == before


def test_warmth_recovery_uses_each_rooms_success_record_without_stars_or_no_damage():
    campaign = Campaign()
    for stage in range(1, 13):
        result = settle(campaign, stage, 9 if stage == 5 else 10)
        assert not result['new_ending']
    assert campaign.memory_progress() == [4, 4, 4]
    # Nine points on repeated successful runs cannot accumulate into ten.
    settle(campaign, 5, 9)
    settle(campaign, 5, 9)
    assert not campaign.extended_ending_unlocked()
    result = settle(campaign, 5, 10)
    assert result['new_ending']
    assert campaign.extended_ending_unlocked()
    assert campaign.total_stars() == 0
    assert not any(badge.endswith(':quiet') for badge in campaign.data['badges'])
    assert campaign.warmth_memories() == list(range(1, 13))
    assert not any(campaign.data['narrative'].values())
    before_regions = campaign.memory_progress()
    before_warmth = campaign.warmth_memories()
    settle(campaign, 5, 0, status='FAILED')
    assert campaign.memory_progress() == before_regions
    assert campaign.warmth_memories() == before_warmth
    assert not settle(campaign, 5, 10)['new_ending']


@pytest.mark.parametrize('value', [None, [], 'seen', 1, {'basic_seen': 1},
                                  {'true_seen': 'false'}, {'liberation_seen': None}])
def test_malformed_narrative_flags_are_rejected(value):
    with pytest.raises(SaveFormatError):
        Campaign({'narrative': value})


def test_narrative_flags_are_independent_and_survive_save_roundtrip(tmp_path):
    campaign = Campaign({'completed': [1], 'photos': [1], 'narrative': {'basic_seen': True}})
    assert campaign.data['narrative'] == {
        'basic_seen': True, 'true_seen': False, 'liberation_seen': False}
    campaign.data['narrative']['liberation_seen'] = True
    campaign.data['narrative']['true_seen'] = True
    before = deepcopy(campaign.data)
    store = SaveStore(tmp_path / 'save.json')
    assert store.save(campaign)
    reloaded = SaveStore(store.path).load()
    assert reloaded.data == before
    assert reloaded.memory_progress() == [1, 0, 0]
    assert not reloaded.extended_ending_unlocked()


def test_silent_stage_contract_and_specified_objects_match_the_three_memories():
    expected = [
        ('수건', 'towel'), ('안내 지도', 'map'), ('발자국 사진', 'footprint_photo'),
        ('정원 안내서', 'book'), ('스카프 고리', 'scarf_hook'), ('씨앗 주머니', 'seeds'),
        ('잃어버린 악보', 'music'), ('찻잔 받침', 'saucer'), ('커튼 끈', 'cord'),
        ('담요', 'cushion'), ('마지막 악보', 'music_final'), ('정원 손잡이', 'handle'),
    ]
    assert PROLOGUE == []
    assert [memory['stages'] for memory in MEMORIES] == [list(range(1, 5)), list(range(5, 9)), list(range(9, 13))]
    assert [memory['name'] for memory in MEMORIES] == ['문이 열리던 밤', '세 개의 자리', '놓고 싶지 않았던 밤']
    for level, stage, (name, kind) in zip(load_levels(), STAGES, expected):
        assert level['quest']['name'] == name
        assert level['quest']['kind'] == stage['emblem'] == kind
        assert stage['name'] == level['name']
        assert stage['lines'] == []
        assert name in stage['before'] and '4점' in stage['before']
        memory_id, region = memory_for_stage(level['stage'])
        assert stage['restoration'] == f'기억 {memory_id} · {region + 1}번 영역'
    assert ENDING == [('보송', '왔네. 수건은 저기 있어.'), ('이파리', '네 자리도 그대로야.'),
                      ('도레', '그럼, 오늘은 작게 연주할게.')]
    assert EXTENDED_ENDING == [('보송', '꼬리 좀…… 내 코에 닿아.'), ('도레', '내 거 아닌데.'),
                               ('이파리', '쉿. 까망이 잔다.')]
