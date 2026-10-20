"""호감도와 별점에 따른 대표 기록·엔딩 선택을 검증한다."""
from copy import deepcopy
import pytest

from kkamang import likes
from kkamang.campaign import Campaign, SaveStore, SaveFormatError, SAVE_VERSION
from kkamang.likes_preview import sample_record, SAMPLES


def full_state(components):
    state = likes.empty_state()
    for stage in range(1, 13):
        likes.offer_memory(state, sample_record(stage, components))
    return state


def test_normalization_and_replacement_not_addition():
    state = likes.empty_state()
    original = sample_record(1, (60, 10, 15))
    assert likes.offer_memory(state, original) == 'registered'
    assert likes.contribution(original) == {'bosong': 100, 'dore': 40, 'ipari': 100}
    before = deepcopy(state)
    replay = sample_record(1, (36, 25, 15), 'replay')
    assert likes.offer_memory(state, replay) == 'pending'
    assert likes.totals(state) == likes.totals(before)
    comparison = likes.comparison(state)
    assert comparison['after']['bosong'] == 5
    assert comparison['after']['dore'] == pytest.approx(100 / 12)
    likes.choose_memory(state, True)
    assert state['representatives']['1']['run_id'] == 'replay'
    assert state['best_stars']['1'] == 3
    assert likes.totals(state) == comparison['after']
    snapshot = deepcopy(state)
    assert not likes.choose_memory(state, True)
    likes.offer_memory(state, replay)
    assert state == snapshot


def test_keep_and_reject_overwriting_unresolved_choice():
    state = likes.empty_state()
    likes.offer_memory(state, sample_record(1, (60, 10, 15)))
    likes.offer_memory(state, sample_record(1, (36, 25, 15), 'replay'))
    with pytest.raises(ValueError):
        likes.offer_memory(state, sample_record(2, (60, 25, 15)))
    likes.choose_memory(state, False)
    assert state['representatives']['1']['run_id'] == 'preview-1'


@pytest.mark.parametrize('ending,components', SAMPLES.items())
def test_six_ending_branches(ending, components):
    state = full_state(components)
    assert likes.select_ending(state, range(1, 13)) == ending


def test_priority_thresholds_two_high_and_unfinished():
    state = full_state((42, 10, 5))
    assert likes.select_ending(state, range(1, 13)) == 'ghost_a'
    assert likes.select_ending(full_state((42, 20, 5)), range(1, 13)) == 'normal'

    assert likes.select_ending(full_state((0, 25, 5)), range(1, 13)) == 'sad'
    assert likes.select_ending(state, range(1, 12)) is None
    likes.offer_memory(state, sample_record(1, (60, 25, 15), 'pending'))
    assert likes.select_ending(state, range(1, 13)) is None


def test_missing_scores_are_unknown_not_sad_or_zero():
    state = full_state((0, 25, 15))
    state['representatives']['1']['scores'] = None
    assert all(v is None for v in likes.totals(state).values())
    assert likes.select_ending(state, range(1, 13)) is None
    state = full_state((0, 25, 15))
    del state['best_stars']['1']
    assert likes.select_ending(state, range(1, 13)) is None


def test_save_migration_and_pending_survive_reload(tmp_path):
    campaign = Campaign({'save_version': 1, 'completed': [1]})
    assert campaign.data['save_version'] == SAVE_VERSION
    assert campaign.data['likes'] == likes.empty_state()
    state = campaign.data['likes']
    likes.offer_memory(state, sample_record(1, (60, 10, 15)))
    likes.offer_memory(state, sample_record(1, (36, 25, 15), 'replay'))
    store = SaveStore(tmp_path / 'save.json')
    store.load()
    assert store.save(campaign)
    loaded = SaveStore(store.path).load()
    assert loaded.data == campaign.data
    likes.choose_memory(loaded.data['likes'], True)
    assert likes.totals(loaded.data['likes'])['bosong'] == 5


def test_corrupt_affection_save_is_rejected():
    data = Campaign().data
    data['completed'] = [1]
    likes.offer_memory(data['likes'], sample_record(1, (60, 25, 15)))
    data['likes']['representatives']['1']['scores']['warmth'] = float('nan')
    with pytest.raises(SaveFormatError):
        Campaign(data)


def test_main_v3_memories_migrate_without_inventing_run_details():
    scores = {'warmth': 36, 'speed': 25, 'courage': 15, 'total': 76, 'stars': 2}
    campaign = Campaign({'save_version': 3, 'completed': [1],
                         'friend_memories': {'stage_01': scores},
                         'star_records': {'stage_01': {'score': 85, 'stars': 3}}})
    record = campaign.data['likes']['representatives']['1']
    assert record['scores']['life'] == 15
    assert record['banked'] is None and record['ticks_used'] is None
    assert campaign.data['likes']['best_stars']['1'] == 3
    assert Campaign(campaign.data).data == campaign.data


def test_likes_v2_preserves_representative_and_cosmetic_unlocks():
    data = Campaign().data
    data['save_version'] = 2
    data['completed'] = [1, 2]
    data['badges'] = ['stage_01:rich', 'stage_01:quiet', 'stage_02:rich', 'stage_02:quiet']
    data['cosmetics']['scarf'] = 'peach'
    del data['legacy_cosmetic_stars']
    likes.offer_memory(data['likes'], sample_record(1, (60, 10, 15)))
    loaded = Campaign(data)
    assert loaded.data['cosmetics']['scarf'] == 'peach'
    assert loaded.data['likes']['representatives'] == data['likes']['representatives']


def test_settlement_is_atomic_and_idempotent_and_excludes_failed_ai():
    from kkamang.levels import load_level
    level = load_level(1)
    outcome = {'status': 'COMPLETE', 'quest_collected': True, 'banked': 14,
               'carried': 14, 'damage_count': 0, 'ticks_used': 40, 'remaining_ticks': 40}
    record = sample_record(1, (60, 25, 15), 'settled')
    campaign = Campaign()
    for controller in ('ai', 'human'):
        campaign.settle(level, outcome, 'preview-only', 'basic', controller, controller)
    assert campaign.data['likes'] == likes.empty_state()
    campaign.settle(level, outcome, 'preview-only', 'basic', 'settled', memory_record=record)
    before = deepcopy(campaign.data)
    campaign.settle(level, outcome, 'preview-only', 'basic', 'settled', memory_record=record)
    assert campaign.data == before
    for status in ('FAILED', 'ABORTED'):
        failed = dict(outcome, status=status, banked=0)
        campaign.settle(level, failed, 'preview-only', 'basic', status)
    assert campaign.data['likes'] == before['likes']
    invalid = deepcopy(record)
    invalid['run_id'] = 'wrong'
    before = deepcopy(campaign.data)
    with pytest.raises(ValueError):
        campaign.settle(level, outcome, 'preview-only', 'basic', 'different', memory_record=invalid)
    assert campaign.data == before
