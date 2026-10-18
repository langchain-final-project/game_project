"""Campaign correctness, progression invariants and actual save failure recovery."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from kkamang.campaign import Campaign, SaveStore, SAVE_VERSION
from kkamang.story import STAGES, PROLOGUE, ENDING


def level(stage=1):
    return {'stage': stage, 'id': f'stage_{stage:02d}', 'target_value': 4, 'total_item_value': 14,
            'star_warmth_cap': 14, 'time_budget_ticks': 80, 'star_fast_ticks': 56,
            'initial_courage': 3}


def outcome(status='COMPLETE', banked=10, damage=0):
    return {'status': status, 'reason': 'returned' if status in ('COMPLETE', 'RETURN_LOW') else 'timeout',
            'banked': banked, 'carried': banked if banked else 10, 'damage_count': damage,
            'remaining_courage': 3 - damage,
            'ticks_used': 40, 'remaining_ticks': 20}


def clear(campaign, stage=1, mode='basic', sid=None, **kwargs):
    return campaign.settle(level(stage), outcome(**kwargs), 'hash-A', mode,
                           sid or f'{stage}-{mode}-{len(campaign.data["settled_ids"])}')


def test_all_twelve_first_completions_unlock_story_without_cutscene_flags():
    c = Campaign()
    for stage in range(1, 13):
        assert c.is_unlocked(stage)
        if stage < 12:
            assert not c.is_unlocked(stage + 1)
        result = clear(c, stage, banked=4, damage=1)
        assert result['first_complete'] and result['new_photo']
        assert c.data['photos'] == list(range(1, stage + 1))
    assert c.optional_badges() == 0
    assert len(c.data['room_states']) == 12
    assert c.data['intro_seen'] is False  # Scene viewing is not a reward condition.


def test_settlement_is_idempotent_even_after_save_reload(tmp_path):
    store = SaveStore(tmp_path / 'save.json')
    c = store.load()
    clear(c, sid='same-id')
    before = deepcopy(c.data)
    assert store.save(c)
    c = SaveStore(store.path).load()
    result = clear(c, sid='same-id')
    assert result['already_settled']
    assert c.data == before


def test_unique_badges_cross_modes_and_cosmetics_thresholds():
    c = Campaign()
    clear(c, 1, banked=10)
    clear(c, 1, mode='relaxed', banked=14)
    assert c.optional_badges() == 2
    assert c.unlocked_cosmetics('scarf') == ['default']
    assert not c.set_cosmetic('scarf', 'peach')
    result = clear(c, 2)
    assert c.optional_badges() == 4
    assert result['new_cosmetics'] == []
    result = clear(c, 3)
    assert result['new_cosmetics'] == ['scarf:lavender', 'scarf:peach']
    assert c.set_cosmetic('scarf', 'peach')
    for stage in range(4, 13):
        clear(c, stage)
    assert c.optional_badges() == 24
    assert len(c.unlocked_cosmetics('rug')) == 3
    assert len(c.unlocked_cosmetics('frame')) == 3
    assert c.set_cosmetic('frame', 'stars')
    c2 = Campaign(c.data)
    assert c2.optional_badges() == 24
    assert c2.data['cosmetics']['scarf'] == 'peach'


def test_stars_keep_only_a_room_best_score_and_friend_memory_is_independent():
    c = Campaign()
    first = outcome(banked=4, damage=0)
    first['ticks_used'] = 0
    first['remaining_courage'] = 3
    reward = c.settle(level(), first, 'hash-A', 'basic', 'first')
    assert reward['score']['total'] == 40
    assert c.stars_for_stage(1) == 0
    original_memory = deepcopy(c.data['friend_memories']['stage_01'])
    retry = outcome(banked=14, damage=2)
    retry['ticks_used'] = 80
    retry['remaining_courage'] = 1
    reward = c.settle(level(), retry, 'hash-A', 'basic', 'retry')
    assert reward['best_star_improved'] and c.stars_for_stage(1) == 2
    assert c.data['friend_memories']['stage_01'] == original_memory
    assert c.remember_friend_run(level(), retry)
    assert c.data['friend_memories']['stage_01'] != original_memory


def test_friendship_normalises_the_three_score_ranges_to_the_same_scale():
    c = Campaign()
    c.data['friend_memories'] = {'stage_01': {
        'warmth': 36, 'speed': 20, 'courage': 15, 'total': 71, 'stars': 2,
    }}
    assert c.friendship_scores() == {'bosong': 60, 'ipari': 100, 'dore': 80}
    assert c.friendship_maximums() == {'bosong': 1200, 'ipari': 1200, 'dore': 1200}


def test_failed_low_aborted_locked_and_ai_runs_do_not_unlock_story():
    c = Campaign()
    for status, banked in [('RETURN_LOW', 3), ('FAILED', 0), ('ABORTED', 0)]:
        clear(c, status=status, banked=banked)
    clear(c, stage=2)
    ai = c.settle(level(), outcome(), 'hash-A', 'basic', 'ai-1', controller='ai')
    assert not ai['first_complete']
    assert c.data['completed'] == []
    assert c.data['photos'] == []
    assert c.optional_badges() == 0
    assert c.data['records']['ai']['basic']['hash-A']['stage_01']['complete'] == 1


def test_records_are_partitioned_by_mode_hash_controller():
    c = Campaign()
    for i, (mode, hash_, ctrl) in enumerate([('basic', 'A', 'human'), ('relaxed', 'A', 'human'),
                                            ('basic', 'B', 'human'), ('basic', 'A', 'ai')]):
        c.settle(level(), outcome(), hash_, mode, str(i), ctrl)
    r = c.data['records']
    assert r['human']['basic']['A']['stage_01']['episodes'] == 1
    assert r['human']['basic']['B']['stage_01']['episodes'] == 1
    assert r['human']['relaxed']['A']['stage_01']['episodes'] == 1
    assert r['ai']['basic']['A']['stage_01']['episodes'] == 1
    assert c.optional_badges() == 2


def test_badge_conditions_independent_and_no_score_spending():
    c = Campaign()
    result = clear(c, banked=8)
    assert result['new_badges'] == ['stage_01:quiet']
    result = clear(c, banked=11, damage=1)
    assert result['new_badges'] == ['stage_01:rich']
    record = c.data['records']['human']['basic']['hash-A']['stage_01']
    assert record['best_banked'] == 11
    assert record['last']['banked'] == 11


def test_impossible_outcome_rejected_without_partial_settlement():
    c = Campaign()
    before = deepcopy(c.data)
    with pytest.raises(ValueError):
        clear(c, banked=3)
    with pytest.raises(ValueError):
        clear(c, status='FAILED', banked=10)
    assert c.data == before


def test_atomic_save_keeps_previous_good_backup(tmp_path):
    store = SaveStore(tmp_path / 'nested' / 'save.json')
    c = store.load()
    clear(c, 1)
    assert store.save(c)
    first_bytes = store.path.read_bytes()
    clear(c, 2)
    assert store.save(c)
    assert store.backup_path.read_bytes() == first_bytes
    reloaded = SaveStore(store.path).load()
    assert reloaded.completed(2)
    assert not list(store.path.parent.glob('*.tmp'))


def test_failed_replace_keeps_memory_and_original_then_retry(tmp_path, monkeypatch):
    store = SaveStore(tmp_path / 'save.json')
    c = store.load()
    clear(c, 1)
    assert store.save(c)
    prior = store.path.read_bytes()
    clear(c, 2)
    import kkamang.campaign as module
    original_replace = module.os.replace

    def fail_primary(src, dst):
        if Path(dst) == store.path:
            raise OSError('simulated disk write failure')
        original_replace(src, dst)

    monkeypatch.setattr(module.os, 'replace', fail_primary)
    assert not store.save(c)
    assert c.completed(2)
    assert store.path.read_bytes() == prior
    assert store.error and not list(tmp_path.glob('*.tmp'))
    monkeypatch.setattr(module.os, 'replace', original_replace)
    assert store.save(c)
    assert SaveStore(store.path).load().completed(2)


def test_future_save_never_overwritten_including_direct_save_call(tmp_path):
    path = tmp_path / 'save.json'
    raw = b'{"save_version": 200, "future_data": "precious"}'
    path.write_bytes(raw)
    store = SaveStore(path)
    c = store.load()
    clear(c)
    assert store.has_save and store.read_only
    assert not store.save(c)
    assert path.read_bytes() == raw
    direct = SaveStore(path)
    assert not direct.save(Campaign())
    assert path.read_bytes() == raw


def test_corrupt_primary_recovers_backup_and_archives_original(tmp_path):
    path = tmp_path / 'save.json'
    store = SaveStore(path)
    c = store.load()
    clear(c)
    assert store.save(c)
    clear(c, 2)
    assert store.save(c)
    corrupt = b'{BROKEN JSON'
    path.write_bytes(corrupt)
    recovered_store = SaveStore(path)
    recovered = recovered_store.load()
    assert recovered_store.recovered_from_backup
    assert recovered.completed(1) and not recovered.completed(2)
    assert recovered_store.error is not None
    clear(recovered, 2)
    assert recovered_store.save(recovered)
    assert len(list(tmp_path.glob('save.json.corrupt-*'))) == 1
    assert next(tmp_path.glob('save.json.corrupt-*')).read_bytes() == corrupt
    assert SaveStore(path).load().completed(2)


def test_corrupt_without_backup_is_read_only_and_preserved(tmp_path):
    path = tmp_path / 'save.json'
    raw = b'{"save_version": 1, "completed": "invalid"}'
    path.write_bytes(raw)
    store = SaveStore(path)
    c = store.load()
    assert store.read_only
    clear(c)
    assert not store.save(c)
    assert path.read_bytes() == raw


def test_external_change_blocks_overwrite(tmp_path):
    path = tmp_path / 'save.json'
    store = SaveStore(path)
    c = store.load()
    assert store.save(c)
    changed = b'{"save_version": 5}'
    path.write_bytes(changed)
    assert not store.save(c)
    assert store.read_only
    assert path.read_bytes() == changed


def test_legacy_migration_is_deterministic_and_scene_independent():
    raw = {'version': 0, 'first_complete': {'stage_01': True, 'stage_02': True},
           'unique_badges': ['stage_01:quiet', 'stage_01:quiet', 'stage_01:rich',
                             'stage_02:quiet', 'stage_02:rich'],
           'cosmetics': {'scarf': 'peach'}}
    c = Campaign(raw)
    assert c.data['save_version'] == SAVE_VERSION
    assert c.data['photos'] == [1, 2]
    assert c.optional_badges() == 4
    assert c.data['cosmetics']['scarf'] == 'peach'
    assert Campaign(c.data).data == c.data
    assert raw['version'] == 0


def test_twelve_story_entries_match_levels_and_real_assets():
    project = Path(__file__).resolve().parents[1]
    for stage, story in enumerate(STAGES, 1):
        assert story['stage'] == stage
        cfg = json.loads((project / f'data/levels/stage_{stage:02d}.json').read_text('utf-8'))
        assert cfg['name'] == story['name']
        assert story['lines'] and story['restoration'] and story['before'] and story['teaser']
        if story['art']:
            # Match the renderer: bare filenames use art/, paths are relative
            # to assets/ (including the reorganized background/ directory).
            image = story['art'] if '/' in story['art'] else 'art/' + story['art']
            assert (project / 'assets' / image).is_file()
        else:
            assert story['emblem']
    assert len(STAGES) == 12 and PROLOGUE and ENDING
    assert STAGES[-1]['lines'][-1] == ('보송', '오늘은 문, 열어 둘게.')


def test_parseable_but_damaged_records_are_preserved_as_corrupt(tmp_path):
    path = tmp_path / 'save.json'
    data = Campaign().data
    data['records'] = {'human': {'basic': {'hash': {'stage_01': {'episodes': 'broken'}}}}}
    raw = json.dumps(data).encode('utf-8')
    path.write_bytes(raw)
    store = SaveStore(path)
    campaign = store.load()
    assert store.read_only and store.error
    clear(campaign)
    assert not store.save(campaign)
    assert path.read_bytes() == raw
