"""Real intermediate-frame hazards, replay witnesses and evaluation time units."""
from copy import deepcopy
import json

import pytest

from kkamang.core import GameCore
from kkamang.levels import load_level
from kkamang.realtime import RealtimeGameCore, REALTIME_OBSERVATION_VERSION
from tools.evaluate import evaluate
from tools.realtime_policy import find_realtime_path, project_frames, RealtimeGreedyPolicy
from tools.replay_validation import replay_game_record
from tools.validate_realtime_routes import validate_routes


def hazard_config(*, invincibility=False):
    config = load_level(1)
    config['ghosts'] = [{'id': 'A', 'route': [[5, 1], [6, 1]],
        'initial_index': 0, 'direction': 1, 'initial_countdown': 1, 'ticks_per_move': 1}]
    if invincibility:
        config['invincibility'] = {'position': [4, 2], 'duration_ticks': 10}
    return config


def test_projection_rejects_mid_frame_contact_even_with_clear_endpoint():
    core = RealtimeGameCore(hazard_config())
    initial = core.snapshot()
    trial = deepcopy(core)
    for _ in range(18):
        result = trial.step(3)
    assert result.snapshot['position'] != result.snapshot['ghosts'][0]['position']
    assert result.snapshot['courage'] == 2
    assert project_frames(core, [3] * 18) is None
    assert core.snapshot() == initial


def test_only_active_item_allows_contact_when_explicitly_selected():
    core = RealtimeGameCore(hazard_config(invincibility=True))
    actions = [1] * 18 + [0] * 18 + [3] * 18
    assert project_frames(core, actions, allow_item_invincibility=True) is not None
    assert project_frames(core, actions, allow_item_invincibility=False) is None


def test_planner_uses_physics_steps_and_does_not_change_live_game():
    core = RealtimeGameCore(load_level(1))
    initial = core.snapshot()
    path = find_realtime_path(core, [5, 1], interact=False, max_nodes=10)
    assert len(path) == 18
    projected = project_frames(core, path)
    assert projected[1].snapshot['motion']['position'] == [5.0, 1.0]
    assert projected[1].snapshot['tick'] == 1
    assert core.snapshot() == initial
    policy = RealtimeGreedyPolicy(core, target_warmth=4, max_search_nodes=10)
    assert 0 <= policy.action(initial) < 6
    assert core.snapshot() == initial
    with pytest.raises(ValueError, match='RealtimeGameCore'):
        find_realtime_path(GameCore(load_level(1)), [5, 1])


def test_frame_search_finds_a_detour_when_shortest_walks_hit_a_ghost():
    config = hazard_config()
    config['ghosts'][0].update(initial_countdown=200, ticks_per_move=200)
    core = RealtimeGameCore(config)
    before = core.snapshot()
    diagnostics = {}
    path = find_realtime_path(core, [6, 1], interact=False, max_nodes=256, diagnostics=diagnostics)
    assert path is not None
    assert diagnostics['expanded_nodes'] > 0
    checked = project_frames(core, path, allow_item_invincibility=False)
    assert checked is not None
    assert checked[1].snapshot['motion']['position'] == [6.0, 1.0]
    assert 1 in path and 0 in path  # Leaves and rejoins the straight corridor.
    assert core.snapshot() == before


def test_realtime_evaluation_cutoff_reports_frames_not_world_ticks(tmp_path):
    rows, report = evaluate(1, [1], 20, 'basic', True, control_mode='realtime',
        target_warmth=4, max_search_nodes=10, max_episode_steps=1, record_dir=tmp_path)
    assert report['errors'] == []
    assert report['control_mode'] == 'realtime'
    assert report['observation_version'] == REALTIME_OBSERVATION_VERSION
    assert report['episode_step_unit'] == 'physics_frame'
    assert rows[0]['initial_phase'] == rows[1]['initial_phase']
    assert rows[0]['seed'] == rows[1]['seed']
    for row in rows:
        assert row['physics_frames'] == row['episode_steps'] == 1
        assert row['ticks_used'] == 0
        assert row['simulated_seconds'] == pytest.approx(1 / 60)
        assert row['truncated'] and not row['terminated']
        assert row['banked'] == row['score_total'] == 0
    files = list(tmp_path.glob('*.json'))
    assert len(files) == 2
    for file in files:
        state = replay_game_record(json.loads(file.read_text(encoding='utf-8')))
        assert state['motion']['frame'] == 1
        assert state['outcome'] is None


def test_grid_evaluation_keeps_legacy_time_units():
    rows, report = evaluate(1, [1], 20, 'basic', False, control_mode='grid',
        max_episode_steps=1, max_search_nodes=10)
    assert not report['errors']
    assert report['control_mode'] == 'grid'
    assert report['episode_step_unit'] == 'world_tick'
    assert all(row['ticks_used'] == row['episode_steps'] == 1 for row in rows)
    assert all(row['physics_frames'] is None for row in rows)
    assert all(row['simulated_seconds'] == pytest.approx(.3) for row in rows)


def test_generated_realtime_witness_replays_and_detects_tampering(tmp_path):
    result = validate_routes([1], [4], max_search_nodes=10, output=tmp_path)
    assert result['all_passed'] and result['passed'] == 1
    row = result['routes'][0]
    assert row['contacts'] == row['damage_count'] == 0
    assert row['banked'] >= 4
    assert row['physics_frames'] > row['ticks_used']
    record = json.loads(next((tmp_path / 'replays').glob('*.json')).read_text(encoding='utf-8'))
    assert replay_game_record(record)['outcome']['status'] == 'COMPLETE'
    record['actions'][0] = 5
    with pytest.raises(AssertionError):
        replay_game_record(record)


def test_bounded_search_failure_is_reported_separately_from_route_safety():
    core = RealtimeGameCore(load_level(1))
    for _ in range(79 * 18):
        core.step(5)
    diagnostics = {}
    assert find_realtime_path(core, [11, 1], diagnostics=diagnostics) is None
    assert diagnostics['status'] == 'insufficient_time'
    assert not core.snapshot()['terminal']


@pytest.mark.parametrize('kwargs', [dict(control_mode='bad'), dict(episodes=0),
                                   dict(target_warmth=3), dict(max_search_nodes=0)])
def test_evaluation_rejects_invalid_protocol(kwargs):
    arguments = dict(episodes=1, stages=[1], seed=0, mode='basic', randomize_phase=False,
                     control_mode='realtime')
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        evaluate(**arguments)
