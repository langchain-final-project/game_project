"""Replay v0.2 recorded witness routes through the implemented game core.

This verifies conformance. It does not search for new routes, train an agent, or
establish that human players find the proposed time budgets comfortable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kkamang.core import GameCore
from kkamang.levels import load_level
from kkamang.realtime import RealtimeGameCore, PHYSICS_HZ, REALTIME_OBSERVATION_VERSION


def replay_game_record(record):
    """Replay an exported desktop record, including turns between world ticks."""
    config = load_level(record['stage_id'])
    realtime = record.get('control_mode') == 'realtime'
    if realtime:
        if (record['replay_version'] != '2.0' or record['physics_hz'] != PHYSICS_HZ
                or record['observation_version'] != REALTIME_OBSERVATION_VERSION):
            raise ValueError('incompatible realtime replay contract')
    elif record.get('control_mode', 'grid') != 'grid' or record['replay_version'] != '1.0':
        raise ValueError('incompatible grid replay contract')
    core_class = RealtimeGameCore if realtime else GameCore
    game = core_class(config, mode=record['mode'], seed=record['seed'])
    if game.config_hash != record['config_hash']:
        raise ValueError('replay level configuration hash mismatch')
    game.reset(seed=record['seed'], phase_overrides=record['initial_phase'])
    for action in record['actions']:
        game.step(action)
    expected = record['outcome']
    if expected and expected['status'] == 'ABORTED':
        game.abort(expected['reason'])
    snapshot = game.snapshot()
    if snapshot['terminal'] != record['terminal'] or snapshot['outcome'] != expected:
        raise AssertionError('replay terminal outcome mismatch')
    return snapshot


def phase_overrides(config, offsets):
    """Advance each authored ghost clock, independently of the game clock."""
    result = {}
    if len(config['ghosts']) != len(offsets):
        raise AssertionError('Incomplete reference phase offsets')
    for ghost, offset in zip(config['ghosts'], offsets):
        index = ghost['initial_index']
        countdown = ghost['initial_countdown']
        # Deliberately use clock replay, not the original solver's formula.
        for _ in range(offset):
            countdown -= 1
            if countdown == 0:
                index = (index + ghost['direction']) % len(ghost['route'])
                countdown = ghost['ticks_per_move']
        result[ghost['id']] = {'index': index, 'countdown': countdown}
    return result


def compare_trace(snapshot, reference):
    """Compare recorded regression state fields on representative runs."""
    actual = {
        'tick': snapshot['tick'],
        'player': snapshot['position'],
        'carried': snapshot['carried'],
        'quest_collected':snapshot['quest']['collected'],
        'collected_ids': sorted(i['id'] for i in snapshot['items'] if not i['active']),
        'opened_shortcut_ids': sorted(s['id'] for s in snapshot['shortcuts'] if s['opened']),
        'ghosts': [{'id': g['id'], 'position': g['position'],
                    'current_index': g['route_index'], 'countdown': g['countdown']}
                   for g in snapshot['ghosts']],
    }
    if 'magnet' in reference:
        actual['magnet'] = snapshot.get('magnet')
    for name in ('invincibility', 'chaser'):
        if name in reference:
            actual[name] = snapshot.get(name)
    expected = {key: reference[key] for key in actual}
    if actual != expected:
        raise AssertionError(f'Trace mismatch at tick {snapshot["tick"]}: {actual!r} != {expected!r}')


def replay(route):
    contact_policy = route.get('contact_policy', 'no_contact')
    if contact_policy not in ('no_contact', 'item_invincible'):
        raise AssertionError(f'Unknown contact policy: {contact_policy}')
    config = load_level(route['level_id'])
    game = GameCore(config, mode='basic', seed=0)
    state = game.reset(seed=0, phase_overrides=phase_overrides(config, route['phase_offsets']))
    trace = route.get('trace')
    if trace:
        compare_trace(state, trace[0])
    if len(route['actions']) != route['ticks']:
        raise AssertionError('Reference action/tick count mismatch')
    contact_events = 0
    for tick, action in enumerate(route['actions'], 1):
        result = game.step(action)
        contact_events += sum(event['type'] == 'contact' for event in result.events)
        for event in result.events:
            if event['type'] == 'contact' and not (
                    contact_policy == 'item_invincible' and event.get('item_invincible', False)):
                raise AssertionError(f'Forbidden contact at tick {tick}')
        if result.snapshot['courage'] != 3:
            raise AssertionError(f'Unexpected damage at tick {tick}')
        if result.terminated != (tick == route['ticks']):
            raise AssertionError(f'Unexpected terminal state at tick {tick}')
        if trace:
            compare_trace(result.snapshot, trace[tick])
    state = game.snapshot()
    outcome = state['outcome']
    if outcome['status'] != 'COMPLETE' or outcome['reason'] != 'returned':
        raise AssertionError(f'Unexpected outcome: {outcome}')
    if outcome['banked'] != route['banked'] or outcome['banked'] < route['target']:
        raise AssertionError(f'Banked value mismatch: {outcome}')
    if outcome['ticks_used'] != route['ticks'] or outcome['damage_count'] != 0:
        raise AssertionError(f'Tick/damage mismatch: {outcome}')
    if contact_events and contact_policy == 'no_contact':
        raise AssertionError(f'No-contact reference had {contact_events} contacts')
    opened = [s['opened'] for s in state['shortcuts']]
    if route['constraint'] == 'all_shortcuts_closed' and any(opened):
        raise AssertionError('Closed-shortcut route opened a shortcut')
    if route['constraint'] == 'must_open_all_shortcuts' and not all(opened):
        raise AssertionError('Required shortcut was not opened')
    return outcome


def validate(representative=False):
    fixture_path = ROOT / 'validation' / 'reference_routes.json'
    fixture = json.loads(fixture_path.read_text(encoding='utf-8'))
    hashes = []
    for expected in fixture['level_hashes']:
        file_path = ROOT / 'data' / expected['file']
        actual = hashlib.sha256(
            file_path.read_text(encoding='utf-8').encode('utf-8')
        ).hexdigest()
        hashes.append({'stage_id': expected['id'], 'expected_sha256': expected['sha256'],
                       'actual_sha256': actual, 'passed': actual == expected['sha256']})
    routes = fixture['routes']
    if representative:
        routes = [route for route in routes if 'trace' in route]
    failures = []
    by_stage = {}
    start = time.perf_counter()
    for index, route in enumerate(routes):
        stats = by_stage.setdefault(route['level_id'], {'routes': 0, 'passed': 0, 'trace_steps': 0})
        stats['routes'] += 1
        try:
            replay(route)
        except Exception as exc:
            failures.append({'index': index, 'stage_id': route['level_id'],
                             'target': route['target'], 'phase_offsets': route['phase_offsets'],
                             'constraint': route['constraint'], 'error': f'{type(exc).__name__}: {exc}'})
        else:
            stats['passed'] += 1
            stats['trace_steps'] += len(route.get('trace', []))
    source_hashes = {}
    for name in ('core.py', 'levels.py'):
        source_hashes[name] = hashlib.sha256((ROOT / 'kkamang' / name).read_bytes()).hexdigest()
    return {
        'scope': 'Executed current GameCore against recorded v0.2 witness action sequences; not independent correctness proof.',
        'limitations': ['No PPO or other learning was performed.',
                       'No claim about human difficulty, real-time input, UI or Windows execution.',
                       'All tested phase combinations use the authored routes and directions.'],
        'reference_fixture_sha256': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        'original_results_sha256': fixture['source_results_sha256'],
        'implementation_sha256': source_hashes,
        'representative_only': representative,
        'route_count': len(routes), 'passed': len(routes) - len(failures),
        'phase_configurations': len({(r['level_id'], tuple(r['phase_offsets'])) for r in routes}),
        'representative_trace_routes': sum('trace' in r for r in routes),
        'representative_trace_states': sum(len(r.get('trace', [])) for r in routes),
        'level_hashes': hashes, 'by_stage': by_stage, 'failures': failures,
        'elapsed_seconds': round(time.perf_counter() - start, 3),
        'all_passed': not failures and all(h['passed'] for h in hashes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--representative', action='store_true', help='Only routes with per-tick regression traces')
    parser.add_argument('--record', type=Path, help='Replay an exported human JSON record instead of grid witnesses')
    parser.add_argument('--output', type=Path, default=ROOT / 'validation' / 'runtime_replay_results.json')
    args = parser.parse_args()
    if args.record:
        record = json.loads(args.record.read_text(encoding='utf-8'))
        snapshot = replay_game_record(record)
        print(f"Exported {record.get('control_mode', 'grid')} replay passed: "
              f"{len(record['actions'])} actions, {snapshot['tick']} world ticks")
        return 0
    report = validate(args.representative)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"GameCore replay: {report['passed']}/{report['route_count']} passed; "
          f"{report['phase_configurations']} phases; "
          f"{sum(h['passed'] for h in report['level_hashes'])}/12 current level hashes match")
    if report['failures']:
        print(json.dumps(report['failures'][:3], ensure_ascii=False, indent=2))
    print(f'Report: {args.output}')
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
