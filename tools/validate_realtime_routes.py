"""Generate and replay current free-movement witnesses in local-only outputs.

No path found means this bounded heuristic failed, not that the level is
impossible. A minimum-warmth target is not a claim of an exact or optimal score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kkamang.env import RealtimeMansionEnv
from kkamang.realtime import PHYSICS_HZ, REALTIME_OBSERVATION_VERSION
from tools.realtime_policy import RealtimeGreedyPolicy, POLICY_VERSION
from tools.replay_validation import replay_game_record, snapshot_digest


def validate_routes(stages, targets, *, seed=0, mode='basic', max_search_nodes=128,
                    contact_policy='no_contact', randomize_phase=False,
                    output=None, progress=None):
    if (not stages or not targets or len(set(stages)) != len(stages)
            or len(set(targets)) != len(targets)
            or any(type(x) is not int or not 1 <= x <= 12 for x in stages)
            or any(type(x) is not int or not 4 <= x <= 14 for x in targets)
            or type(seed) is not int or seed < 0
            or type(max_search_nodes) is not int or max_search_nodes < 1
            or contact_policy not in ('no_contact', 'item_invincible')):
        raise ValueError('invalid stages, targets, seed, search budget or contact policy')
    output = Path(output) if output is not None else None
    if output is not None:
        (output / 'replays').mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for stage in stages:
        for target in targets:
            env = RealtimeMansionEnv(stage, mode=mode, randomize_phase=randomize_phase)
            try:
                _, info = env.reset(seed=seed + stage * 100000)
                policy = RealtimeGreedyPolicy(env.core, target_warmth=target,
                    max_search_nodes=max_search_nodes,
                    allow_item_invincibility=contact_policy == 'item_invincible')
                done = False
                contacts = forbidden_contacts = 0
                while not done:
                    obs, _, terminated, truncated, info = env.step(policy.action(info['snapshot']))
                    if not env.observation_space.contains(obs):
                        raise AssertionError('realtime route escaped observation contract')
                    done = terminated or truncated
                    for event in info['events']:
                        if event['type'] == 'contact':
                            contacts += 1
                            forbidden_contacts += int(contact_policy == 'no_contact'
                                or not event.get('item_invincible', False))
                record = env.core.replay_record()
                record['final_snapshot_sha256'] = snapshot_digest(info['snapshot'])
                # A second fresh core replays the exported 60Hz action sequence.
                replayed = replay_game_record(record)
                if replayed != info['snapshot']:
                    raise AssertionError('full realtime replay snapshot mismatch')
                outcome = info['outcome']
                passed = (outcome['status'] == 'COMPLETE' and outcome['quest_collected']
                          and outcome['banked'] >= target and outcome['damage_count'] == 0
                          and forbidden_contacts == 0)
                row = dict(stage=stage, target_warmth=target, passed=passed,
                    status=outcome['status'], reason=outcome['reason'], banked=outcome['banked'],
                    damage_count=outcome['damage_count'], contacts=contacts,
                    forbidden_contacts=forbidden_contacts, physics_frames=info['episode_steps'],
                    ticks_used=info['snapshot']['tick'], remaining_ticks=info['snapshot']['remaining_ticks'],
                    simulated_seconds=info['episode_steps']/PHYSICS_HZ,
                    score_total=outcome['score']['total'], stars=outcome['score']['stars'],
                    config_hash=info['config_hash'], initial_phase=record['initial_phase'],
                    search_counts=dict(policy.search_counts), replay_verified=True)
                rows.append(row)
                if output is not None:
                    record['route_validation'] = dict(target_warmth=target,
                        contact_policy=contact_policy, passed=passed, policy_version=POLICY_VERSION)
                    path = output / 'replays' / f'stage_{stage:02d}_target_{target:02d}.json'
                    path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                if progress is not None:
                    progress(row)
            except Exception as exc:
                errors.append(dict(stage=stage, target_warmth=target, error=repr(exc)))
            finally:
                env.close()
    result = dict(validation_kind='frame_verified_realtime_witnesses_not_optimal_paths',
        control_mode='realtime', observation_version=REALTIME_OBSERVATION_VERSION,
        physics_hz=PHYSICS_HZ, policy_version=POLICY_VERSION, contact_policy=contact_policy,
        seed=seed, mode=mode, randomize_phase=randomize_phase, max_search_nodes=max_search_nodes,
        expected_routes=len(stages)*len(targets), passed=sum(row['passed'] for row in rows),
        all_passed=not errors and len(rows) == len(stages)*len(targets)
                   and all(row['passed'] for row in rows),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('tools/realtime_policy.py', 'tools/validate_realtime_routes.py',
             'kkamang/core.py', 'kkamang/realtime.py', 'kkamang/env.py')},
        notes=['Every physics frame is simulated; no grid witness is reused.',
               'No policy was trained; no claim of optimality or human difficulty.',
               'A failed witness search is not proof of level impossibility.',
               'The center-to-center baseline does not search all possible mid-cell turns.',
               'Complete replay snapshots are verified in a second fresh core.'],
        routes=rows, errors=errors)
    if output is not None:
        (output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--targets', type=int, nargs='+', default=[4])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mode', choices=('basic', 'relaxed'), default='basic')
    parser.add_argument('--contact-policy', choices=('no_contact', 'item_invincible'), default='no_contact')
    parser.add_argument('--randomize-phase', action='store_true')
    parser.add_argument('--max-search-nodes', type=int, default=128)
    parser.add_argument('--output', type=Path, default=Path('reports/local/realtime_routes'))
    args = parser.parse_args()
    try:
        report = validate_routes(args.stages, args.targets, seed=args.seed, mode=args.mode,
            max_search_nodes=args.max_search_nodes, contact_policy=args.contact_policy,
            randomize_phase=args.randomize_phase, output=args.output,
            progress=lambda row: print(f"stage {row['stage']:02d} target {row['target_warmth']}: "
                f"{'PASS' if row['passed'] else 'NO WITNESS'} / banked {row['banked']} / "
                f"damage {row['damage_count']} / {row['physics_frames']} frames", flush=True))
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps({'passed': report['passed'], 'expected': report['expected_routes'],
                      'errors': report['errors'], 'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
