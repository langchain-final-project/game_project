"""Reproducible random-vs-planner smoke evaluation, NOT an RL training result.

Run: python tools/evaluate.py --episodes 10 --output reports/evaluation
Both policies use the same stage/episode seed. A timed greedy planner uses only
the fully visible snapshot: known routes, current phase, map, items and time.
It is a useful sanity baseline, not an optimal solver or a learned policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import json
import math
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gymnasium.utils.env_checker import check_env

from kkamang.env import MansionEnv, OBSERVATION_VERSION

MOVES = ((0, 0, -1), (1, 0, 1), (2, -1, 0), (3, 1, 0), (5, 0, 0))


def ghost_position(ghost: dict, ticks: int) -> tuple[int, int]:
    """Predict from visible deterministic route/countdown, never a hidden state."""
    moves = 0 if ticks < ghost['countdown'] else 1 + (
        ticks - ghost['countdown']) // ghost['ticks_per_move']
    index = (ghost['route_index'] + moves * ghost['direction']) % len(ghost['route'])
    return tuple(ghost['route'][index])


def safe_transition(snapshot: dict, old: tuple, new: tuple, tick: int) -> bool:
    for ghost in snapshot['ghosts']:
        before = ghost_position(ghost, tick)
        after = ghost_position(ghost, tick + 1)
        if new == after or (old == after and new == before):
            return False
    return True


def safe_path(snapshot: dict, goal: tuple[int, int], *, offset: int = 0,
              start: tuple[int, int] | None = None, horizon: int | None = None) -> list[int] | None:
    """BFS over position/known ghost phase, including one safe E at destination.

    Walls and currently closed gates are impassable. The planner conservatively
    avoids contact even while protected. No assumption of invulnerability is
    used to claim a safe route. Horizon includes the final interact tick.
    """
    start = tuple(snapshot['position']) if start is None else start
    goal = tuple(goal)
    horizon = snapshot['remaining_ticks'] - offset if horizon is None else horizon
    if horizon < 1:
        return None
    period = math.lcm(*(len(g['route']) * g['ticks_per_move'] for g in snapshot['ghosts']))
    closed = {tuple(g['gate']) for g in snapshot['shortcuts'] if not g['opened']}
    queue = deque([(start, 0)])
    root = (start, offset % period)
    parents: dict[tuple, tuple[tuple, int] | None] = {root: None}
    while queue:
        position, elapsed = queue.popleft()
        node = (position, (offset + elapsed) % period)
        if position == goal and elapsed < horizon and safe_transition(
                snapshot, position, position, offset + elapsed):
            actions = [4]
            while parents[node] is not None:
                previous, action = parents[node]
                actions.append(action)
                node = previous
            return list(reversed(actions))
        if elapsed + 1 >= horizon:
            continue
        for action, dx, dy in MOVES:
            new = (position[0] + dx, position[1] + dy)
            if not (0 <= new[0] < snapshot['width'] and 0 <= new[1] < snapshot['height']):
                continue
            if snapshot['grid'][new[1]][new[0]] == '#' or new in closed:
                continue
            if not safe_transition(snapshot, position, new, offset + elapsed):
                continue
            next_node = (new, (offset + elapsed + 1) % period)
            if next_node not in parents:
                parents[next_node] = (node, action)
                queue.append((new, elapsed + 1))
    return None


class TimedGreedyPolicy:
    """Recover the mandatory object, then collect heat with a safe return reserve.

    Replan after every collection. Prefer value per travel cost; leave a two-tick
    slack for the return. Does not search all item orders or choose shortcuts,
    so its score can legitimately fall below an optimal or trained policy.
    """
    def __init__(self):
        self.actions = deque()

    def action(self, snapshot: dict) -> int:
        if self.actions:
            return self.actions.popleft()
        if not snapshot['quest']['collected']:
            quest_path = safe_path(snapshot,tuple(snapshot['quest']['position']))
            if quest_path:
                self.actions.extend(quest_path)
                return self.actions.popleft()
        exit_pos = tuple(snapshot['exit'])
        return_path = safe_path(snapshot, exit_pos)
        candidates = []
        if snapshot['courage'] > 0:
            for item in snapshot['items']:
                if not item['active']:
                    continue
                target = tuple(item['position'])
                outward = safe_path(snapshot, target)
                if outward is None:
                    continue
                home = safe_path(snapshot, exit_pos, start=target, offset=len(outward))
                if home is None or len(outward) + len(home) + 2 > snapshot['remaining_ticks']:
                    continue
                # Completing the four-point task receives priority over extras.
                useful = min(item['value'], max(0, snapshot['target'] - snapshot['carried']))
                priority = (item['value'] + useful) / len(outward)
                candidates.append((-priority, len(outward), item['id'], outward))
        if candidates:
            self.actions.extend(min(candidates)[3])
        elif return_path:
            self.actions.extend(return_path)
        else:
            # No safe route was found; do not invent successful banked points.
            return 5
        return self.actions.popleft()


def evaluate(episodes: int, stages: list[int], seed: int, mode: str,
             randomize_phase: bool) -> tuple[list[dict], dict]:
    rows = []
    errors = []
    for stage in stages:
        for policy_name in ('random', 'timed_greedy'):
            env = MansionEnv(stage, mode=mode, randomize_phase=randomize_phase)
            try:
                check_env(env, skip_render_check=True)
                for episode in range(episodes):
                    episode_seed = seed + stage * 100000 + episode
                    _, info = env.reset(seed=episode_seed)
                    random = np.random.default_rng(episode_seed)
                    planner = TimedGreedyPolicy()
                    cumulative_reward = 0.0
                    terminated = truncated = False
                    while not (terminated or truncated):
                        action = (int(random.integers(6)) if policy_name == 'random'
                                  else planner.action(info['snapshot']))
                        obs, reward, terminated, truncated, info = env.step(action)
                        if not env.observation_space.contains(obs):
                            raise AssertionError('rollout escaped observation space')
                        cumulative_reward += reward
                    outcome = info['outcome'] or {}
                    rows.append({
                        'stage': stage, 'policy': policy_name, 'episode': episode,
                        'seed': episode_seed, 'mode': mode, 'randomize_phase': randomize_phase,
                        'status': outcome.get('status', 'TRUNCATED'),
                        'reason': outcome.get('reason', info.get('truncation_reason', 'unknown')),
                        'complete': outcome.get('status') == 'COMPLETE',
                        'returned': outcome.get('status') in ('COMPLETE', 'RETURN_LOW'),
                        'banked': outcome.get('banked', 0),
                        'carried': info['snapshot']['carried'],
                        'damage_count': outcome.get('damage_count', 0),
                        'ticks_used': info['episode_steps'],
                        'remaining_ticks': info['snapshot']['remaining_ticks'],
                        'reward': cumulative_reward, 'terminated': terminated,
                        'truncated': truncated, 'config_hash': info['config_hash'],
                        'observation_version': info['observation_version'],
                    })
            except Exception as exc:
                errors.append({'stage': stage, 'policy': policy_name, 'error': repr(exc),
                               'traceback': traceback.format_exc()})
            finally:
                env.close()
    summaries = []
    for stage in stages:
        for policy in ('random', 'timed_greedy'):
            subset = [r for r in rows if r['stage'] == stage and r['policy'] == policy]
            if not subset:
                continue
            n = len(subset)
            summaries.append({'stage': stage, 'policy': policy, 'episodes': n,
                'complete_rate': sum(r['complete'] for r in subset) / n,
                'return_rate': sum(r['returned'] for r in subset) / n,
                'mean_banked': sum(r['banked'] for r in subset) / n,
                'mean_damage': sum(r['damage_count'] for r in subset) / n,
                'mean_ticks': sum(r['ticks_used'] for r in subset) / n,
                'statuses': dict(Counter(r['status'] for r in subset))})
    report = {'evaluation_kind': 'headless_smoke_baselines_not_trained_RL',
              'observation_version': OBSERVATION_VERSION, 'base_seed': seed,
              'episodes_per_stage_policy': episodes, 'mode': mode,
              'randomize_phase': randomize_phase,
              'notes': ['No policy was trained.', 'Failed or externally truncated runs bank zero.',
                        'Timed greedy uses public map, route and phase data; it is not optimal.',
                        'Same seed and fixed map are reproducibility checks, not unseen-map generalization.',
                        'Training must reserve separate validation/test seeds; compare fixed protocol and budgets.'],
              'summary': summaries, 'errors': errors}
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--stages', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--mode', choices=('basic', 'relaxed'), default='basic')
    parser.add_argument('--fixed-phase', action='store_true',
                        help='Keep designed ghost starting phases instead of seeded variations')
    parser.add_argument('--output', type=Path, default=Path('reports/evaluation'))
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0 or any(s < 1 or s > 12 for s in args.stages):
        parser.error('episodes >= 1, seed >= 0, and stages in 1..12 are required')
    rows, report = evaluate(args.episodes, args.stages, args.seed, args.mode, not args.fixed_phase)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / 'episodes.csv'
    if rows:
        with csv_path.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / 'summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'episodes': len(rows), 'errors': len(report['errors']),
                      'output': str(args.output)}, ensure_ascii=False))
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
