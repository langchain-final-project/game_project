"""Reproducible random-vs-planner smoke evaluation, NOT an RL training result.

Run: python tools/evaluate.py --control-mode realtime --episodes 10
Both policies use the same stage/episode seed. A timed greedy planner uses only
the fully visible snapshot: known routes, current phase, map, items and time.
It is a useful sanity baseline, not an optimal solver or a learned policy.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from copy import deepcopy
import csv
import heapq
import hashlib
from itertools import count
import json
import math
from pathlib import Path
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gymnasium.utils.env_checker import check_env

from kkamang.env import MansionEnv, RealtimeMansionEnv, OBSERVATION_VERSION
from kkamang.realtime import PHYSICS_HZ, FRAMES_PER_TICK, REALTIME_OBSERVATION_VERSION
from tools.realtime_policy import RealtimeGreedyPolicy, POLICY_VERSION as REALTIME_POLICY_VERSION
from tools.replay_validation import snapshot_digest

MOVES = ((0, 0, -1), (1, 0, 1), (2, -1, 0), (3, 1, 0), (5, 0, 0))
POLICY_VERSION = 'unified_core_v1'
DEFAULT_SEARCH_NODES = 1500


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
              start: tuple[int, int] | None = None, horizon: int | None = None,
              interact: bool = True, forbid_rich: bool = False) -> list[int] | None:
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
    transient = max((g['countdown'] for g in snapshot['ghosts']), default=0)
    magnet = snapshot.get('magnet')
    initial_magnet = magnet['remaining_ticks'] if magnet else 0
    used = magnet['collected'] if magnet else True
    rich = [tuple(item['position']) for item in snapshot['items']
            if item['active'] and item['value'] == 3] if forbid_rich else []

    def phase(elapsed):
        tick = offset + elapsed
        return tick if tick < transient else transient + (tick - transient) % period

    def effect(point, remaining, collected):
        if magnet and not collected and point == tuple(magnet['position']):
            remaining, collected = magnet['duration_ticks'], True
        # 자석은 3점 온기를 끌어오지 않으므로 직접 밟는 칸만 피합니다.
        if point in rich:
            return None
        return max(0, remaining - 1), collected

    closed = {tuple(g['gate']) for g in snapshot['shortcuts'] if not g['opened']}
    root = (start, phase(0), max(0, initial_magnet - offset), used)
    queue = deque([(root, 0)])
    parents = {root: None}
    while queue:
        node, elapsed = queue.popleft()
        position, _, remaining_magnet, collected_magnet = node
        if position == goal and (not interact or (
                elapsed < horizon and safe_transition(snapshot, position, position, offset + elapsed)
                and effect(position, remaining_magnet, collected_magnet) is not None)):
            actions = [4] if interact else []
            while parents[node] is not None:
                previous, action = parents[node]
                actions.append(action)
                node = previous
            return list(reversed(actions))
        if elapsed + 1 > horizon - int(interact):
            continue
        for action, dx, dy in MOVES:
            new = (position[0] + dx, position[1] + dy)
            if not (0 <= new[0] < snapshot['width'] and 0 <= new[1] < snapshot['height']):
                continue
            if snapshot['grid'][new[1]][new[0]] == '#' or new in closed:
                continue
            if not safe_transition(snapshot, position, new, offset + elapsed):
                continue
            next_effect = effect(new, remaining_magnet, collected_magnet)
            if next_effect is None:
                continue
            next_node = (new, phase(elapsed + 1), *next_effect)
            if next_node not in parents:
                parents[next_node] = (node, action)
                queue.append((next_node, elapsed + 1))
    return None


def collection_positions(snapshot: dict, item: dict) -> list[tuple[int, int]]:
    """Only one-point heat can be collected from a neighboring tile by a magnet."""
    x, y = item['position']
    magnet = snapshot.get('magnet')
    radius = (1 if magnet and magnet['remaining_ticks'] > 0 and item['value'] == 1
              else 0)
    return [(nx, ny) for ny in range(y - radius, y + radius + 1)
            for nx in range(x - radius, x + radius + 1)
            if 0 <= nx < snapshot['width'] and 0 <= ny < snapshot['height']
            and snapshot['grid'][ny][nx] == '.']


def acceptable_step(result, *, forbid_rich=False, allow_item_invincibility=False):
    """Only an active item permits contact; damage protection never does."""
    if any(event['type'] == 'damage' or (
            event['type'] == 'contact' and not (
                allow_item_invincibility and event.get('item_invincible', False)))
           for event in result.events):
        return False
    return not (forbid_rich and any(not item['active'] and item['value'] == 3
                                   for item in result.snapshot['items']))


def project_path(core, path, *, forbid_rich=False, allow_item_invincibility=False):
    """Verify candidates in the actual core, including both automatic items."""
    test = deepcopy(core)
    result = None
    for index, action in enumerate(path):
        result = test.step(action)
        if not acceptable_step(result, forbid_rich=forbid_rich,
                               allow_item_invincibility=allow_item_invincibility):
            return None
        if result.terminated and (index != len(path) - 1
                                  or result.outcome['status'] != 'COMPLETE'):
            return None
    return (test, result) if result is not None else None


def simulation_key(s):
    chaser = s.get('chaser')
    return (tuple(s['position']), tuple(i['active'] for i in s['items']),
            s['quest']['collected'], tuple(g['opened'] for g in s['shortcuts']),
            tuple((g['route_index'], g['countdown']) for g in s['ghosts']),
            None if chaser is None else (chaser['active'], tuple(chaser['position']), chaser['countdown']),
            tuple(None if s.get(name) is None else
                  (s[name]['collected'], s[name]['remaining_ticks'])
                  for name in ('magnet', 'invincibility')))


def find_path(core, goal, *, interact=True, forbid_rich=False,
              allow_item_invincibility=False, max_nodes=DEFAULT_SEARCH_NODES,
              diagnostics=None):
    """Verified candidate walks, then bounded A*, on every stage.

    Each branch simulates pickup, expiry, pursuit and contact stuns. A node limit
    returns None (no witness found), not a proof that a route is impossible.
    Optional diagnostics distinguish the search limit from an exhausted search
    and a static distance that exceeds the remaining time. None is not a proof
    that a different target order could not complete the stage.
    """
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')

    def finish(path, status, expanded=0):
        if diagnostics is not None:
            diagnostics.update(status=status, expanded_nodes=expanded)
        return path

    snapshot = core.snapshot()
    if 'motion' in snapshot:
        raise ValueError('find_path supports grid movement; realtime needs a frame-based policy')
    if snapshot['terminal']:
        return finish(None, 'terminal')
    goal = tuple(goal)
    closed = {tuple(g['gate']) for g in snapshot['shortcuts'] if not g['opened']}
    if (len(goal) != 2 or not 0 <= goal[0] < snapshot['width']
            or not 0 <= goal[1] < snapshot['height']
            or snapshot['grid'][goal[1]][goal[0]] != '.' or goal in closed):
        return finish(None, 'unreachable')
    distances = {goal: 0}
    queue = deque([goal])
    while queue:
        x, y = queue.popleft()
        for _, dx, dy in MOVES[:4]:
            point = (x + dx, y + dy)
            if (point not in distances and point not in closed
                    and 0 <= point[0] < snapshot['width'] and 0 <= point[1] < snapshot['height']
                    and snapshot['grid'][point[1]][point[0]] == '.'):
                distances[point] = distances[(x, y)] + 1
                queue.append(point)
    if tuple(snapshot['position']) not in distances:
        return finish(None, 'unreachable')
    if distances[tuple(snapshot['position'])] + int(interact) > snapshot['remaining_ticks']:
        return finish(None, 'insufficient_time')
    # Most legs have a valid shortest walk. Verify it before expanding a tree.
    point = tuple(snapshot['position'])
    direct = []
    while point != goal:
        for action, dx, dy in MOVES[:4]:
            neighbor = (point[0] + dx, point[1] + dy)
            if distances.get(neighbor) == distances[point] - 1:
                direct.append(action)
                point = neighbor
                break
    direct += [4] if interact else []
    if direct and project_path(core, direct, forbid_rich=forbid_rich,
                               allow_item_invincibility=allow_item_invincibility) is not None:
        return finish(direct, 'found')
    # Patrol BFS is only a cheap candidate generator. It cannot authorise a
    # route: pursuit and both items are always verified with actual transitions.
    hint = safe_path(snapshot, goal, interact=interact, forbid_rich=forbid_rich)
    if hint and project_path(core, hint, forbid_rich=forbid_rich,
                             allow_item_invincibility=allow_item_invincibility) is not None:
        return finish(hint, 'found')
    serial = count()
    root = simulation_key(snapshot)
    best = {root: 0}
    frontier = [(distances[tuple(snapshot['position'])], 0, next(serial), core, [])]
    expanded = 0
    for _ in range(max_nodes):
        if not frontier:
            break
        _, _, _, current, path = heapq.heappop(frontier)
        cost = len(path)
        state = current.snapshot()
        if cost != best.get(simulation_key(state)):
            continue
        expanded += 1
        if tuple(state['position']) == goal:
            if not interact:
                return finish(path, 'found', expanded)
            checked = project_path(current, [4], forbid_rich=forbid_rich,
                                   allow_item_invincibility=allow_item_invincibility)
            if checked is not None:
                return finish(path + [4], 'found', expanded)
        if state['remaining_ticks'] < distances[tuple(state['position'])] + int(interact):
            continue
        for action, dx, dy in MOVES:
            point = (state['position'][0] + dx, state['position'][1] + dy)
            if point not in distances:
                continue
            # Geometry and RNG are read-only in step(). These search clones are
            # never returned or reset; mutable state/actions still get copied.
            memo = {id(current._config): current._config,
                    id(current._rng): current._rng,
                    id(current._state['grid']): current._state['grid']}
            memo.update((id(g['route']), g['route']) for g in current._state['ghosts'])
            candidate = deepcopy(current, memo)
            result = candidate.step(action)
            if result.terminated or not acceptable_step(
                    result, forbid_rich=forbid_rich,
                    allow_item_invincibility=allow_item_invincibility):
                continue
            if result.snapshot['remaining_ticks'] < distances[point] + int(interact):
                continue
            key = simulation_key(result.snapshot)
            if cost + 1 >= best.get(key, math.inf):
                continue
            best[key] = cost + 1
            heapq.heappush(frontier, (cost + 1 + distances[point], distances[point],
                                     next(serial), candidate, path + [action]))
    return finish(None, 'node_limit' if frontier else 'exhausted', expanded)


class TimedGreedyPolicy:
    """Recover the mandatory object, then collect heat with a safe return reserve.

    Replan after every collection. Prefer value per travel cost; leave a two-tick
    slack for the return. Does not search all item orders or choose shortcuts,
    so its score can legitimately fall below an optimal or trained policy.
    """
    def __init__(self, core, *, max_search_nodes=DEFAULT_SEARCH_NODES):
        if core is None:
            raise ValueError('TimedGreedyPolicy requires the live GameCore')
        if 'motion' in core.snapshot():
            raise ValueError('TimedGreedyPolicy supports grid movement, not realtime')
        if type(max_search_nodes) is not int or max_search_nodes < 1:
            raise ValueError('max_search_nodes must be a positive integer')
        self.actions = deque()
        self.core = core
        self.max_search_nodes = max_search_nodes
        self.search_counts = Counter()
        self.expanded_nodes = 0
        self.fallback_counts = Counter()

    def _path(self, core, point, *, interact=True):
        diagnostics = {}
        path = find_path(core, point, interact=interact, allow_item_invincibility=True,
                         max_nodes=self.max_search_nodes, diagnostics=diagnostics)
        self.search_counts[diagnostics['status']] += 1
        self.expanded_nodes += diagnostics['expanded_nodes']
        return path

    def _fallback(self):
        # Avoid a blind WAIT when the chaser would hit on this very tick.
        for action in (5, 0, 1, 2, 3):
            if project_path(self.core, [action], allow_item_invincibility=True) is not None:
                self.fallback_counts['no_verified_plan'] += 1
                return [action]
        self.fallback_counts['no_safe_action'] += 1
        return [5]

    def _planned_actions(self, snapshot):
        """Rank real collection gains, leaving a verified route back home."""
        exit_pos = tuple(snapshot['exit'])
        if snapshot['quest']['collected'] and snapshot['carried'] == 14:
            home = self._path(self.core, exit_pos)
            return home if home and project_path(self.core, home, allow_item_invincibility=True) is not None else self._fallback()
        goals = {}
        if not snapshot['quest']['collected']:
            goals[(tuple(snapshot['quest']['position']), True)] = 'quest'
        for name in ('magnet', 'invincibility'):
            item = snapshot.get(name)
            if item and not item['collected']:
                goals[(tuple(item['position']), False)] = name
        for item in snapshot['items']:
            if item['active']:
                for point in collection_positions(snapshot, item):
                    goals.setdefault((point, False), 'heat')
        candidates = []
        for (point, interact), kind in goals.items():
            outward = self._path(self.core, point, interact=interact)
            if not outward:
                continue
            checked = project_path(self.core, outward, allow_item_invincibility=True)
            if checked is None or checked[1].terminated:
                continue
            projected, result = checked
            after = result.snapshot
            gain = after['carried'] - snapshot['carried']
            got_quest = after['quest']['collected'] and not snapshot['quest']['collected']
            got_item = sum(bool(after.get(name) and after[name]['collected']
                                and not snapshot[name]['collected'])
                           for name in ('magnet', 'invincibility'))
            if not (gain or got_quest or got_item):
                continue
            home = self._path(projected, exit_pos)
            if not home or len(home) + 2 > after['remaining_ticks']:
                continue
            if project_path(projected, home, allow_item_invincibility=True) is None:
                continue
            # Acquiring the quest and a useful timed effect has value beyond
            # heat gained on the outward route itself; this remains a heuristic.
            utility = gain + min(gain, max(0, snapshot['target'] - snapshot['carried']))
            utility += 8 * got_quest + 5 * got_item
            candidates.append((-utility / len(outward), len(outward), point, outward))
        if candidates:
            return min(candidates)[3]
        home = self._path(self.core, exit_pos)
        if snapshot['quest']['collected'] and snapshot['carried'] >= snapshot['target'] and home:
            if project_path(self.core, home, allow_item_invincibility=True) is not None:
                return home
        return self._fallback()

    def action(self, snapshot: dict) -> int:
        if self.actions:
            return self.actions.popleft()
        self.actions.extend(self._planned_actions(snapshot))
        return self.actions.popleft()


def evaluate(episodes: int, stages: list[int], seed: int, mode: str,
             randomize_phase: bool, *, max_search_nodes=DEFAULT_SEARCH_NODES,
             progress=None, control_mode='grid', target_warmth=14,
             max_episode_steps=None, record_dir=None) -> tuple[list[dict], dict]:
    if control_mode not in ('grid', 'realtime'):
        raise ValueError('control_mode must be grid or realtime')
    if (type(episodes) is not int or episodes < 1 or not stages or len(set(stages)) != len(stages)
            or any(type(stage) is not int or not 1 <= stage <= 12 for stage in stages)
            or type(seed) is not int or seed < 0
            or type(max_search_nodes) is not int or max_search_nodes < 1
            or type(target_warmth) is not int or not 4 <= target_warmth <= 14):
        raise ValueError('invalid episode, stage, seed, target or search budget')
    if control_mode == 'grid' and target_warmth != 14:
        raise ValueError('target_warmth is configurable only in realtime mode')
    realtime = control_mode == 'realtime'
    env_class = RealtimeMansionEnv if realtime else MansionEnv
    if record_dir is not None:
        record_dir = Path(record_dir)
        record_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []
    for stage in stages:
        for policy_name in ('random', 'timed_greedy'):
            group_start = len(rows)
            env = env_class(stage, mode=mode, randomize_phase=randomize_phase,
                            max_episode_steps=max_episode_steps)
            try:
                # Gym's checker deliberately rejects a first-step cutoff. Check
                # the game interface before applying the external rollout limit.
                limit = env.max_episode_steps
                env.max_episode_steps = None
                try:
                    check_env(env, skip_render_check=True)
                finally:
                    env.max_episode_steps = limit
                for episode in range(episodes):
                    episode_seed = seed + stage * 100000 + episode
                    _, info = env.reset(seed=episode_seed)
                    random = np.random.default_rng(episode_seed)
                    initial_phase = json.dumps(env.core.replay_record()['initial_phase'], sort_keys=True)
                    planner = (RealtimeGreedyPolicy(core=env.core, max_search_nodes=max_search_nodes,
                                                    target_warmth=target_warmth) if realtime else
                               TimedGreedyPolicy(core=env.core, max_search_nodes=max_search_nodes))
                    cumulative_reward = 0.0
                    policy_seconds = 0.0
                    max_policy_seconds = 0.0
                    contacts = Counter()
                    magnetic_heat = 0
                    terminated = truncated = False
                    while not (terminated or truncated):
                        started = perf_counter()
                        action = (int(random.integers(6)) if policy_name == 'random'
                                  else planner.action(info['snapshot']))
                        elapsed = perf_counter() - started
                        policy_seconds += elapsed
                        max_policy_seconds = max(max_policy_seconds, elapsed)
                        obs, reward, terminated, truncated, info = env.step(action)
                        if not env.observation_space.contains(obs):
                            raise AssertionError('rollout escaped observation space')
                        cumulative_reward += reward
                        for event in info['events']:
                            if event['type'] == 'contact':
                                contacts['total'] += 1
                                contacts['item_protected'] += int(event.get('item_invincible', False))
                                contacts['chaser'] += int('C' in event['ghost_ids'])
                                contacts['patrol'] += int(any(g != 'C' for g in event['ghost_ids']))
                            elif event['type'] == 'collect' and event.get('magnetic', False):
                                magnetic_heat += event['value']
                    outcome = info['outcome'] or {}
                    score = outcome.get('score') or {}
                    snapshot = info['snapshot']
                    rows.append({
                        'stage': stage, 'policy': policy_name, 'episode': episode,
                        'seed': episode_seed, 'mode': mode, 'randomize_phase': randomize_phase,
                        'status': outcome.get('status', 'TRUNCATED'),
                        'reason': outcome.get('reason', info.get('truncation_reason', 'unknown')),
                        'complete': outcome.get('status') == 'COMPLETE',
                        'target_reached': outcome.get('status') == 'COMPLETE'
                                          and outcome.get('banked', 0) >= target_warmth,
                        'returned': outcome.get('status') in ('COMPLETE', 'RETURN_LOW'),
                        'banked': outcome.get('banked', 0),
                        'carried': info['snapshot']['carried'],
                        'damage_count': outcome.get('damage_count', 0),
                        'ticks_used': snapshot['tick'],
                        'episode_steps': info['episode_steps'],
                        'physics_frames': info['episode_steps'] if realtime else None,
                        'simulated_seconds': (info['episode_steps'] / PHYSICS_HZ if realtime
                                              else snapshot['tick'] * env.core.config['tick_seconds']),
                        'policy_compute_seconds': policy_seconds,
                        'mean_policy_action_ms': 1000 * policy_seconds / info['episode_steps'],
                        'max_policy_action_ms': 1000 * max_policy_seconds,
                        'control_mode': control_mode,
                        'remaining_ticks': info['snapshot']['remaining_ticks'],
                        'initial_ticks': snapshot['initial_ticks'],
                        'initial_phase': initial_phase,
                        'quest_collected': snapshot['quest']['collected'],
                        'remaining_courage': snapshot['courage'],
                        'score_warmth': score.get('warmth', 0),
                        'score_speed': score.get('speed', 0),
                        'score_courage': score.get('courage', 0),
                        'score_total': score.get('total', 0),
                        'stars': score.get('stars', 0),
                        'contacts': contacts['total'],
                        'chaser_contacts': contacts['chaser'],
                        'patrol_contacts': contacts['patrol'],
                        'item_protected_contacts': contacts['item_protected'],
                        'magnet_present': snapshot.get('magnet') is not None,
                        'magnet_collected': bool(snapshot.get('magnet') and snapshot['magnet']['collected']),
                        'magnetic_heat': magnetic_heat,
                        'invincibility_present': snapshot.get('invincibility') is not None,
                        'invincibility_collected': bool(snapshot.get('invincibility') and snapshot['invincibility']['collected']),
                        'search_found': planner.search_counts['found'],
                        'search_node_limit': planner.search_counts['node_limit'],
                        'search_exhausted': planner.search_counts['exhausted'],
                        'search_insufficient_time': planner.search_counts['insufficient_time'],
                        'search_unreachable': planner.search_counts['unreachable'],
                        'search_expanded_nodes': planner.expanded_nodes,
                        'no_verified_plan_steps': planner.fallback_counts['no_verified_plan'],
                        'no_safe_action_steps': planner.fallback_counts['no_safe_action'],
                        'reward': cumulative_reward, 'terminated': terminated,
                        'truncated': truncated, 'config_hash': info['config_hash'],
                        'observation_version': info['observation_version'],
                    })
                    if record_dir is not None:
                        record = env.core.replay_record()
                        record['final_snapshot_sha256'] = snapshot_digest(snapshot)
                        record['evaluation'] = {'policy': policy_name, 'episode_seed': episode_seed,
                            'target_warmth': target_warmth, 'truncated': truncated,
                            'truncation_reason': info.get('truncation_reason')}
                        filename = f'stage_{stage:02d}_{policy_name}_{episode:04d}.json'
                        (record_dir / filename).write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            except Exception as exc:
                errors.append({'stage': stage, 'policy': policy_name, 'error': repr(exc),
                               'traceback': traceback.format_exc()})
            finally:
                env.close()
            if progress is not None:
                group = rows[group_start:]
                progress(stage, policy_name, sum(r['complete'] for r in group), len(group))
    summaries = []
    for stage in stages:
        for policy in ('random', 'timed_greedy'):
            subset = [r for r in rows if r['stage'] == stage and r['policy'] == policy]
            if not subset:
                continue
            n = len(subset)
            complete = [r for r in subset if r['complete']]
            summaries.append({'stage': stage, 'policy': policy, 'episodes': n,
                'complete_rate': sum(r['complete'] for r in subset) / n,
                'target_rate': sum(r['target_reached'] for r in subset) / n,
                'return_rate': sum(r['returned'] for r in subset) / n,
                'mean_banked': sum(r['banked'] for r in subset) / n,
                'mean_damage': sum(r['damage_count'] for r in subset) / n,
                'mean_ticks': sum(r['ticks_used'] for r in subset) / n,
                'mean_episode_steps': sum(r['episode_steps'] for r in subset) / n,
                'mean_physics_frames': (sum(r['physics_frames'] for r in subset) / n if realtime else None),
                'mean_simulated_seconds': sum(r['simulated_seconds'] for r in subset) / n,
                'mean_policy_compute_seconds': sum(r['policy_compute_seconds'] for r in subset) / n,
                'mean_complete_ticks': (sum(r['ticks_used'] for r in complete) / len(complete)
                                        if complete else None),
                'mean_complete_remaining_ticks': (sum(r['remaining_ticks'] for r in complete) / len(complete)
                                                   if complete else None),
                'mean_score_warmth': sum(r['score_warmth'] for r in subset) / n,
                'mean_score_speed': sum(r['score_speed'] for r in subset) / n,
                'mean_score_courage': sum(r['score_courage'] for r in subset) / n,
                'mean_score_total': sum(r['score_total'] for r in subset) / n,
                'mean_stars': sum(r['stars'] for r in subset) / n,
                'magnet_collection_rate': sum(r['magnet_collected'] for r in subset) / n,
                'invincibility_collection_rate': sum(r['invincibility_collected'] for r in subset) / n,
                'mean_contacts': sum(r['contacts'] for r in subset) / n,
                'mean_item_protected_contacts': sum(r['item_protected_contacts'] for r in subset) / n,
                'search_node_limit': sum(r['search_node_limit'] for r in subset),
                'search_exhausted': sum(r['search_exhausted'] for r in subset),
                'search_insufficient_time': sum(r['search_insufficient_time'] for r in subset),
                'no_verified_plan_steps': sum(r['no_verified_plan_steps'] for r in subset),
                'no_safe_action_steps': sum(r['no_safe_action_steps'] for r in subset),
                'statuses': dict(Counter(r['status'] for r in subset)),
                'end_reasons': dict(Counter(r['reason'] for r in subset))})
    report = {'evaluation_kind': 'headless_smoke_baselines_not_trained_RL',
              'control_mode': control_mode,
              'observation_version': REALTIME_OBSERVATION_VERSION if realtime else OBSERVATION_VERSION,
              'physics_hz': PHYSICS_HZ if realtime else None,
              'episode_step_unit': 'physics_frame' if realtime else 'world_tick',
              'target_warmth': target_warmth,
              'max_episode_steps': max_episode_steps,
              'base_seed': seed,
              'episodes_per_stage_policy': episodes, 'mode': mode,
              'randomize_phase': randomize_phase,
              'policy_version': REALTIME_POLICY_VERSION if realtime else POLICY_VERSION,
              'policy_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in ('tools/evaluate.py', 'tools/realtime_policy.py', 'kkamang/core.py',
                               'kkamang/realtime.py', 'kkamang/env.py')},
              'max_search_nodes': max_search_nodes,
              'contact_policy': 'item_invincible_only',
              'return_reserve_ticks': 2,
              'notes': ['No policy was trained.',
                        ('This evaluates the realtime desktop controller, one action per physics frame.' if realtime else
                         'This evaluates grid movement, not the realtime desktop controller.'),
                        'ticks_used counts world ticks; episode_steps counts environment actions.',
                        'Policy compute time includes planning and excludes environment stepping; it is not target-device FPS.',
                        'This is an offline planning baseline: planning wall time does not advance the simulated game.',
                        'Uniform random samples a new action every environment step, including every 1/60s frame in realtime.',
                        'The realtime planner searches tile-center walks and waits, not every mid-cell turn.',
                        'Failed or externally truncated runs bank zero.',
                        'Timed greedy uses public map, route and phase data; it is not optimal.',
                        'All stages use the selected actual core with the same search budget and contact rule.',
                        'Search counters describe attempted legs, not proven causes of episode failure.',
                        'Fallback counters count planning decisions, not physics frames.',
                        'Score averages include failed runs as zero; completed-run time averages are separate.',
                        'Same seed and fixed map are reproducibility checks, not unseen-map generalization.',
                        'Training must reserve separate validation/test seeds; compare fixed protocol and budgets.'],
              'summary': summaries, 'errors': errors}
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--control-mode', choices=('realtime', 'grid'), default='realtime',
                        help='Desktop free movement by default; grid retains legacy checks')
    parser.add_argument('--target-warmth', type=int, default=14,
                        help='Realtime greedy collection target in 4..14; may legally return less')
    parser.add_argument('--max-episode-steps', type=int,
                        help='External cutoff in physics frames (realtime) or world ticks (grid)')
    parser.add_argument('--stages', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--max-search-nodes', type=int,
                        help='A* nodes per target: realtime default 128, grid default 1500')
    parser.add_argument('--mode', choices=('basic', 'relaxed'), default='basic')
    parser.add_argument('--fixed-phase', action='store_true',
                        help='Keep designed ghost starting phases instead of seeded variations')
    parser.add_argument('--output', type=Path, default=Path('reports/local/evaluation'))
    args = parser.parse_args()
    if args.max_search_nodes is None:
        args.max_search_nodes = 128 if args.control_mode == 'realtime' else DEFAULT_SEARCH_NODES
    if (args.episodes < 1 or args.seed < 0 or args.max_search_nodes < 1
            or any(s < 1 or s > 12 for s in args.stages) or len(set(args.stages)) != len(args.stages)
            or not 4 <= args.target_warmth <= 14
            or (args.control_mode == 'grid' and args.target_warmth != 14)
            or (args.max_episode_steps is not None and args.max_episode_steps < 1)):
        parser.error('positive budgets, seed >= 0, stages in 1..12 and target in 4..14 required; grid target is 14')
    def progress(stage, policy, complete, count):
        print(f'stage {stage:02d} {policy}: {complete}/{count} complete', flush=True)
    rows, report = evaluate(args.episodes, args.stages, args.seed, args.mode, not args.fixed_phase,
                            max_search_nodes=args.max_search_nodes, progress=progress,
                            control_mode=args.control_mode, target_warmth=args.target_warmth,
                            max_episode_steps=args.max_episode_steps, record_dir=args.output / 'replays')
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
