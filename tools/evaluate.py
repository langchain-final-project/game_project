"""같은 맵·시드에서 무작위 정책과 공개 상태 기반 탐욕 경로 정책을 비교한다.

강화학습 훈련 결과나 최적해를 뜻하지 않으며, 환경 연결 확인용 기준 정책이다."""
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
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from gymnasium.utils.env_checker import check_env

from kkamang.env import MansionEnv, OBSERVATION_VERSION

MOVES = ((0, 0, -1), (1, 0, 1), (2, -1, 0), (3, 1, 0), (5, 0, 0))
POLICY_VERSION = 'unified_core_v1'
DEFAULT_SEARCH_NODES = 1500


def ghost_position(ghost: dict, ticks: int) -> tuple[int, int]:
    """공개된 순찰 경로와 남은 이동 시간만으로 위치를 예측한다."""
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
    """위치와 공개 순찰 위상의 너비 우선 탐색으로 마지막 상호작용까지 안전한 경로를 구한다.

    닫힌 문과 벽은 통과할 수 없고 보호 중에도 접촉을 피한다. 제한 시간에는 마지막 상호작용 1틱이 포함된다."""
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
    """자석의 인접 수집은 1점 온기에만 적용한다."""
    x, y = item['position']
    magnet = snapshot.get('magnet')
    radius = (1 if magnet and magnet['remaining_ticks'] > 0 and item['value'] == 1
              else 0)
    return [(nx, ny) for ny in range(y - radius, y + radius + 1)
            for nx in range(x - radius, x + radius + 1)
            if 0 <= nx < snapshot['width'] and 0 <= ny < snapshot['height']
            and snapshot['grid'][ny][nx] == '.']


def acceptable_step(result, *, forbid_rich=False, allow_item_invincibility=False):
    """아이템 무적만 접촉을 허용하며 피해 후 보호 시간은 안전 경로 근거로 쓰지 않는다."""
    if any(event['type'] == 'damage' or (
            event['type'] == 'contact' and not (
                allow_item_invincibility and event.get('item_invincible', False)))
           for event in result.events):
        return False
    return not (forbid_rich and any(not item['active'] and item['value'] == 3
                                   for item in result.snapshot['items']))


def project_path(core, path, *, forbid_rich=False, allow_item_invincibility=False):
    """아이템 자동 수집과 만료까지 실제 코어로 후보 경로를 검증한다."""
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
    """후보 경로를 실제로 검증하고 실패하면 노드 수가 제한된 탐색을 수행한다.

    탐색 한도에 따른 실패는 불가능성의 증명이 아니며, 다른 목표 순서는 성공할 수 있다."""
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
    # 순찰 탐색은 후보 생성용이며 추격과 아이템 효과는 실제 코어 전이로 검증한다.
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
            # 탐색 복제는 읽기 전용 맵과 난수를 공유하되 변경 가능한 상태와 행동 기록은 복사한다.
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
    """필수 물건을 먼저 회수하고 안전한 귀환 시간을 확보하며 온기를 수집한다.

    모든 수집 순서와 지름길을 탐색하지 않는 기준 정책이다."""
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
        for action in (5, 0, 1, 2, 3):
            if project_path(self.core, [action], allow_item_invincibility=True) is not None:
                self.fallback_counts['no_verified_plan'] += 1
                return [action]
        self.fallback_counts['no_safe_action'] += 1
        return [5]

    def _planned_actions(self, snapshot):
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
             progress=None) -> tuple[list[dict], dict]:
    rows = []
    errors = []
    for stage in stages:
        for policy_name in ('random', 'timed_greedy'):
            group_start = len(rows)
            env = MansionEnv(stage, mode=mode, randomize_phase=randomize_phase)
            try:
                check_env(env, skip_render_check=True)
                for episode in range(episodes):
                    episode_seed = seed + stage * 100000 + episode
                    _, info = env.reset(seed=episode_seed)
                    random = np.random.default_rng(episode_seed)
                    initial_phase = json.dumps(env.core.replay_record()['initial_phase'], sort_keys=True)
                    planner = TimedGreedyPolicy(core=env.core, max_search_nodes=max_search_nodes)
                    cumulative_reward = 0.0
                    contacts = Counter()
                    magnetic_heat = 0
                    terminated = truncated = False
                    while not (terminated or truncated):
                        action = (int(random.integers(6)) if policy_name == 'random'
                                  else planner.action(info['snapshot']))
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
                        'returned': outcome.get('status') in ('COMPLETE', 'RETURN_LOW'),
                        'banked': outcome.get('banked', 0),
                        'carried': info['snapshot']['carried'],
                        'damage_count': outcome.get('damage_count', 0),
                        'ticks_used': info['episode_steps'],
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
                'return_rate': sum(r['returned'] for r in subset) / n,
                'mean_banked': sum(r['banked'] for r in subset) / n,
                'mean_damage': sum(r['damage_count'] for r in subset) / n,
                'mean_ticks': sum(r['ticks_used'] for r in subset) / n,
                'mean_complete_ticks': (sum(r['ticks_used'] for r in complete) / len(complete)
                                        if complete else None),
                'mean_complete_remaining_ticks': (sum(r['remaining_ticks'] for r in complete) / len(complete)
                                                   if complete else None),
                'mean_score_warmth': sum(r['score_warmth'] for r in subset) / n,
                'mean_score_speed': sum(r['score_speed'] for r in subset) / n,
                'mean_score_courage': sum(r['score_courage'] for r in subset) / n,
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
              'control_mode': 'grid',
              'observation_version': OBSERVATION_VERSION, 'base_seed': seed,
              'episodes_per_stage_policy': episodes, 'mode': mode,
              'randomize_phase': randomize_phase,
              'policy_version': POLICY_VERSION,
              'policy_implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'max_search_nodes': max_search_nodes,
              'contact_policy': 'item_invincible_only',
              'return_reserve_ticks': 2,
              'notes': ['No policy was trained.', 'This evaluates grid movement, not the realtime desktop controller.',
                        'Failed or externally truncated runs bank zero.',
                        'Timed greedy uses public map, route and phase data; it is not optimal.',
                        'All stages use actual GameCore transitions with the same search budget and contact rule.',
                        'Search counters describe attempted legs, not proven causes of episode failure.',
                        'Score averages include failed runs as zero; completed-run time averages are separate.',
                        'Same seed and fixed map are reproducibility checks, not unseen-map generalization.',
                        'Training must reserve separate validation/test seeds; compare fixed protocol and budgets.'],
              'summary': summaries, 'errors': errors}
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--stages', type=int, nargs='+', default=list(range(1, 13)))
    parser.add_argument('--seed', type=int, default=20261014)
    parser.add_argument('--max-search-nodes', type=int, default=DEFAULT_SEARCH_NODES,
                        help='Same A* node budget per target for every stage')
    parser.add_argument('--mode', choices=('basic', 'relaxed'), default='basic')
    parser.add_argument('--fixed-phase', action='store_true',
                        help='Keep designed ghost starting phases instead of seeded variations')
    parser.add_argument('--output', type=Path, default=Path('reports/evaluation'))
    args = parser.parse_args()
    if (args.episodes < 1 or args.seed < 0 or args.max_search_nodes < 1
            or any(s < 1 or s > 12 for s in args.stages)):
        parser.error('episodes/search nodes >= 1, seed >= 0, and stages in 1..12 are required')
    def progress(stage, policy, complete, count):
        print(f'stage {stage:02d} {policy}: {complete}/{count} complete', flush=True)
    rows, report = evaluate(args.episodes, args.stages, args.seed, args.mode, not args.fixed_phase,
                            max_search_nodes=args.max_search_nodes, progress=progress)
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
