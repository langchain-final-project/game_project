"""Bounded, frame-verified baseline for the desktop controller, not learned RL.

Search uses center-to-center walks (18 physics steps), short waits and one-frame
E presses. Every intermediate frame is checked in RealtimeGameCore. This is a
restricted subset of free movement, not an optimal continuous-motion solver.
"""
from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
import heapq
from itertools import count
import math

from kkamang.realtime import FRAMES_PER_TICK, RealtimeGameCore

POLICY_VERSION = 'realtime_verified_v1'
MOVES = ((0, 0, -1), (1, 0, 1), (2, -1, 0), (3, 1, 0))


def project_frames(core, actions, *, allow_item_invincibility=True):
    """Return a simulated copy only if *every* frame meets the contact rule."""
    if not isinstance(core, RealtimeGameCore):
        raise ValueError('frame projection requires RealtimeGameCore')
    trial = deepcopy(core)
    result = None
    for index, action in enumerate(actions):
        result = trial.step(action)
        if any(event['type'] == 'damage' or (
                event['type'] == 'contact' and not (
                    allow_item_invincibility and event.get('item_invincible', False)))
               for event in result.events):
            return None
        if result.terminated and (index != len(actions) - 1
                                  or result.outcome['status'] != 'COMPLETE'):
            return None
    return (trial, result) if result is not None else None


def _distances(snapshot, goal):
    closed = {tuple(g['gate']) for g in snapshot['shortcuts'] if not g['opened']}
    distance = {}
    if (0 <= goal[0] < snapshot['width'] and 0 <= goal[1] < snapshot['height']
            and snapshot['grid'][goal[1]][goal[0]] == '.' and goal not in closed):
        distance[goal] = 0
    queue = deque(distance)
    while queue:
        x, y = queue.popleft()
        for _, dx, dy in MOVES:
            point = (x + dx, y + dy)
            if (point not in distance and point not in closed
                    and 0 <= point[0] < snapshot['width']
                    and 0 <= point[1] < snapshot['height']
                    and snapshot['grid'][point[1]][point[0]] == '.'):
                distance[point] = distance[(x, y)] + 1
                queue.append(point)
    return distance


def _key(snapshot):
    chaser = snapshot.get('chaser')
    return (tuple(round(x, 8) for x in snapshot['motion']['position']),
            snapshot['motion']['frame'] % FRAMES_PER_TICK,
            snapshot['remaining_ticks'], snapshot['quest']['collected'],
            tuple(item['active'] for item in snapshot['items']),
            tuple(g['opened'] for g in snapshot['shortcuts']),
            tuple((g['route_index'], g['countdown']) for g in snapshot['ghosts']),
            None if chaser is None else (chaser['active'], tuple(chaser['position']), chaser['countdown']),
            tuple(None if snapshot.get(name) is None else
                  (snapshot[name]['collected'], snapshot[name]['remaining_ticks'])
                  for name in ('magnet', 'invincibility')))


def find_realtime_path(core, goal, *, interact=True, max_nodes=1500,
                       allow_item_invincibility=True, diagnostics=None):
    """Find a witnessed safe leg; failure is not a proof of impossibility.

    The planner starts at tile centers. It does not consume grid witness routes,
    convert grid safety claims, open shortcuts, or change the live game.
    """
    if not isinstance(core, RealtimeGameCore):
        raise ValueError('realtime search requires RealtimeGameCore')
    if type(max_nodes) is not int or max_nodes < 1:
        raise ValueError('max_nodes must be a positive integer')

    def finish(actions, status, expanded=0):
        if diagnostics is not None:
            diagnostics.update(status=status, expanded_nodes=expanded)
        return actions

    snapshot = core.snapshot()
    if snapshot['terminal']:
        return finish(None, 'terminal')
    center = snapshot['motion']['position']
    if any(abs(x - round(x)) > 1e-8 for x in center):
        raise ValueError('baseline search starts at tile centers, not mid-cell')
    goal = tuple(goal)
    distances = _distances(snapshot, goal)
    start = tuple(snapshot['position'])
    if start not in distances:
        return finish(None, 'unreachable')
    available = snapshot['remaining_ticks'] * FRAMES_PER_TICK - (
        snapshot['motion']['frame'] % FRAMES_PER_TICK)
    if distances[start] * FRAMES_PER_TICK + int(interact) > available:
        return finish(None, 'insufficient_time')
    # Try several clock offsets on a shortest spatial path before bounded A*.
    point, direct = start, []
    while point != goal:
        for action, dx, dy in MOVES:
            new = point[0] + dx, point[1] + dy
            if distances.get(new) == distances[point] - 1:
                direct.extend([action] * FRAMES_PER_TICK)
                point = new
                break
    if interact:
        direct.append(4)
    for delay in (0, 3, 6, 9, 12, 15, 18):
        actions = [5] * delay + direct
        if actions and project_frames(core, actions,
                allow_item_invincibility=allow_item_invincibility) is not None:
            return finish(actions, 'found')
        if not actions:
            return finish([], 'found')

    serial = count()
    frontier = [(distances[start] * FRAMES_PER_TICK, 0, next(serial), deepcopy(core), [])]
    visited = {_key(snapshot): 0}
    expanded = 0
    while frontier and expanded < max_nodes:
        _, cost, _, current, path = heapq.heappop(frontier)
        state = current.snapshot()
        if visited.get(_key(state)) != cost:
            continue
        expanded += 1
        if tuple(state['position']) == goal:
            ending = [4] if interact else []
            if not ending or project_frames(current, ending,
                    allow_item_invincibility=allow_item_invincibility) is not None:
                return finish(path + ending, 'found', expanded)
        if cost >= available - int(interact):
            continue
        options = [(action, FRAMES_PER_TICK) for action, _, _ in MOVES]
        options.extend([(5, FRAMES_PER_TICK), (5, 1)])
        for action, frames in options:
            if cost + frames + int(interact) > available:
                continue
            if action < 4:
                _, dx, dy = MOVES[action]
                destination = state['position'][0] + dx, state['position'][1] + dy
                if destination not in distances:
                    continue
            segment = [action] * frames
            projected = project_frames(current, segment,
                                       allow_item_invincibility=allow_item_invincibility)
            if projected is None or projected[1].terminated:
                continue
            trial, result = projected
            after = result.snapshot
            if any(abs(x - round(x)) > 1e-8 for x in after['motion']['position']):
                continue
            key = _key(after)
            next_cost = cost + frames
            if next_cost >= visited.get(key, math.inf):
                continue
            visited[key] = next_cost
            estimate = next_cost + distances[tuple(after['position'])] * FRAMES_PER_TICK
            heapq.heappush(frontier, (estimate, next_cost, next(serial), trial, path + segment))
    return finish(None, 'node_limit' if frontier else 'exhausted', expanded)


class RealtimeGreedyPolicy:
    """Quest first, then heat with a frame-verified return and two-tick reserve.

    A failed attempt to reach target_warmth may return a lower but legal score.
    This deterministic, public-information baseline is not a learned policy.
    """
    def __init__(self, core, *, max_search_nodes=1500, target_warmth=14,
                 allow_item_invincibility=True):
        if not isinstance(core, RealtimeGameCore):
            raise ValueError('RealtimeGreedyPolicy requires RealtimeGameCore')
        if type(target_warmth) is not int or not 4 <= target_warmth <= 14:
            raise ValueError('target_warmth must be an integer in 4..14')
        if type(max_search_nodes) is not int or max_search_nodes < 1:
            raise ValueError('max_search_nodes must be a positive integer')
        self.core = core
        self.max_search_nodes = max_search_nodes
        self.target_warmth = target_warmth
        self.allow_item_invincibility = allow_item_invincibility
        self.actions = deque()
        self.search_counts = Counter()
        self.expanded_nodes = 0
        self.fallback_counts = Counter()

    def _path(self, core, point, interact=True):
        diagnostics = {}
        actions = find_realtime_path(core, point, interact=interact,
            max_nodes=self.max_search_nodes,
            allow_item_invincibility=self.allow_item_invincibility, diagnostics=diagnostics)
        self.search_counts[diagnostics['status']] += 1
        self.expanded_nodes += diagnostics['expanded_nodes']
        return actions

    def _project(self, core, actions):
        return project_frames(core, actions,
                              allow_item_invincibility=self.allow_item_invincibility)

    def _fallback(self):
        self.fallback_counts['no_verified_plan'] += 1
        for action in (5, 0, 1, 2, 3):
            actions = [action] * FRAMES_PER_TICK
            projected = self._project(self.core, actions)
            if projected is not None and all(abs(x - round(x)) < 1e-8
                    for x in projected[1].snapshot['motion']['position']):
                return actions
        self.fallback_counts['no_safe_action'] += 1
        return [5] * FRAMES_PER_TICK

    def _plan(self, snapshot):
        home = tuple(snapshot['exit'])
        if snapshot['quest']['collected'] and snapshot['carried'] >= self.target_warmth:
            route = self._path(self.core, home)
            return route if route is not None else self._fallback()
        goals = []
        if not snapshot['quest']['collected']:
            goals.append((tuple(snapshot['quest']['position']), True, 'quest'))
        goals.extend((tuple(item['position']), False, 'heat')
                     for item in snapshot['items'] if item['active'])
        for name in ('magnet', 'invincibility'):
            item = snapshot.get(name)
            if item and not item['collected']:
                goals.append((tuple(item['position']), False, name))
        candidates = []
        for point, interact, kind in goals:
            route = self._path(self.core, point, interact)
            if not route:
                continue
            projected = self._project(self.core, route)
            if projected is None or projected[1].terminated:
                continue
            trial, result = projected
            after = result.snapshot
            gain = after['carried'] - snapshot['carried']
            item_gain = kind in ('magnet', 'invincibility')
            quest_gain = after['quest']['collected'] and not snapshot['quest']['collected']
            if not (gain or item_gain or quest_gain):
                continue
            back = self._path(trial, home)
            available = after['remaining_ticks'] * FRAMES_PER_TICK - (
                after['motion']['frame'] % FRAMES_PER_TICK)
            if back is None or len(back) + 2 * FRAMES_PER_TICK > available:
                continue
            utility = gain + 5 * item_gain + 8 * quest_gain
            candidates.append((-utility / len(route), len(route), point, route))
            if quest_gain:
                return route
        if candidates:
            return min(candidates)[3]
        if snapshot['quest']['collected'] and snapshot['carried'] >= snapshot['target']:
            route = self._path(self.core, home)
            if route is not None:
                return route
        return self._fallback()

    def action(self, snapshot):
        if not self.actions:
            self.actions.extend(self._plan(snapshot))
        return self.actions.popleft()
