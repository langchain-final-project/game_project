"""온기·필수 물건·귀환을 함께 고려한 Stage 3 목표 안내입니다."""

from collections import deque
from functools import lru_cache
import math

from rl.ppo_path_distance import path_distance, next_waypoint
from kkamang.levels import load_level
from kkamang.scoring import score_outcome


@lru_cache(maxsize=1)
def _stage3_score_level():
    return load_level(3)


@lru_cache(maxsize=12)
def _tour_solver(grid, gates, positions, values, terminal):
    """정적 지형의 목표 간 거리와 부분집합별 남은 최소 이동 비용을 재사용합니다."""
    blocked = set(gates) | {(x, y) for y, row in enumerate(grid) for x, cell in enumerate(row) if cell == '#'}
    def distances(origin):
        result = {origin: 0}
        queue = deque([origin])
        while queue:
            x, y = queue.popleft()
            for point in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                px, py = point
                if (0 <= py < len(grid) and 0 <= px < len(grid[0])
                        and point not in blocked and point not in result):
                    result[point] = result[(x,y)] + 1
                    queue.append(point)
        return result
    maps = [distances(position) for position in positions]
    matrix = tuple(tuple(m.get(p, math.inf) for p in positions) for m in maps)
    tx, ty = terminal
    end = tuple(m.get(terminal, min((m.get(p,math.inf)+1 for p in
                                    ((tx-1,ty),(tx+1,ty),(tx,ty-1),(tx,ty+1))),default=math.inf)) for m in maps)

    @lru_cache(maxsize=180000)
    def solve(current, mask, needed, required):
        if needed <= 0 and not mask & required:
            return end[current]
        if sum(value for i, value in enumerate(values) if mask & (1 << i)) < needed:
            return math.inf
        best = math.inf
        remaining = mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            index = bit.bit_length() - 1
            cost = matrix[current][index] + solve(index, mask ^ bit, max(0, needed-values[index]), required)
            best = min(best, cost)
        return best
    return solve


def plan_target(snapshot, warmth_target, terminal, *, require_quest=True, small_only=False):
    """다음 목표만 반환하고 실제 이동·대기 행동은 정책이 선택하게 합니다."""
    items = sorted(snapshot['items'], key=lambda item: item['id'])
    positions = tuple(tuple(item['position']) for item in items) + (tuple(snapshot['quest']['position']),)
    values = tuple(item['value'] for item in items) + (0,)
    mask = sum(1 << i for i, item in enumerate(items) if item['active'] and (not small_only or item['value'] == 1))
    required = 0
    if require_quest and not snapshot['quest']['collected']:
        required = 1 << len(items)
        mask |= required
    needed = max(0, warmth_target - snapshot['carried'])
    if needed == 0 and not required:
        return list(terminal)
    gates = tuple(sorted(tuple(door['gate']) for door in snapshot.get('shortcuts',[]) if not door['opened']))
    solve = _tour_solver(tuple(snapshot['grid']), gates, positions, values, tuple(terminal))
    candidates = []
    for i, position in enumerate(positions):
        if not mask & (1 << i):
            continue
        distance = path_distance(snapshot, position)
        if distance is not None:
            cost = distance + solve(i, mask ^ (1 << i), max(0,needed-values[i]), required)
            if math.isfinite(cost):
                candidates.append((cost, distance, i))
    return list(positions[min(candidates)[2]]) if candidates else None


def stage3_warmth_target(snapshot, settings, safe):
    if not safe:
        return snapshot['target']
    if settings.get('stage3_score_aware_warmth', False):
        # 현재 용기와 귀환 거리를 기준으로 점수를 추정해 필요한 추가 수집만 안내합니다.
        level = _stage3_score_level()
        distance = path_distance(snapshot, snapshot['exit'])
        arrival = snapshot['tick'] + math.ceil(distance or 0) + settings.get('stage3_return_reserve_ticks', 3)
        for warmth in range(settings['warmth_target'], level['star_warmth_cap'] + 1):
            score = score_outcome(level, {
                'status': 'COMPLETE', 'banked': warmth, 'ticks_used': arrival,
                'remaining_courage': snapshot['courage'],
            })
            if score['stars'] == 3:
                return warmth
        return level['star_warmth_cap']
    return max(settings['warmth_target'],14 if snapshot['damage_count']>=2 else 0)


def efficient_goal(snapshot, settings, safe):
    """무적으로 중앙 고가치 온기를 수집한 뒤 남은 수집·물건·귀환 비용을 봅니다."""
    enough = snapshot['carried'] >= snapshot['target']
    desired = stage3_warmth_target(snapshot,settings,safe)
    routing = snapshot
    if snapshot['quest']['collected'] and snapshot['carried'] < desired:
        # 추가 수집 중 출구를 경유해 낮은 별로 조기 완료하는 경로는 안내하지 않습니다.
        grid = list(snapshot['grid'])
        ex, ey = snapshot['exit']
        grid[ey] = grid[ey][:ex] + '#' + grid[ey][ex+1:]
        routing = {**snapshot, 'grid': grid}
    invincible = snapshot.get('invincibility')
    phase = 1
    if safe and invincible and not invincible['collected']:
        target = plan_target(routing, snapshot['target'], invincible['position'],
                             require_quest=False, small_only=True)
    elif safe and invincible and invincible['remaining_ticks'] > 0 and snapshot['carried'] < desired:
        central = [item for item in snapshot['items'] if item['active'] and item['value'] == 3]
        reachable = [(distance, item['position']) for item in central
                     if (distance := path_distance(routing,item['position'])) is not None]
        target = min(reachable)[1] if reachable else None
        if target is None:
            target = plan_target(routing, desired, snapshot['exit'])
    else:
        target = plan_target(routing, desired, snapshot['exit'])
    if target is None:
        target = snapshot['quest']['position'] if not snapshot['quest']['collected'] else snapshot['exit']
    if list(target) == list(snapshot['quest']['position']) and not snapshot['quest']['collected']:
        phase = 0
    elif snapshot['quest']['collected'] and snapshot['carried'] >= desired:
        phase, target = 2, snapshot['exit']
    if settings.get('path_waypoints',False):
        target = next_waypoint(routing, target,
                               lookahead=settings.get('stage3_waypoint_mode') == 'corridor_lookahead_v1')
    return phase, target, enough
