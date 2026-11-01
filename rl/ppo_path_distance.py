"""벽·닫힌 문을 반영한 격자 최단 경로를 연속 좌표에 연결합니다."""

from collections import deque
from functools import lru_cache
import math

from kkamang.realtime import PLAYER_RADIUS


@lru_cache(maxsize=64)
def _distance_map(grid, gates, target):
    height, width = len(grid), len(grid[0])
    blocked = frozenset({(x, y) for y, row in enumerate(grid) for x, cell in enumerate(row)
                         if cell == "#"} | set(gates))
    distances = {}
    if target not in blocked and 0 <= target[0] < width and 0 <= target[1] < height:
        distances[target] = 0
        queue = deque([target])
        while queue:
            x, y = queue.popleft()
            for point in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
                if (0 <= point[0] < width and 0 <= point[1] < height
                        and point not in blocked and point not in distances):
                    distances[point] = distances[(x, y)] + 1
                    queue.append(point)
    return distances, blocked


def _clear_segment(start, end, grid, blocked):
    # 몸체 크기를 반영해 벽 모서리를 가로질러 중심점에 연결하지 않습니다.
    lo = [math.floor(min(start[i], end[i]) + .5 - PLAYER_RADIUS + 1e-10) for i in (0, 1)]
    hi = [math.floor(max(start[i], end[i]) + .5 + PLAYER_RADIUS - 1e-10) for i in (0, 1)]
    return all(0 <= x < len(grid[0]) and 0 <= y < len(grid) and (x, y) not in blocked
               for y in range(lo[1], hi[1] + 1) for x in range(lo[0], hi[0] + 1))


def path_distance(snapshot, target):
    """연결 가능한 인접 중심점까지의 거리 + BFS 거리. 경로가 없으면 None."""
    grid = tuple(snapshot["grid"])
    gates = tuple(sorted(tuple(d["gate"]) for d in snapshot.get("shortcuts", []) if not d["opened"]))
    distances, blocked = _distance_map(grid, gates, tuple(target))
    x, y = snapshot["motion"]["position"]
    start = (x, y)
    candidates = []
    for cx in {math.floor(x), math.ceil(x)}:
        for cy in {math.floor(y), math.ceil(y)}:
            center = (cx, cy)
            if center not in distances:
                continue
            connected = any(
                _clear_segment(start, corner, grid, blocked)
                and _clear_segment(corner, center, grid, blocked)
                for corner in ((cx, y), (x, cy))
            )
            if connected:
                candidates.append(distances[center] + abs(cx-x) + abs(cy-y))
    return min(candidates) if candidates else None


def next_waypoint(snapshot, target, *, lookahead=False):
    """최단 경로의 다음 중심점을 안내합니다. 실제 행동은 정책이 선택합니다."""
    grid = tuple(snapshot["grid"])
    gates = tuple(sorted(tuple(d["gate"]) for d in snapshot.get("shortcuts", []) if not d["opened"]))
    distances, blocked = _distance_map(grid, gates, tuple(target))
    start = tuple(snapshot["motion"]["position"])
    candidates = []
    for x in {math.floor(start[0]), math.ceil(start[0])}:
        for y in {math.floor(start[1]), math.ceil(start[1])}:
            center = (x, y)
            if center not in distances:
                continue
            for corner in ((x, start[1]), (start[0], y)):
                if _clear_segment(start, corner, grid, blocked) and _clear_segment(corner, center, grid, blocked):
                    candidates.append((distances[center] + abs(x-start[0]) + abs(y-start[1]), center, corner))
    if not candidates:
        return list(target)
    if lookahead:
        # 반 칸씩 행동을 유지할 때 작은 중심 정렬 목표를 왕복하지 않도록,
        # 같은 최단 경로 위에서 몸체가 통과할 수 있는 직선 구간을 먼저 안내합니다.
        best_cost = min(cost for cost, _, _ in candidates)
        nodes = {center for _, center, _ in candidates}
        for x, y in tuple(nodes):
            nodes.update(point for point in ((x-1,y), (x+1,y), (x,y-1), (x,y+1))
                         if point in distances and distances[point] < distances[(x,y)])
        forward = []
        for node in nodes:
            for axis in (0, 1):
                point = list(start)
                point[axis] = node[axis]
                travel = abs(point[axis] - start[axis])
                if travel <= 1e-8:
                    continue
                cost = travel + abs(point[1-axis] - node[1-axis]) + distances[node]
                if (cost <= best_cost + 1e-8
                        and _clear_segment(start, point, grid, blocked)
                        and _clear_segment(point, node, grid, blocked)):
                    forward.append((-round(travel, 8), round(cost, 8), axis, point))
        if forward:
            return min(forward)[3]
    _, center, corner = min(candidates)
    for point in (corner, center):
        if max(abs(point[i]-start[i]) for i in (0, 1)) > 1e-8:
            return list(point)
    x, y = center
    neighbors = [p for p in ((x-1,y), (x+1,y), (x,y-1), (x,y+1))
                 if p in distances and distances[p] < distances[center]]
    return list(min(neighbors, key=lambda p: (distances[p], p))) if neighbors else list(target)
