"""벽과 현재 닫힌 문을 반영한 퀘스트 거리. 행동이나 기준 경로를 만들지 않습니다.

네 방향으로 연결된 칸 중심의 최단 거리를 계산하고, 연속 위치는 근처 중심에
벽을 통과하지 않는 L자 연결로 붙입니다. 열린 공간에서는 Manhattan 거리와
같지만, 몸체 반경·코너 보정·유령·미래의 문 개방을 예측하는 물리 경로는 아닙니다.
정적 거리장은 지도/닫힌 문/목표가 같으면 재사용하므로 프레임마다 BFS하지 않습니다.
"""
from __future__ import annotations

from collections import deque
from functools import lru_cache
import math

DISTANCE_METRIC_VERSION = 'walkable_grid_l1_v1'


@lru_cache(maxsize=256)
def _distance_field(grid, closed_gates, target):
    # 목표에서 네 방향 BFS를 한 번 수행해 모든 통행 가능 칸의 거리를 재사용합니다.
    height, width = len(grid), len(grid[0])
    blocked = frozenset(closed_gates) | frozenset(
        (x, y) for y, row in enumerate(grid) for x, cell in enumerate(row) if cell == '#')
    distances = [-1] * (width * height)
    tx, ty = target
    if target in blocked:
        return tuple(distances), blocked, width, height, 2.
    distances[ty * width + tx] = 0
    frontier = deque([target])
    while frontier:
        x, y = frontier.popleft()
        following = distances[y * width + x] + 1
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if (0 <= nx < width and 0 <= ny < height and (nx, ny) not in blocked
                    and distances[ny * width + nx] == -1):
                distances[ny * width + nx] = following
                frontier.append((nx, ny))
    # 연속 위치에서 칸 중심까지의 L 연결(<2타일)을 포함하는 거리 상한.
    return tuple(distances), blocked, width, height, float(max(distances) + 2)


def _cell(coordinate):
    # 원래 게임과 같이 정수 좌표를 칸 중심으로 사용합니다.
    return math.floor(round(coordinate, 10) + .5)


def _axis_clear(start, end, blocked, width, height):
    """축 방향 선분이 통과하는 칸을 검사합니다. 몸체 반경은 다루지 않습니다."""
    if start[0] == end[0]:
        cells = ((_cell(start[0]), y)
                 for y in range(_cell(min(start[1], end[1])), _cell(max(start[1], end[1])) + 1))
    elif start[1] == end[1]:
        cells = ((x, _cell(start[1]))
                 for x in range(_cell(min(start[0], end[0])), _cell(max(start[0], end[0])) + 1))
    else:
        raise ValueError('거리 연결은 수평 또는 수직 선분이어야 합니다.')
    return all(0 <= x < width and 0 <= y < height and (x, y) not in blocked for x, y in cells)


def _connection_clear(position, anchor, blocked, width, height):
    for bend in ((anchor[0], position[1]), (position[0], anchor[1])):
        if (_axis_clear(position, bend, blocked, width, height)
                and _axis_clear(bend, anchor, blocked, width, height)):
            return True
    return False


def quest_walk_distance_and_bound(snapshot, level) -> tuple[float | None, float]:
    """길 거리와 같은 연결 성분의 거리 상한을 함께 반환합니다.

    반환값은 타일 단위이며 지도/문 상태만 사용합니다. 게임 상태·난수·행동을
    변경하지 않습니다. 도달 불가 거리는 None이며 상한으로 대체하지 않습니다.
    상한은 목표에서 연결된 칸 중심까지의 최대 거리+2입니다. 같은 지도/문 상태의
    모든 위치에 동일하고, 연속 위치까지 포함하여 선형 잠재값을 0..1에 둡니다.
    """
    grid = tuple(snapshot.get('grid', level.get('grid', ())))
    if (not grid or not grid[0] or any(len(row) != len(grid[0]) or set(row) - {'#', '.'} for row in grid)):
        raise ValueError('직사각형의 #/. 지도가 필요합니다.')
    height, width = len(grid), len(grid[0])
    target = tuple(snapshot['quest']['position'])
    if (len(target) != 2 or any(type(value) is not int for value in target)
            or not 0 <= target[0] < width or not 0 <= target[1] < height):
        raise ValueError('퀘스트 위치는 지도 안의 정수 좌표여야 합니다.')
    position = tuple(float(value) for value in snapshot['motion']['position'])
    if (len(position) != 2 or any(not math.isfinite(value) for value in position)
            or not 0 <= position[0] <= width - 1 or not 0 <= position[1] <= height - 1):
        raise ValueError('플레이어 위치는 지도 안의 유한 연속 좌표여야 합니다.')
    closed = tuple(sorted({tuple(door['gate'])
                          for door in snapshot.get('shortcuts', level.get('shortcuts', ()))
                          if not door.get('opened', False)}))
    if any(len(gate) != 2 or any(type(value) is not int for value in gate)
           or not 0 <= gate[0] < width or not 0 <= gate[1] < height for gate in closed):
        raise ValueError('닫힌 문 위치는 지도 안의 정수 좌표여야 합니다.')
    distances, blocked, width, height, bound = _distance_field(grid, closed, target)
    if (_cell(position[0]), _cell(position[1])) in blocked:
        return None, bound
    best = math.inf
    for x in {math.floor(position[0]), math.ceil(position[0])}:
        for y in {math.floor(position[1]), math.ceil(position[1])}:
            if not 0 <= x < width or not 0 <= y < height:
                continue
            value = distances[y * width + x]
            if value < 0 or not _connection_clear(position, (x, y), blocked, width, height):
                continue
            candidate = value + abs(position[0] - x) + abs(position[1] - y)
            best = min(best, candidate)
    return (None if not math.isfinite(best) else float(best)), bound


def quest_walk_distance(snapshot, level) -> float | None:
    """현재 연속 위치에서 퀘스트까지의 길 거리, 연결할 수 없으면 None."""
    return quest_walk_distance_and_bound(snapshot, level)[0]


def clear_distance_cache():
    _distance_field.cache_clear()


def distance_cache_info():
    return _distance_field.cache_info()
