"""기존 관측에 출구 거리장과 귀환 문맥 4값을 더합니다.

원래 14채널·224값은 같은 순서로 보존합니다. 추가 정보는 현재 지도·닫힌 문과
플레이어 위치에서 계산하며, 다음 행동·정답 경로·미래 유령 위치를 제공하지 않습니다.
시간은 한 타일/게임 틱을 가정한 기하학적 근사로, 충돌·유령 회피 시간을 보장하지 않습니다.
"""
from __future__ import annotations

import math

from gymnasium import spaces
import numpy as np

from kkamang.scoring import score_outcome
from .distance import _distance_field, quest_walk_distance_and_bound
from . import observation as legacy

NAVIGATION_PROFILE = 'exit_distance_field_v1'
NAVIGATION_OBSERVATION_VERSION = 'navigation_native630_exitfield_state228_v1'
MAP_SHAPE = (legacy.MAP_SHAPE[0] + 1, *legacy.MAP_SHAPE[1:])
MAP_FIELDS = (*legacy.MAP_FIELDS, 'walkable_exit_distance')
NAVIGATION_STATE_FIELDS = ('exit_distance_fraction', 'estimated_return_ticks_fraction',
                           'return_time_margin_fraction', 'return_ready')
STATE_FIELDS = (*legacy.STATE_FIELDS, *NAVIGATION_STATE_FIELDS)
STATE_SIZE = len(STATE_FIELDS)
OBSERVATION_DIMENSIONS = int(np.prod(MAP_SHAPE)) + STATE_SIZE


def observation_space():
    """거리·예상 틱·준비는[0,1], 남은 시간 여유는[-1,1]입니다."""
    old = legacy.observation_space()['state']
    low = np.concatenate((old.low, np.array([0., 0., -1., 0.], dtype=np.float32)))
    high = np.concatenate((old.high, np.ones(4, dtype=np.float32)))
    return spaces.Dict({
        'map': spaces.Box(0., 1., shape=MAP_SHAPE, dtype=np.float32),
        'state': spaces.Box(low, high, dtype=np.float32),
    })


def _exit_geometry(snapshot, level):
    # 기존 길 거리 함수로 연속 위치·목표·지도·문 유효성을 함께 확인합니다.
    probe = {**snapshot, 'quest': {**snapshot['quest'], 'position': snapshot['exit']}}
    distance, bound = quest_walk_distance_and_bound(probe, level)
    grid = tuple(snapshot.get('grid', level.get('grid', ())))
    closed = tuple(sorted({tuple(door['gate'])
                           for door in snapshot.get('shortcuts', level.get('shortcuts', ()))
                           if not door.get('opened', False)}))
    values, _, width, height, field_bound = _distance_field(grid, closed, tuple(snapshot['exit']))
    if (height, width) != legacy.MAP_SHAPE[1:]:
        raise ValueError('Navigation observation requires the original9×16 map.')
    if field_bound != bound:
        raise RuntimeError('Scalar and spatial exit distance bounds differ.')
    distances = np.asarray(values, dtype=np.float32).reshape(height, width)
    # 1은 벽·닫힌 문·출구와 단절된 칸의 표지입니다. 정상 도달 칸은 항상1 미만입니다.
    field = np.ones((height, width), dtype=np.float32)
    reachable = distances >= 0
    field[reachable] = distances[reachable] / bound
    return field, distance, bound


def _metrics(snapshot, level, distance, bound):
    initial_ticks = snapshot['initial_ticks']
    if not isinstance(initial_ticks, (int, float)) or initial_ticks <= 0:
        raise ValueError('A positive initial native time budget is required.')
    # native1틱은0.3초이며 기본18프레임 정책 행동1회와 같습니다. 몸체·유령으로
    # 실제 소요시간은 더 길 수 있으므로 안전한 ETA나 실제 귀환 점수로 사용하지 않습니다.
    estimated_ticks = None if distance is None else math.ceil(distance)
    distance_fraction = 1. if distance is None else min(1., max(0., distance / bound))
    ticks_fraction = (1. if estimated_ticks is None
                      else min(1., max(0., estimated_ticks / initial_ticks)))
    margin = (-1. if estimated_ticks is None else
              max(-1., min(1., (snapshot['remaining_ticks'] - estimated_ticks) / initial_ticks)))
    qualified = bool(snapshot['quest']['collected']
                     and snapshot['carried'] >= snapshot['target'])
    estimated_score = 0
    if not snapshot['terminal'] and qualified and estimated_ticks is not None:
        estimated_score = score_outcome(level, {
            'status': 'COMPLETE', 'banked': snapshot['carried'],
            'ticks_used': initial_ticks - snapshot['remaining_ticks'] + estimated_ticks,
            'remaining_courage': snapshot['courage'],
        })['total']
    ready = bool(not snapshot['terminal'] and qualified and distance is not None
                 and (estimated_score >= 85 or snapshot['carried'] >= level['star_warmth_cap']))
    return {
        'exit_distance': distance, 'exit_bound': bound,
        'estimated_return_ticks': estimated_ticks,
        'estimated_return_score': estimated_score,
        'exit_distance_fraction': distance_fraction,
        'estimated_return_ticks_fraction': ticks_fraction,
        'return_time_margin_fraction': margin,
        'return_ready': ready,
    }


def navigation_metrics(snapshot, level):
    """현재 상태의 기하학적 귀환 근사입니다. 실제 귀환·점수 성공 판정은 아닙니다."""
    _, distance, bound = _exit_geometry(snapshot, level)
    return _metrics(snapshot, level, distance, bound)


def encode_navigation_observation(snapshot, level, core):
    """기존2,240값을 보존하고 지도144값·문맥4값을 추가해2,388값을 만듭니다."""
    old = legacy.encode_observation(snapshot, level, core)
    field, distance, bound = _exit_geometry(snapshot, level)
    metrics = _metrics(snapshot, level, distance, bound)
    added = np.asarray([metrics[name] for name in NAVIGATION_STATE_FIELDS], dtype=np.float32)
    return {
        'map': np.concatenate((old['map'], field[np.newaxis]), axis=0),
        'state': np.concatenate((old['state'], added)),
    }
