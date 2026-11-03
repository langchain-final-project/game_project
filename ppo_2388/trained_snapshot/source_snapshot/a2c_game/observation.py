"""원본 게임 상태를 지도와 상태 벡터로 바꿔 A2C 입력 2,240값을 만듭니다.

원본 630값 중 앞 432값은 지도 첫 3개 층에, 나머지 198값은 상태에 보존합니다.
추가 층과 보조값은 현재 관측 가능한 정보를 재구성하며 정답 경로나 행동은 없습니다.
"""
from __future__ import annotations

from gymnasium import spaces
import numpy as np

from kkamang.env import REALTIME_OBSERVATION_SIZE, encode_realtime_observation
from kkamang.scoring import score_outcome

OBSERVATION_VERSION = 'unified_native630_spatial_state224_v1'
MAP_FIELDS = ('walls', 'closed_gates', 'exit', 'pending_quest', 'warmth_value',
              'player', 'patrol_a', 'patrol_b', 'active_chaser', 'shortcut_terminal',
              'uncollected_magnet', 'uncollected_invincibility', 'patrol_a_route',
              'patrol_b_route')
MAP_SHAPE = (14, 9, 16)
# 배열 순서는 층·세로·가로입니다. 지도 2,016값 + 상태 224값 = 총 2,240값입니다.
NATIVE_MAP_SIZE = 432
NATIVE_STATE_SIZE = REALTIME_OBSERVATION_SIZE - NATIVE_MAP_SIZE
HELPER_FIELDS = ('remaining_time_fraction', 'star_fast_fraction', 'native_target_fraction',
                 'warmth_cap_fraction', 'warmth_progress', 'projected_score_fraction',
                 'exit_dx', 'exit_dy', 'quest_dx', 'quest_dy',
                 'can_move_up', 'can_move_down', 'can_move_left', 'can_move_right')
STATE_FIELDS = (*(f'native_{n}' for n in range(NATIVE_MAP_SIZE, REALTIME_OBSERVATION_SIZE)),
                *(f'stage_{stage:02d}' for stage in range(1, 13)), *HELPER_FIELDS)
STATE_SIZE = len(STATE_FIELDS)
OBSERVATION_DIMENSIONS = int(np.prod(MAP_SHAPE)) + STATE_SIZE
HELPER_OFFSET = NATIVE_STATE_SIZE + 12


def observation_space():
    low = np.zeros(STATE_SIZE, dtype=np.float32)
    low[HELPER_OFFSET + 6:HELPER_OFFSET + 10] = -1.
    return spaces.Dict({'map': spaces.Box(0., 1., shape=MAP_SHAPE, dtype=np.float32),
                        'state': spaces.Box(low, np.ones(STATE_SIZE, dtype=np.float32),
                                            dtype=np.float32)})


def _put_point(channel, position):
    # 소수 좌표를 주변 4칸에 나누어 담아 칸 경계 사이의 이동도 표현합니다.
    x, y = map(float, position)
    left, top = int(np.floor(x)), int(np.floor(y))
    for ix, wx in ((left, 1 - (x - left)), (left + 1, x - left)):
        for iy, wy in ((top, 1 - (y - top)), (top + 1, y - top)):
            if 0 <= ix < 16 and 0 <= iy < 9:
                channel[iy, ix] += wx * wy


def _relative(position, target):
    if target is None:
        return 0., 0.
    return tuple((float(goal) - float(actual)) / bound
                 for actual, goal, bound in zip(position, target, (15, 8)))


def projected_score(snapshot, level):
    """현재 즉시 귀환한다고 가정한 참고 점수입니다. 실제 종료 점수가 아닙니다."""
    if not snapshot['quest']['collected'] or snapshot['carried'] < snapshot['target']:
        return 0
    return score_outcome(level, {'status': 'COMPLETE', 'banked': snapshot['carried'],
        'ticks_used': snapshot['initial_ticks'] - snapshot['remaining_ticks'],
        'remaining_courage': snapshot['courage']})['total']


def encode_observation(snapshot, level, core):
    """현재 지도·수집물·유령·시간·이동 가능 상태를 신경망 입력으로 묶습니다."""
    stage = snapshot['stage']
    if type(stage) is not int or not 1 <= stage <= 12 or stage != level['stage']:
        raise ValueError('Native snapshot and level stage must agree (1..12).')
    native = encode_realtime_observation(snapshot)
    spatial = np.zeros(MAP_SHAPE, dtype=np.float32)
    # 원본 지형 3개 층을 유지하고 그 뒤에 물체의 위치와 상태를 별도 층으로 놓습니다.
    spatial[:3] = native[:NATIVE_MAP_SIZE].reshape(3, 9, 16)
    quest = snapshot['quest']
    if not quest['collected']:
        x, y = quest['position']
        spatial[3, y, x] = 1.
    for item in snapshot['items']:
        if item['active']:
            x, y = item['position']
            spatial[4, y, x] = item['value'] / 3.
    motion = snapshot['motion']
    _put_point(spatial[5], motion['position'])
    for ghost in snapshot['ghosts']:
        index = ('A', 'B').index(ghost['id'])
        _put_point(spatial[6 + index], motion['ghost_positions'][ghost['id']])
        for x, y in ghost['route']:
            spatial[12 + index, y, x] = 1.
    if motion['chaser_position'] is not None:
        _put_point(spatial[8], motion['chaser_position'])
    for shortcut in snapshot['shortcuts']:
        if not shortcut['opened']:
            x, y = shortcut['terminal']
            spatial[9, y, x] = 1.
    for name, channel in (('magnet', 10), ('invincibility', 11)):
        item = snapshot.get(name)
        if item is not None and not item['collected']:
            x, y = item['position']
            spatial[channel, y, x] = 1.
    target, cap = level['target_value'], level['star_warmth_cap']
    # can_move는 현재 한 물리 프레임의 이동 가능 여부이며, 18프레임 전체의 보장은 아닙니다.
    helpers = np.asarray([
        snapshot['remaining_ticks'] / snapshot['initial_ticks'],
        level['star_fast_ticks'] / level['time_budget_ticks'], target / 14., cap / 14.,
        max(0., min(1., (snapshot['carried'] - target) / (cap - target))),
        projected_score(snapshot, level) / 100.,
        *_relative(motion['position'], snapshot['exit']),
        *_relative(motion['position'], None if quest['collected'] else quest['position']),
        *(float(core.can_move(action)) for action in range(4)),
    ], dtype=np.float32)
    # 상태 224값 = 원본 198값 + Stage 표시 12값 + 계산한 보조 상태 14값입니다.
    state = np.concatenate((native[NATIVE_MAP_SIZE:], np.eye(12, dtype=np.float32)[stage - 1],
                            helpers))
    return {'map': spatial, 'state': state}
