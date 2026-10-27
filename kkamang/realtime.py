"""공통 탐사 규칙에 연속 이동을 연결한다. 행동은 1/60초, 세계 진행은 18행동마다 한 틱이며 방향 전환도 같은 시계를 쓴다."""
from copy import deepcopy
import math
from numbers import Integral, Real

from .core import Action, GameCore, StepResult

PHYSICS_HZ = 60
FRAME_SECONDS = 1 / PHYSICS_HZ
FRAMES_PER_TICK = 18
PLAYER_RADIUS = .28
REALTIME_OBSERVATION_VERSION = 'vector_v6_realtime'


class RealtimeGameCore(GameCore):
    def reset(self, seed=None, phase_overrides=None):
        super().reset(seed=seed, phase_overrides=phase_overrides)
        self._motion_position = list(self._state['position'])
        self._frames = 0
        self._frame_remainder = 0.
        self._distance = 0.
        self._moving = False

        self._ghost_origins = {
            ghost['id']: list(ghost['position'])
            for ghost in self._state['ghosts']
        }
        chaser = self._state['chaser']
        self._chaser_origin = (
            list(chaser['position']) if chaser is not None else None
        )

        return self.snapshot()

    def snapshot(self):
        snapshot = super().snapshot()
        frame = getattr(self, '_frames', 0)
        phase = frame % FRAMES_PER_TICK / FRAMES_PER_TICK
        origins = getattr(self, '_ghost_origins', {})
        ghost_positions = {}

        for ghost in snapshot['ghosts']:
            origin = origins.get(ghost['id'], ghost['position'])
            ghost_positions[ghost['id']] = [
                origin[axis] + (ghost['position'][axis] - origin[axis]) * phase
                for axis in (0, 1)
            ]

        chaser_position = None
        chaser = snapshot.get('chaser')

        if chaser is not None and chaser['active']:
            origin = getattr(self, '_chaser_origin', None)
            if origin is None:
                origin = chaser['position']
            chaser_position = [
                origin[axis] + (chaser['position'][axis] - origin[axis]) * phase
                for axis in (0, 1)
            ]

        snapshot['motion'] = {
            'position': list(getattr(self, '_motion_position', snapshot['position'])),
            'frame': frame,
            'phase': phase,
            'moving': getattr(self, '_moving', False),
            'distance': getattr(self, '_distance', 0.),
            'ghost_positions': ghost_positions,
            'chaser_position': chaser_position,
        }
        return snapshot

    def _blocked(self, x, y):
        state = self._state
        return (not (0 <= x < state['width'] and 0 <= y < state['height'])
                or state['grid'][y][x] == '#'
                or any(not door['opened'] and door['gate'] == [x, y]
                       for door in state['shortcuts']))

    def _destination(self, action):
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(action)]
        axis = 0 if dx else 1
        direction = dx or dy
        old = self._motion_position
        target = list(old)
        target[axis] += direction / FRAMES_PER_TICK
        target[axis] = max(0., min(self._state['width' if axis == 0 else 'height'] - 1,
                                  target[axis]))
        # 이동 후보의 몸 가장자리를 벽·닫힌 문에 맞춰 제한해 칸을 관통하지 않게 한다.
        lo = [math.floor(v + .5 - PLAYER_RADIUS + 1e-10) for v in target]
        hi = [math.floor(v + .5 + PLAYER_RADIUS - 1e-10) for v in target]
        for y in range(lo[1], hi[1] + 1):
            for x in range(lo[0], hi[0] + 1):
                if not self._blocked(x, y):
                    continue
                edge = (x, y)[axis] - direction * (.5 + PLAYER_RADIUS)
                if direction > 0:
                    target[axis] = max(old[axis], min(target[axis], edge))
                else:
                    target[axis] = min(old[axis], max(target[axis], edge))
        return [float(round(v)) if abs(v - round(v)) < 1e-10 else v for v in target]

    def can_move(self, action):
        if (not isinstance(action, Integral) or isinstance(action, bool)
                or not 0 <= action <= 3):
            raise ValueError('movement action must be an integer 0..3')
        return self._destination(action) != self._motion_position

    def step(self, action):
        """칸 중간의 방향 전환을 포함해 1/60초 행동 하나를 적용한다."""
        self._ensure_active()
        if (not isinstance(action, Integral) or isinstance(action, bool)
                or not 0 <= action <= 5):
            raise ValueError('action must be an integer 0..5 from Action')
        old_cell = list(self._state['position'])
        old = list(self._motion_position)
        if action <= Action.RIGHT:
            self._motion_position = self._destination(action)
        new = self._motion_position
        self._moving = new != old
        self._distance += math.dist(old, new)
        self._state['position'] = [math.floor(round(v, 10) + .5) for v in new]
        self._frames += 1
        self._actions.append(int(action))
        world_tick = self._frames % FRAMES_PER_TICK == 0
        result = self._transition(
            Action.INTERACT if action == Action.INTERACT else Action.WAIT,
            previous_position=old_cell, advance_world=world_tick, record_action=False,
        )
        events = list(result.events)
        if self._moving:
            events.insert(0, {'type': 'move', 'from': old, 'to': list(new)})
        return StepResult(result.snapshot, events, result.terminated, result.outcome)

    def advance(self, action, seconds):
        """누적 시간을 고정 간격으로 처리해 같은 행동열의 결과를 화면 주기와 분리한다."""
        self._ensure_active()
        if (not isinstance(action, Integral) or isinstance(action, bool)
                or not 0 <= action <= 5):
            raise ValueError('action must be an integer 0..5 from Action')
        if (not isinstance(seconds, Real) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('seconds must be finite and nonnegative')
        self._frame_remainder += float(seconds)
        events = []
        while self._frame_remainder + 1e-12 >= FRAME_SECONDS and not self._state['terminal']:
            self._frame_remainder = max(0., self._frame_remainder - FRAME_SECONDS)
            events.extend(self.step(action).events)
        return self._result(events)

    def replay_record(self):
        record = super().replay_record()
        record.update(replay_version='2.0', control_mode='realtime',
                      physics_hz=PHYSICS_HZ,
                      observation_version=REALTIME_OBSERVATION_VERSION)
        return deepcopy(record)
