"""Continuous cat movement with the same pickup, contact and scoring rules.

One action lasts 1/60 second. E and WAIT do not translate the cat. World clocks
advance every 18 actions; changing direction never grants another world tick.
The integer position is the occupied cell; motion.position is the body center
in tile-center coordinates. No wall clock or Pygame is used here.
"""
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
        return self.snapshot()

    def snapshot(self):
        snapshot = super().snapshot()
        snapshot['motion'] = {
            'position': list(getattr(self, '_motion_position', snapshot['position'])),
            'frame': getattr(self, '_frames', 0),
            'phase': getattr(self, '_frames', 0) % FRAMES_PER_TICK / FRAMES_PER_TICK,
            'moving': getattr(self, '_moving', False),
            'distance': getattr(self, '_distance', 0.),
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
        # The short, fixed step cannot tunnel across a tile. Clamp the leading
        # body edge against any wall/closed gate touched by the candidate box.
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
        """Apply one 1/60-second command, including a mid-cell direction change."""
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
        """Headless fixed-step driver; identical commands are frame-rate invariant."""
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
