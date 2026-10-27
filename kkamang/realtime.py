"""공통 탐사 규칙에 연속 이동을 연결한다. 행동은 1/60초, 세계 진행은 18행동마다 한 틱이며 방향 전환도 같은 시계를 쓴다."""
from copy import deepcopy
import math
from numbers import Integral, Real

from .core import Action, GameCore, StepResult

PHYSICS_HZ = 60
FRAME_SECONDS = 1 / PHYSICS_HZ
FRAMES_PER_TICK = 18
PLAYER_RADIUS = .28
CORNER_ASSIST_DISTANCE = .4
# Contact bodies exclude tails, hats, sprite padding and visual bobbing.
PLAYER_CONTACT_RADIUS = .32
GHOST_CONTACT_RADIUS = .32
# B is drawn ten pixels wider than A/C and has a wider visible body.
GHOST_B_CONTACT_RADIUS = .42
REALTIME_OBSERVATION_VERSION = 'vector_v9_realtime_corner'


class RealtimeGameCore(GameCore):
    def reset(self, seed=None, phase_overrides=None):
        super().reset(seed=seed, phase_overrides=phase_overrides)
        self._motion_position = list(self._state['position'])
        self._frames = 0
        self._frame_remainder = 0.
        self._distance = 0.
        self._moving = False
        self._chaser_stunned = False

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
        ghost_positions, chaser_position = self._ghost_motion(phase)
        snapshot['motion'] = {
            'position': list(getattr(self, '_motion_position', snapshot['position'])),
            'frame': frame,
            'phase': phase,
            'moving': getattr(self, '_moving', False),
            'distance': getattr(self, '_distance', 0.),
            'ghost_positions': ghost_positions,
            'chaser_position': chaser_position,
            'chaser_stunned': getattr(self, '_chaser_stunned', False),
        }
        return snapshot

    def _ghost_motion(self, phase=None):
        """Return the physical centers used by both rendering and contact checks."""
        if phase is None:
            phase = self._frames % FRAMES_PER_TICK / FRAMES_PER_TICK
        origins = getattr(self, '_ghost_origins', {})
        ghost_positions = {}

        for ghost in self._state['ghosts']:
            origin = origins.get(ghost['id'], ghost['position'])
            ghost_positions[ghost['id']] = [
                origin[axis] + (ghost['position'][axis] - origin[axis]) * phase
                for axis in (0, 1)
            ]

        chaser_position = None
        chaser = self._state.get('chaser')

        if chaser is not None and chaser['active']:
            origin = getattr(self, '_chaser_origin', None)
            if origin is None:
                origin = chaser['position']
            chaser_phase = 0. if getattr(self, '_chaser_stunned', False) else phase
            chaser_position = [
                origin[axis] + (chaser['position'][axis] - origin[axis]) * chaser_phase
                for axis in (0, 1)
            ]
        return ghost_positions, chaser_position

    @staticmethod
    def _bodies_contact(player_before, player_after, ghost_before, ghost_after,
                        ghost_radius=GHOST_CONTACT_RADIUS):
        # Sweep the relative segment so head-on crossings cannot tunnel through.
        start = [player_before[i] - ghost_before[i] for i in (0, 1)]
        end = [player_after[i] - ghost_after[i] for i in (0, 1)]
        delta = [end[i] - start[i] for i in (0, 1)]
        length_squared = sum(value * value for value in delta)
        t = (max(0., min(1., -sum(start[i] * delta[i] for i in (0, 1)) / length_squared))
             if length_squared else 0.)
        distance_squared = sum((start[i] + delta[i] * t) ** 2 for i in (0, 1))
        return distance_squared <= (PLAYER_CONTACT_RADIUS + ghost_radius) ** 2 + 1e-12

    def _ghost_contacts(self, previous_position, previous_ghosts, previous_chaser):
        current, chaser_position = self._ghost_motion()
        contacts = []
        for ghost in self._state['ghosts']:
            position = current[ghost['id']]
            old = self._previous_ghost_motion.get(ghost['id'], position)
            radius = (GHOST_B_CONTACT_RADIUS if ghost['id'] == 'B'
                      else GHOST_CONTACT_RADIUS)
            if self._bodies_contact(self._previous_player_motion, self._motion_position,
                                    old, position, radius):
                contacts.append({'ghost_id': ghost['id'], 'kind': 'body'})
        if chaser_position is not None:
            old = self._previous_chaser_motion
            if self._bodies_contact(self._previous_player_motion, self._motion_position,
                                    chaser_position if old is None else old, chaser_position):
                chaser = self._state['chaser']
                contacts.append({'ghost_id': chaser['id'], 'kind': 'body'})
                was_touching = old is not None and self._bodies_contact(
                    self._previous_player_motion, self._previous_player_motion, old, old)
                if not was_touching:
                    chaser['countdown'] = 4
                    self._chaser_origin = list(chaser_position)
                    self._chaser_stunned = True
        return contacts

    def _update_chaser(self, events):
        chaser = self._state['chaser']
        if (chaser is not None and chaser['active'] and self._chaser_stunned
                and self._state['quest']['collected']):
            chaser['countdown'] -= 1
            if chaser['countdown'] > 0:
                return
            self._chaser_stunned = False
            if math.dist(self._chaser_origin, chaser['position']) > 1e-10:
                # Finish the interrupted cardinal segment before planning a turn.
                chaser['countdown'] = chaser['ticks_per_move']
                return
            chaser['countdown'] = 1
        super()._update_chaser(events)

    def _blocked(self, x, y):
        state = self._state
        return (not (0 <= x < state['width'] and 0 <= y < state['height'])
                or state['grid'][y][x] == '#'
                or any(not door['opened'] and door['gate'] == [x, y]
                       for door in state['shortcuts']))

    @staticmethod
    def _body_cells(position):
        lo = [math.floor(v + .5 - PLAYER_RADIUS + 1e-10) for v in position]
        hi = [math.floor(v + .5 + PLAYER_RADIUS - 1e-10) for v in position]
        return [(x, y) for y in range(lo[1], hi[1] + 1)
                for x in range(lo[0], hi[0] + 1)]

    def _axis_destination(self, old, axis, amount):
        """Clip a cardinal movement against the entire swept body bounds."""
        direction = 1 if amount > 0 else -1
        target = list(old)
        target[axis] += amount
        target[axis] = max(0., min(self._state['width' if axis == 0 else 'height'] - 1,
                                  target[axis]))
        lo = [math.floor(min(old[i], target[i]) + .5 - PLAYER_RADIUS + 1e-10)
              for i in (0, 1)]
        hi = [math.floor(max(old[i], target[i]) + .5 + PLAYER_RADIUS - 1e-10)
              for i in (0, 1)]
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

    def _destination(self, action):
        dx, dy = ((0, -1), (0, 1), (-1, 0), (1, 0))[int(action)]
        axis = 0 if dx else 1
        amount = (dx or dy) / FRAMES_PER_TICK
        old = self._motion_position
        direct = self._axis_destination(old, axis, amount)
        if abs(direct[axis] - old[axis]) > 1e-10:
            return direct

        # A blocked turn near a corner can first slide toward a clear lane.
        # Validate the full correction and forward exit before moving at normal
        # speed. This is stateless, so desktop, RL and replay choose identically.
        desired = list(old)
        desired[axis] += amount
        side_axis = 1 - axis
        candidates = set()
        for cell in self._body_cells(desired):
            if self._blocked(*cell):
                for side in (-1, 1):
                    alignment = cell[side_axis] + side * (.5 + PLAYER_RADIUS)
                    correction = alignment - old[side_axis]
                    if 1e-10 < abs(correction) <= CORNER_ASSIST_DISTANCE + 1e-10:
                        candidates.add(correction)
        for correction in sorted(candidates, key=lambda value: (abs(value), value)):
            aligned = self._axis_destination(old, side_axis, correction)
            if abs(aligned[side_axis] - old[side_axis] - correction) > 1e-10:
                continue
            forward = self._axis_destination(aligned, axis, amount)
            if abs(forward[axis] - aligned[axis] - amount) > 1e-10:
                continue
            slide = math.copysign(min(abs(correction), abs(amount)), correction)
            return self._axis_destination(old, side_axis, slide)
        return direct

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
        self._previous_player_motion = list(old)
        self._previous_ghost_motion, self._previous_chaser_motion = self._ghost_motion()
        if action <= Action.RIGHT:
            self._motion_position = self._destination(action)
        new = self._motion_position
        self._moving = new != old
        self._distance += math.dist(old, new)
        self._state['position'] = [math.floor(round(v, 10) + .5) for v in new]
        self._frames += 1
        self._actions.append(int(action))
        world_tick = self._frames % FRAMES_PER_TICK == 0
        if world_tick:
            # Complete the previous segment, then let the grid kernel choose
            # the next destination. Phase zero starts at this physical endpoint.
            self._ghost_origins, endpoint = self._ghost_motion(phase=1.)
            if endpoint is not None:
                self._chaser_origin = endpoint
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
