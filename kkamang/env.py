"""Headless Gymnasium adapter; gameplay and learning use the same GameCore.

No Pygame import, campaign save or cosmetic reward is used here. The versioned
observation layout includes timed item states and excludes visual effects.
"""
from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .core import GameCore, OBSERVATION_VERSION
from .levels import MAGNET_DURATION_TICKS, INVINCIBILITY_DURATION_TICKS, load_level
from .realtime import RealtimeGameCore, REALTIME_OBSERVATION_VERSION, FRAMES_PER_TICK

OBSERVATION_SIZE = 620
REALTIME_OBSERVATION_SIZE = 623
TIME_BOUND = 512


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]: {value!r}")
    return int(value)


def _position(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have two coordinates")
    return (_integer(value[0], 0, 15, name + '.x'),
            _integer(value[1], 0, 8, name + '.y'))


def encode_observation(snapshot: dict) -> np.ndarray:
    """Encode the documented, fixed-slot 620-float fully observed state.

    Invalid semantic values fail visibly rather than being clipped into a valid
    Box. Stable IDs prevent collecting an item from shifting later item slots.
    Returned arrays are new and never alias internal game state.
    """
    s = snapshot
    if s['width'] != 16 or s['height'] != 9:
        raise ValueError(f'{OBSERVATION_VERSION} requires a 16 × 9 map')
    grid = s['grid']
    if len(grid) != 9 or any(len(row) != 16 or set(row) - {'#', '.'} for row in grid):
        raise ValueError('grid must have 9 rows of 16 #/. cells')
    obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            obs[y * 16 + x] = cell == '#'
    ex, ey = _position(s['exit'], 'exit')
    obs[288 + ey * 16 + ex] = 1
    x, y = _position(s['position'], 'position')
    courage = _integer(s['courage'], 0, 3, 'courage')
    protection = _integer(s['protection'], 0, 3, 'protection')
    carried = _integer(s['carried'], 0, 14, 'carried')
    initial = _integer(s['initial_ticks'], 1, TIME_BOUND, 'initial_ticks')
    remaining = _integer(s['remaining_ticks'], 0, initial, 'remaining_ticks')
    target = _integer(s['target'], 1, 14, 'target')
    if s['mode'] not in ('basic', 'relaxed'):
        raise ValueError('unknown mode')
    obs[432:437] = (x / 15, y / 8, courage / 3, protection / 3, carried / 14)
    obs[437:441] = (remaining / TIME_BOUND, initial / TIME_BOUND,
                    target / 14, int(s['mode'] == 'relaxed'))

    items = {item['id']: item for item in s['items']}
    if len(s['items']) != 10 or set(items) != {f'heat_{i:02d}' for i in range(10)}:
        raise ValueError('exactly heat_00 through heat_09 item IDs are required')
    for i in range(10):
        item = items[f'heat_{i:02d}']
        ix, iy = _position(item['position'], f'item {i}')
        value = _integer(item['value'], 1, 3, f'item {i} value')
        if value not in (1, 3) or type(item['active']) is not bool:
            raise ValueError('item value must be 1 or 3 and active must be boolean')
        if item['active']:
            obs[441 + 4*i:445 + 4*i] = (1, ix / 15, iy / 8, value / 3)

    ghosts = s['ghosts']
    if len(ghosts) > 2 or len({g['id'] for g in ghosts}) != len(ghosts):
        raise ValueError('at most two uniquely identified ghosts are supported')
    for ghost in ghosts:
        if ghost['id'] not in ('A', 'B'):
            raise ValueError('ghost slots must be A or B')
        base = 481 + 53 * ('A', 'B').index(ghost['id'])
        route = ghost['route']
        length = _integer(len(route), 1, 16, 'route length')
        index = _integer(ghost['route_index'], 0, length - 1, 'route_index')
        if ghost['direction'] not in (-1, 1) or type(ghost['direction']) is bool:
            raise ValueError('ghost direction must be -1 or +1')
        speed = _integer(
            ghost['ticks_per_move'], 1, TIME_BOUND, 'ghost ticks_per_move'
        )
        countdown = _integer(
            ghost['countdown'], 1, TIME_BOUND, 'ghost countdown'
        )
        obs[base] = 1
        for j, position in enumerate(route):
            gx, gy = _position(position, f'ghost route {j}')
            obs[base + 1 + 3*j:base + 4 + 3*j] = (gx / 15, gy / 8, 1)
        if _position(ghost['position'], 'ghost position') != tuple(route[index]):
            raise ValueError('ghost position and route_index disagree')
        # obs[base + 49:base + 53] = (length / 16, index / 15, int(ghost['direction'] == 1), countdown / 2)
        obs[base + 49:base + 53] = (
            length / 16,
            index / 15,
            int(ghost['direction'] == 1),
            countdown / TIME_BOUND,
        )
        speed_slot = 602 + ('A', 'B').index(ghost['id'])
        obs[speed_slot] = speed / TIME_BOUND

    shortcuts = s['shortcuts']
    if len(shortcuts) > 2 or len({g['id'] for g in shortcuts}) != len(shortcuts):
        raise ValueError('at most two uniquely identified shortcuts are supported')
    for shortcut in shortcuts:
        if shortcut['id'] not in ('A', 'B') or type(shortcut['opened']) is not bool:
            raise ValueError('shortcut slots must be A/B with boolean opened')
        base = 587 + 6 * ('A', 'B').index(shortcut['id'])
        tx, ty = _position(shortcut['terminal'], 'terminal')
        gx, gy = _position(shortcut['gate'], 'gate')
        if not shortcut['opened']:
            obs[144 + gy * 16 + gx] = 1
        obs[base:base+6] = (1, tx/15, ty/8, gx/15, gy/8, int(shortcut['opened']))
    q = s['quest']
    qx, qy = _position(q['position'], 'quest')
    if type(q['collected']) is not bool:
        raise ValueError('quest.collected must be boolean')
    obs[599:602] = (qx / 15, qy / 8, int(q['collected']))

    chaser = s.get('chaser')
    if chaser is not None:
        if not isinstance(chaser, dict) or chaser.get('id') != 'C':
            raise ValueError('chaser must use ID C')

        if type(chaser['active']) is not bool:
            raise ValueError('chaser.active must be boolean')

        cx, cy = _position(chaser['position'], 'chaser position')

        countdown = _integer(
            chaser['countdown'], 1, TIME_BOUND, 'chaser countdown'
        )
        speed = _integer(
            chaser['ticks_per_move'], 1, TIME_BOUND, 'chaser ticks_per_move'
        )

        obs[604:610] = (
            1,
            int(chaser['active']),
            cx / 15,
            cy / 8,
            countdown / TIME_BOUND,
            speed / TIME_BOUND,
        )

    for name, offset, duration_ticks in (('magnet', 610, MAGNET_DURATION_TICKS),
                                        ('invincibility', 615, INVINCIBILITY_DURATION_TICKS)):
        item = s.get(name)
        if item is None:
            continue
        if not isinstance(item, dict) or type(item['collected']) is not bool:
            raise ValueError(f'{name} must have a boolean collected flag')
        x, y = _position(item['position'], f'{name} position')
        duration = _integer(item['duration_ticks'], duration_ticks, duration_ticks,
                            f'{name} duration_ticks')
        remaining = _integer(item['remaining_ticks'], 0, duration, f'{name} remaining_ticks')
        if not item['collected'] and remaining:
            raise ValueError(f'uncollected {name} cannot have an active effect')
        obs[offset:offset + 5] = (1, int(item['collected']), x / 15, y / 8,
                                  remaining / duration_ticks)

    if not np.isfinite(obs).all() or not ((obs >= 0) & (obs <= 1)).all():
        raise ValueError('observation encoding escaped the declared bounds')
    return obs


def encode_realtime_observation(snapshot: dict) -> np.ndarray:
    """Add the true center and world-clock phase to the legacy cell features."""
    motion = snapshot['motion']
    position = motion['position']
    if not isinstance(position, (list, tuple)) or len(position) != 2:
        raise ValueError('motion position must have two coordinates')
    values = []
    for value, ceiling in zip(position, (15, 8)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                or not np.isfinite(value) or not 0 <= value <= ceiling):
            raise ValueError('motion position must be finite and inside the map')
        values.append(value / ceiling)
    frame = _integer(motion['frame'], 0, 2**31 - 1, 'motion frame')
    if frame // FRAMES_PER_TICK != snapshot['tick']:
        raise ValueError('world tick must match its physics frame')
    phase = frame % FRAMES_PER_TICK / FRAMES_PER_TICK
    if motion['phase'] != phase:
        raise ValueError('motion phase must match its physics frame')
    if [int(np.floor(round(value, 10) + .5)) for value in position] != snapshot['position']:
        raise ValueError('occupied cell must match the continuous center')
    return np.concatenate((encode_observation(snapshot),
                           np.array([*values, phase], dtype=np.float32)))


class MansionEnv(gym.Env):
    """Six-action, fixed-budget exploration task (without human progression).

    `max_episode_steps` is an optional *experiment* cutoff; the game's own time
    budget is a terminal failure and is included in the observation.
    Reward: terminal ``2 * COMPLETE + banked / 14``; optionally subtract
    ``safety_coefficient`` for every actual damage event (default zero).
    """

    metadata = {'render_modes': ['ansi']}
    core_class = GameCore
    observation_size = OBSERVATION_SIZE
    observation_contract = OBSERVATION_VERSION
    encode = staticmethod(encode_observation)

    def __init__(self, stage: int | str = 1, *, config: dict | None = None,
                 mode: str = 'basic', randomize_phase: bool = False,
                 safety_coefficient: float = 0.0,
                 max_episode_steps: int | None = None,
                 render_mode: str | None = None,
                 expected_observation_version: str = OBSERVATION_VERSION):
        super().__init__()
        if expected_observation_version != self.observation_contract:
            raise ValueError(f'observation version mismatch: {expected_observation_version!r}')
        if render_mode not in (None, 'ansi'):
            raise ValueError('only headless or ansi rendering is supported')
        if not np.isfinite(safety_coefficient) or safety_coefficient < 0:
            raise ValueError('safety_coefficient must be finite and nonnegative')
        if max_episode_steps is not None:
            _integer(max_episode_steps, 1, 2**31 - 1, 'max_episode_steps')
        self.render_mode = render_mode
        self.safety_coefficient = float(safety_coefficient)
        self.max_episode_steps = max_episode_steps
        self.observation_version = self.observation_contract
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(0.0, 1.0, (self.observation_size,), np.float32)
        self.core = self.core_class(deepcopy(config) if config is not None else load_level(stage),
                             mode=mode, seed=0, randomize_phase=randomize_phase)
        self.encode(self.core.snapshot())  # Fail incompatible config at construction.
        self._finished = True
        self._steps = 0

    def _info(self, snapshot: dict, events: list | None = None) -> dict:
        return {'snapshot': deepcopy(snapshot), 'events': deepcopy(events or []),
                'outcome': deepcopy(snapshot.get('outcome')),
                'config_hash': self.core.config_hash,
                'observation_version': self.observation_version,
                'controller': 'ai', 'episode_steps': self._steps}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = dict(options or {})
        unknown = set(options) - {'phase_overrides'}
        if unknown:
            raise ValueError(f'unknown reset options: {sorted(unknown)}')
        core_seed = int(self.np_random.integers(0, 2**31 - 1))
        snapshot = self.core.reset(seed=core_seed, phase_overrides=options.get('phase_overrides'))
        self._finished = False
        self._steps = 0
        return self.encode(snapshot), self._info(snapshot)

    def step(self, action):
        if self._finished:
            raise RuntimeError('call reset() before stepping a new or finished episode')
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f'action must be an integer from 0 through 5, got {action!r}')
        result = self.core.step(int(action))
        self._steps += 1
        terminated = bool(result.terminated)
        truncated = bool(not terminated and self.max_episode_steps is not None
                         and self._steps >= self.max_episode_steps)
        self._finished = terminated or truncated
        outcome = result.outcome
        utility = 0.0
        if terminated and outcome is not None:
            utility = 2.0 * (outcome['status'] == 'COMPLETE') + outcome['banked'] / 14.0
        damage_events = sum(event['type'] == 'damage' for event in result.events)
        penalty = self.safety_coefficient * damage_events
        info = self._info(result.snapshot, result.events)
        info['reward_terms'] = {'terminal_utility': utility, 'damage_penalty': penalty}
        if truncated:
            info['truncation_reason'] = 'external_step_limit'
        return self.encode(result.snapshot), float(utility - penalty), terminated, truncated, info

    def render(self):
        if self.render_mode != 'ansi':
            return None
        s = self.core.snapshot()
        grid = [list(row) for row in s['grid']]
        def mark(position, char):
            x, y = position
            grid[y][x] = char
        mark(s['exit'], 'E')
        for item in s['items']:
            if item['active']:
                mark(item['position'], str(item['value']))
        magnet = s.get('magnet')
        if magnet is not None and not magnet['collected']:
            mark(magnet['position'], 'M')
        invincibility = s.get('invincibility')
        if invincibility is not None and not invincibility['collected']:
            mark(invincibility['position'], 'I')
        for shortcut in s['shortcuts']:
            mark(shortcut['gate'], '/' if shortcut['opened'] else '|')
        for ghost in s['ghosts']:
            mark(ghost['position'], 'G')
        mark(s['position'], 'C')
        return '\n'.join(''.join(row) for row in grid) + (
            f"\nHP={s['courage']} carried={s['carried']} ticks={s['remaining_ticks']}")

    def close(self):
        pass


class RealtimeMansionEnv(MansionEnv):
    """The desktop game's free movement: six actions, each lasting 1/60 second.

    Existing grid planners/models use MansionEnv; they cannot be used as a
    correctness or difficulty evaluation of this continuous controller.
    """
    core_class = RealtimeGameCore
    observation_size = REALTIME_OBSERVATION_SIZE
    observation_contract = REALTIME_OBSERVATION_VERSION
    encode = staticmethod(encode_realtime_observation)

    def __init__(self, *args, expected_observation_version=REALTIME_OBSERVATION_VERSION, **kwargs):
        super().__init__(*args, expected_observation_version=expected_observation_version, **kwargs)
