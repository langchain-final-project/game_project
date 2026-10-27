"""게임 코어를 Gymnasium 환경으로 연결한다. 관측·보상·종료를 제공하며 캠페인 저장과 화면 효과는 제외한다."""
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
REALTIME_OBSERVATION_SIZE = 630
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
    """고정 식별자 슬롯의 620차원 완전 관측을 새 배열로 만든다. 잘못된 값은 보정하지 않고 거부한다."""
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
    """격자 관측에 연속 몸체 좌표·세계 위상·추격형 정지를 추가한다."""
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
    ghost_centers = motion['ghost_positions']
    expected_ids = {ghost['id'] for ghost in snapshot['ghosts']}
    if set(ghost_centers) != expected_ids:
        raise ValueError('continuous ghost slots must match the active patrol IDs')
    centers = [ghost_centers.get(ident) for ident in ('A', 'B')]
    chaser_active = snapshot['chaser'] is not None and snapshot['chaser']['active']
    if (motion['chaser_position'] is not None) != chaser_active:
        raise ValueError('continuous chaser position must match its active state')
    centers.append(motion['chaser_position'])
    continuous = []
    for center in centers:
        if center is None:
            continuous.extend((0., 0.))
            continue
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError('continuous ghost position must have two coordinates')
        for value, ceiling in zip(center, (15, 8)):
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
                    or not np.isfinite(value) or not 0 <= value <= ceiling):
                raise ValueError('continuous ghost position must be finite and inside the map')
            continuous.append(value / ceiling)
    stunned = motion['chaser_stunned']
    if type(stunned) is not bool or (stunned and not chaser_active):
        raise ValueError('chaser stun must be boolean and requires an active chaser')
    return np.concatenate((encode_observation(snapshot),
                           np.array([*values, phase, *continuous, int(stunned)], dtype=np.float32)))


class MansionEnv(gym.Env):
    """여섯 행동의 격자 탐사 환경. 게임 시간 소진은 종료, max_episode_steps는 실험용 중단이다.

    필수 물건 첫 획득 +1, 온기 첫 4점은 점당 +0.1, 성공은 10 + 점수/100,
    목숨·시간 실패는 -2이며 safety_coefficient는 실제 피해마다 차감한다.
    """

    metadata = {'render_modes': ['ansi']}
    core_class = GameCore
    observation_size = OBSERVATION_SIZE
    observation_contract = OBSERVATION_VERSION
    encode = staticmethod(encode_observation)

    def __init__(self, stage: int | str = 1, *, config: dict | None = None,
                 mode: str = 'basic', randomize_phase: bool = False,
                 safety_coefficient: float = 0.1,
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
        self.encode(self.core.snapshot())  # 관측 계약과 맞지 않는 설정은 환경 생성 시 거부한다.
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

        # 행동 전 상태: 이번 행동으로 무엇을 새로 얻었는지 비교
        before = self.core.snapshot()
        result = self.core.step(int(action))
        after = result.snapshot
        self._steps += 1

        terminated = bool(result.terminated)
        truncated = bool(
            not terminated
            and self.max_episode_steps is not None
            and self._steps >= self.max_episode_steps
        )
        self._finished = terminated or truncated

        # 1. 필수 물건을 처음 획득하면 +1
        quest_reward = float(
            not before['quest']['collected']
            and after['quest']['collected']
        )

        # 2. 처음 모은 온기 4점까지만, 1점당 +0.1
        warmth_gain = max(
            0,
            min(after['carried'], 4) - min(before['carried'], 4),
        )
        warmth_reward = 0.1 * warmth_gain

        # 3. 성공 또는 게임 규칙에 따른 실패 보상
        success_reward = 0.0
        failure_reward = 0.0
        outcome = result.outcome

        if terminated and outcome is not None:
            if outcome['status'] == 'COMPLETE':
                success_reward = 10.0 + outcome['score']['total'] / 100.0
            elif (
                outcome['status'] == 'FAILED'
                and outcome['reason'] in ('courage', 'timeout')
            ):
                failure_reward = -2.0

        # 4. 실제 피해만 차감한다. 보호 중 접촉에는 차감하지 않는다.
        damage_count = sum(
            event['type'] == 'damage'
            for event in result.events
        )
        damage_reward = -self.safety_coefficient * damage_count

        # 각 항목은 부호를 포함한 보상이다.
        reward_terms = {
            'quest': quest_reward,
            'warmth': warmth_reward,
            'success': success_reward,
            'failure': failure_reward,
            'damage': damage_reward,
        }
        reward = float(sum(reward_terms.values()))

        info = self._info(after, result.events)
        info['reward_terms'] = reward_terms

        if truncated:
            info['truncation_reason'] = 'external_step_limit'

        return self.encode(after), reward, terminated, truncated, info

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
    """데스크톱과 같은 1/60초 연속 이동 환경. 격자용 모델·경로 결과를 이 제어 방식의 난이도 검증으로 대체하지 않는다."""
    core_class = RealtimeGameCore
    observation_size = REALTIME_OBSERVATION_SIZE
    observation_contract = REALTIME_OBSERVATION_VERSION
    encode = staticmethod(encode_realtime_observation)

    def __init__(self, *args, expected_observation_version=REALTIME_OBSERVATION_VERSION, **kwargs):
        super().__init__(*args, expected_observation_version=expected_observation_version, **kwargs)
