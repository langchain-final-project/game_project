"""원래 실시간 게임을 Gymnasium 환경으로 연결하고 학습 보상을 계산합니다.

하나의 A2C 정책이 Stage1~12의 원래 출발점부터 실제 종료까지 행동합니다.
이 파일은 관측·학습 보상만 구성하며 이동, 시간, 유령, 자동 상호작용,
점수·종료 판정은 RealtimeGameCore가 담당합니다. 과거 정책은 불러오지 않습니다.
"""
from __future__ import annotations

from copy import deepcopy
import math
from collections.abc import Mapping

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from kkamang.levels import load_level
from kkamang.realtime import RealtimeGameCore
from kkamang.scoring import score_outcome
from .distance import quest_walk_distance_and_bound
from .observation import encode_observation, observation_space, OBSERVATION_VERSION
from .rollout_core import RolloutRealtimeGameCore

ACTION_NAMES = ('up', 'down', 'left', 'right', 'wait')
POLICY_TO_CORE = (0, 1, 2, 3, 5)
# 정책의 대기(4)를 게임 코어의 대기(5)에 연결합니다. 별도 E 행동은 없습니다.
REWARD_VERSION = 'native85_score_event_potential_capped_novelty_v1'
TERM_NAMES = ('quest_reward', 'pickup_reward', 'score_reward', 'highscore_bonus',
              'failure_penalty', 'damage_penalty', 'wall_stall_penalty',
              'potential_shaping', 'novelty_reward')
DEFAULT_REWARD = {'quest': .05, 'pickup_per_value': .05, 'score_scale': 1.,
                  'highscore_bonus': 1., 'failure': -1., 'damage': -.02,
                  'wall_stall': -.001, 'potential_scale': .2,
                  'novelty_per_state': .01, 'novelty_cap': .2}


def validate_stages(stages):
    if (not isinstance(stages, (list, tuple)) or not stages
            or any(type(stage) is not int or not 1 <= stage <= 12 for stage in stages)
            or len(set(stages)) != len(stages)):
        raise ValueError('Stage pool must contain unique integer stages 1..12.')


def native85(snapshot):
    outcome = snapshot.get('outcome') or {}
    return bool(outcome.get('status') == 'COMPLETE'
                and outcome.get('score', {}).get('total', 0) >= 85)


def _distance(snapshot, level, target):
    probe = {**snapshot, 'quest': {**snapshot['quest'], 'position': target}}
    return quest_walk_distance_and_bound(probe, level)


def potential(snapshot, level):
    """통행 가능한 거리로 퀘스트·수집·귀환의 진전을 수치화합니다.

    행동이나 정답 경로를 반환하지 않습니다. 예상 귀환 점수는 보조 보상의
    맥락을 고를 때만 쓰며, 실제 성공은 게임 종료 후 85점 이상인지로 판정합니다.
    """
    if snapshot['terminal']:
        return 0.
    if not snapshot['quest']['collected']:
        distance, bound = quest_walk_distance_and_bound(snapshot, level)
        return 0. if distance is None else .25 * max(0., 1. - distance / bound)
    exit_distance, exit_bound = _distance(snapshot, level, snapshot['exit'])
    estimate = 0
    if exit_distance is not None and snapshot['carried'] >= snapshot['target']:
        estimate = score_outcome(level, {'status': 'COMPLETE', 'banked': snapshot['carried'],
            'ticks_used': snapshot['initial_ticks'] - snapshot['remaining_ticks']
                          + math.ceil(exit_distance),
            'remaining_courage': snapshot['courage']})['total']
    near, reachable = 0., False
    if estimate < 85:
        for item in snapshot['items']:
            if not item['active']:
                continue
            distance, bound = _distance(snapshot, level, item['position'])
            if distance is not None:
                reachable = True
                near = max(near, item['value'] / 3. * max(0., 1. - distance / bound))
        if reachable:
            return .25 + .5 * snapshot['carried'] / level['star_warmth_cap'] + .25 * near
    return (0. if exit_distance is None
            else 1. + .5 * max(0., 1. - exit_distance / exit_bound))


def frame_reward_terms(previous, snapshot, events, reward_config):
    """퀘스트·수집·피해·실제 종료 사건마다 일회성 보상을 계산합니다."""
    cfg = reward_config
    terms = dict.fromkeys(TERM_NAMES, 0.)
    terms['quest_reward'] = cfg['quest'] * int(
        snapshot['quest']['collected'] and not previous['quest']['collected'])
    terms['pickup_reward'] = cfg['pickup_per_value'] * sum(
        event['value'] for event in events if event['type'] == 'collect')
    terms['damage_penalty'] = cfg['damage'] * sum(event['type'] == 'damage' for event in events)
    outcome = snapshot.get('outcome') or {}
    if outcome.get('status') == 'FAILED':
        terms['failure_penalty'] = cfg['failure']
    elif outcome.get('status') == 'COMPLETE':
        score = outcome.get('score', {}).get('total', 0)
        terms['score_reward'] = cfg['score_scale'] * score / 100.
        terms['highscore_bonus'] = cfg['highscore_bonus'] * int(score >= 85)
    return terms


class UnifiedMansionEnv(gym.Env):
    """정책 행동 1번을 최대 18물리 프레임으로 실행하는 공통 환경입니다."""
    metadata = {'render_modes': []}

    def __init__(self, config=None, *, stages=None, seed=0, training=False):
        super().__init__()
        self.config = deepcopy(config or {})
        env = self.config.get('environment', {})
        self.action_repeat = env.get('action_repeat', 18)
        if type(self.action_repeat) is not int or not 1 <= self.action_repeat <= 18:
            raise ValueError('action_repeat must be an integer in1..18.')
        self.mode = env.get('mode', 'basic')
        self.randomize_phase = env.get('randomize_phase', False)
        self.fast_rollout = env.get('fast_rollout', True)
        if type(self.fast_rollout) is not bool:
            raise ValueError('environment.fast_rollout must be a bool.')
        if self.mode not in ('basic', 'relaxed') or type(self.randomize_phase) is not bool:
            raise ValueError('Native mode or randomize_phase is invalid.')
        if env.get('auto_interaction', True) is not True:
            raise ValueError('The unified experiment requires native automatic interaction.')
        if env.get('max_policy_steps') is not None:
            raise ValueError('Use the original native time limit, not an artificial episode limit.')
        self.gamma = self.config.get('a2c', {}).get('gamma', .999 ** 3)
        if type(self.gamma) not in (int, float) or not 0 < self.gamma <= 1:
            raise ValueError('gamma must be finite in(0,1].')
        self.reward_config = {**DEFAULT_REWARD, **self.config.get('reward', {})}
        if set(self.reward_config) != set(DEFAULT_REWARD):
            raise ValueError('Unknown unified reward term.')
        for key, value in self.reward_config.items():
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f'Invalid finite reward setting: {key}')
            if (key in ('failure', 'damage', 'wall_stall')) != (value <= 0):
                if value != 0:
                    raise ValueError(f'Reward sign is invalid: {key}')
        self.training = bool(training)
        self.training_progress = 0.
        self.action_space = spaces.Discrete(5)
        self.observation_profile = env.get('observation_profile', 'native')
        if self.observation_profile == 'native':
            self.observation_space = observation_space()
            self.observation_version = OBSERVATION_VERSION
        elif self.observation_profile == 'exit_distance_field_v1':
            from . import navigation_observation as navigation
            self.observation_space = navigation.observation_space()
            self.observation_version = navigation.NAVIGATION_OBSERVATION_VERSION
        else:
            raise ValueError('Unknown observation_profile.')
        self.reward_version = REWARD_VERSION
        self.np_random = np.random.default_rng(seed)
        self.set_stage_pool(stages if stages is not None else env.get('stages', [1]))
        self.core = None
        self._finished = True

    def set_training_progress(self, fraction):
        """학습 진도에 따라 새 위치 방문 보너스를 줄입니다. 게임 목표는 유지합니다."""
        if type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('Training progress must be a finite fraction in0..1.')
        self.training_progress = float(fraction)

    def set_stage_pool(self, stages, weights=None):
        """다음 회차에서 고를 Stage 분포만 바꿉니다. 진행 중인 판은 유지합니다."""
        validate_stages(stages)
        if weights is None:
            probabilities = np.full(len(stages), 1. / len(stages))
        else:
            if isinstance(weights, Mapping):
                weights = [weights.get(stage, weights.get(str(stage), 0.)) for stage in stages]
            if (not isinstance(weights, (list, tuple, np.ndarray)) or len(weights) != len(stages)
                    or any(type(weight) not in (int, float) or not math.isfinite(weight) or weight < 0
                           for weight in weights) or sum(weights) <= 0):
                raise ValueError('Stage weights must be finite, nonnegative, nonzero and match stages.')
            probabilities = np.asarray(weights, dtype=np.float64) / sum(weights)
        self._pending_stages = tuple(stages)
        self._pending_probabilities = probabilities

    def _observation(self, snapshot):
        if self.observation_profile == 'exit_distance_field_v1':
            from .navigation_observation import encode_navigation_observation
            return encode_navigation_observation(snapshot, self.level, self.core)
        return encode_observation(snapshot, self.level, self.core)

    def reset(self, *, seed=None, options=None):
        """선택한 Stage의 원래 출발점에서 새 게임을 시작하고 회차 기록을 초기화합니다."""
        super().reset(seed=seed)
        options = dict(options or {})
        if set(options) - {'stage'}:
            raise ValueError('Only an original native stage may be selected at reset.')
        stage = options.get('stage')
        if stage is None:
            stage = int(self.np_random.choice(self._pending_stages, p=self._pending_probabilities))
        validate_stages([stage])
        self.stage, self.level = stage, load_level(stage)
        core_type = RolloutRealtimeGameCore if self.fast_rollout else RealtimeGameCore
        self.core = core_type(self.level, mode=self.mode,
            seed=int(self.np_random.integers(0, 2**31 - 1)), randomize_phase=self.randomize_phase,
            auto_interaction=True)
        snapshot = self.core.snapshot()
        self.policy_steps = self.physical_frames = self.wall_stall_decisions = 0
        self.learning_return = self.extrinsic_return = self.novelty_paid = 0.
        self.action_counts = [0] * 5
        self.visited_cells = {tuple(snapshot['position'])}
        self._novelty_states = {(stage, *snapshot['position'],
                                 bool(snapshot['quest']['collected']), snapshot['carried'])}
        self.reward_totals = dict.fromkeys(TERM_NAMES, 0.)
        self._potential_scale = self.reward_config['potential_scale'] if self.training else 0.
        self._potential = self._potential_scale * potential(snapshot, self.level)
        self._finished = False
        return self._observation(snapshot), self._info(snapshot, [], {})

    def _info(self, snapshot, events, terms):
        return {'stage': self.stage, 'snapshot': snapshot, 'events': events,
            'reward_terms': terms, 'native_outcome': deepcopy(snapshot.get('outcome')),
            'observation_version': self.observation_version, 'reward_version': self.reward_version,
            'original_native_start': True, 'training': self.training}

    def _novelty_reward(self, snapshot):
        # 같은 수집 상태의 같은 위치에는 중복 지급하지 않고, 회차별 상한도 적용합니다.
        if not self.training or snapshot['terminal']:
            return 0.
        key = (self.stage, *snapshot['position'], bool(snapshot['quest']['collected']), snapshot['carried'])
        if key in self._novelty_states:
            return 0.
        self._novelty_states.add(key)
        remaining = max(0., self.reward_config['novelty_cap'] - self.novelty_paid)
        coefficient = self.reward_config['novelty_per_state'] * max(.1, 1. - self.training_progress)
        value = min(coefficient, remaining)
        self.novelty_paid += value
        return value

    def step(self, action):
        """행동 실행 → 관측·보상 계산 → 실제 종료 정보 기록 순서로 처리합니다."""
        if self._finished:
            raise RuntimeError('reset() is required before step().')
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError('Policy action must be an integer in0..4.')
        previous = self.core.snapshot()
        before_distance = previous['motion']['distance']
        core_action = POLICY_TO_CORE[int(action)]
        gamma_frame = self.gamma ** (1. / self.action_repeat)
        terms, events = dict.fromkeys(TERM_NAMES, 0.), []
        raw_extrinsic = 0.
        frame_step = getattr(self.core, 'step_frame', self.core.step)
        # 한 정책 행동 안에서도 게임은 물리 프레임마다 진행되며 종료 즉시 멈춥니다.
        for offset in range(self.action_repeat):
            result = frame_step(core_action)
            snapshot = result.snapshot
            frame_terms = frame_reward_terms(previous, snapshot, result.events, self.reward_config)
            if not self.training:
                for key in ('quest_reward', 'pickup_reward', 'damage_penalty'):
                    frame_terms[key] = 0.
            for name, value in frame_terms.items():
                terms[name] += gamma_frame ** offset * value
            raw_extrinsic += sum(frame_terms.values())
            events.extend(result.events)
            self.visited_cells.add(tuple(snapshot['position']))
            previous = snapshot
            if result.terminated:
                break
        executed = offset + 1
        # 빌린 중간 프레임은 여기서 끝내고 정책·로그·외부 호출자에는 독립 복사본을 줍니다.
        snapshot = self.core.snapshot() if self.fast_rollout else result.snapshot
        wall = bool(int(action) < 4 and snapshot['motion']['distance'] - before_distance <= 1e-9)
        self.wall_stall_decisions += int(wall)
        terms['wall_stall_penalty'] = (self.reward_config['wall_stall']
                                       if self.training and wall else 0.)
        raw_extrinsic += terms['wall_stall_penalty']
        # 잠재값 보상은 할인된 다음 잠재값−현재 잠재값입니다. 실제 게임 점수와 별개입니다.
        following = 0. if result.terminated else self._potential_scale * potential(snapshot, self.level)
        terms['potential_shaping'] = gamma_frame ** executed * following - self._potential
        self._potential = following
        terms['novelty_reward'] = self._novelty_reward(snapshot)
        reward = float(sum(terms.values()))
        self.learning_return += reward
        self.extrinsic_return += raw_extrinsic
        self.policy_steps += 1
        self.physical_frames += executed
        self.action_counts[int(action)] += 1
        for name, value in terms.items():
            self.reward_totals[name] += value
        self._finished = bool(result.terminated)
        info = self._info(snapshot, events, terms)
        info.update(policy_action=int(action), core_action=core_action, executed_frames=executed,
                    native_terminated=self._finished, wall_stall=wall,
                    potential=following, novelty_paid=self.novelty_paid)
        if self._finished:
            # 학습 보상의 합과 실제 귀환 점수를 분리해 실패한 0점도 기록합니다.
            outcome = snapshot.get('outcome') or {}
            returned = outcome.get('status') == 'COMPLETE'
            score = outcome.get('score', {}).get('total', 0) if returned else 0
            info['episode'] = {'stage': self.stage, 'game_return': returned,
                'highscore_success': native85(snapshot), 'score': score,
                'stars': outcome.get('score', {}).get('stars', 0) if returned else 0,
                'quest_collected': bool(snapshot['quest']['collected']),
                'warmth': snapshot['carried'], 'damage_count': snapshot.get('damage_count', 0),
                'policy_steps': self.policy_steps, 'physical_frames': self.physical_frames,
                'wall_stall_decisions': self.wall_stall_decisions,
                'visited_cells': len(self.visited_cells), 'action_counts': list(self.action_counts),
                'reason': outcome.get('reason', 'unknown'), 'learning_return': self.learning_return,
                'extrinsic_return': self.extrinsic_return, 'novelty_paid': self.novelty_paid,
                'reward_terms': dict(self.reward_totals), 'native_outcome': deepcopy(outcome),
                'r': self.learning_return, 'l': self.policy_steps, 't': 0.}
        return self._observation(snapshot), reward, self._finished, False, info

    def close(self):
        self._finished = True
