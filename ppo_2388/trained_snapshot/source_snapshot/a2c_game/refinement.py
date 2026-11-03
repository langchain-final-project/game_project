"""같은 A2C로 원래 게임을 끝까지 실행하며 귀환 병목을 진단하는 선택 실험입니다.

추가 보상 실험에서는 선택한 거리·정체·이동 순환 비용만 더합니다.
정답 행동을 고르거나 원래 귀환·피해·시간·게임 점수를 변경하지 않습니다.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from contextlib import contextmanager
import math

import numpy as np

from kkamang.scoring import score_outcome
from .config import stage_pool
from .distance import quest_walk_distance_and_bound
from .environment import UnifiedMansionEnv
from .evaluation import isolated_rng
from . import evaluation as evaluator

REFINEMENT_VERSION = 'native85_reward_profiles_and_progress_diagnostics_v4'
SAMPLER_VERSION = 'rotating_return_focus_v1'
EXTRA_TERMS = ('return_guidance_delta', 'prolonged_idle_penalty', 'return_loop_penalty')
DEFAULT_REFINEMENT = {'return_potential_gain': 1., 'idle_after': 6,
                      'idle_penalty': 0., 'loop_window': 12,
                      'loop_min_repeats': 3, 'loop_penalty': 0.,
                      'loop_penalty_cap': .15, 'loop_safety_radius': 1.5}
REWARD_PROFILES = {
    'allocation_only': {**DEFAULT_REFINEMENT, 'experimental_combined_reward': False},
    'return_guidance': {'return_potential_gain': 4., 'idle_after': 6,
                        'idle_penalty': -.01, 'experimental_combined_reward': True},
    # 기존 보상의 배율은 유지하고, 준비 후 이동 순환 비용만 명시적으로 추가합니다.
    'loop_guard': {**DEFAULT_REFINEMENT, 'loop_penalty': -.005,
                   'sampler_version': SAMPLER_VERSION, 'focus_rotate_steps': 25_000,
                   'experimental_combined_reward': True},
}
# 같은 출발 모델·배분·순환 비용에서 귀환 거리 신호의 효과만 비교합니다.
REWARD_PROFILES['return_loop_guard'] = {
    **REWARD_PROFILES['loop_guard'], 'return_potential_gain': 4.,
}
REWARD_PROFILES['return_progress_guard'] = {
    **REWARD_PROFILES['return_loop_guard'], 'loop_detection': 'no_progress',
}


def reward_profile_settings(name):
    if name not in REWARD_PROFILES:
        raise ValueError('Choose a declared reward profile: ' + ', '.join(REWARD_PROFILES))
    return {'reward_profile': name, **deepcopy(REWARD_PROFILES[name])}


def return_diagnostics(snapshot, level):
    """통행 가능한 길 거리와 귀환 예상 점수를 계산합니다. 실제 종료 점수와는 다릅니다.

    온기를 모두 모았다면 시간·피해 때문에85점이 불가능해졌어도 귀환 준비를
    유지합니다. 더 모을 온기가 없는데 계속 수집하도록 판단하지 않기 위함입니다.
    """
    probe = {**snapshot, 'quest': {**snapshot['quest'], 'position': snapshot['exit']}}
    distance, bound = quest_walk_distance_and_bound(probe, level)
    outcome = snapshot.get('outcome') or {}
    qualified = bool(snapshot['quest']['collected']
                     and snapshot['carried'] >= snapshot['target'])
    estimate = 0
    if outcome.get('status') == 'COMPLETE':
        estimate = int(outcome.get('score', {}).get('total', 0))
    elif not snapshot['terminal'] and qualified and distance is not None:
        estimate = score_outcome(level, {'status': 'COMPLETE',
            'banked': snapshot['carried'],
            'ticks_used': snapshot['initial_ticks'] - snapshot['remaining_ticks']
                          + math.ceil(distance),
            'remaining_courage': snapshot['courage']})['total']
    ready = bool(not snapshot['terminal'] and qualified and distance is not None
                 and (estimate >= 85 or snapshot['carried'] >= level['star_warmth_cap']))
    return {'exit_distance': distance, 'exit_bound': bound,
            'estimated_return_score': estimate, 'return_ready': ready}


class _DiagnosticCore:
    """원래 게임 동작을 그대로 실행하면서 실제 프레임의 진단 정보만 관찰합니다."""
    def __init__(self, core, observe):
        self.core, self.observe = core, observe

    def __getattr__(self, name):
        return getattr(self.core, name)

    def step(self, action):
        result = self.core.step(action)
        self.observe(result.snapshot, result.events)
        return result

    def step_frame(self, action):
        # 빠른 코어의 빌린 프레임 상태는 다음 프레임 전에 즉시 읽습니다.
        result = getattr(self.core, 'step_frame', self.core.step)(action)
        self.observe(result.snapshot, result.events)
        return result


class RefinementEnv(UnifiedMansionEnv):
    """같은 정책이 원래 출발점부터 실제 게임 종료까지 행동하도록 유지합니다."""
    def __init__(self, config=None, *, stages=None, seed=0, training=False):
        super().__init__(config, stages=stages, seed=seed, training=training)
        self.refinement = {**DEFAULT_REFINEMENT, **self.config.get('refinement', {})}
        gain = self.refinement['return_potential_gain']
        threshold = self.refinement['idle_after']
        idle = self.refinement['idle_penalty']
        if (type(gain) not in (int, float) or not math.isfinite(gain) or gain < 1
                or type(threshold) is not int or threshold < 1
                or type(idle) not in (int, float) or not math.isfinite(idle) or idle > 0):
            raise ValueError('Return gain>=1, positive integer idle_after, and nonpositive idle cost are required.')
        window, repeats = self.refinement['loop_window'], self.refinement['loop_min_repeats']
        if (type(window) is not int or window < 4 or type(repeats) is not int
                or not 2 <= repeats <= window):
            raise ValueError('Loop window>=4 and repetitions in2..window are required.')
        for key in ('loop_penalty', 'loop_penalty_cap', 'loop_safety_radius'):
            value = self.refinement[key]
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or (value > 0 if key == 'loop_penalty' else value < 0)):
                raise ValueError(f'Invalid loop guard setting: {key}')
        detection = self.refinement.get('loop_detection', 'short_cycle')
        if not isinstance(detection, str) or detection not in ('short_cycle', 'no_progress'):
            raise ValueError('loop_detection must be short_cycle or no_progress.')
        if self.refinement.get('sampler_version') == SAMPLER_VERSION:
            interval = self.refinement.get('focus_rotate_steps')
            origin = self.refinement.get('sampling_start_steps', 0)
            if type(interval) is not int or interval < 1 or type(origin) is not int or origin < 0:
                raise ValueError('Positive focus rotation interval and nonnegative sampling origin are required.')
        self.reward_version = REFINEMENT_VERSION
        self._clear_diagnostics()

    def _clear_diagnostics(self):
        self.return_ready_seen = False
        self.ready_at_tick = self.ready_exit_distance = None
        self.min_exit_distance_after_ready = None
        self.post_ready_wait_steps = self.post_ready_stall_steps = 0
        self.damage_after_ready = 0
        self._idle_streak = 0
        self._return_potential = 0.
        self.post_ready_loop_steps = self.post_ready_risk_exempt_steps = 0
        self.loop_penalty_paid = 0.
        self._loop_history = deque(maxlen=self.refinement['loop_window'])
        self._loop_best_distance = self._loop_goal = None
        self._loop_no_progress_steps = 0

    def _return_loop_cost(self, previous, snapshot, diagnostic, moved, event_progress):
        """반복 또는 선택한 장기 무진전을 잡으며 짧은 대기·유령 회피는 제외합니다.

        시간은 방문 키에 넣지 않습니다. 퀘스트·온기·문 상태 변경이나 피해는
        기록을 초기화합니다. 이 기록은 보상·진단용이며 관측 차원을 바꾸지 않습니다.
        """
        goal = (bool(snapshot['quest']['collected']), snapshot['carried'],
                tuple(bool(door['opened']) for door in snapshot['shortcuts']))
        distance = diagnostic['exit_distance']
        damaged = snapshot.get('damage_count', 0) != previous.get('damage_count', 0)
        ready = diagnostic['return_ready'] and not snapshot['terminal'] and distance is not None
        changed = (not ready or event_progress or damaged or goal != self._loop_goal)
        if changed:
            self._loop_history.clear()
            self._loop_best_distance = distance if ready else None
            self._loop_no_progress_steps = 0
            self._loop_goal = goal if ready else None
        if not ready:
            return 0.
        if self._loop_best_distance is None or distance < self._loop_best_distance - .25:
            self._loop_best_distance = distance
            self._loop_no_progress_steps = 0
            self._loop_history.clear()
        elif not changed:
            self._loop_no_progress_steps += 1
        cell = tuple(snapshot['position'])
        self._loop_history.append((goal, cell))
        window = self.refinement['loop_window']
        repeated = self._loop_history.count((goal, cell)) >= self.refinement['loop_min_repeats']
        loop = (moved and self._loop_no_progress_steps >= window
                and len(self._loop_history) == window and repeated
                and len(set(self._loop_history)) >= 2)
        if self.refinement.get('loop_detection') == 'no_progress':
            # 큰 순환과 장기 대기도 같은 출구 최저거리의 정체로 봅니다.
            # 처음 window행동과 새로운 거리 진전 직후에는 비용을 주지 않습니다.
            loop = self._loop_no_progress_steps > window
        if not loop:
            return 0.
        position = snapshot['motion']['position']
        threats = list(snapshot['motion']['ghost_positions'].values())
        chaser = snapshot['motion']['chaser_position']
        if chaser is not None and not snapshot['motion'].get('chaser_stunned', False):
            threats.append(chaser)
        if any(math.dist(position, threat) <= self.refinement['loop_safety_radius'] for threat in threats):
            self.post_ready_risk_exempt_steps += 1
            return 0.
        self.post_ready_loop_steps += 1
        remaining = max(0., self.refinement['loop_penalty_cap'] - self.loop_penalty_paid)
        cost = -min(-self.refinement['loop_penalty'], remaining) if self.training else 0.
        self.loop_penalty_paid -= cost
        return cost

    def _observe_snapshot(self, snapshot, events=()):
        diagnostic = return_diagnostics(snapshot, self.level)
        if diagnostic['return_ready'] and not self.return_ready_seen:
            self.return_ready_seen = True
            self.ready_at_tick = int(snapshot['tick'])
            self.ready_exit_distance = diagnostic['exit_distance']
        if self.return_ready_seen:
            distance = diagnostic['exit_distance']
            if distance is not None:
                self.min_exit_distance_after_ready = (distance if self.min_exit_distance_after_ready is None
                    else min(self.min_exit_distance_after_ready, distance))
            self.damage_after_ready += sum(event['type'] == 'damage' for event in events)
        return diagnostic

    def _extra_potential(self, snapshot):
        if not self.training or snapshot['terminal']:
            return 0.
        diagnostic = return_diagnostics(snapshot, self.level)
        if not diagnostic['return_ready']:
            return 0.
        proximity = max(0., 1. - diagnostic['exit_distance'] / diagnostic['exit_bound'])
        # 기존 귀환 거리 보상의 기울기는0.5×potential_scale입니다.
        # 상태 기반 보상 형식을 유지하면서 지정 배율에서 부족한 부분만 더합니다.
        return ((self.refinement['return_potential_gain'] - 1.)
                * self.reward_config['potential_scale'] * .5 * proximity)

    def diagnostics(self):
        return {'return_ready_seen': self.return_ready_seen,
                'ready_at_tick': self.ready_at_tick,
                'ready_exit_distance': self.ready_exit_distance,
                'min_exit_distance_after_ready': self.min_exit_distance_after_ready,
                'post_ready_wait_steps': self.post_ready_wait_steps,
                'post_ready_stall_steps': self.post_ready_stall_steps,
                'consecutive_ready_idle_steps': self._idle_streak,
                'post_ready_loop_steps': self.post_ready_loop_steps,
                'post_ready_risk_exempt_steps': self.post_ready_risk_exempt_steps,
                'consecutive_ready_no_progress_steps': self._loop_no_progress_steps,
                'loop_penalty_paid': self.loop_penalty_paid,
                'damage_after_ready': self.damage_after_ready}

    def reset(self, *, seed=None, options=None):
        self._clear_diagnostics()
        observation, info = super().reset(seed=seed, options=options)
        self.reward_totals.update(dict.fromkeys(EXTRA_TERMS, 0.))
        self._return_potential = self._extra_potential(info['snapshot'])
        self.core = _DiagnosticCore(self.core, self._observe_snapshot)
        self._observe_snapshot(info['snapshot'])
        info.update(self.diagnostics())
        info['return_diagnostics'] = return_diagnostics(info['snapshot'], self.level)
        return observation, info

    def step(self, action):
        previous = self.core.snapshot()
        before_ready = return_diagnostics(previous, self.level)['return_ready']
        observation, reward, terminal, truncated, info = super().step(action)
        snapshot = info['snapshot']
        current = return_diagnostics(snapshot, self.level)
        moved = snapshot['motion']['distance'] - previous['motion']['distance'] > 1e-9
        event_progress = (snapshot['carried'] != previous['carried']
                          or snapshot['quest']['collected'] != previous['quest']['collected'])
        if self.return_ready_seen:
            self.post_ready_wait_steps += int(action == 4)
            self.post_ready_stall_steps += int(not moved and not event_progress)
        ready_idle = bool((before_ready or current['return_ready']) and not moved and not event_progress)
        self._idle_streak = self._idle_streak + 1 if ready_idle else 0
        idle_cost = (self.refinement['idle_penalty'] if self.training and not terminal
                     and self._idle_streak > self.refinement['idle_after'] else 0.)
        loop_cost = self._return_loop_cost(previous, snapshot, current, moved, event_progress)
        next_potential = self._extra_potential(snapshot)
        discount = self.gamma ** (info['executed_frames'] / self.action_repeat)
        guidance = discount * next_potential - self._return_potential
        self._return_potential = next_potential
        # 기본 보상·게임 사건·종료 신호는 그대로 둡니다.
        # 선언한 학습 보상만 더하고 종료 시 거리 보상도 정리합니다.
        extra = guidance + idle_cost + loop_cost
        reward += extra
        self.learning_return += extra
        self.extrinsic_return += idle_cost + loop_cost
        terms = {'return_guidance_delta': guidance, 'prolonged_idle_penalty': idle_cost,
                 'return_loop_penalty': loop_cost}
        for key, value in terms.items():
            self.reward_totals[key] += value
        info['reward_terms'].update(terms)
        info.update(self.diagnostics())
        info['return_diagnostics'] = current
        info['refinement_potential'] = next_potential
        info['reward_version'] = self.reward_version
        if 'episode' in info:
            episode = info['episode']
            episode.update(self.diagnostics(), learning_return=self.learning_return,
                extrinsic_return=self.extrinsic_return, r=self.learning_return,
                reward_terms=dict(self.reward_totals), reward_version=self.reward_version)
        return observation, float(reward), terminal, truncated, info


def _mode_metrics(metrics, stage, mode):
    data = metrics.get(mode, {})
    data = data.get('per_stage', data)
    return data.get(str(stage), data.get(stage, {}))


@contextmanager
def collect_evaluation_diagnostics():
    """기존 평가와 같은 판을 재실행하지 않고 부가 진단만 함께 기록합니다.

    관측·실제 물리·평가 보상은 같습니다. 기본 평가기가 내보내지 않는 정보만
    한시적으로 수집하고, 중단되더라도 원래 평가 환경으로 되돌립니다.
    """
    records = []
    original = evaluator.UnifiedMansionEnv

    class RecordedEnvironment(RefinementEnv):
        def step(self, action):
            result = super().step(action)
            info = result[4]
            if 'episode' in info:
                row = deepcopy(info['episode'])
                row['diagnostic_exit_bound'] = info['return_diagnostics']['exit_bound']
                records.append(row)
            return result

    try:
        evaluator.UnifiedMansionEnv = RecordedEnvironment
        yield records
    finally:
        evaluator.UnifiedMansionEnv = original


def summarize_diagnostics(records, stages, episodes, *, deterministic, seed, randomize_phase=False):
    """모든 실패를 분모에 포함해 성공률·대기·피해·귀환 접근 지표를 집계합니다.

    귀환 준비 이후 거리와 전체 시도 기준 거리를 구분합니다. 준비에 실패한 판은
    맵의 거리 상한으로 계산하여 수집 능력 하락이 귀환 개선으로 보이지 않게 합니다.
    """
    per_stage = {}
    for stage in stages:
        rows = [row for row in records if row['stage'] == stage]
        if len(rows) != episodes:
            raise ValueError(f'Diagnostic Stage{stage}: expected {episodes} complete attempts, got {len(rows)}.')
        ready = [row for row in rows if row['return_ready_seen']]
        distances = [row['min_exit_distance_after_ready'] for row in ready
                     if row['min_exit_distance_after_ready'] is not None]
        all_distances = [row['min_exit_distance_after_ready']
                         if row['return_ready_seen'] and row['min_exit_distance_after_ready'] is not None
                         else row['diagnostic_exit_bound'] for row in rows]
        per_stage[str(stage)] = {
            'stage': stage, 'episodes': episodes,
            'mean_score': sum(row['score'] for row in rows) / episodes,
            'return_rate': sum(row['game_return'] for row in rows) / episodes,
            'star3_rate': sum(row['highscore_success'] for row in rows) / episodes,
            'quest_rate': sum(row['quest_collected'] for row in rows) / episodes,
            'mean_warmth': sum(row['warmth'] for row in rows) / episodes,
            'return_ready_rate': len(ready) / episodes,
            'mean_exit_distance_after_ready': sum(distances) / len(distances) if distances else None,
            'min_exit_distance_after_ready': min(distances) if distances else None,
            'mean_exit_distance_all_attempts': sum(all_distances) / episodes,
            'mean_post_ready_wait_steps': sum(row['post_ready_wait_steps'] for row in rows) / episodes,
            'mean_post_ready_stall_steps': sum(row['post_ready_stall_steps'] for row in rows) / episodes,
            'mean_post_ready_loop_steps': sum(row.get('post_ready_loop_steps', 0) for row in rows) / episodes,
            'mean_post_ready_risk_exempt_steps': sum(row.get('post_ready_risk_exempt_steps', 0) for row in rows) / episodes,
            'mean_loop_penalty_paid': sum(row.get('loop_penalty_paid', 0.) for row in rows) / episodes,
            'mean_damage_after_ready': sum(row['damage_after_ready'] for row in rows) / episodes,
            'reasons': {reason: sum(row['reason'] == reason for row in rows)
                        for reason in {row['reason'] for row in rows}},
        }
    return {'version': REFINEMENT_VERSION, 'original_native_start': True,
            'deterministic': deterministic, 'seed': seed, 'randomize_phase': randomize_phase,
            'training_updates_added': 0, 'per_stage': per_stage, 'episodes': records,
            'denominator': 'all native attempts; conditional distance explicitly labelled',
            'scope': 'fixed-phase development checks, not unseen-phase generalization'}


def stage_sampler(config, state, recent_metrics=None):
    """4개 환경을 복습1개·집중2개·다른 맵1개로 배분하는 선택 실험입니다.

    recent_metrics는 G/S 방식별 Stage 평가값입니다.
    새 맵을 도입했다고 다른 맵이 해결됐다고 판정하지 않습니다.
    """
    steps = int(state.get('steps', state.get('timesteps', 0)))
    pool = stage_pool(config, steps)
    metrics = recent_metrics or state.get('refinement_metrics') or {}
    mastered, stalled = [], []
    exposure = state.get('stage_exposure', {})
    for stage in pool:
        greedy = _mode_metrics(metrics, stage, 'greedy')
        stochastic = _mode_metrics(metrics, stage, 'stochastic')
        if (greedy.get('star3_rate', 0.) >= .9 and greedy.get('return_rate', 0.) >= .9
                and stochastic.get('star3_rate', 0.) >= .8 and stochastic.get('return_rate', 0.) >= .8):
            mastered.append(stage)
        seen = exposure.get(str(stage), exposure.get(stage, {})).get('learner_steps', 0)
        row = stochastic or greedy
        # 첫 저득점 귀환이나 가끔 성공한 것은 진전입니다.
        # 고득점 정책이 확보되기 전까지 해당 맵의 집중 학습 후보를 유지합니다.
        if (stage not in mastered and seen >= 50_000 and row.get('quest_rate', 0.) >= .8
                and row.get('mean_warmth', 0.) >= 12):
            stalled.append(stage)
    old = pool[:-1] or pool
    retention = mastered or old
    remaining = [stage for stage in pool if stage not in mastered]
    settings = config.get('refinement', {})
    if settings.get('sampler_version') == SAMPLER_VERSION:
        interval = settings.get('focus_rotate_steps', 25_000)
        origin = settings.get('sampling_start_steps', state.get('refinement_start_steps', 0))
        if type(interval) is not int or interval < 1 or type(origin) is not int or origin < 0:
            raise ValueError('Invalid focus rotation interval/origin.')
        # 수집률이 일시 하락한 실패 맵도 배제하지 않습니다(예: Stage9).
        candidates = sorted(remaining or pool)
        epoch = max(0, steps - origin) // interval
        # 두 집중 환경에 서로 다른 실패 맵을 배정해 한 맵에 영구 고정되지 않게 합니다.
        first = candidates[(2 * epoch) % len(candidates)]
        second = candidates[(2 * epoch + 1) % len(candidates)]
        exploration = [stage for stage in remaining if stage not in (first, second)] or pool
        return [list(retention), [first], [second], list(exploration)]
    focus = min(stalled) if stalled else (remaining[-1] if remaining else pool[-1])
    exploration = [stage for stage in remaining if stage != focus] or pool
    return [list(retention), [focus], [focus], list(exploration)]


def evaluate_diagnostics(model, config, stages, episodes=1, seed=25000,
                         deterministic=True):
    """원래 게임을 한 판씩 끝까지 실행하여 귀환·안전 지표를 진단합니다.

    실패한 판을 재시도하거나 제외하지 않고 추가 학습 보상은 끕니다.
    고득점 성능 평가를 대체하지 않는 읽기 전용 보조 진단입니다.
    """
    stages = list(stages)
    if (not stages or len(set(stages)) != len(stages)
            or any(type(stage) is not int or not 1 <= stage <= 12 for stage in stages)
            or type(episodes) is not int or episodes < 1
            or type(seed) is not int or not 0 <= seed < 2**31 - 13_000_000
            or type(deterministic) is not bool):
        raise ValueError('Invalid original-start diagnostic stage, episodes, seed or mode.')
    policy = getattr(model, 'policy', None)
    previous_mode = getattr(policy, 'training', None)
    records = []
    try:
        with isolated_rng(seed):
            if policy is not None and hasattr(policy, 'set_training_mode'):
                policy.set_training_mode(False)
            for stage in stages:
                env = RefinementEnv(config, stages=[stage], training=False)
                try:
                    if (hasattr(model, 'observation_space') and model.observation_space != env.observation_space
                            or hasattr(model, 'action_space') and model.action_space != env.action_space):
                        raise ValueError('Diagnostic observation/action space differs from the policy.')
                    for index in range(episodes):
                        with isolated_rng(seed + stage * 1_000_000 + index):
                            observation, _ = env.reset(seed=seed + stage * 10_000 + index)
                            done = False
                            while not done:
                                action, _ = model.predict(observation, deterministic=deterministic)
                                observation, _, done, truncated, info = env.step(int(np.asarray(action).item()))
                                if truncated:
                                    raise RuntimeError('A diagnostic ended before its native game ending.')
                        episode = deepcopy(info['episode'])
                        episode['episode_seed'] = seed + stage * 10_000 + index
                        episode['diagnostic_exit_bound'] = info['return_diagnostics']['exit_bound']
                        records.append(episode)
                finally:
                    env.close()
    finally:
        if previous_mode is not None and hasattr(policy, 'set_training_mode'):
            policy.set_training_mode(previous_mode)
    return summarize_diagnostics(records, stages, episodes, deterministic=deterministic,
                                 seed=seed, randomize_phase=config.get('environment', {}).get('randomize_phase', False))
