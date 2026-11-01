"""실제 물리 이동으로 만든 귀환 시작 상태와 공통 커리큘럼입니다."""

from collections import deque
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from . import config as cfg


def exit_distance(snapshot, level):
    from a2c_game.distance import quest_walk_distance_and_bound
    probe = {**snapshot, 'quest': {**snapshot['quest'], 'position': snapshot['exit']}}
    return quest_walk_distance_and_bound(probe, level)[0]


def snapshot_hash(snapshot):
    payload = json.dumps(snapshot, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _cell_route(snapshot):
    start, goal = tuple(snapshot['position']), tuple(snapshot['exit'])
    blocked = {tuple(door['gate']) for door in snapshot['shortcuts'] if not door['opened']}
    queue, previous = deque([start]), {start: None}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            break
        for point in ((x,y-1), (x,y+1), (x-1,y), (x+1,y)):
            px, py = point
            if (0 <= py < len(snapshot['grid']) and 0 <= px < len(snapshot['grid'][0])
                    and snapshot['grid'][py][px] != '#' and point not in blocked and point not in previous):
                previous[point] = (x,y)
                queue.append(point)
    if goal not in previous:
        raise ValueError('실제 문 상태에서 출구 경로를 찾을 수 없습니다.')
    route = []
    while goal != start:
        route.append(goal)
        goal = previous[goal]
    return route[::-1]


def build_bank(settings, recipe):
    """기준 정책으로 수집하고 실제 코어로 귀환 경로를 검증합니다. 학습하지 않습니다."""
    from .runtime import make_env, load_baseline
    from .evaluate import isolated_rng
    model = load_baseline()
    records = []
    outcomes = {}
    with isolated_rng():
        for stage in recipe['stages']:
            env = make_env(settings, stages=[stage], training=False)
            try:
                observation, info = env.reset(seed=recipe['preparation_seed'], options={'stage':stage})
                actions = []
                for _ in range(101):
                    snapshot = info['snapshot']
                    if snapshot['quest']['collected'] and snapshot['carried'] >= 14:
                        break
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, done, _, info = env.step(int(action))
                    actions.extend([int(action) if int(action) < 4 else 5] * info['executed_frames'])
                    if done:
                        raise RuntimeError(f'Stage {stage}: 기준 정책의 수집 상태 준비 실패')
                else:
                    raise RuntimeError(f'Stage {stage}: 물건·온기 14점 준비 시간 초과')
                checkpoints = []

                def remember():
                    state = env.core.snapshot()
                    checkpoints.append({'stage':stage, 'reset_seed':recipe['preparation_seed'],
                        'distance':exit_distance(state, env.level), 'prefix':list(actions),
                        'snapshot_sha256':snapshot_hash(state),
                        'native_frame':state['motion']['frame'], 'damage_at_start':state['damage_count'],
                        'remaining_ticks':state['remaining_ticks']})

                remember()
                result = None
                for goal in _cell_route(env.core.snapshot()):
                    for _ in range(80):
                        position = env.core.snapshot()['motion']['position']
                        dx, dy = goal[0]-position[0], goal[1]-position[1]
                        if max(abs(dx), abs(dy)) < 1e-8:
                            break
                        action = (3 if dx > 0 else 2) if abs(dx) > abs(dy) else (1 if dy > 0 else 0)
                        result = env.core.step(action)
                        actions.append(action)
                        if result.terminated:
                            break
                    else:
                        raise RuntimeError(f'Stage {stage}: 준비 경로에서 이동 정체')
                    if result is not None and result.terminated:
                        break
                    remember()
                outcome = env.core.snapshot()['outcome']
                if not outcome or outcome.get('status') != 'COMPLETE' or outcome['score']['stars'] != 3:
                    raise RuntimeError(f'Stage {stage}: 실제 별 3개 귀환 경로 준비 실패: {outcome}')
                outcomes[str(stage)] = outcome
                selected = {}
                for band in recipe['distance_bands']:
                    item = min(checkpoints, key=lambda c:abs(c['distance']-band))
                    selected[item['native_frame']] = item
                for index, item in enumerate(sorted(selected.values(), key=lambda c:c['distance'])):
                    item.update(id=f'{stage}_{index}', continuation=actions[len(item['prefix']):])
                    records.append(item)
            finally:
                env.close()
    return {'version':recipe['version'], 'baseline_sha256':cfg.BASELINE_SHA256,
            'preparation_seed':recipe['preparation_seed'], 'distance_bands':recipe['distance_bands'],
            'stages':recipe['stages'], 'states':records, 'verified_outcomes':outcomes}


def ensure_bank(settings, recipe):
    from .prepare import sha256
    from .runtime import write_json
    path = Path(recipe['bank_file'])
    if path.is_file():
        bank = json.loads(path.read_text(encoding='utf-8'))
    else:
        bank = build_bank(settings, recipe)
        temporary = path.with_suffix('.pending.json')
        write_json(temporary, bank)
        temporary.replace(path)
    for key, value in {'version':recipe['version'], 'baseline_sha256':cfg.BASELINE_SHA256,
                       'stages':recipe['stages'], 'distance_bands':recipe['distance_bands'],
                       'preparation_seed':recipe['preparation_seed']}.items():
        if bank.get(key) != value:
            raise ValueError('귀환 준비 상태의 조건이 다릅니다: ' + key)
    if any(not [s for s in bank['states'] if s['stage']==stage] for stage in recipe['stages']):
        raise ValueError('귀환 준비 상태에 빠진 스테이지가 있습니다.')
    return bank, sha256(path)


class ReturnProgress:
    """한 알고리즘의 네 환경이 진급 기록을 공유합니다. 두 알고리즘 간에는 공유하지 않습니다."""

    def __init__(self, bank, recipe):
        if (not 0<=recipe['practice_probability']<=1 or not 0<=recipe['review_probability']<=1
                or type(recipe['history_size']) is not int or type(recipe['required_successes']) is not int
                or not 1<=recipe['required_successes']<=recipe['history_size']):
            raise ValueError('귀환 연습 확률과 진급 조건을 확인하세요.')
        self.recipe = deepcopy(recipe)
        self.states = {s:sorted([x for x in bank['states'] if x['stage']==s],
                                key=lambda x:x['distance']) for s in recipe['stages']}
        self.frontier = {stage:0 for stage in self.states}
        self.history = {stage:deque(maxlen=recipe['history_size']) for stage in self.states}
        self.attempts = {stage:0 for stage in self.states}
        self.successes = {stage:0 for stage in self.states}
        self.templates = {}

    def choose(self, rng):
        stage = int(rng.choice(list(self.states)))
        index = self.frontier[stage]
        if index and rng.random() < self.recipe['review_probability']:
            index = int(rng.integers(index))
        return self.states[stage][index]

    def record(self, state, success):
        stage = state['stage']
        self.attempts[stage] += 1
        self.successes[stage] += int(success)
        current = self.frontier[stage]
        if state['id'] != self.states[stage][current]['id']:
            return
        self.history[stage].append(bool(success))
        if (current+1 < len(self.states[stage])
                and len(self.history[stage]) == self.recipe['history_size']
                and sum(self.history[stage]) >= self.recipe['required_successes']):
            self.frontier[stage] += 1
            self.history[stage].clear()

    def report(self):
        return {str(stage):{'frontier':self.frontier[stage],
                'distance':self.states[stage][self.frontier[stage]]['distance'],
                'state_count':len(self.states[stage]), 'attempts':self.attempts[stage],
                'three_star_successes':self.successes[stage]} for stage in self.states}


class ReturnPractice(gym.Wrapper):
    def __init__(self, env, settings, bank, recipe, progress=None):
        super().__init__(env)
        self.settings, self.recipe = deepcopy(settings), deepcopy(recipe)
        self.progress = ReturnProgress(bank, recipe) if progress is None else progress
        self.start_state = None

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.start_state = None
        if self.env.np_random.random() < self.recipe['practice_probability']:
            state = self.progress.choose(self.env.np_random)
            self.start_state = state
            if state['id'] not in self.progress.templates:
                from .runtime import make_env
                probe = make_env(self.settings, stages=[state['stage']], training=False)
                try:
                    probe.reset(seed=state['reset_seed'], options={'stage':state['stage']})
                    for action in state['prefix']:
                        result = probe.core.step(action)
                        if result.terminated:
                            raise ValueError('귀환 시작 상태를 준비하는 중 게임이 끝났습니다.')
                    if snapshot_hash(probe.core.snapshot()) != state['snapshot_sha256']:
                        raise ValueError('귀환 시작 상태의 실제 물리 재생 결과가 다릅니다.')
                    self.progress.templates[state['id']] = (deepcopy(probe.core), deepcopy(probe.level))
                finally:
                    probe.close()
            core, level = self.progress.templates[state['id']]
            self.env.core, self.env.level = deepcopy(core), deepcopy(level)
            self.env.stage = state['stage']
            snapshot = self.env.core.snapshot()
            from a2c_game.environment import TERM_NAMES, potential
            # 준비 구간의 세계 시간·유령·피해·아이템은 보존하고 학습 회차 기록만 초기화합니다.
            self.env.policy_steps = self.env.physical_frames = self.env.wall_stall_decisions = 0
            self.env.learning_return = self.env.extrinsic_return = self.env.novelty_paid = 0.
            self.env.action_counts = [0]*5
            self.env.visited_cells = {tuple(snapshot['position'])}
            self.env._novelty_states = {(self.env.stage,*snapshot['position'],True,snapshot['carried'])}
            self.env.reward_totals = dict.fromkeys(TERM_NAMES,0.)
            self.env._potential_scale = self.env.reward_config['potential_scale']
            self.env._potential = self.env._potential_scale * potential(snapshot,self.env.level)
            self.env._finished = False
            observation = self.env._observation(snapshot)
            info = self.env._info(snapshot,[],{})
        self.extra_return = 0.
        self.extra_terms = {'return_progress_reward':0.,'return_time_cost':0.,'return_success_bonus':0.}
        return observation, self._tag(info)

    def _tag(self, info):
        state = self.start_state
        return {**info, 'original_native_start':state is None,
                'start_type':'return_practice' if state else 'full_start',
                'start_distance':state['distance'] if state else None,
                'prepared_frames':state['native_frame'] if state else 0,
                'quest_collected_at_start':state is not None,
                'damage_at_start':state['damage_at_start'] if state else 0}

    def step(self, action):
        before = self.env.core.snapshot()
        observation, reward, done, truncated, info = self.env.step(action)
        snapshot = info['snapshot']
        terms = {'return_progress_reward':0., 'return_time_cost':0., 'return_success_bonus':0.}
        if (snapshot['stage'] in self.recipe['stages']
                and before['quest']['collected'] and before['carried'] >= 12):
            old_distance, new_distance = exit_distance(before,self.env.level), exit_distance(snapshot,self.env.level)
            if old_distance is not None and new_distance is not None:
                terms['return_progress_reward'] = self.recipe['return_progress_per_cell']*(old_distance-new_distance)
            terms['return_time_cost'] = self.recipe['time_per_frame']*info['executed_frames']
            outcome = snapshot.get('outcome') or {}
            if outcome.get('status')=='COMPLETE' and outcome['score']['stars']==3:
                terms['return_success_bonus'] = self.recipe['three_star_return_bonus']
        extra = sum(terms.values())
        self.extra_return += extra
        for key,value in terms.items():
            self.extra_terms[key] += value
        info = self._tag(info)
        info['reward_terms'] = {**info['reward_terms'],**terms}
        if 'episode' in info:
            episode = dict(info['episode'])
            if self.start_state:
                self.progress.record(self.start_state,episode['stars']==3 and episode['game_return'])
            episode.update({key:info[key] for key in ('start_type','start_distance','prepared_frames',
                                                     'quest_collected_at_start','damage_at_start')})
            episode.update(base_learning_return=episode['learning_return'],return_extra_reward=self.extra_return)
            episode['reward_terms'] = {**episode['reward_terms'],**self.extra_terms}
            episode['learning_return'] += self.extra_return
            episode['r'] = episode['learning_return']
            info['episode'] = episode
        return observation, reward+extra, done, truncated, info
