"""전12맵 모델 선정과 학습 진행 로그에 필요한 공통 함수."""
from collections import defaultdict, deque
from . import train as runner
BASE_LOG = runner.LearningLog

def coverage_rank(result):
    """통과 맵 수와 낮은 맵의 성공률을 평균보다 우선합니다."""
    rows = result['metrics']
    if any(set(rows[mode]) != {str(s) for s in range(1, 13)}
           for mode in ('greedy', 'stochastic')):
        raise ValueError('Selection requires both modes on all twelve stages.')
    greedy = [rows['greedy'][str(s)] for s in range(1, 13)]
    stochastic = [rows['stochastic'][str(s)] for s in range(1, 13)]
    passed = [g['mean_score'] >= 85 and g['star3_rate'] >= .9 for g in greedy]
    stable = sum(ok and s['star3_rate'] >= .9 for ok, s in zip(passed, stochastic))
    return [stable, sum(passed),
            *sorted(s['star3_rate'] for s in stochastic),
            *sorted(s['return_rate'] for s in stochastic),
            sum(s['mean_score'] for s in stochastic) / 12,
            sum(g['mean_score'] for g in greedy) / 12]

def development_goal(result, required=.9):
    coverage_rank(result)  # Reject incomplete evaluation cohorts.
    return all(row['star3_rate'] >= required and row['return_rate'] >= required
               and (mode != 'greedy' or row['mean_score'] >= 85)
               for mode in ('greedy', 'stochastic')
               for row in result['metrics'][mode].values())

class StageProgressLog(BASE_LOG):
    """실패0점도 포함하여 Stage별 점수·귀환·퀘스트·수집 현황을 출력합니다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.by_stage = defaultdict(lambda: deque(maxlen=100))

    def _on_step(self):
        proceed = super()._on_step()
        for info in self.locals['infos']:
            if info.get('episode') is not None:
                self.by_stage[int(info['stage'])].append(info['episode'])
        if self.n_calls % 800 == 0:
            progress = []
            for stage, episodes in sorted(self.by_stage.items()):
                n = len(episodes)
                row = {'stage': stage, 'episodes': n,
                       'mean_score': sum(e['score'] for e in episodes) / n,
                       'return_rate': sum(e['game_return'] for e in episodes) / n,
                       'star3_rate': sum(e['highscore_success'] for e in episodes) / n,
                       'quest_rate': sum(e['quest_collected'] for e in episodes) / n,
                       'mean_warmth': sum(e['warmth'] for e in episodes) / n,
                       'mean_post_ready_wait_steps': sum(e.get('post_ready_wait_steps', 0) for e in episodes) / n,
                       'mean_post_ready_loop_steps': sum(e.get('post_ready_loop_steps', 0) for e in episodes) / n,
                       'mean_loop_penalty_paid': sum(e.get('loop_penalty_paid', 0) for e in episodes) / n}
                progress.append(row)
                print(f'  Stage{stage}: 최근{n}판 평균 {row["mean_score"]:.1f}점 / '
                      f'귀환 {row["return_rate"]:.0%} / 85점 {row["star3_rate"]:.0%} / '
                      f'퀘스트 {row["quest_rate"]:.0%} / 온기 {row["mean_warmth"]:.1f} / '
                      f'수집 후 대기 {row["mean_post_ready_wait_steps"]:.1f}스텝 / '
                      f'귀환 정체 {row["mean_post_ready_loop_steps"]:.1f}스텝', flush=True)
            runner.append_json(self.run / 'train/stage_progress.jsonl',
                               {'steps': self.num_timesteps, 'per_stage': progress})
        return proceed
