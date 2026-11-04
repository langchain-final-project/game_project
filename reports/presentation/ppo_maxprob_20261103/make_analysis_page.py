"""저장 로그를 근거로 관찰 사실과 가설을 구분한 발표용 분석 페이지를 만듭니다."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
PROJECT = OUTPUT.parents[2]
SOURCE = PROJECT / 'ppo_2388'
LOCAL_RUN = SOURCE / 'runs/seed_11_20261102_230005_281926_io_retry_v1'
RUN = LOCAL_RUN if LOCAL_RUN.is_dir() else SOURCE / 'trained_snapshot'
os.environ['MPLCONFIGDIR'] = str(OUTPUT / '.mplconfig')

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main():
    evidence = {}
    sources = {}
    for algorithm, base in [('A2C', SOURCE / 'shared_snapshot'), ('PPO', RUN)]:
        path = base / 'evaluation/history.jsonl'
        sources[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        groups = defaultdict(dict)
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if row['mode'] == 'greedy' and row['steps'] >= 8_000_000:
                assert row['episodes'] == 1
                groups[row['steps']][int(row['stage'])] = row
        assert len(groups) == 21
        assert all(set(rows) == set(range(1, 13)) for rows in groups.values())
        counts = [int(sum(row['star3_rate'] for row in rows.values())) for rows in groups.values()]
        exposure = json.loads((base / 'stageexposure.json').read_text(encoding='utf-8'))
        evidence[algorithm] = {
            'late_greedy_evaluations': len(groups),
            'late_passed_maps_min': min(counts), 'late_passed_maps_max': max(counts),
            'stage9_greedy_return_count': int(sum(rows[9]['return_rate'] for rows in groups.values())),
            'stage9_training_exposure': exposure['9'],
        }
    assert evidence['A2C']['stage9_greedy_return_count'] == 21
    assert evidence['PPO']['stage9_greedy_return_count'] == 0
    assert (evidence['A2C']['late_passed_maps_min'], evidence['A2C']['late_passed_maps_max']) == (11, 12)
    assert (evidence['PPO']['late_passed_maps_min'], evidence['PPO']['late_passed_maps_max']) == (3, 9)
    (OUTPUT / 'cause_analysis_evidence.json').write_text(
        json.dumps({'scope': 'greedy, 8M to 10M policy steps, 21 saved evaluations',
                    'sources': sources, 'observations': evidence,
                    'causal_attribution_confirmed': False}, ensure_ascii=False, indent=2), encoding='utf-8')

    fonts = SOURCE / 'game_source/assets/fonts'
    for name in ('NotoSansKR-Regular.ttf', 'NotoSansKR-Bold.ttf'):
        font_manager.fontManager.addfont(str(fonts / name))
    family = font_manager.FontProperties(fname=str(fonts / 'NotoSansKR-Regular.ttf')).get_name()
    plt.rcParams.update({'font.family': family, 'svg.fonttype': 'path'})
    cream, ink, muted = '#FFF9EF', '#241E3B', '#746C7D'
    purple, teal, gold = '#8452BD', '#4C9487', '#D5A35B'
    fig = plt.figure(figsize=(16, 9), dpi=180, facecolor=cream)
    canvas = fig.add_axes((0, 0, 1, 1))
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis('off')
    fig.text(.055, .94, '05 학습 과정과 PPO 비교', fontsize=15, color=purple, weight='bold')
    fig.text(.055, .855, '동일 환경에서 PPO 성적이 낮았던 이유는?',
             fontsize=31, color=ink, weight='bold')
    fig.text(.055, .794, '저장된 기록으로 확인한 현상과, 이를 설명할 수 있는 원인 후보를 구분했습니다.',
             fontsize=16, color=muted)

    cards = [
        (.055, purple, '01  반복 최적화 설정',
         '같은 새 경험 128개', 'A2C 1회  /  PPO 10회 갱신',
         '확인한 조건',
         '같은 경험을 반복 학습합니다.\n반복 횟수만큼 새로운 경험이\n추가되는 것은 아닙니다.',
         '가능한 해석',
         '제한된 경험에 치우치거나\n정책이 크게 바뀌었을 가능성'),
        (.36, teal, '02  수집 경험의 차이',
         'Stage 9 · 후반 21개 학습 시점 평가', 'A2C 21/21회  /  PPO 0/21회',
         '확인한 현상',
         '같은 환경에서도 두 정책의\n수행 결과는 달랐습니다.\n학습 중 수집한 경험도 달라집니다.',
         '가능한 해석',
         '성공 경험의 축적 차이가\n귀환 전략 학습에 영향을 줄 가능성'),
        (.665, purple, '03  학습 시점별 변동',
         '후반 85점 이상 귀환 맵 수', 'A2C 11~12개  /  PPO 3~9개',
         '확인한 현상',
         'PPO는 최고 9/12맵에서\n마지막 5/12맵으로 감소했습니다.\n학습량과 성적이 함께 늘지 않았습니다.',
         '가능한 해석',
         '12개 맵을 담당하는 공통 정책의\n갱신이 기존 성능에 영향을 줄 가능성'),
    ]
    for x, color, title, metric_title, metric, heading1, body1, heading2, body2 in cards:
        canvas.add_patch(FancyBboxPatch((x, .215), .28, .505,
                         boxstyle='round,pad=0.012,rounding_size=0.012',
                         facecolor='#FFFFFF', edgecolor='#E6D9C7', linewidth=1.3))
        fig.text(x + .012, .671, title, fontsize=19, color=color, weight='bold')
        fig.text(x + .012, .62, metric_title, fontsize=13.5, color=muted)
        fig.text(x + .012, .58, metric, fontsize=16, color=ink, weight='bold')
        fig.text(x + .012, .521, heading1, fontsize=13, color=color, weight='bold')
        fig.text(x + .012, .483, body1, fontsize=15, color=ink, va='top', linespacing=1.65)
        fig.text(x + .012, .343, heading2, fontsize=13, color=color, weight='bold')
        fig.text(x + .012, .305, body2, fontsize=15, color=ink, va='top', linespacing=1.65)

    canvas.add_patch(FancyBboxPatch((.055, .104), .89, .063,
                     boxstyle='round,pad=0.01,rounding_size=0.012',
                     facecolor='#F1EAF8', edgecolor=gold, linewidth=1.3))
    fig.text(.071, .129, '이번 설정에서는 PPO의 반복 최적화가 A2C보다 높은 고득점 귀환 성능으로 이어지지 않았습니다.',
             fontsize=17.2, color=ink, weight='bold', va='center')
    fig.text(.055, .058, '분석 범위: 최대확률 행동 · 800만~1,000만 스텝의 21개 평가 · 맵당 1판 · 학습 seed 11의 단일 실행',
             fontsize=11.5, color=muted)
    fig.text(.055, .029, '해석의 한계: 각 원인 후보의 기여도는 검증하지 않았으므로, 성능 차이의 원인으로 확정하지 않습니다.',
             fontsize=11.5, color=muted)
    fig.savefig(OUTPUT / 'cause_analysis_page.png', dpi=180, facecolor=cream)
    fig.savefig(OUTPUT / 'cause_analysis_page.svg', facecolor=cream)
    plt.close(fig)
    print(json.dumps({'image': str(OUTPUT / 'cause_analysis_page.png'),
                      'evidence': evidence}, ensure_ascii=True))


if __name__ == '__main__':
    main()
