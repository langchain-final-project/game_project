"""저장된 greedy 평가만 사용해 발표용 학습 곡선과 원자료를 만듭니다."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
SOURCE = PROJECT / 'ppo_2388'
LOCAL_RUN = SOURCE / 'runs/seed_11_20261102_230005_281926_io_retry_v1'
RUN = LOCAL_RUN if LOCAL_RUN.is_dir() else SOURCE / 'trained_snapshot'
os.environ['MPLCONFIGDIR'] = str(OUTPUT / '.mplconfig')

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, StrMethodFormatter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def history(path, algorithm):
    groups = defaultdict(dict)
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row['mode'] != 'greedy':
            continue
        key = (int(row['steps']), row['evaluation_label'])
        stage = int(row['stage'])
        if stage in groups[key] and groups[key][stage] != row:
            raise ValueError(f'Conflicting repeated evaluation: {algorithm}, {key}, {stage}')
        groups[key][stage] = row
    result = []
    for (steps, label), rows in sorted(groups.items()):
        assert set(rows) == set(range(1, 13)), (algorithm, label)
        assert all(row['episodes'] == 1 for row in rows.values())
        passed = sum(row['return_rate'] == 1 and row['mean_score'] >= 85 for row in rows.values())
        assert passed == sum(row['star3_rate'] for row in rows.values())
        result.append({'algorithm': algorithm, 'steps': steps,
                       'evaluation_label': label, 'passed_maps': passed,
                       'returned_maps': sum(row['return_rate'] == 1 for row in rows.values()),
                       'mean_score': sum(row['mean_score'] for row in rows.values()) / 12})
    assert len(result) == 101 and result[0]['steps'] == 0
    return result


def main():
    fonts = SOURCE / 'game_source/assets/fonts'
    for name in ('NotoSansKR-Regular.ttf', 'NotoSansKR-Bold.ttf'):
        font_manager.fontManager.addfont(str(fonts / name))
    family = font_manager.FontProperties(fname=str(fonts / 'NotoSansKR-Regular.ttf')).get_name()
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'svg.fonttype': 'path', 'font.size': 15})

    paths = {'A2C': SOURCE / 'shared_snapshot/evaluation/history.jsonl',
             'PPO': RUN / 'evaluation/history.jsonl'}
    curves = {name: history(path, name) for name, path in paths.items()}
    a2c_model = read_json(SOURCE / 'shared_snapshot/selected_evaluation.json')
    ppo_model = read_json(RUN / 'checks/round_096.json')
    comparison = {'A2C': next(row for row in curves['A2C'] if row['steps'] == a2c_model['steps']),
                  'PPO': next(row for row in curves['PPO'] if row['steps'] == ppo_model['steps'])}
    best_ppo = max(curves['PPO'], key=lambda row: (row['passed_maps'], row['mean_score']))
    assert best_ppo == comparison['PPO']
    assert comparison['A2C']['passed_maps'] == 12
    assert comparison['PPO']['passed_maps'] == 9
    assert curves['PPO'][-1]['passed_maps'] == 5
    assert curves['A2C'][-1]['passed_maps'] == 12

    with (OUTPUT / 'learning_curve_data.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curves['A2C'][0]))
        writer.writeheader()
        writer.writerows(curves['A2C'] + curves['PPO'])

    cream, ink, muted = '#FFF9EF', '#241E3B', '#746C7D'
    purple, teal = '#8452BD', '#4C9487'
    fig, ax = plt.subplots(figsize=(15, 8), dpi=220)
    fig.patch.set_facecolor(cream)
    ax.set_facecolor(cream)
    fig.subplots_adjust(left=.095, right=.945, top=.79, bottom=.20)
    fig.text(.095, .945, '학습 진행에 따른 85점 이상 귀환 맵 수',
             fontsize=25, fontweight='bold', color=ink, ha='left')
    fig.text(.095, .882, '최대확률 행동 평가  ·  전12맵 각각 1판  ·  학습 seed 11',
             fontsize=15.5, color=muted, ha='left')

    ax.set_xlim(0, 1070)
    ax.set_ylim(-.2, 13.8)
    ax.set_xticks(range(0, 1001, 200))
    ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.set_yticks(range(0, 13, 2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(axis='y', which='major', color='#DED8D0', linewidth=.8)
    ax.tick_params(axis='both', labelsize=14, length=0, pad=10, colors=muted)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#B5ACA2')
        ax.spines[side].set_linewidth(.9)
    ax.set_xlabel('누적 학습 스텝 (만)', fontsize=17, color=ink, labelpad=16)
    ax.set_ylabel('85점 이상 귀환한 맵 수 (개)', fontsize=17, color=ink, labelpad=17)
    ax.axhline(12, color='#C6B79B', linewidth=1.2, linestyle=(0, (5, 5)), zorder=1)
    ax.text(8, 12.23, '전12맵 목표', color='#8B7C63', fontsize=13)

    for name, color in (('PPO', teal), ('A2C', purple)):
        rows = curves[name]
        ax.plot([row['steps'] / 10_000 for row in rows],
                [row['passed_maps'] for row in rows], color=color,
                linewidth=2.5, label=name, zorder=3 if name == 'A2C' else 2,
                marker='o', markersize=3.1, markeredgewidth=0)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index('A2C'), labels.index('PPO')]
    ax.legend([handles[index] for index in order], ['A2C', 'PPO'],
              loc='upper left', bbox_to_anchor=(.0, 1.12), ncol=2,
              frameon=False, fontsize=16, handlelength=2.5, columnspacing=1.8)

    for name, color, label, text_position in (
        ('A2C', purple, '발표 비교 모델  12/12', (675, 13.05)),
        ('PPO', teal, '발표 비교 모델  9/12', (710, 10.45)),
    ):
        row = comparison[name]
        point = (row['steps'] / 10_000, row['passed_maps'])
        ax.scatter(*point, s=105, facecolor=cream, edgecolor=color, linewidth=2.4, zorder=6)
        ax.annotate(label, xy=point, xytext=text_position, fontsize=14.5,
                    color=color, fontweight='bold', ha='left', va='center',
                    arrowprops={'arrowstyle': '-', 'color': color, 'lw': 1.2,
                                'connectionstyle': 'angle3,angleA=0,angleB=90'})
    ax.scatter(1000, 5, s=75, color=teal, zorder=6)
    ax.annotate('종료 5/12', xy=(1000, 5), xytext=(795, 2.3),
                fontsize=14.5, color=teal, fontweight='bold', ha='left',
                arrowprops={'arrowstyle': '-', 'color': teal, 'lw': 1.2})
    fig.text(.095, .043, '실선은 저장된 평가값을 연결한 것임 · 스텝은 4개 환경의 정책 결정 수 합계',
             fontsize=12.5, color=muted, ha='left')

    fig.savefig(OUTPUT / 'maxprob_learning_curve.png', dpi=220, facecolor=cream)
    fig.savefig(OUTPUT / 'maxprob_learning_curve_transparent.png', dpi=220, transparent=True)
    fig.savefig(OUTPUT / 'maxprob_learning_curve.svg', facecolor=cream)
    plt.close(fig)
    metadata = {
        'metric': 'greedy evaluation: maps returned with score >= 85 out of 12',
        'training_seed': 11, 'episodes_per_stage_per_checkpoint': 1,
        'fixed_native_ghost_phase': True,
        'source_files': {str(path.relative_to(PROJECT)): sha(path) for path in paths.values()},
        'evaluation_points': {name: len(rows) for name, rows in curves.items()},
        'comparison_models': comparison,
        'last_evaluations': {name: rows[-1] for name, rows in curves.items()},
        'smoothing': False, 'seed_average_or_uncertainty_band': False,
    }
    (OUTPUT / 'graph_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'points': metadata['evaluation_points'],
                      'comparison': {name: row['passed_maps'] for name, row in comparison.items()},
                      'last': {name: rows[-1]['passed_maps'] for name, rows in curves.items()}}, ensure_ascii=True))


if __name__ == '__main__':
    main()
