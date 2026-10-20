"""분리된 사람 플레이 세션을 실행하고 보존된 재생 기록의 결과를 집계한다."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def summarize(folder: Path) -> tuple[Path, Path]:
    """완료된 사람 플레이 기록만 읽고 원본 저장과 재생 파일은 변경하지 않는다."""
    rows, skipped = [], []
    for path in sorted((folder / 'replays').glob('*.json')):
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            outcome = record.get('outcome')
            if record.get('controller_type') != 'human' or not record.get('terminal') or not isinstance(outcome, dict):
                skipped.append(path.name)
                continue
            score = outcome.get('score') or {}
            rows.append({
                'file': path.name, 'stage': record['stage'], 'mode': record['mode'],
                'config_hash': record['config_hash'], 'status': outcome['status'],
                'reason': outcome['reason'], 'banked': outcome['banked'],
                'carried': outcome['carried'], 'score': score.get('total', ''),
                'damage': outcome['damage_count'],
                'remaining_courage': outcome.get('remaining_courage', ''),
                'quest_collected': outcome.get('quest_collected', ''),
                'ticks_used': outcome['ticks_used'], 'remaining_ticks': outcome['remaining_ticks'],
                'initial_ticks': record['initial_ticks'],
                'tester_note': '',
            })
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            skipped.append(path.name)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, summary_path = folder / 'balance.csv', folder / 'summary.txt'
    fields = list(rows[0]) if rows else ['file', 'stage', 'mode', 'status', 'tester_note']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['stage'], row['mode'], row['config_hash'])].append(row)
    lines = ['밸런스 테스트 결과', '',
             '사람 플레이 기록이며 난이도 적정성이나 최적 경로를 증명하지 않습니다.',
             '방·모드·설정 해시별로 구분합니다. 중단은 성공률 분모에서 제외합니다.',
             '시간은 게임 틱입니다. 실제 플레이 경과 초가 아닙니다.',
             '원본 리플레이는 앱 정책상 세션별 최근 50개까지만 보존됩니다.',
             'CSV 재생성 시 수동 메모가 초기화되므로 메모를 쓴 CSV는 별도로 보관하세요.', '']
    for (stage, mode, config_hash), runs in sorted(groups.items()):
        finished = [r for r in runs if r['status'] != 'ABORTED']
        successes = [r for r in finished if r['status'] == 'COMPLETE']
        rate = f'{len(successes) / len(finished):.0%}' if finished else '평가 없음'
        failures = Counter(r['reason'] for r in finished if r['status'] != 'COMPLETE')
        lines.append(f'{stage:02d}번 방 / {mode} / {config_hash[:12]}')
        lines.append(f'  성공 {len(successes)}/{len(finished)} ({rate}), 중단 {len(runs)-len(finished)}회')
        lines.append(f'  실패 원인: {dict(failures) or "없음"}')
        if successes:
            mean_damage = sum(r['damage'] for r in successes) / len(successes)
            mean_left = sum(r['remaining_ticks'] for r in successes) / len(successes)
            lines.append(f'  성공 판 평균 피해 {mean_damage:.1f}회 / 남은 시간 {mean_left:.1f}틱')
        lines.append('')
    if not rows:
        lines.append('집계할 완료 기록이 없습니다. 한 판을 완료하거나 게임 메뉴에서 탐사를 중단하세요.')
    if skipped:
        lines.append('제외한 파일: ' + ', '.join(skipped))
    summary_path.write_text('\n'.join(lines) + '\n', encoding='utf-8-sig')
    return csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=range(1, 13), default=1)
    parser.add_argument('--mode', choices=['basic', 'relaxed'], default='basic')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--report', type=Path, help='기존 테스트 세션 폴더를 다시 집계')
    parser.add_argument('--headless', action='store_true', help='창 없는 실행 확인')
    args = parser.parse_args()
    if args.report:
        if not args.report.is_dir():
            parser.error('--report requires an existing session directory')
        folder = args.report
    else:
        if args.headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            os.environ['SDL_AUDIODRIVER'] = 'dummy'
        from kkamang.app import GameApp
        folder = ROOT / 'reports' / 'local' / ('balance-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
        folder.mkdir(parents=True)
        print(f'Test session: {folder}', flush=True)
        app = GameApp(save_path=folder / 'save.json', headless=args.headless,
                      no_audio=args.headless, seed=args.seed)
        # 테스트 세션만 전체 방을 열며 사용자 저장에는 적용하지 않는다.
        app.campaign.data['completed'] = list(range(1, 13))
        app.campaign.data['photos'] = list(range(1, 13))
        app.campaign.data['intro_seen'] = True
        app.campaign.data['tutorial_seen'] = True
        app.campaign.data['settings']['mode'] = args.mode
        app.mode = args.mode
        app.prepare(args.stage)
        try:
            app.run(max_frames=3 if args.headless else None)
        finally:
            for path in summarize(folder):
                print(path)
        return
    for path in summarize(folder):
        print(path)


if __name__ == '__main__':
    main()
