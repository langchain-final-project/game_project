import csv
import json

from kkamang.core import GameCore
from kkamang.levels import load_level
from tools.balance_test import summarize


def test_report_retains_real_outcome_and_excludes_ai_and_corrupt_files(tmp_path):
    replay_dir = tmp_path / 'replays'
    replay_dir.mkdir()
    core = GameCore(load_level(1))
    core.abort()
    replay = core.replay_record()
    replay['controller_type'] = 'human'
    (replay_dir / 'human.json').write_text(json.dumps(replay), encoding='utf-8')
    replay['controller_type'] = 'ai'
    (replay_dir / 'ai.json').write_text(json.dumps(replay), encoding='utf-8')
    (replay_dir / 'broken.json').write_text('{', encoding='utf-8')
    csv_path, summary_path = summarize(tmp_path)
    with csv_path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]['status'] == 'ABORTED'
    assert rows[0]['score'] == '0'
    summary = summary_path.read_text(encoding='utf-8-sig')
    assert '성공 0/0 (평가 없음), 중단 1회' in summary
    assert 'ai.json' in summary and 'broken.json' in summary


def test_empty_session_reports_no_observations(tmp_path):
    _, report = summarize(tmp_path)
    assert '집계할 완료 기록이 없습니다' in report.read_text(encoding='utf-8-sig')
