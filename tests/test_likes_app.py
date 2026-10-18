# 테스트 케이스 호감도 관련 기능
from copy import deepcopy
import json
import os
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import pygame
import pytest

from kkamang.app import GameApp
from kkamang.campaign import SaveStore
from kkamang.likes_preview import configure_preview, sample_record

ROOT = Path(__file__).resolve().parents[1]
ROUTES = json.loads((ROOT / 'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']


@pytest.fixture
def app(tmp_path):
    instance = GameApp(save_path=tmp_path / 'save.json', headless=True, no_audio=True)
    yield instance
    instance.close()


def complete(app, stage=1):
    app.stage = stage
    app.start_run()
    route = next(r for r in ROUTES if r['level_id'] == f'stage_{stage:02d}' and r['target'] == 4)
    for action in route['actions']:
        result = app.core.advance(action, .3)
    assert result.outcome['status'] == 'COMPLETE'
    app.finish_run(result)


def test_real_completion_provider_choice_escape_and_no_star_loss(app):
    calls = []
    def provider(level, outcome, snapshot, mode):
        calls.append((level['stage'], outcome['banked'], snapshot['courage'], mode))
        return sample_record(1, (60, 10, 15) if len(calls) == 1 else (36, 25, 15))['scores']
    app.score_provider = provider
    complete(app)
    assert app.scene == 'result'
    original = deepcopy(app.campaign.data['likes']['representatives']['1'])
    complete(app)
    assert len(calls) == 2  # called even though the replay's total is lower
    assert app.scene == 'memory_choice'
    app.draw()
    assert [c['id'] for c in app.controls] == ['memory_keep', 'memory_replace']
    app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE, mod=0))
    assert app.scene == 'result'
    assert app.campaign.data['likes']['representatives']['1'] == original
    complete(app)
    before_badges = deepcopy(app.campaign.data['badges'])
    app.dispatch('memory_replace')
    assert app.campaign.data['likes']['representatives']['1']['scores']['total'] == 76
    assert app.campaign.data['likes']['best_stars']['1'] == 3
    assert app.campaign.data['badges'] == before_badges
    assert SaveStore(app.store.path).load().data == app.campaign.data


def test_real_scores_persist_and_reload_resumes_choice(app):
    complete(app)
    score = app.campaign.data['likes']['representatives']['1']['scores']
    assert score['total'] == app.last_outcome['score']['total']
    assert score['life'] == app.last_outcome['score']['courage']
    assert app.campaign.data['likes']['best_stars']['1'] == app.campaign.stars_for_stage(1)
    complete(app)
    restored = GameApp(save_path=app.store.path, headless=True, no_audio=True)
    try:
        restored.dispatch('continue')
        assert restored.scene == 'memory_choice'
        assert restored.last_outcome is None
        restored.dispatch('memory_keep')
        assert restored.scene == 'likes'
        restored.draw()
        assert restored.campaign.data['likes']['pending'] is None
    finally:
        restored.close()


def test_real_replay_keeps_star_best_and_syncs_memory(app):
    complete(app)
    first = deepcopy(app.campaign.data['likes']['representatives']['1'])
    app.start_run()
    route = next(r for r in ROUTES if r['level_id'] == 'stage_01' and r['target'] == 14)
    for action in route['actions']:
        result = app.core.advance(action, .3)
    app.finish_run(result)
    best = app.campaign.star_record(1)
    assert app.scene == 'memory_choice'
    app.dispatch('memory_keep')
    assert app.campaign.data['likes']['representatives']['1'] == first
    assert app.campaign.star_record(1) == best
    complete(app)
    app.dispatch('memory_replace')
    selected = app.campaign.data['likes']['representatives']['1']['scores']
    legacy = app.campaign.data['friend_memories']['stage_01']
    assert legacy['courage'] == selected['life']
    assert legacy['total'] == selected['total']
    assert app.campaign.star_record(1) == best
    app.draw()
    assert not {'remember_friend', 'keep_friend'} & {c['id'] for c in app.controls}


def test_save_failure_retains_choice_then_can_retry(app):
    complete(app)
    complete(app)
    with patch.object(app.store, '_atomic_write', side_effect=OSError('disk full')):
        app.dispatch('memory_replace')
    assert app.scene == 'save_error'
    selected = deepcopy(app.campaign.data['likes']['representatives'])
    app.dispatch('save_retry')
    assert app.scene == 'result'
    assert app.campaign.data['likes']['representatives'] == selected
    assert SaveStore(app.store.path).load().data['likes']['pending'] is None


def test_ending_waits_for_choice_and_preserves_seen_history(app):
    configure_preview(app, 'choice')
    app.start_ending()
    assert app.scene == 'memory_choice'
    assert app.campaign.data['likes']['seen_endings'] == []
    app.dispatch('memory_keep')
    app.dispatch('show_ending')
    assert app.ending_id == 'normal'
    assert app.scene == 'ending'
    app.start_ending()
    assert app.campaign.data['likes']['seen_endings'] == ['normal']
    app.campaign.data['likes']['representatives']['1']['scores'] = None
    app.start_ending()
    assert app.ending_id is None
    assert app.campaign.data['likes']['seen_endings'] == ['normal']


@pytest.mark.parametrize('preview', ['overview', 'choice', 'happy_all', 'sad', 'ghost_a', 'ghost_b', 'ghost_c', 'normal'])
def test_new_screens_read_only_and_controls_on_canvas(app, preview):
    configure_preview(app, preview)
    before = deepcopy(app.campaign.data)
    app.draw()
    app.draw()
    assert app.campaign.data == before
    assert len({c['id'] for c in app.controls}) == len(app.controls)
    for c in app.controls:
        assert pygame.Rect(0, 0, 1280, 720).contains(c['rect'])
