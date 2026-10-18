"""v0.2 mission, replay, observation, save and rendering boundary checks."""
import json,os
from pathlib import Path
from copy import deepcopy
import pytest
import numpy as np
from kkamang.core import GameCore
from kkamang.levels import load_level,load_levels,validate_level
from kkamang.env import MansionEnv,encode_observation,OBSERVATION_SIZE
from kkamang.campaign import Campaign
from tools.replay_validation import replay
ROOT=Path(__file__).resolve().parents[1]

def test_quest_only_without_heat_blocks_return():
    g=GameCore(load_level(1));g._state['quest']['collected']=True
    r=g.step(4)
    assert not r.terminated and r.outcome is None
    assert r.events[0]['missing_heat']==4 and not r.events[0]['missing_quest']

def test_quest_pickup_once_reset_and_snapshot_isolation():
    g=GameCore(load_level(1));g._state['position']=g.snapshot()['quest']['position']
    a=g.step(4);b=g.step(4)
    assert a.snapshot['quest']['collected'] and sum(e['type']=='quest_collect' for e in a.events)==1
    assert not any(e['type']=='quest_collect' for e in b.events)
    assert b.snapshot['carried']==0 and b.snapshot['tick']==2
    snap=g.snapshot();snap['quest']['collected']=False
    assert g.snapshot()['quest']['collected']
    assert not g.reset()['quest']['collected']

def test_quest_observation_cannot_alias_uncollected_state():
    g=GameCore(load_level(1));s=g.snapshot();a=encode_observation(s)
    s['quest']['collected']=True;b=encode_observation(s)
    assert OBSERVATION_SIZE==620 and b[601]==1 and a[601]==0
    np.testing.assert_array_equal(a[:601],b[:601])
    assert a[599]==pytest.approx(s['quest']['position'][0]/15)
    assert a[600]==pytest.approx(s['quest']['position'][1]/8)

def test_campaign_rejects_forged_completion_without_object_atomically():
    c=Campaign();before=deepcopy(c.data)
    with pytest.raises(ValueError):c.settle(load_level(1),{'status':'COMPLETE','banked':14},'hash','basic','fake')
    assert c.data==before

@pytest.mark.parametrize('stage',range(1,13))
def test_safe_route_needs_no_rich_items_or_gate(stage):
    routes=json.loads((ROOT/'validation/safe_routes.json').read_text())
    route=next(r for r in routes if r['level_id']==f'stage_{stage:02}')
    outcome=replay(route)
    assert outcome['quest_collected'] and outcome['banked'] >= 4
    g=GameCore(load_level(stage))
    for a in route['actions']:s=g.step(a).snapshot
    assert all(i['active'] for i in s['items'] if i['value']==3)
    assert not any(x['opened'] for x in s['shortcuts'])

@pytest.mark.parametrize('stage',range(1,13))
def test_human_core_and_ai_env_equal_for_rich_route(stage):
    routes=json.loads((ROOT/'validation/reference_routes.json').read_text())['routes']
    route=next(r for r in routes if r['level_id']==f'stage_{stage:02}' and r['target']==14)
    env=MansionEnv(stage);g=GameCore(load_level(stage));env.reset(seed=0)
    for a in route['actions']:
        human=g.step(a);obs,reward,term,trunc,info=env.step(a)
        assert info['snapshot']==human.snapshot and term==human.terminated and not trunc
        np.testing.assert_array_equal(obs,encode_observation(human.snapshot))
        if not term: assert reward==0
    assert reward==3 and human.outcome['quest_collected']
    assert human.outcome['damage_count']==0

@pytest.mark.parametrize('field',['exit','gate','heat'])
def test_quest_overlap_rejected(field):
    l=load_level(5)
    l['quest']['position']=l['exit'] if field=='exit' else l['shortcuts'][0]['gate'] if field=='gate' else l['items'][0]['position']
    with pytest.raises(ValueError):validate_level(l)

def test_png_cat_directions_walk_and_fixed_appearance():
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    import pygame
    from kkamang.ui.illustration import cat_frame
    raw=lambda **kw:pygame.image.tobytes(cat_frame(160,**kw),'RGBA')
    assert len({raw(direction=d) for d in ['up','down','left','right']})==4
    for d in ['up','down','left','right']:
        assert raw(direction=d,mood='walk',phase=2)!=raw(direction=d,mood='walk',phase=6)
    # The first PNG set has no expressions or scarf colour variants yet.
    assert len({raw(mood=m) for m in ['idle','happy','collect','surprised']})==1
    assert raw(scarf='default')==raw(scarf='peach')==raw(scarf='lavender')


def test_hurt_cat_can_walk_and_walk_phase_depends_on_progress():
    import pygame
    from unittest.mock import patch
    from kkamang.ui.illustration import cat, cat_frame
    raw = lambda **kw: pygame.image.tobytes(cat_frame(160, **kw), 'RGBA')
    assert raw(mood='surprised', walking=True, phase=2) != raw(
        mood='surprised', walking=True, phase=6)
    assert raw(mood='surprised', walking=True, phase=2) != raw(
        mood='surprised', walking=False, phase=2)
    surface = pygame.Surface((200, 200))
    with patch('kkamang.ui.illustration.cat_frame', wraps=cat_frame) as frame:
        cat(surface, (100, 180), time=1., mood='surprised',
            walking=True, walk_progress=.125)
        first = frame.call_args
        cat(surface, (100, 180), time=100., mood='surprised',
            walking=True, walk_progress=.125)
        assert frame.call_args == first
        cat(surface, (100, 180), time=100., mood='surprised',
            walking=True, walk_progress=.25)
        assert frame.call_args.args[2] != first.args[2]

def test_home_restore_and_cosmetics_do_not_mutate_campaign_or_core():
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    import pygame
    from kkamang.ui.illustration import shelter
    surface=pygame.Surface((1280,720));g=GameCore(load_level(7));before=g.snapshot()
    c=Campaign();data=deepcopy(c.data)
    shelter(surface,[],c.data['cosmetics'],time=1);a=pygame.image.tobytes(surface,'RGB')
    shelter(surface,list(range(1,13)),{'scarf':'peach','rug':'leaves','frame':'stars'},time=1)
    assert a!=pygame.image.tobytes(surface,'RGB')
    assert c.data==data and g.snapshot()==before


def test_observation_contract_matches_item_layout():
    from kkamang.env import OBSERVATION_VERSION
    contract = json.loads((ROOT / 'data/levels/observation_contract.json').read_text(encoding='utf-8'))
    assert contract['observation_version'] == OBSERVATION_VERSION
    assert contract['shape'] == [OBSERVATION_SIZE]
    assert sum(field['length'] for field in contract['fields']) == OBSERVATION_SIZE
    assert contract['fields'][-2]['offset'] == 610
    assert contract['fields'][-1]['offset'] == 615


def test_magnet_flights_animate_without_mutating_rules_pause_and_reset(tmp_path):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from unittest.mock import patch
    from kkamang.app import GameApp
    from kkamang.ui import field
    app = GameApp(save_path=tmp_path / 'magnet.json', headless=True, no_audio=True)
    try:
        app.campaign.data['completed'] = [1]
        app.stage = 2
        app.start_run()
        app.core._state['position'] = [3, 3]
        app.core._motion_position = [3, 3]
        app.previous = app.core.snapshot()
        result = app.core.advance(1, .3)
        app._process_result(result)
        assert result.snapshot['carried'] == 2
        assert len(app.magnet_flights) == 2
        assert 'icons/items/magnet_item.png' == field.FIELD_IMAGES['magnet']
        before = app.core.snapshot()
        with patch.object(field, '_draw_warmth', wraps=field._draw_warmth) as draw:
            app.draw()
            origins = [call.args[2] for call in draw.call_args_list
                       if call.kwargs.get('show_value') is False]
        app.advance(.12)
        with patch.object(field, '_draw_warmth', wraps=field._draw_warmth) as draw:
            app.draw()
            moving = [call.args[2] for call in draw.call_args_list
                      if call.kwargs.get('show_value') is False]
        assert origins != moving and len(moving) == 2
        assert {k: v for k, v in app.core.snapshot().items() if k != 'motion'} == {
            k: v for k, v in before.items() if k != 'motion'}
        ages = [flight['age'] for flight in app.magnet_flights]
        paused = app.core.snapshot()
        app.transition('pause')
        app.advance(1.)
        assert [flight['age'] for flight in app.magnet_flights] == ages
        assert app.core.snapshot() == paused
        app.transition('game')
        app.advance(.23)
        assert not app.magnet_flights
        app.start_run()
        assert not app.magnet_flights
        assert not app.core.snapshot()['magnet']['collected']
    finally:
        app.close()


def test_safe_planner_rejects_remote_rich_collection_not_just_rich_tiles():
    from tools.evaluate import safe_path
    from tools.plan_routes import rollout
    config = load_level(1)
    config['magnet'] = {'position': [3, 3], 'duration_ticks': 13}
    game = GameCore(config)
    game._state['position'] = [3, 2]
    assert rollout(game, [1]) is not None
    assert rollout(game, [1], forbid_rich=True) is None
    assert safe_path(game.snapshot(), (3, 3), interact=False, forbid_rich=True) is None


def test_planner_finds_a_complete_magnet_route_without_mutating_input():
    from tools.plan_routes import plan
    game = GameCore(load_level(2))
    initial = game.snapshot()
    actions = plan(game, target=14, width=8)
    assert actions and game.snapshot() == initial
    for action in actions:
        result = game.step(action)
        assert not any(event['type'] == 'contact' for event in result.events)
    assert result.outcome['status'] == 'COMPLETE'
    assert result.outcome['banked'] == 14
    assert game.snapshot()['magnet']['collected']


@pytest.mark.parametrize('stage', [2, 7])
def test_magnet_reference_routes_use_effect_without_contact(stage):
    routes = json.loads((ROOT / 'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']
    route = next(r for r in routes if r['level_id'] == f'stage_{stage:02}' and r['target'] == 14)
    game = GameCore(load_level(stage))
    attracted = []
    for action in route['actions']:
        result = game.step(action)
        assert not any(event['type'] == 'contact' for event in result.events)
        attracted.extend(event for event in result.events
                         if event['type'] == 'collect' and event['magnetic'])
    assert result.outcome['banked'] == 14
    assert game.snapshot()['magnet']['collected']
    assert attracted


def test_invincibility_renderer_tracks_effect_expiry_pause_and_reset(tmp_path):
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    from unittest.mock import patch
    from kkamang.app import GameApp
    from kkamang.ui import field
    app = GameApp(save_path=tmp_path / 'invincibility.json', headless=True, no_audio=True)
    try:
        app.campaign.data['completed'] = [1, 2]
        app.stage = 3
        app.start_run()
        assert field.FIELD_IMAGES['invincibility'] == 'icons/items/invincibility_item.png'
        app.core._state['position'] = [7, 6]
        app.core._motion_position = [7, 6]
        app.previous = app.core.snapshot()
        app._process_result(app.core.advance(0, .3))
        assert '무적 획득' in app.toast
        assert app.core.snapshot()['protection'] == 0
        for remaining in range(9, 0, -1):
            before = app.core.snapshot()
            with patch.object(field, '_draw_png', wraps=field._draw_png) as draw:
                app.draw()
                assert any(call.args[2] == 'protection' for call in draw.call_args_list)
            assert app.core.snapshot() == before
            assert before['invincibility']['remaining_ticks'] == remaining
            if remaining == 9:
                app.transition('pause')
                app.advance(1.)
                assert app.core.snapshot() == before
                app.transition('game')
            app._process_result(app.core.advance(5, .3))
        with patch.object(field, '_draw_png', wraps=field._draw_png) as draw:
            app.draw()
            assert not any(call.args[2] == 'protection' for call in draw.call_args_list)
        app.core._state['protection'] = 2
        with patch.object(field, '_draw_png', wraps=field._draw_png) as draw:
            app.draw()
            assert any(call.args[2] == 'protection' for call in draw.call_args_list)
        app.start_run()
        assert not app.core.snapshot()['invincibility']['collected']
        app.stage = 4
        app.campaign.data['completed'] = [1, 2, 3]
        app.start_run()
        assert app.core.snapshot()['invincibility'] is None
    finally:
        app.close()


def test_path_search_allows_only_item_protected_contacts_and_preserves_live_game():
    from tools.evaluate import find_path, project_path
    config = load_level(1)
    config['invincibility'] = {'position': [3, 3], 'duration_ticks': 10}
    config['ghosts'] = [{'id': 'A', 'route': [[3, 4], [3, 5]], 'direction': 1,
                         'initial_index': 0, 'initial_countdown': 100, 'ticks_per_move': 1}]
    game = GameCore(config)
    game._state['position'] = [3, 2]
    before = game.snapshot()
    path = find_path(game, (3, 5), interact=False, allow_item_invincibility=True)
    assert path == [1, 1, 1] and game.snapshot() == before
    assert project_path(game, path) is None
    checked = project_path(game, path, allow_item_invincibility=True)
    assert checked and checked[0].snapshot()['damage_count'] == 0
    strict = find_path(game, (3, 5), interact=False, allow_item_invincibility=False)
    assert strict and len(strict) > len(path)
    assert project_path(game, strict) is not None
    game.step(1)
    game._state['invincibility']['remaining_ticks'] = 1
    checked = project_path(game, [1], allow_item_invincibility=True)
    assert checked and checked[1].snapshot['invincibility']['remaining_ticks'] == 0
    assert next(e for e in checked[1].events if e['type'] == 'contact')['item_invincible']
    assert project_path(game, [5, 1], allow_item_invincibility=True) is None
    game._state['invincibility']['remaining_ticks'] = 0
    game._state['protection'] = 3
    assert project_path(game, [1], allow_item_invincibility=True) is None
    del config['invincibility']
    ordinary = GameCore(config)
    ordinary._state.update(position=[3, 3], protection=3)
    assert project_path(ordinary, [1], allow_item_invincibility=True) is None


def test_path_search_simulates_chaser_stun_on_invincibility_pickup():
    from tools.evaluate import find_path, project_path
    game = GameCore(load_level(5))
    game._state.update(position=[12, 1])
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=[12, 3], countdown=1)
    before = game.snapshot()
    path = find_path(game, (13, 2), interact=False, allow_item_invincibility=True)
    assert path == [1, 3] and game.snapshot() == before
    projected, result = project_path(game, path, allow_item_invincibility=True)
    assert result.snapshot['chaser']['position'] == [12, 2]
    assert result.snapshot['chaser']['countdown'] == 3
    assert result.snapshot['courage'] == 3


@pytest.mark.parametrize('stage', [3, 5, 11])
def test_invincibility_reference_routes_collect_item_and_return_without_damage(stage):
    routes = json.loads((ROOT / 'validation/reference_routes.json').read_text(encoding='utf-8'))['routes']
    route = next(r for r in routes if r['level_id'] == f'stage_{stage:02}' and r['target'] == 14)
    assert route['contact_policy'] == 'item_invincible'
    assert replay(route)['damage_count'] == 0
    game = GameCore(load_level(stage))
    contacts = []
    for action in route['actions']:
        result = game.step(action)
        contacts.extend(e for e in result.events if e['type'] == 'contact')
    assert game.snapshot()['invincibility']['collected']
    assert all(e['item_invincible'] for e in contacts)
    if stage == 3:
        assert contacts
        strict_route = {**route, 'contact_policy': 'no_contact'}
        with pytest.raises(AssertionError, match='Forbidden contact'):
            replay(strict_route)


@pytest.mark.parametrize('magnet', [False, True])
def test_search_avoids_chaser_without_invincibility_and_preserves_live_state(magnet):
    from tools.evaluate import find_path, project_path
    config = load_level(1)
    config['chaser'] = {'id': 'C', 'spawn_position': [8, 6],
                         'initial_countdown': 1, 'ticks_per_move': 1}
    if magnet:
        config['magnet'] = {'position': [3, 3], 'duration_ticks': 13}
    game = GameCore(config)
    game._state['position'] = [3, 3]
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=[3, 4], countdown=1)
    initial, actions = game.snapshot(), game.actions
    initial_rng = game._rng.getstate()
    assert project_path(game, [1]) is None
    diagnostics = {}
    path = find_path(game, (3, 4), interact=False, diagnostics=diagnostics)
    assert path and len(path) > 1
    assert diagnostics['status'] == 'found'
    assert project_path(game, path) is not None
    assert game.snapshot() == initial and game.actions == actions
    assert game._rng.getstate() == initial_rng


def test_search_diagnostics_distinguish_limits_time_and_blocked_goals():
    from tools.evaluate import find_path
    config = load_level(1)
    config['chaser'] = {'id': 'C', 'spawn_position': [8, 6],
                         'initial_countdown': 1, 'ticks_per_move': 1}
    game = GameCore(config)
    game._state['position'] = [3, 3]
    game._state['quest']['collected'] = True
    game._state['chaser'].update(active=True, position=[3, 4], countdown=1)
    info = {}
    assert find_path(game, (3, 4), interact=False, max_nodes=1, diagnostics=info) is None
    assert info == {'status': 'node_limit', 'expanded_nodes': 1}
    assert find_path(game, (3, 4), interact=False) is not None
    game._state['remaining_ticks'] = 1
    assert find_path(game, (3, 5), interact=False, diagnostics=info) is None
    assert info['status'] == 'insufficient_time'
    assert find_path(game, (0, 0), diagnostics=info) is None
    assert info['status'] == 'unreachable'
    game.abort()
    assert find_path(game, (3, 5), diagnostics=info) is None
    assert info['status'] == 'terminal'


def test_completed_timeout_is_not_a_verified_route():
    from tools.evaluate import project_path
    game = GameCore(load_level(1))
    game._state['remaining_ticks'] = 1
    assert project_path(game, [5]) is None
    assert not game.snapshot()['terminal']
