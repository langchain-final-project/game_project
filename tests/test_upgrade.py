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
    assert OBSERVATION_SIZE==610 and b[601]==1 and a[601]==0
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
    assert outcome['quest_collected'] and outcome['banked']==4
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

def test_character_directions_paws_moods_and_scarf_have_distinct_frames():
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    import pygame
    from kkamang.ui.illustration import cat_frame
    raw=lambda **kw:pygame.image.tobytes(cat_frame(160,**kw),'RGBA')
    assert len({raw(direction=d) for d in ['up','down','left','right']})==4
    for d in ['up','down','left','right']:
        assert raw(direction=d,mood='walk',phase=2)!=raw(direction=d,mood='walk',phase=6)
    assert len({raw(mood=m) for m in ['idle','happy','collect','surprised']})==4
    assert raw(scarf='default')!=raw(scarf='peach')

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
