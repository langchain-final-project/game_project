"""Public-state beam planner. Replayable witnesses, not optimality proofs."""
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from kkamang.core import GameCore
from kkamang.levels import load_levels
from tools.evaluate import safe_path

def plan(game,target=4,width=18,allow_shortcuts=True,forbid_rich=False):
 beam=[(game,[])];best=None
 for depth in range(15):
  candidates={}
  for core,actions in beam:
   s=core.snapshot()
   if s['quest']['collected'] and s['carried']>=target:
    path=safe_path(s,tuple(s['exit']))
    if path and (best is None or len(actions)+len(path)<len(best)):
     test=deepcopy(core)
     for a in path:res=test.step(a)
     if res.outcome and res.outcome['status']=='COMPLETE':best=actions+path
    continue
   goals=[]
   if not s['quest']['collected']:goals.append(('q',s['quest']['position']))
   if s['carried']<target:goals += [(i['id'],i['position']) for i in s['items'] if i['active'] and not(forbid_rich and i['value']==3)]
   if allow_shortcuts:goals += [('s'+g['id'],g['terminal']) for g in s['shortcuts'] if not g['opened']]
   for ident,pos in goals:
    path=safe_path(s,tuple(pos))
    if not path or len(actions)+len(path)>s['initial_ticks']-2:continue
    test=deepcopy(core)
    for a in path:
     res=test.step(a)
     if res.terminated:break
    if res.terminated:continue
    n=res.snapshot;p=n['position'];e=n['exit'];home=abs(p[0]-e[0])+abs(p[1]-e[1])+1
    if n['remaining_ticks']<home:continue
    if best is not None and len(actions)+len(path)+home>=len(best):continue
    key=(tuple(p),tuple(i['active'] for i in n['items']),n['quest']['collected'],tuple(g['opened'] for g in n['shortcuts']),tuple((g['route_index'],g['countdown']) for g in n['ghosts']))
    aa=actions+path
    if key not in candidates or len(aa)<len(candidates[key][1]):candidates[key]=(test,aa)
  def rank(pair):
   s=pair[0].snapshot();q=s['quest'];p=s['position']
   qdist=0 if q['collected'] else abs(p[0]-q['position'][0])+abs(p[1]-q['position'][1])+1
   return (len(pair[1])+qdist+max(0,target-s['carried'])*2,-s['carried'])
  beam=sorted(candidates.values(),key=rank)[:width]
  if not beam:break
 return best

def trace(s):
 return {'tick':s['tick'],'player':s['position'],'carried':s['carried'],'quest_collected':s['quest']['collected'],
 'collected_ids':sorted(i['id'] for i in s['items'] if not i['active']),
 'opened_shortcut_ids':sorted(g['id'] for g in s['shortcuts'] if g['opened']),
 'ghosts':[{'id':g['id'],'position':g['position'],'current_index':g['route_index'],'countdown':g['countdown']} for g in s['ghosts']]}

def main():
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--include-safe',action='store_true')
 args=parser.parse_args()
 routes=[];hashes=[]
 for l in load_levels():
  for target in (4,10,14):
   g=GameCore(l);actions=plan(g,target)
   if actions is None:actions=plan(g,target,width=72)
   if actions is None:raise RuntimeError(f"No witness: {l['id']} target={target}")
   tt=[trace(g.snapshot())]
   for a in actions:result=g.step(a);tt.append(trace(result.snapshot))
   o=result.outcome
   assert o['status']=='COMPLETE' and o['damage_count']==0 and o['quest_collected']
   routes.append({'level_id':l['id'],'target':target,'actions':actions,'ticks':len(actions),'banked':o['banked'],'phase_offsets':[0]*len(l['ghosts']),'constraint':'unconstrained','trace':tt})
   print(l['id'],target,len(actions),'/',l['time_budget_ticks'],flush=True)
  p=ROOT/'data'/'levels'/f"{l['id']}.json"
  hashes.append({'id':l['id'],'file':'levels/'+p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
 result={'source':'v0.2 beam planner + GameCore rollout; traces are regression fixtures, not independent oracle','source_results_sha256':None,'routes':routes,'level_hashes':hashes}
 (ROOT/'validation/reference_routes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if args.include_safe:
  safe=[]
  for l in load_levels():
   g=GameCore(l);a=plan(g,4,allow_shortcuts=False,forbid_rich=True)
   if a is None:a=plan(g,4,width=72,allow_shortcuts=False,forbid_rich=True)
   if a is None:raise RuntimeError(f"No safe witness: {l['id']}")
   for x in a:result=g.step(x)
   safe.append({'level_id':l['id'],'target':4,'constraint':'all_shortcuts_closed','forbid_rich':True,'phase_offsets':[0]*len(l['ghosts']),'actions':a,'ticks':len(a),'banked':result.outcome['banked']})
  (ROOT/'validation/safe_routes.json').write_text(json.dumps(safe,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
