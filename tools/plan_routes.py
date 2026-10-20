"""공개 상태 기반 빔 탐색으로 재생 가능한 경로를 만들며 최적성은 보장하지 않는다."""
from copy import deepcopy
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from kkamang.core import GameCore
from kkamang.levels import load_levels
from tools.evaluate import collection_positions, project_path, find_path
from collections import deque
from functools import lru_cache

def rollout(core, path, *, forbid_rich=False, allow_item_invincibility=False):
    """원본 게임을 보존하고, 후보 경로의 안전성을 확인합니다."""
    return project_path(core, path, forbid_rich=forbid_rich,
                        allow_item_invincibility=allow_item_invincibility)

def make_lower_bound(game, target, forbid_rich):
    """남은 임무에 필요한 최소 틱 수를 낙관적으로 추정합니다."""
    initial = game.snapshot()

    items = [
        item for item in initial['items']
        if not (forbid_rich and item['value'] == 3)
    ]

    points = (
        [tuple(item['position']) for item in items]
        + [tuple(initial['quest']['position'])]
    )
    values = [item['value'] for item in items] + [0]
    item_ids = [item['id'] for item in items]
    quest_bit = 1 << len(items)
    exit_pos = tuple(initial['exit'])

    @lru_cache(None)
    def distances(origin):
        queue = deque([origin])
        found = {origin: 0}

        while queue:
            x, y = queue.popleft()

            for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                point = (x + dx, y + dy)
                nx, ny = point

                if point in found:
                    continue
                if not (0 <= nx < initial['width'] and 0 <= ny < initial['height']):
                    continue
                if initial['grid'][ny][nx] == '#':
                    continue

                found[point] = found[(x, y)] + 1
                queue.append(point)

        return found

    @lru_cache(None)
    def cost(position, mask, needed):
        if needed <= 0 and not (mask & quest_bit):
            return distances(position).get(exit_pos, 10000) + 1

        best = 10000

        for index, point in enumerate(points):
            bit = 1 << index

            if not (mask & bit):
                continue
            if index < len(items) and needed <= 0:
                continue

            travel = distances(position).get(point, 10000)
            remaining = cost(
                point,
                mask ^ bit,
                max(0, needed - values[index]),
            )
            # 필수 물건에는 E키 1틱이 필요합니다.
            # 온기는 이동하면서 수집하므로 추가 행동 비용이 없습니다.
            interaction = int(index == len(items))
            best = min(best, travel + interaction + remaining)

        return best

    def estimate(snapshot):
        magnet = snapshot.get('magnet')
        if magnet and (not magnet['collected'] or magnet['remaining_ticks'] > 0):
            # 자석으로 원격 수집할 수 있으므로 온기 칸 방문 비용은 하한이 아닙니다.
            # 필수 물건과 귀환까지의 거리만 사용해 과대 추정을 피합니다.
            position = tuple(snapshot['position'])
            if snapshot['quest']['collected']:
                return distances(position).get(exit_pos, 10000) + 1
            quest = tuple(snapshot['quest']['position'])
            return (distances(position).get(quest, 10000)
                    + distances(quest).get(exit_pos, 10000) + 2)
        active = {
            item['id']
            for item in snapshot['items']
            if item['active']
        }

        mask = sum(
            1 << index
            for index, ident in enumerate(item_ids)
            if ident in active
        )

        if not snapshot['quest']['collected']:
            mask |= quest_bit

        return cost(
            tuple(snapshot['position']),
            mask,
            max(0, target - snapshot['carried']),
        )

    return estimate


def plan(game, target=4, width=18, allow_shortcuts=True, forbid_rich=False,
         allow_item_invincibility=True):
    if 'motion' in game.snapshot():
        raise ValueError('plan supports grid movement; realtime needs a frame-based policy')

    def route_path(core, goal, interact=True):
        # forbid_rich는 3점 온기 칸의 직접 방문을 막고 자석의 1점 수집은 허용합니다.
        return find_path(core, goal, interact=interact, forbid_rich=forbid_rich,
                         allow_item_invincibility=allow_item_invincibility)

    lower_bound = make_lower_bound(game, target, forbid_rich)
    beam = [(game, [])]
    best = None

    for depth in range(15):
        candidates = {}

        for core, actions in beam:
            s = core.snapshot()

            # 필수 물건과 목표 온기를 모았다면 귀환 경로를 검사합니다.
            if s['quest']['collected'] and s['carried'] >= target:
                path = route_path(core, tuple(s['exit']))

                if path and (best is None or len(actions) + len(path) < len(best)):
                    checked = rollout(core, path, forbid_rich=forbid_rich,
                                      allow_item_invincibility=allow_item_invincibility)

                    if checked is not None:
                        _, result = checked
                        if result.outcome and result.outcome['status'] == 'COMPLETE':
                            best = actions + path

                continue

            goals = {}

            if not s['quest']['collected']:
                goals[(tuple(s['quest']['position']), True)] = 'q'

            magnet = s.get('magnet')
            if magnet and not magnet['collected']:
                goals[(tuple(magnet['position']), False)] = 'magnet'
            invincibility = s.get('invincibility')
            if invincibility and not invincibility['collected']:
                goals[(tuple(invincibility['position']), False)] = 'invincibility'

            if s['carried'] < target:
                for item in s['items']:
                    if item['active'] and not (forbid_rich and item['value'] == 3):
                        for point in collection_positions(s, item):
                            goals.setdefault((point, magnet is None and invincibility is None), item['id'])

            if allow_shortcuts:
                for gate in s['shortcuts']:
                    if not gate['opened']:
                        goals[(tuple(gate['terminal']), True)] = 's' + gate['id']

            for (position, interact), ident in goals.items():
                path = route_path(core, position, interact)

                if not path or len(actions) + len(path) > s['initial_ticks'] - 2:
                    continue

                # 실제 게임으로 실행해 안전한 후보만 남깁니다.
                checked = rollout(core, path, forbid_rich=forbid_rich,
                                  allow_item_invincibility=allow_item_invincibility)
                if checked is None:
                    continue

                test, result = checked
                if result.terminated:
                    continue

                n = result.snapshot
                if (n['carried'] == s['carried']
                        and n['quest']['collected'] == s['quest']['collected']
                        and [g['opened'] for g in n['shortcuts']] == [g['opened'] for g in s['shortcuts']]
                        and (not magnet or n['magnet']['collected'] == magnet['collected'])
                        and (not invincibility or n['invincibility']['collected'] == invincibility['collected'])):
                    continue
                p = n['position']
                e = n['exit']
                home = abs(p[0] - e[0]) + abs(p[1] - e[1]) + 1

                if n['remaining_ticks'] < home:
                    continue
                if best is not None and len(actions) + len(path) + home >= len(best):
                    continue

                # 추격 유령의 상태가 다르면 서로 다른 후보로 취급합니다.
                chaser = n.get('chaser')
                chase_key = None if chaser is None else (
                    chaser['active'],
                    tuple(chaser['position']),
                    chaser['countdown'],
                )

                key = (
                    tuple(p),
                    tuple(item['active'] for item in n['items']),
                    n['quest']['collected'],
                    tuple(gate['opened'] for gate in n['shortcuts']),
                    tuple((g['route_index'], g['countdown']) for g in n['ghosts']),
                    chase_key,
                    None if n.get('magnet') is None else (
                        n['magnet']['collected'], n['magnet']['remaining_ticks']),
                    None if n.get('invincibility') is None else (
                        n['invincibility']['collected'], n['invincibility']['remaining_ticks']),
                )

                new_actions = actions + path
                if key not in candidates or len(new_actions) < len(candidates[key][1]):
                    candidates[key] = (test, new_actions)


        def rank(pair):
            s = pair[0].snapshot()
            return (
                len(pair[1]) + lower_bound(s),
                -s['carried'],
            )
        beam = sorted(candidates.values(), key=rank)[:width]
        if not beam:
            break

    return best

def trace(s):
 return {'tick':s['tick'],'player':s['position'],'carried':s['carried'],'quest_collected':s['quest']['collected'],
 'collected_ids':sorted(i['id'] for i in s['items'] if not i['active']),
 'opened_shortcut_ids':sorted(g['id'] for g in s['shortcuts'] if g['opened']),
 'ghosts':[{'id':g['id'],'position':g['position'],'current_index':g['route_index'],'countdown':g['countdown']} for g in s['ghosts']],
 'magnet':deepcopy(s.get('magnet')), 'invincibility':deepcopy(s.get('invincibility')),
 'chaser':deepcopy(s.get('chaser'))}

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
   routes.append({'level_id':l['id'],'target':target,'actions':actions,'ticks':len(actions),'banked':o['banked'],'phase_offsets':[0]*len(l['ghosts']),'constraint':'unconstrained','contact_policy':'item_invincible' if 'invincibility' in l else 'no_contact','trace':tt})
   print(l['id'],target,len(actions),'/',l['time_budget_ticks'],flush=True)
  p=ROOT/'data'/'levels'/f"{l['id']}.json"
  hashes.append({'id':l['id'],'file':'levels/'+p.name,'sha256':hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest()})
 result={'source':'v0.2 beam planner + GameCore rollout; traces are regression fixtures, not independent oracle','source_results_sha256':None,'routes':routes,'level_hashes':hashes}
 (ROOT/'validation/reference_routes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 if args.include_safe:
  safe=[]
  for l in load_levels():
   g=GameCore(l);a=plan(g,4,allow_shortcuts=False,forbid_rich=True,allow_item_invincibility=False)
   if a is None:a=plan(g,4,width=72,allow_shortcuts=False,forbid_rich=True,allow_item_invincibility=False)
   if a is None:raise RuntimeError(f"No safe witness: {l['id']}")
   for x in a:result=g.step(x)
   safe.append({'level_id':l['id'],'target':4,'constraint':'all_shortcuts_closed','forbid_rich':True,'phase_offsets':[0]*len(l['ghosts']),'actions':a,'ticks':len(a),'banked':result.outcome['banked']})
  (ROOT/'validation/safe_routes.json').write_text(json.dumps(safe,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
