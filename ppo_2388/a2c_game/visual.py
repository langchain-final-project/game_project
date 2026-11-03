"""실제 게임 상태를 브라우저 재생용 JSON으로 변환합니다.

게임 상태나 점수를 변경하지 않으며, 진행 중인 회차의 점수를 추정하지 않습니다.
"""
from __future__ import annotations

from copy import deepcopy


def _point(value):
    return None if value is None else [float(value[0]), float(value[1])]


def frame_from_snapshot(snapshot, action=None):
    """원래 코어의 한 순간을 저장합니다. 프레임을 학습 데이터로 사용하지 않습니다."""
    # 칸 중심 좌표보다 실제 소수 위치를 우선하여 게임에서 이동한 모습을 기록합니다.
    motion = snapshot.get('motion') or {}
    ghost_positions = motion.get('ghost_positions') or {}
    ghosts = [
        {'id': str(ghost['id']),
         'position': _point(ghost_positions.get(ghost['id'], ghost['position'])),
         'kind': 'patrol'}
        for ghost in snapshot.get('ghosts', [])
    ]
    chaser = snapshot.get('chaser')
    if chaser and chaser.get('active'):
        ghosts.append({'id': str(chaser.get('id', 'chaser')),
                       'position': _point(motion.get('chaser_position') or chaser['position']),
                       'kind': 'chaser'})
    # 진행 중 가정 점수를 만들지 않고, 원래 게임이 종료 후 확정한 점수만 가져옵니다.
    outcome = deepcopy(snapshot.get('outcome'))
    score = ((outcome or {}).get('score') or {}).get('total')
    quest = snapshot.get('quest') or {}
    frame = {
        'stage': int(snapshot.get('stage', 0)),
        'grid': list(snapshot.get('grid', [])),
        'width': int(snapshot.get('width', 0)),
        'height': int(snapshot.get('height', 0)),
        'player': _point(motion.get('position', snapshot.get('position'))),
        'ghosts': ghosts,
        'items': [{'id': str(item.get('id', '')), 'position': _point(item['position']),
                   'value': int(item['value']), 'active': bool(item['active'])}
                  for item in snapshot.get('items', [])],
        'quest': {'position': _point(quest.get('position')),
                  'collected': bool(quest.get('collected', False))},
        'exit': _point(snapshot.get('exit')),
        'shortcuts': [{'gate': _point(door.get('gate')),
                       'terminal': _point(door.get('terminal')),
                       'opened': bool(door.get('opened', False))}
                      for door in snapshot.get('shortcuts', [])],
        'courage': int(snapshot.get('courage', 0)),
        'carried': int(snapshot.get('carried', 0)),
        'target': int(snapshot.get('target', 0)),
        'remaining_ticks': int(snapshot.get('remaining_ticks', 0)),
        'initial_ticks': int(snapshot.get('initial_ticks', 0)),
        'tick': int(snapshot.get('tick', 0)),
        'terminal': bool(snapshot.get('terminal', False)),
        'outcome': outcome,
        'score': None if score is None else int(score),
        'native_status': (outcome or {}).get('status'),
        'action': None if action is None else int(action),
    }
    for name in ('magnet', 'invincibility'):
        item = snapshot.get(name)
        frame[name] = None if item is None else {
            'position': _point(item.get('position')), 'collected': bool(item.get('collected')),
            'remaining_ticks': int(item.get('remaining_ticks', 0)),
        }
    return frame
