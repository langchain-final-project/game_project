"""수집장의 영구 해금과 표지 기록을 관리한다. 탐사 점수와 이야기 진행은 변경하지 않는다."""
from functools import lru_cache
import json

from .paths import ASSET_DIR


@lru_cache(maxsize=1)
def catalog():
    path = ASSET_DIR / 'art' / 'collection' / 'catalog.json'
    return tuple(json.loads(path.read_text(encoding='utf-8')))


def empty_state():
    return {'unlocked': [], 'seen': [], 'favorite': None,
            'practice_rooms': [], 'discoveries': []}


def normalize(value):
    if not isinstance(value, dict):
        raise ValueError('수집장 기록이 올바르지 않아요.')
    state = empty_state()
    for key, limit in (('unlocked', 24), ('seen', 24), ('practice_rooms', 12)):
        entries = value.get(key, [])
        if not isinstance(entries, list) or any(type(n) is not int or not 1 <= n <= limit for n in entries):
            raise ValueError('수집장 번호가 올바르지 않아요.')
        state[key] = sorted(set(entries))
    favorite = value.get('favorite')
    if favorite is not None and (type(favorite) is not int or not 1 <= favorite <= 24):
        raise ValueError('수집장 표지 번호가 올바르지 않아요.')
    state['favorite'] = favorite
    discoveries = value.get('discoveries', [])
    allowed = ('shelter_before', 'shelter_return', 'memory_paw')
    if not isinstance(discoveries, list) or any(v not in allowed for v in discoveries):
        raise ValueError('수집장 발견 기록이 올바르지 않아요.')
    state['discoveries'] = sorted(set(discoveries))
    return state


def progress(data, entry):
    state = data.get('collection', empty_state())
    rooms = set(data.get('completed', [])) | set(state['practice_rooms'])
    rule = entry['rule']
    if rule in ('first_success', 'rooms'):
        return len(rooms)
    if rule == 'stars':
        return sum(record['stars'] for record in data.get('star_records', {}).values())
    if rule in ('rich', 'quiet'):
        return len({badge for badge in data.get('badges', []) if badge.endswith(':' + rule)})
    return int(rule in state['discoveries'])


def available(data):
    saved = data.get('collection', empty_state())['unlocked']
    return set(saved) | {entry['id'] for entry in catalog() if progress(data, entry) >= entry['goal']}


def sync(data):
    state = data.setdefault('collection', empty_state())
    unlocked = available(data)
    new = sorted(unlocked - set(state['unlocked']))
    state['unlocked'] = sorted(unlocked)
    state['seen'] = sorted(set(state['seen']) & unlocked)
    if state['favorite'] not in unlocked:
        state['favorite'] = None
    return new


def record_success(data, stage, *, practice=False):
    if practice:
        state = data['collection']
        state['practice_rooms'] = sorted(set(state['practice_rooms']) | {stage})
    return sync(data)


def discover(data, event):
    state = data['collection']
    if event == 'shelter_before' and data['completed']:
        pass
    elif event == 'shelter_return' and 'shelter_before' in state['discoveries']:
        pass
    elif event == 'memory_paw' and any(
            set(range(start, start + 4)) <= set(data['completed']) for start in (1, 5, 9)):
        pass
    else:
        return []
    state['discoveries'] = sorted(set(state['discoveries']) | {event})
    return sync(data)


def mark_seen(data, number):
    if number not in available(data):
        return False
    sync(data)
    state = data['collection']
    state['seen'] = sorted(set(state['seen']) | {number})
    return True


def set_favorite(data, number):
    if not mark_seen(data, number):
        return False
    data['collection']['favorite'] = number
    return True


def requirement(data, entry):
    value = min(progress(data, entry), entry['goal'])
    rule, goal = entry['rule'], entry['goal']
    if rule == 'first_success':
        return '기본 또는 연습 모드에서 첫 성공 귀환'
    if rule == 'rooms':
        return f'서로 다른 방 {goal}개 성공 · {value}/{goal} (기본·연습 합산)'
    if rule == 'stars':
        return f'방별 최고 별 합계 {goal}개 · {value}/{goal}'
    if rule in ('rich', 'quiet'):
        name = '온기' if rule == 'rich' else '무피해'
        return f'{name} 배지 {goal}개 · {value}/{goal}'
    return entry['hint']
