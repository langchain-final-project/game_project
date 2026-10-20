"""방별 대표 기록으로 호감도와 엔딩을 계산한다. 미평가 점수는 None으로 구분하고 탐사 규칙은 변경하지 않는다."""
from copy import deepcopy
import math

ROOM_COUNT = 12
HIGH_AFFECTION = 70.0
CHARACTERS = (
    ('bosong', '보송', 'warmth', 60, '함께 나눌 온기를 좋아해요.'),
    ('dore', '도레', 'speed', 25, '다시 만나는 시간을 기다려요.'),
    ('ipari', '이파리', 'life', 15, '무사히 돌아오면 안심해요.'),
)
ENDING_IDS = ('happy_all', 'sad', 'ghost_a', 'ghost_b', 'ghost_c', 'normal')
CHARACTER_ENDINGS = dict(zip((c[0] for c in CHARACTERS), ENDING_IDS[2:5]))


def empty_state():
    return {'representatives': {}, 'pending': None, 'best_stars': {}, 'seen_endings': []}


def _int(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return value


def validate_scores(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError('score_breakdown must be a dictionary or None')
    version = value.get('version')
    if not isinstance(version, str) or not version:
        raise ValueError('score_breakdown requires a version')
    result = {'version': version}
    for _, _, key, maximum, _ in CHARACTERS:
        result[key] = _int(value.get(key), 0, maximum, key)
    result['total'] = _int(value.get('total'), 0, 100, 'total')
    result['stars'] = _int(value.get('stars'), 0, 3, 'stars')
    if result['total'] != sum(result[key] for key in ('warmth', 'speed', 'life')):
        raise ValueError('total does not equal the three component scores')
    if result['stars'] != sum(result['total'] >= t for t in (45, 65, 85)):
        raise ValueError('stars must follow the agreed 45/65/85 thresholds')
    return result


def from_game_score(score):
    """점수를 다시 계산하지 않고 용기 항목 이름을 호감도 계약에 맞춘다."""
    if score is None:
        return None
    return validate_scores({'version': 'score_v1', 'warmth': score['warmth'],
                            'speed': score['speed'], 'life': score['courage'],
                            'total': score['total'], 'stars': score['stars']})


def legacy_record(stage, score):
    """이전 저장의 점수를 옮기되 당시 기록하지 않은 탐사 세부 정보는 만들지 않는다."""
    return validate_record({'stage': stage, 'run_id': f'legacy-{stage}',
                            'mode': 'legacy', 'config_hash': 'legacy', 'rules_version': 'legacy',
                            'banked': None, 'ticks_used': None, 'initial_courage': None,
                            'remaining_courage': None, 'scores': from_game_score(score)})


def validate_record(value):
    if not isinstance(value, dict):
        raise ValueError('memory record must be a dictionary')
    result = {}
    for key in ('run_id', 'config_hash', 'rules_version'):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f'memory requires {key}')
        result[key] = value[key]
    result['stage'] = _int(value.get('stage'), 1, ROOM_COUNT, 'stage')
    if value.get('mode') not in ('basic', 'relaxed', 'legacy'):
        raise ValueError('unknown memory mode')
    result['mode'] = value['mode']
    for key in ('banked', 'ticks_used', 'initial_courage', 'remaining_courage'):
        if result['mode'] == 'legacy' and value.get(key) is None:
            result[key] = None
        else:
            result[key] = _int(value.get(key), 0, 1000000, key)
    if result['mode'] != 'legacy' and not 0 < result['remaining_courage'] <= result['initial_courage']:
        raise ValueError('a completed memory requires remaining courage')
    result['scores'] = validate_scores(value.get('scores'))
    return result


def make_record(level, outcome, snapshot, mode, run_id, config_hash, scores=None):
    if outcome.get('status') != 'COMPLETE' or not outcome.get('quest_collected'):
        raise ValueError('only a completed quest can become a memory')
    return validate_record({
        'stage': level['stage'], 'run_id': run_id, 'mode': mode,
        'config_hash': config_hash, 'rules_version': level['rules_version'],
        'banked': outcome['banked'], 'ticks_used': outcome['ticks_used'],
        'initial_courage': level['initial_courage'],
        'remaining_courage': snapshot['courage'], 'scores': scores,
    })


def normalize_state(value):
    if not isinstance(value, dict):
        raise ValueError('likes must be a dictionary')
    result = empty_state()
    reps = value.get('representatives', {})
    best = value.get('best_stars', {})
    if not isinstance(reps, dict) or not isinstance(best, dict):
        raise ValueError('invalid room memories or star mirror')
    for key, record in reps.items():
        record = validate_record(record)
        if key != str(record['stage']):
            raise ValueError('memory stage/key mismatch')
        result['representatives'][key] = record
    for key, stars in best.items():
        if key not in {str(n) for n in range(1, ROOM_COUNT + 1)}:
            raise ValueError('invalid star room')
        result['best_stars'][key] = _int(stars, 0, 3, 'best stars')
    pending = value.get('pending')
    if pending is not None:
        pending = validate_record(pending)
        if str(pending['stage']) not in reps:
            raise ValueError('pending memory requires an existing representative')
        result['pending'] = pending
    seen = value.get('seen_endings', [])
    if not isinstance(seen, list) or any(e not in ENDING_IDS for e in seen):
        raise ValueError('invalid ending history')
    result['seen_endings'] = list(dict.fromkeys(seen))
    records = list(result['representatives'].values())
    if pending is not None:
        records.append(pending)
    for record in records:
        if record['scores'] is not None and best.get(str(record['stage']), -1) < record['scores']['stars']:
            raise ValueError('star mirror cannot be below a scored memory')
    return result


def offer_memory(state, record):
    """첫 성공은 대표 기록으로 등록하고 재도전 기록은 명시적인 교체 선택을 기다린다."""
    record = validate_record(record)
    key = str(record['stage'])
    old = state['representatives'].get(key)
    if old is not None and old['run_id'] == record['run_id']:
        return 'unchanged'
    if state['pending'] is not None:
        if state['pending']['run_id'] == record['run_id']:
            return 'pending'
        raise ValueError('resolve the pending memory before another run')
    if record['scores'] is not None:
        state['best_stars'][key] = max(state['best_stars'].get(key, 0), record['scores']['stars'])
    if old is None:
        state['representatives'][key] = deepcopy(record)
        return 'registered'
    state['pending'] = deepcopy(record)
    return 'pending'


def choose_memory(state, replace):
    pending = state['pending']
    if pending is None:
        return False
    if replace:
        state['representatives'][str(pending['stage'])] = deepcopy(pending)
    state['pending'] = None
    return True


def contribution(record):
    if record is None or record['scores'] is None:
        return {c[0]: None for c in CHARACTERS}
    return {ident: record['scores'][key] / maximum * 100
            for ident, _, key, maximum, _ in CHARACTERS}


def totals(state):
    """미완료 방은 0으로 계산하되 미평가 기록이 있으면 전체 호감도를 미정으로 둔다."""
    values = [contribution(r) for r in state['representatives'].values()]
    return {
        ident: (None if any(v[ident] is None for v in values)
                else math.fsum(v[ident] for v in values) / ROOM_COUNT)
        for ident, *_ in CHARACTERS
    }


def comparison(state):
    pending = state['pending']
    if pending is None:
        return None
    after = deepcopy(state)
    choose_memory(after, True)
    return {'old': deepcopy(state['representatives'][str(pending['stage'])]),
            'new': deepcopy(pending), 'before': totals(state), 'after': totals(after)}


def select_ending(state, completed):
    if set(completed) != set(range(1, ROOM_COUNT + 1)) or state['pending'] is not None:
        return None
    keys = {str(n) for n in range(1, ROOM_COUNT + 1)}
    if set(state['representatives']) != keys or set(state['best_stars']) != keys:
        return None
    affection = totals(state)
    if any(v is None for v in affection.values()):
        return None
    high = [ident for ident, value in affection.items() if value >= HIGH_AFFECTION]
    if len(high) == 3:
        return 'happy_all'
    if sum(state['best_stars'].values()) == 0:
        return 'sad'
    if len(high) == 1:
        return CHARACTER_ENDINGS[high[0]]
    return 'normal'
