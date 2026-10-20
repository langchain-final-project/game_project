"""완료한 방에 대응하는 기억 그림 영역과 표시 정보를 구성하고 여섯 엔딩 내레이션을 연결한다."""
from .levels import load_levels


PROLOGUE = []

MEMORIES = [
    {'id': 1, 'name': '문이 열리던 밤', 'stages': [1, 2, 3, 4]},
    {'id': 2, 'name': '세 개의 자리', 'stages': [5, 6, 7, 8]},
    {'id': 3, 'name': '놓고 싶지 않았던 밤', 'stages': [9, 10, 11, 12]},
]


def memory_for_stage(stage: int) -> tuple[int, int]:
    """방에 대응하는 기억 번호와 0부터 시작하는 영역 번호를 반환한다."""
    if type(stage) is not int or not 1 <= stage <= 12:
        raise ValueError('stage must be an integer in 1..12')
    return (stage - 1) // 4 + 1, (stage - 1) % 4


STAGES = []
for _level in load_levels():
    _stage = _level['stage']
    _quest = _level['quest']
    _memory_id, _region_index = memory_for_stage(_stage)
    STAGES.append({
        'stage': _stage, 'name': _level['name'],
        'photo': f'기억 {_memory_id} · {_region_index + 1}번 영역',
        'title': f'기억 {_memory_id}',
        'before': f"아이템: {_quest['name']} · 온기 4점 이상",
        'lines': [], 'restoration': _quest['restoration'], 'teaser': '',
        'art': '', 'emblem': _quest['kind'], 'act': _memory_id,
    })

# 호환 이름도 공통 내레이션을 참조하며 실제 분기는 ENDING_CONTENT의 여섯 엔딩을 사용한다.
from .ending_content import ENDING_CONTENT
ENDING = ENDING_CONTENT['normal']['lines']
EXTENDED_ENDING = ENDING_CONTENT['happy_all']['lines']

SPEAKERS = {'보송': '보송', '이파리': '이파리', '도레': '도레'}
COSMETIC_LABELS = {
    'scarf': {'default': '세이지 스카프', 'peach': '복숭아 스카프', 'lavender': '라벤더 스카프'},
    'rug': {'default': '따뜻한 기본 러그', 'stars': '별빛 러그', 'leaves': '잎사귀 러그'},
    'frame': {'default': '나무 액자', 'leaves': '잎사귀 액자', 'stars': '별빛 액자'},
}
