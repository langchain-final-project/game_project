"""호감도·엔딩 화면용 개발 미리보기를 구성한다. 정상 플레이의 점수와 저장에는 사용하지 않는다."""
from . import likes

PREVIEWS = ('overview', 'choice', *likes.ENDING_IDS)
SAMPLES = {
    'happy_all': (48, 20, 15), 'sad': (0, 25, 15),
    'ghost_a': (60, 5, 5), 'ghost_b': (24, 25, 5),
    'ghost_c': (30, 5, 15), 'normal': (30, 10, 10),
}


def sample_record(stage, components, run_id=None):
    warmth, speed, life = components
    total = sum(components)
    return likes.validate_record({
        'stage': stage, 'run_id': run_id or f'preview-{stage}',
        'mode': 'basic', 'config_hash': 'preview-only', 'rules_version': 'preview-only',
        'banked': round(4 + warmth / 6), 'ticks_used': 40,
        'initial_courage': 3, 'remaining_courage': max(1, life // 5),
        'scores': {'version': 'preview-only', 'warmth': warmth, 'speed': speed,
                   'life': life, 'total': total, 'stars': sum(total >= t for t in (45, 65, 85))},
    })


def configure_preview(app, name):
    state = likes.empty_state()
    components = SAMPLES.get(name, SAMPLES['normal'])
    for stage in range(1, 13):
        likes.offer_memory(state, sample_record(stage, components))
    app.campaign.data['likes'] = state
    app.campaign.data['completed'] = list(range(1, 13))
    app.campaign.data['photos'] = list(range(1, 13))
    app.campaign.data['intro_seen'] = True
    if name == 'choice':
        likes.offer_memory(state, sample_record(1, (12, 25, 15), 'preview-replay'))
        app.transition('memory_choice')
    elif name == 'overview':
        app.transition('likes')
    else:
        app.start_ending()
        app.dialogue_time = 99.
