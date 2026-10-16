"""Story-owner handoff: replace lines/art here without changing affection rules."""
ENDING_CONTENT = {
    'happy_all': {'title': '언제든 돌아올 집', 'lines': [('이야기', '세 친구 모두와 깊은 기억을 나누었어요.')]},
    'sad': {'title': '남겨진 불빛', 'lines': [('이야기', '문은 열렸지만, 저택에는 아직 나누지 못한 온기가 남았어요.')]},
    'ghost_a': {'title': '보송과 남긴 기억', 'lines': [('보송', '네가 나누어 준 온기를 오래 기억할게.')]},
    'ghost_b': {'title': '도레와 남긴 기억', 'lines': [('도레', '다시 만나는 시간이 늘 기다려졌어.')]},
    'ghost_c': {'title': '이파리와 남긴 기억', 'lines': [('이파리', '무사히 돌아오는 발소리에 안심했어.')]},
    'normal': {'title': '함께 남긴 밤', 'lines': [('이야기', '까망이는 저택에 작은 기억들을 남기고 아침을 맞았어요.')]},
}
# All six are connection placeholders, not final story drafts.
for _ending in ENDING_CONTENT.values():
    _ending.setdefault('art', 'ending_scene.png')
    _ending.setdefault('subtitle', '함께 남긴 기억 · 임시 엔딩')
