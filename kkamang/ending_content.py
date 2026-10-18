"""Artwork and epilogue dialogue from kkamang_endings_6; no branch rule changes."""
ENDING_CONTENT = {
    'happy_all': {'title': '아무도 남겨두지 않은 아침', 'subtitle': '베스트',
                  'lines': [('보송', '꼬리 좀…… 내 코에 닿아.'), ('도레', '내 거 아닌데.'), ('이파리', '쉿. 까망이 잔다.')]},
    'sad': {'title': '돌아와도, 아무도', 'subtitle': '새드', 'lines': []},
    'ghost_a': {'title': '너에게 남긴 온도', 'subtitle': '보송 · 온기',
                'lines': [('보송', '이제는 내가 데워 줄 수 있네.'), ('도레', '불은 좀 줄일까?'), ('보송', '응. 오늘은 괜찮아.')]},
    'ghost_b': {'title': '늦지 않은 약속', 'subtitle': '도레 · 속도',
                'lines': [('도레', '기다려. 이번엔 나도 갈 수 있어.'), ('도레', '이번엔 안 늦었지?')]},
    'ghost_c': {'title': '함께 살아갈 계절', 'subtitle': '이파리 · 생명',
                'lines': [('이파리', '살아 있네.'), ('보송', '내년에도 꽃이 필까?'), ('이파리', '그때도 같이 보자.')]},
    'normal': {'title': '네 자리는 여기', 'subtitle': '노멀',
               'lines': [('보송', '왔네. 수건은 저기 있어.'), ('이파리', '네 자리도 그대로야.'), ('도레', '그럼, 오늘은 작게 연주할게.')]},
}
# Production notes and silent actions are not narrated by the dialogue UI.
for _ident, _ending in ENDING_CONTENT.items():
    _ending['art'] = f'ending_{_ident}.png'
