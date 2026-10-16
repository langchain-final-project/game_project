"""Read-only affection cards and representative-memory comparison."""
import pygame

from ..likes import CHARACTERS, ROOM_COUNT, HIGH_AFFECTION, contribution
from .illustration import ghost
from .theme import CREAM, INK, PAPER, GOLD, SAGE, LAVENDER, text, panel

VARIANTS = {'bosong': 0, 'dore': 2, 'ipari': 1}
COLORS = {'bosong': GOLD, 'dore': LAVENDER, 'ipari': SAGE}


def number(value):
    return '평가 대기' if value is None else f'{value:.1f}'


def _bar(surface, rect, value, color):
    pygame.draw.rect(surface, (219, 207, 193), rect, border_radius=9)
    if value is not None and value > 0:
        fill = pygame.Rect(rect)
        fill.w = max(1, round(fill.w * min(100, value) / 100))
        pygame.draw.rect(surface, color, fill, border_radius=9)


def draw_likes(surface, assets, ctx, ui):
    state = ctx['campaign'].data['likes']
    totals = ctx['likes_totals']
    count = len(state['representatives'])
    unscored = sum(r['scores'] is None for r in state['representatives'].values())
    for i, (ident, name, key, maximum, description) in enumerate(CHARACTERS):
        r = pygame.Rect(40 + i * 410, 132, 380, 420)
        panel(surface, r, fill=PAPER, border=COLORS[ident], radius=22)
        text(surface, assets, name, (r.centerx, 154), 28, INK, True, 'midtop')
        ghost(surface, (r.centerx, 340), 125, ctx.get('elapsed', 0),
              resident=True, variant=VARIANTS[ident])
        text(surface, assets, description, (r.centerx, 362), 17, INK, anchor='midtop')
        value = totals[ident]
        label = '기억 평가를 기다리고 있어요' if value is None else f'호감도 {value:.1f} / 100'
        text(surface, assets, label, (r.centerx, 405), 22, INK, True, 'midtop')
        _bar(surface, pygame.Rect(r.x + 30, 449, 320, 18), value, COLORS[ident])
        message = ('다음 기억도 함께 남겨요.' if value is None or value < HIGH_AFFECTION
                   else '아주 소중한 친구가 되었어요.')
        text(surface, assets, message, (r.centerx, 493), 17, INK, anchor='midtop')
    message = f'이 방에 남긴 기억 {count} / {ROOM_COUNT} · 방마다 하나의 기억이 마음에 남아요.'
    if unscored:
        message = f'아직 평가되지 않은 기억 {unscored}개 · 온기와 발걸음 기록은 보관되어 있어요.'
    text(surface, assets, message, (40, 574), 18, CREAM)
    ui.control('home', (40, 619, 195, 45), '쉼터로', primary=True)
    if state['pending'] is not None:
        ui.control('choose_memory', (470, 619, 300, 45), '남길 기억 선택하기')
    complete = len(ctx['campaign'].data['completed']) == ROOM_COUNT
    ui.control('show_ending', (965, 619, 275, 45), '현재 기억의 결말 보기', disabled=not complete)


def draw_choice(surface, assets, ctx, ui):
    compare = ctx.get('memory_comparison')
    if compare is None:
        text(surface, assets, '선택할 기억이 없어요.', (640, 330), 25, CREAM, anchor='center')
        ui.control('home', (465, 612, 350, 50), '쉼터로', primary=True)
        return
    for i, (label, record) in enumerate((('기존 기억', compare['old']), ('이번 기억', compare['new']))):
        r = pygame.Rect(40 + i * 620, 132, 580, 345)
        panel(surface, r, fill=PAPER, border=GOLD, radius=22)
        text(surface, assets, label, (r.x + 28, 150), 27, INK, True)
        mode = '기본' if record['mode'] == 'basic' else '여유'
        text(surface, assets, f'{record["stage"]:02d}번 방 · {mode} 모드', (r.right - 28, 159), 17, INK, anchor='topright')
        summary = f'온기 {record["banked"]}점  ·  {record["ticks_used"]}틱  ·  용기 {record["remaining_courage"]}/{record["initial_courage"]}'
        text(surface, assets, summary, (r.x + 28, 207), 19, INK)
        scores = record['scores']
        text(surface, assets, '점수 평가 대기' if scores is None else f'이 판의 점수 {scores["total"]} · 별 {scores["stars"]}개',
             (r.x + 28, 242), 18, INK)
        values = contribution(record)
        for j, (ident, name, *_) in enumerate(CHARACTERS):
            y = 292 + j * 53
            text(surface, assets, name, (r.x + 28, y), 19, INK, True)
            _bar(surface, pygame.Rect(r.x + 108, y + 6, 310, 14), values[ident], COLORS[ident])
            text(surface, assets, number(values[ident]), (r.right - 26, y), 18, INK, anchor='topright')
    text(surface, assets, '이번 기억을 선택하면 전체 호감도가 이렇게 바뀌어요.', (40, 497), 20, CREAM, True)
    for i, (ident, name, *_) in enumerate(CHARACTERS):
        before, after = compare['before'][ident], compare['after'][ident]
        label = f'{name}  {number(before)} → {number(after)}'
        if before is not None and after is not None:
            label += f'  ({after - before:+.1f})'
        text(surface, assets, label, (40 + i * 410, 537), 20, CREAM)
    text(surface, assets, '세 친구의 기억을 함께 교체해요. 최고 별점과 꾸미기 해금은 유지돼요.', (40, 578), 17, CREAM)
    ui.control('memory_keep', (40, 619, 580, 45), '기존 기억 유지', primary=True)
    ui.control('memory_replace', (660, 619, 580, 45), '이번 기억으로 교체')
