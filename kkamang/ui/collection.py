"""수집장에 24개 모습·12개 물건과 즐겨찾기 표지를 표시한다."""
import math
import pygame

from .. import collection
from .theme import CREAM, INK, GOLD, PAPER, text, fit_text, wrapped
from .illustration import _png_image, QUEST_RECTS

PLUM = (49, 39, 61)
MUTED = (151, 133, 153)
LINE = (177, 148, 109)


def _panel(surface, rect, fill, border=LINE, radius=15):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    pygame.draw.rect(surface, border, rect, 1, border_radius=radius)

def _lock(surface, center, color=GOLD):
    x, y = center
    pygame.draw.arc(surface, color, (x-6, y-10, 12, 15), 0, math.pi, 2)
    pygame.draw.rect(surface, color, (x-9, y-2, 18, 14), border_radius=3)
    pygame.draw.circle(surface, PLUM, (x, y+4), 2)

def draw_paw(surface, rect, color=GOLD):
    r = pygame.Rect(rect)
    pygame.draw.ellipse(surface, color, (r.x+r.w*.28, r.y+r.h*.46, r.w*.48, r.h*.4))
    for x, y in ((.09, .30), (.29, .13), (.53, .13), (.73, .30)):
        pygame.draw.ellipse(surface, color, (r.x+r.w*x, r.y+r.h*y, r.w*.19, r.h*.26))

def _portrait(surface, assets, entry, rect, unlocked=True):
    r = pygame.Rect(rect)
    source = assets.icon('art/collection/' + entry['file'], r.size)
    if not unlocked:
        source = source.copy()
        source.fill((107, 92, 119, 255), special_flags=pygame.BLEND_RGBA_MULT)
        # 알파 외곽선만 남겨 아직 잠긴 의상의 원래 색을 숨긴다.
        mask = pygame.mask.from_surface(source, 10)
        source = mask.to_surface(setcolor=(107, 92, 119, 255), unsetcolor=(0, 0, 0, 0))
    surface.blit(source, source.get_rect(center=r.center))

def _item(surface, quest, rect, unlocked=True):
    r = pygame.Rect(rect)
    source = _png_image('Essential_item.png', r.size, QUEST_RECTS[quest['id']])
    if not unlocked:
        source = pygame.mask.from_surface(source, 10).to_surface(
            setcolor=(107, 92, 119, 255), unsetcolor=(0, 0, 0, 0))
    surface.blit(source, r)

def _base(surface):
    veil = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    veil.fill((*PLUM, 235))
    surface.blit(veil, (0, 0))
    pygame.draw.rect(surface, (128, 103, 88), (18, 12, 1244, 696), 1, border_radius=20)

def _detail(surface, assets, ctx, ui):
    data = ctx['campaign'].data
    kind, number = ctx['collection_detail']
    look = kind == 'looks'
    entry = collection.catalog()[number-1] if look else ctx['levels'][number-1]['quest']
    unlocked = number in (collection.available(data) if look else data['completed'])
    _panel(surface, (46, 38, 535, 626), PAPER if unlocked else (62, 50, 75))
    if look:
        _portrait(surface, assets, entry, (80, 65, 467, 558), unlocked)
    else:
        _item(surface, entry, (95, 122, 437, 420), unlocked)
    if not unlocked:
        _lock(surface, (313, 344))
    text(surface, assets, f'{"특별한 모습" if look else "소중한 물건"}  /  {number:02d}',
         (625, 122), 18, GOLD)
    name = entry['name'] if unlocked else '아직 잠긴 모습' if look else '아직 찾지 못한 물건'
    fit_text(surface, assets, name, (625, 166, 565, 54), 34, CREAM, True, 'midleft')
    pygame.draw.line(surface, LINE, (625, 237), (1190, 237))
    if look:
        message = collection.requirement(data, entry)
        text(surface, assets, '발견한 순간' if unlocked else '발견하는 방법', (625, 275), 18, GOLD)
        wrapped(surface, assets, message, pygame.Rect(625, 319, 550, 110), 24, CREAM, 9)
        if unlocked:
            favorite = data['collection']['favorite'] == number
            ui.control('collection_favorite', (625, 478, 330, 52),
                       '지금 수집장의 표지예요' if favorite else '이 그림을 표지로',
                       primary=not favorite, disabled=favorite)
            text(surface, assets, '수집장을 열 때 가장 먼저 만나요.', (625, 549), 17, MUTED)
    else:
        room = ctx['stories'][number-1]['name']
        text(surface, assets, f'{number:02d}번 방 · {room}', (625, 283), 23, CREAM)
        wrapped(surface, assets, '이 물건과 함께 무사히 돌아온 밤의 기록이에요.' if unlocked
                else '기본 모드에서 이 방의 물건을 챙겨 성공 귀환하면 공개돼요.',
                pygame.Rect(625, 340, 550, 120), 23, CREAM, 9)
    ui.control('collection_close', (1000, 601, 205, 48), '수집장으로')

def _cover(surface, assets, ctx, ui):
    data = ctx['campaign'].data
    entry = collection.catalog()[data['collection']['favorite']-1]
    pygame.draw.rect(surface, (29, 23, 40), (91, 62, 469, 609), border_radius=24)
    _panel(surface, (76, 47, 469, 609), PAPER, GOLD, 24)
    pygame.draw.rect(surface, LINE, (89, 60, 443, 583), 1, border_radius=18)
    _portrait(surface, assets, entry, (105, 83, 411, 495))
    fit_text(surface, assets, entry['name'], (112, 595, 397, 28), 21, INK, True)
    text(surface, assets, '나만의 작은 보물들', (619, 153), 19, GOLD)
    title = assets.title_font(61).render('수집장', True, CREAM)
    surface.blit(title, (613, 201))
    pygame.draw.line(surface, LINE, (619, 299), (1178, 299))
    text(surface, assets, '함께 걸어온 밤마다, 하나씩.', (619, 330), 23, CREAM)
    counts = f'특별한 모습  {len(collection.available(data))}/24     소중한 물건  {len(data["completed"])}/12'
    text(surface, assets, counts, (619, 380), 19, GOLD)
    ui.control('collection_open:looks', (619, 453, 275, 54), '특별한 모습', primary=True)
    ui.control('collection_open:items', (911, 453, 275, 54), '소중한 물건')
    text(surface, assets, '다른 그림을 감상하며 표지를 바꿀 수 있어요.', (619, 540), 17, MUTED)
    ui.control('back', (1001, 612, 185, 42), '쉼터로')

def render_collection(surface, assets, ctx, ui):
    _base(surface)
    data = ctx['campaign'].data
    if ctx.get('collection_detail'):
        _detail(surface, assets, ctx, ui)
        return
    if ctx.get('collection_cover') and data['collection']['favorite'] in collection.available(data):
        _cover(surface, assets, ctx, ui)
        return
    tab = ctx.get('collection_tab', 'looks')
    looks = tab == 'looks'
    count = len(collection.available(data)) if looks else len(data['completed'])
    title = assets.title_font(32).render('수집장', True, CREAM)
    surface.blit(title, (40, 23))
    text(surface, assets, '밤마다 모은 작은 보물', (170, 33), 17, MUTED)
    text(surface, assets, f'{"발견한 모습" if looks else "모은 물건"}  {count} / {24 if looks else 12}',
         (1238, 37), 20, GOLD, anchor='midright')
    ui.control('collection_tab:looks', (40, 73, 160, 34), '특별한 모습', primary=looks)
    ui.control('collection_tab:items', (211, 73, 160, 34), '소중한 물건', primary=not looks)
    if data['collection']['favorite'] is not None:
        ui.control('collection_cover', (1092, 73, 146, 34), '내 표지 보기')
    unlocked = collection.available(data)
    unseen = unlocked - set(data['collection']['seen'])
    total = 24 if looks else 12
    for index in range(total):
        number = index + 1
        # 까망이 네 장은 첫 세로줄에 고정하고 번호도 위에서 아래로 매긴다.
        col, row = (index // 4, index % 4) if looks else (index % 4, index // 4)
        w, h, dx, dy = (190, 128, 202, 136) if looks else (291, 173, 303, 181)
        rect = pygame.Rect(40+col*dx, 121+row*dy, w, h)
        owned = number in (unlocked if looks else data['completed'])
        _panel(surface, rect, PAPER if owned else (60, 49, 73),
               GOLD if looks and number == data['collection']['favorite'] else (119, 99, 116))
        entry = collection.catalog()[index] if looks else ctx['levels'][index]['quest']
        art = pygame.Rect(rect.x+19, rect.y+7, rect.w-38, rect.h-33)
        if looks:
            _portrait(surface, assets, entry, art, owned)
        else:
            _item(surface, entry, art.inflate(-32, -3), owned)
        text(surface, assets, f'{number:02d}', (rect.x+10, rect.y+8), 13,
             (141, 116, 99) if owned else MUTED)
        if not owned:
            _lock(surface, (rect.right-17, rect.y+17))
        elif looks and number in unseen:
            pygame.draw.circle(surface, (180, 105, 77), (rect.right-13, rect.y+13), 4)
        if looks and number == data['collection']['favorite']:
            draw_paw(surface, (rect.right-26, rect.y+5, 22, 22), (173, 118, 42))
        fit_text(surface, assets, entry['name'] if owned else '???',
                 (rect.x+8, rect.bottom-26, rect.w-16, 21), 15 if looks else 17,
                 INK if owned else MUTED, owned)
        ui.control(f'collection_card:{tab}:{number}', rect,
                   entry['name'] if owned else f'{number}번 잠긴 카드', custom=True)
    ui.control('back', (40, 675, 139, 29), '쉼터로')
    text(surface, assets, '방향키 선택 · Enter 감상 · Esc 뒤로', (1238, 690), 14,
         MUTED, anchor='midright')

def move_focus(controls, focus, key):
    """수집장 카드의 실제 배치에 맞춰 방향키로 이동한다."""
    if not controls:
        return 0
    origin = controls[min(focus, len(controls)-1)]['rect'].center
    axis = 0 if key in (pygame.K_LEFT, pygame.K_RIGHT) else 1
    sign = -1 if key in (pygame.K_LEFT, pygame.K_UP) else 1
    candidates = []
    for index, control in enumerate(controls):
        if index == focus or control.get('disabled'):
            continue
        point = control['rect'].center
        forward = (point[axis] - origin[axis]) * sign
        if forward > 0:
            side = abs(point[1-axis] - origin[1-axis])
            candidates.append((forward + side*3, index))
    return min(candidates)[1] if candidates else focus
