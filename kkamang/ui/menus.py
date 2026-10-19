"""Live, data-driven menu screens. Rendering has no gameplay side effects.

Every interactive area is returned to the app as a control. Artwork is decorative;
all progress, scores, labels and prompts are drawn from the current context.
"""
from __future__ import annotations

import math
import pygame

from .theme import (CREAM, INK, NIGHT, PAPER, GOLD, LAVENDER, SAGE,
                    text, wrapped, panel, button)

from .illustration import shelter, cat as draw_cat, ghost, prop
from .likes import draw_likes, draw_choice

W, H = 1280, 720
_COVER_CACHE: dict = {}


def _overlay(surface, rect, color=(25, 19, 37), alpha=175, radius=0):
    r = pygame.Rect(rect)
    layer = pygame.Surface(r.size, pygame.SRCALPHA)
    if radius:
        pygame.draw.rect(layer, (*color[:3], alpha), layer.get_rect(), border_radius=radius)
    else:
        layer.fill((*color[:3], alpha))
    surface.blit(layer, r)


def _art(surface, assets, name, rect):
    r = pygame.Rect(rect)
    if not name:
        return
    name = name if '/' in name else 'art/' + name
    key = (id(assets), name, r.size)
    if key not in _COVER_CACHE:
        src = assets.image(name)
        if src.get_width() == 0 or src.get_height() == 0:
            return
        scale = max(r.w / src.get_width(), r.h / src.get_height())
        scaled = pygame.transform.smoothscale(src, (max(1, round(src.get_width()*scale)),
                                                    max(1, round(src.get_height()*scale))))
        image = pygame.Surface(r.size, pygame.SRCALPHA)
        image.blit(scaled, ((r.w-scaled.get_width())//2, (r.h-scaled.get_height())//2))
        _COVER_CACHE[key] = image
    surface.blit(_COVER_CACHE[key], r)


def _icon(surface, assets, name, rect):
    r = pygame.Rect(rect)
    surface.blit(assets.icon('icons/' + name + '.png', r.size), r)


def _fit_text(surface, assets, value, rect, size=22, color=INK, bold=False, anchor='midleft'):
    r = pygame.Rect(rect)
    value = str(value)
    while size > 13 and assets.font(size, bold).size(value)[0] > r.w:
        size -= 1
    return text(surface, assets, value, getattr(r, anchor), size=size,
                color=color, bold=bold, anchor=anchor)


def _data(ctx):
    camp = ctx.get('campaign')
    return getattr(camp, 'data', camp if isinstance(camp, dict) else {}) or {}


def _stories(ctx):
    entries = ctx.get('stories')
    if entries:
        return entries
    try:
        from ..story import STAGES
        return STAGES
    except ImportError:
        return []


def _story(ctx, number=None):
    n = int(number or ctx.get('stage', 1))
    if number is None and ctx.get('story'):
        return ctx['story']
    all_stories = _stories(ctx)
    return next((s for s in all_stories if int(s.get('stage', 0)) == n),
                {'stage': n, 'name': f'{n}번째 방', 'title': '조금 더 따뜻한 밤',
                 'photo': '함께한 순간', 'before': '흩어진 온기를 찾아 돌아와요.',
                 'lines': [], 'restoration': '방에 작은 불빛이 돌아왔어요.', 'teaser': ''})


def _level(ctx, number=None):
    n = int(number or ctx.get('stage', 1))
    return next((l for l in ctx.get('levels', []) if int(l.get('stage', 0)) == n), {})


def _home_art(completed):
    count = len(completed)
    if count >= 12:
        return 'shelter_after.png'
    if count >= 8:
        return 'tea_memory.png'
    if count >= 4:
        return 'memory_scene.png'
    return 'shelter_before.png'


def _background(surface, assets, name='title_background.png', shade=0, ctx=None, actors=True):
    ctx = ctx or {}
    data = _data(ctx)
    shelter(surface, data.get('completed', []), data.get('cosmetics', {}),
            float(ctx.get('elapsed', 0)), before=ctx.get('home_view')=='before',
            show_actors=actors)
    if shade:
        _overlay(surface, (0, 0, W, H), alpha=min(shade,140))


def _header(surface, assets, title, subtitle='', right=''):
    _overlay(surface, (0, 0, W, 105), alpha=196)
    text(surface, assets, title, (38, 20), size=32, bold=True)
    if subtitle:
        text(surface, assets, subtitle, (40, 67), size=17, color=(224, 213, 203))
    if right:
        panel(surface, (1016, 27, 226, 44), fill=NIGHT, border=GOLD, radius=22)
        text(surface, assets, right, (1129, 49), size=18, anchor='center')


def _footer(surface, assets, note='방향키 선택 · Enter 확인 · Esc 뒤로'):
    _overlay(surface, (0, 676, W, 44), alpha=195)
    text(surface, assets, note, (34, 689), size=15, color=(227, 215, 199))
    text(surface, assets, '까망이와 잠 못 드는 저택', (1246, 689), size=14,
         color=(205, 186, 166), anchor='topright')


def _chip(surface, assets, label, rect, fill=NIGHT, color=CREAM, size=17):
    panel(surface, rect, fill=fill, border=GOLD, radius=18)
    _fit_text(surface, assets, label, pygame.Rect(rect).inflate(-16, -4),
              size=size, color=color, anchor='center')


def _lanterns(surface, assets, completed, rect):
    r = pygame.Rect(rect)
    panel(surface, r, fill=NIGHT, border=GOLD, radius=16)
    cell = r.w / 12
    for i in range(12):
        _icon(surface, assets, 'lantern_on' if i+1 in completed else 'lantern_off',
              (r.x+i*cell+6, r.y+7, cell-12, r.h-20))
        text(surface, assets, str(i+1), (r.x+cell*(i+.5), r.bottom-8), size=11,
             color=GOLD if i+1 in completed else (152, 138, 159), anchor='midbottom')


def _emblem(surface, assets, kind, rect, locked=False, large=False):
    """Original prop illustrations, deliberately distinct from full story scenes."""
    r = pygame.Rect(rect)
    color = (95, 79, 108) if not locked else (156, 142, 139)
    panel(surface, r, fill=color, border=(196, 174, 139), radius=8)
    # Quiet wallpaper and arched window create a consistent album frame.
    for x in range(r.x+16, r.right, 30):
        pygame.draw.line(surface, (104, 88, 116) if not locked else (162, 149, 143),
                         (x, r.y+5), (x, r.bottom-5), 1)
    cx, cy = r.center
    s = min(r.w, r.h) / 125
    def rr(x, y, w, h, col, width=0, radius=3):
        pygame.draw.rect(surface, col, (round(cx+x*s), round(cy+y*s),
                                       max(1, round(w*s)), max(1, round(h*s))),
                         width=width, border_radius=max(1, round(radius*s)))
    def line(p, q, col=GOLD, width=3):
        pygame.draw.line(surface, col, (int(cx+p[0]*s), int(cy+p[1]*s)),
                         (int(cx+q[0]*s), int(cy+q[1]*s)), max(1, round(width*s)))
    def circle(x, y, rad, col):
        pygame.draw.circle(surface, col, (round(cx+x*s), round(cy+y*s)), max(1,round(rad*s)))
    if locked:
        pygame.draw.arc(surface, (224, 210, 181),
                        pygame.Rect(cx-16*s, cy-24*s, 32*s, 36*s), 0, math.pi, max(2, round(5*s)))
        rr(-23, -7, 46, 35, (224, 210, 181), radius=7)
        circle(0, 6, 4, color)
        line((0, 8), (0, 16), color, 4)
        return
    if kind in ('towel', 'scarf'):
        rr(-44, -33, 88, 63, (197, 177, 211), radius=10)
        for x in range(-35, 40, 10):
            line((x, -30), (x, 28), (220, 205, 228), 1)
        for x in range(-37, 42, 8):
            line((x, 30), (x, 39), (226, 214, 229), 2)
        circle(22, -8, 5, (238, 207, 141))
    elif kind in ('sign', 'frame'):
        rr(-45, -35, 90, 68, GOLD, radius=5)
        rr(-38, -28, 76, 54, (116, 94, 135), radius=2)
        circle(0, 6, 12, (237, 208, 157))
        for x,y in [(-16,-7),(-6,-15),(7,-15),(18,-6)]:
            circle(x,y,6,(237,208,157))
    elif kind == 'plant':
        pygame.draw.polygon(surface, (198, 137, 113),
                            [(cx-26*s,cy+10*s),(cx+26*s,cy+10*s),(cx+19*s,cy+41*s),(cx-19*s,cy+41*s)])
        line((0, 10), (0, -35), (174, 199, 148), 4)
        for x, y, angle in [(-17,-15,-25),(15,-28,25),(-13,-34,-20)]:
            pygame.draw.ellipse(surface, (158, 184, 136), (cx+x*s-12*s, cy+y*s-6*s,26*s,13*s))
    elif kind in ('music', 'ensemble'):
        rr(-42, -20, 84, 55, (195, 165, 125), radius=6)
        rr(-36, 8, 72, 20, (247, 227, 185), radius=2)
        for i in range(7):
            line((-34+i*10, 8), (-34+i*10, 27), (107, 88, 98), 1)
        line((7,-15),(7,-43),GOLD,3)
        line((7,-43),(23,-38),GOLD,3)
        circle(0,-15,7,GOLD)
        if kind=='ensemble':
            circle(-22,-36,5,(184,210,169))
    elif kind == 'window':
        rr(-35, -42, 70, 84, (215, 185, 137), radius=26)
        rr(-28, -35, 56, 70, (86, 103, 146), radius=20)
        line((0,-32),(0,35),(220,190,144),4)
        line((-28,0),(28,0),(220,190,144),4)
        circle(13,-22,7,(245,226,172))
    elif kind == 'cushion':
        pygame.draw.ellipse(surface, (177, 175, 141), (cx-48*s,cy-19*s,96*s,61*s))
        pygame.draw.ellipse(surface, (209, 205, 165), (cx-40*s,cy-14*s,80*s,43*s), max(1,round(2*s)))
        circle(0,7,9,(141,135,118))
        for x,y in [(-13,-2),(-5,-9),(6,-9),(15,-1)]:circle(x,y,4,(141,135,118))
    else:
        # Tea, or a book when the chapter has a dedicated scene.
        rr(-34, -29, 68, 57, (206, 178, 136), radius=3)
        rr(-28, -23, 27, 44, (246, 227, 190), radius=1)
        rr(2, -23, 26, 44, (246, 227, 190), radius=1)
        line((0,-27),(0,27),(155,124,109),2)
        for yy in [-12,-3,6]:line((8,yy),(21,yy),(192,169,140),1)
    if large:
        text(surface, assets, '물건 곁에 남은 작은 기억', (cx, r.bottom-15), size=15,
             color=(232,218,193), anchor='midbottom')


def _photo(surface, assets, story, rect, locked=False, large=False, frame='default', cosmetics=None):
    r = pygame.Rect(rect)
    panel(surface, r, fill=(255, 249, 230), border=(179, 153, 123), radius=3)
    inner = r.inflate(-14, -14)
    if not locked:
        n = int(story.get('stage',1))
        key=('memory_v2',n,inner.size,tuple(sorted((cosmetics or {}).items())))
        if key not in _COVER_CACHE:
            scene=pygame.Surface((1280,720),pygame.SRCALPHA)
            shelter(scene,range(1,n+1),cosmetics or {},time=1.7)
            _COVER_CACHE[key]=pygame.transform.smoothscale(scene,inner.size)
        surface.blit(_COVER_CACHE[key],inner)
    else:
        _emblem(surface, assets, story.get('emblem', 'book'), inner, True, large)
    tape = pygame.Surface((max(28, int(r.w*.22)), 12), pygame.SRCALPHA)
    tape.fill((197, 176, 129, 155))
    surface.blit(tape, (r.centerx-tape.get_width()//2, r.y-4))
    if frame != 'default' and not locked:
        frame_color = SAGE if frame == 'leaves' else LAVENDER
        pygame.draw.rect(surface, frame_color, r.inflate(-2,-2), width=4, border_radius=4)
        for x,y in [r.topleft,r.topright,r.bottomleft,r.bottomright]:
            x=max(r.left+10,min(r.right-10,x));y=max(r.top+10,min(r.bottom-10,y))
            if frame=='leaves':
                pygame.draw.ellipse(surface,(191,211,163),(x-8,y-4,17,8))
                pygame.draw.line(surface,(106,137,110),(x-6,y+2),(x+6,y-2),1)
            else:
                pygame.draw.line(surface,GOLD,(x-7,y),(x+7,y),3)
                pygame.draw.line(surface,GOLD,(x,y-7),(x,y+7),3)


def _home_keepsakes(surface, assets, ctx, data):
    """Live vignette makes selected cosmetics visible beside the painted room."""
    r=pygame.Rect(988,409,250,134)
    _overlay(surface,r,alpha=201,radius=15)
    pygame.draw.rect(surface,(185,155,121),r,width=1,border_radius=15)
    text(surface,assets,'까망이가 고른 작은 취향',(1004,419),size=15,color=CREAM)
    cosmetics=data.get('cosmetics',{})
    rug=cosmetics.get('rug','default')
    color={'default':(156,136,172),'stars':(124,117,160),'leaves':(131,156,135)}.get(rug,(156,136,172))
    pygame.draw.ellipse(surface,color,(1005,502,112,27))
    pygame.draw.ellipse(surface,GOLD,(1009,506,104,19),1)
    if rug=='stars':
        for x in (1020,1100):
            pygame.draw.line(surface,GOLD,(x-3,515),(x+3,515),1)
            pygame.draw.line(surface,GOLD,(x,512),(x,518),1)
    elif rug=='leaves':
        for x in (1020,1100):pygame.draw.ellipse(surface,(203,219,176),(x-4,511,8,4))
    if hasattr(assets,'cat'):
        cat=assets.cat((93,89),cosmetics.get('scarf','default'))
    else:cat=assets.icon('art/kkamang_character.png',(93,89))
    surface.blit(cat,(1012,439))
    photos=data.get('photos',[])
    favorite=ctx.get('preferred_photo') or (max(photos) if photos else None)
    if favorite in photos:
        _photo(surface,assets,_story(ctx,favorite),(1123,449,99,73),frame=cosmetics.get('frame','default'),cosmetics=cosmetics)
    else:
        _photo(surface,assets,{'emblem':'frame'},(1123,449,99,73),locked=True,
               frame=cosmetics.get('frame','default'),cosmetics=cosmetics)


class _UI:
    def __init__(self, surface, assets, focus, mouse):
        self.s, self.a, self.focus, self.mouse = surface, assets, focus, mouse
        self.controls = []

    def control(self, id, rect, label, primary=False, disabled=False, custom=False):
        r = pygame.Rect(rect)
        focused = len(self.controls) == self.focus or r.collidepoint(self.mouse)
        self.controls.append({'id': id, 'rect': r, 'label': label, 'disabled': disabled})
        if not custom:
            button(self.s, self.a, r, label, focused=focused, disabled=disabled, primary=primary)
        elif focused and not disabled:
            pygame.draw.rect(self.s, (238, 218, 255), r.inflate(6, 6), width=3, border_radius=12)
        return focused


def _stars(count) -> str:
    count = max(0, min(3, int(count)))
    return '★' * count + '☆' * (3 - count)


def render_menu(surface, assets, screen, ctx, focus=0, mouse=(-1,-1)):
    """Draw one logical 1280×720 screen; return all focusable controls."""
    ui = _UI(surface, assets, focus, mouse)
    data = _data(ctx)
    completed = set(int(x) for x in data.get('completed', []))
    photos = set(int(x) for x in data.get('photos', []))
    badges = data.get('badges', [])
    elapsed = float(ctx.get('elapsed', 0))
    stage = int(ctx.get('stage', 1))
    story = _story(ctx)
    level = _level(ctx)
    mode = ctx.get('mode', data.get('settings', {}).get('mode', 'basic'))
    mode_label = '여유 모드' if mode == 'relaxed' else '기본 모드'

    if screen == 'title':
        _background(surface, assets,ctx=ctx,actors=False)
        draw_cat(surface,(877,595),280,time=elapsed,scarf=data.get('cosmetics',{}).get('scarf','default'))
        ghost(surface,(1110,515),125,time=elapsed,resident=True)
        # The art was composed with quiet space at left for the live title.
        _overlay(surface, (0, 0, 660, H), alpha=115)
        text(surface, assets, '작은 발걸음이 남긴 따뜻한 밤', (68, 110), size=19,
             color=(224, 204, 177))
        surface.blit(assets.title_font(54).render('까망이와',True,CREAM),(64,154))
        surface.blit(assets.title_font(51).render('잠 못 드는 저택',True,CREAM),(64,224))
        pygame.draw.line(surface, GOLD, (70, 312), (410, 312), 2)
        wrapped(surface, assets, '폭풍을 피해 들어온 낯선 집.\n잃어버린 물건과 온기를 찾아, 돌아올 자리를 만들어 주세요.',
                pygame.Rect(70, 337, 470, 90), size=20, color=CREAM, line_gap=9)
        has_save = bool(ctx.get('has_save'))
        if has_save:
            ui.control('continue', (70, 450, 360, 62), '이어서 걷기', primary=True)
            ui.control('new_game', (70, 528, 174, 46), '새로운 밤')
            ui.control('settings', (256, 528, 174, 46), '설정')
        else:
            ui.control('new_game', (70, 450, 360, 62), '저택의 문 열기', primary=True)
            ui.control('settings', (70, 528, 174, 46), '설정')
            ui.control('tutorial', (256, 528, 174, 46), '어떻게 걷나요?')
        ui.control('quit', (70, 590, 174, 44), '나가기')
        if has_save:
            _chip(surface, assets, f'간직한 기억  {len(photos)} / 12', (900, 43, 320, 43))
        _footer(surface, assets, '방향키 선택 · Enter 확인 · 마우스로도 선택할 수 있어요')

    elif screen == 'home':
        old_view = ctx.get('home_view') == 'before'
        _background(surface, assets, 'shelter_before.png' if old_view else _home_art(completed),ctx=ctx)
        title = '처음 만났던 밤' if old_view else '까망이의 쉼터'
        subtitle = '비를 피할 자리는 처음부터 여기 있었어요.' if not completed else '작은 일이 모여, 돌아오고 싶은 집이 되었어요.'
        _header(surface, assets, title, subtitle,
                f'기억 {len(photos)} / 12 · 별 {int(ctx.get("total_stars", 0))} / 36')
        _lanterns(surface, assets, completed, (662, 124, 575, 65))
        if completed:
            latest = _story(ctx, max(completed))
            _overlay(surface, (37, 123, 570, 98), alpha=190, radius=16)
            text(surface, assets, '이 집에 돌아온 변화', (57, 139), size=16, color=GOLD)
            wrapped(surface, assets, latest.get('restoration', ''),
                    pygame.Rect(57, 166, 530, 50), size=19, color=CREAM)
            ui.control('toggle_home_view', (1004, 205, 233, 38),
                       '지금의 쉼터 보기' if old_view else '처음의 쉼터 보기')
        else:
            _overlay(surface, (37, 133, 425, 103), alpha=185, radius=16)
            text(surface, assets, '보송', (57, 148), size=18, color=GOLD, bold=True)
            wrapped(surface, assets, '젖었구나. 괜찮아, 여기서 잠깐 쉬어 가.',
                    pygame.Rect(57, 178, 383, 53), size=21, color=CREAM)
        # Nonintrusive fireflies are decorative and never overlap controls.
        for i in range(9):
            x = 700+i*53
            y = 330+math.sin(elapsed*.7+i*2)*28+(i%3)*50
            pygame.draw.circle(surface, (235, 209, 148), (int(x),int(y)), 2)
        _home_keepsakes(surface,assets,ctx,data)
        _overlay(surface, (0, 560, W, 160), alpha=190)
        note = '열두 방의 불빛이 돌아왔어요. 언제든 다시 만나요.' if len(completed)==12 else '방의 소중한 물건과 온기 4점 이상을 가져오면, 연결등이 켜져요.'
        text(surface, assets, note, (40, 575), size=18)
        ui.control('explore', (38, 613, 300, 52), '기억 속으로 다시 걷기' if len(completed)==12 else '다음 방으로 걷기', primary=True)
        ui.control('album', (352, 613, 215, 52), '기억 앨범')
        ui.control('appearance', (581, 613, 215, 52), '쉼터 꾸미기')
        # ui.control('rooms', (810, 613, 215, 52), '구역 살펴보기')
        ui.control('likes', (810, 613, 215, 52), '유령 호감도')
        ui.control('settings', (1039, 613, 199, 52), '설정')
        _footer(surface, assets)
    elif screen in ('likes', 'ending_conditions'):
        _background(surface, assets, ctx=ctx, actors=False)
        _header(surface, assets, '유령 호감도', '이 방에 남긴 기억이 친구들의 마음에 쌓여요.')
        draw_likes(surface, assets, ctx, ui)
        _footer(surface, assets)

    elif screen == 'memory_choice':
        _background(surface, assets, ctx=ctx, actors=False)
        _header(surface, assets, '이 방에 남길 기억', '두 번의 발걸음 중 어떤 기억을 남길까요?')
        draw_choice(surface, assets, ctx, ui)
        _footer(surface, assets, '방향키 선택 · Enter 확인 · Esc 기존 기억 유지')

    elif screen == 'rooms':
        _background(surface, assets, _home_art(completed), 140,ctx=ctx)
        _header(surface, assets, '열두 방을 잇는 불빛', '가까운 방에서 시작해, 천천히 저택 안쪽으로.',
                f'{len(completed)} / 12 완료 · 별 {int(ctx.get("total_stars", 0))} / 36')
        for n in range(1, 13):
            col, row = (n-1)%4, (n-1)//4
            rect = pygame.Rect(38+col*304, 128+row*160, 288, 143)
            st = _story(ctx, n)
            unlocked = n == 1 or n-1 in completed
            fill = PAPER if unlocked else (104, 92, 112)
            panel(surface, rect, fill=fill, border=GOLD if n in completed else (159, 138, 150), radius=15)
            _icon(surface, assets, 'lantern_on' if n in completed else 'lantern_off', (rect.x+12, rect.y+15, 47, 61))
            text(surface, assets, f'{n:02d}', (rect.x+69, rect.y+14), size=19,
                 color=(128,104,105) if unlocked else (208,196,210), bold=True)
            _fit_text(surface, assets, st.get('name', f'{n}번째 방'), (rect.x+69,rect.y+45,207,31),
                      size=24, color=INK if unlocked else CREAM, bold=True)
            record = data.get('star_records', {}).get(f'stage_{n:02d}', {})
            badge_text = (f'완료  {_stars(record.get("stars", 0))}  최고 {record.get("score", 0)}점'
                          if n in completed else ('이제 들어갈 수 있어요' if unlocked else '앞방의 연결등을 켜 주세요'))
            text(surface, assets, badge_text, (rect.x+16,rect.y+96), size=16,
                 color=(124,103,108) if unlocked else (210,198,213))
            if f'stage_{n:02d}:rich' in badges:
                _icon(surface,assets,'badge_warmth',(rect.right-78,rect.y+8,29,29))
            if f'stage_{n:02d}:quiet' in badges:
                _icon(surface,assets,'badge_unhurt',(rect.right-42,rect.y+8,29,29))
            ui.control(f'stage:{n}', rect, st.get('name', ''), disabled=not unlocked, custom=True)
        ui.control('back', (38, 626, 174, 40), '쉼터로')
        text(surface,assets,'완료한 방은 다시 도전해 별과 최고 점수를 높일 수 있어요.',(240,635),size=17)
        _footer(surface, assets)

    elif screen == 'briefing':
        _background(surface, assets, _home_art(completed), 175,ctx=ctx)
        _header(surface, assets, f'{stage:02d}  {story.get("name", "탐사 준비")}', '필수 물건과 온기를 챙겨, 종이 울리기 전에 돌아와요.', mode_label)
        panel(surface, (38, 126, 805, 492), fill=(55,45,65), border=GOLD, radius=17)
        snapshot = ctx.get('snapshot')
        if snapshot:
            from .field import draw_field
            draw_field(surface, assets, snapshot, pygame.Rect(54, 151, 773, 435), elapsed=elapsed,
                       cosmetics=data.get('cosmetics'))
        else:
            text(surface, assets, '지도를 준비하고 있어요.', (440,370), anchor='center')
        panel(surface, (864, 126, 377, 492), fill=PAPER, border=GOLD, radius=17)
        text(surface, assets, '이번 방에서', (889,147), size=23, color=INK,bold=True)
        wrapped(surface,assets,story.get('before','흩어진 온기를 찾아요.'),pygame.Rect(889,186,326,87),size=18,color=INK)
        pygame.draw.line(surface,(196,174,145),(888,289),(1215,289),1)
        tick_seconds=float(level.get('tick_seconds',.3))
        ticks=(snapshot or {}).get('initial_ticks',level.get('time_budget_ticks',0))
        rows=[('필수 물건',level.get('quest',{}).get('name','물건')),('제한 시간',f'{ticks*tick_seconds:.1f}초'),
              ('순찰 잔상',f'{len(level.get("ghosts",[]))}명'),('선택 샛문',f'{len(level.get("shortcuts",[]))}개')]
        for i,(lab,val) in enumerate(rows):
            y=311+i*39
            text(surface,assets,lab,(889,y),size=17,color=(131,112,112))
            text(surface,assets,val,(1216,y),size=18,color=INK,anchor='topright',bold=True)
        wrapped(surface,assets,'물건 + 온기 4점 이상 → 귀환\n E: 수집 · 장치 · 귀환',pygame.Rect(889,482,326,70),size=17,color=INK)
        _chip(surface,assets,'멈춰 있어도 탐사 시간은 흘러요',(888,568,327,32),fill=(228,217,196),color=INK,size=15)
        ui.control('back',(38,630,165,39),'방 선택으로')
        ui.control('tutorial',(218,630,202,39),'조작 다시 보기')
        ui.control('mode',(866,630,172,39),mode_label+' 변경')
        ui.control('start',(1052,630,189,39),'탐사 시작',primary=True)
        briefing_note = ('여유 모드는 연습 전용이에요 · 별 · 호감도 · 진행 기록이 남지 않아요'
                         if mode == 'relaxed' else
                         '준비 화면에서는 시간이 흐르지 않아요 · Enter로 선택')
        _footer(surface,assets,briefing_note)

    elif screen == 'result':
        outcome=ctx.get('outcome') or {}
        reward=ctx.get('reward') or {}
        practice=bool(reward.get('practice_mode'))
        status=outcome.get('status','FAILED')
        success=status=='COMPLETE'
        low=status=='RETURN_LOW'
        _background(surface,assets,'shelter_before.png',110,ctx=ctx)
        _header(surface,assets,'연습을 마쳤어요' if practice and success else
                ('무사히 돌아왔어요' if success else ('잠깐 쉬어 가요' if low else '괜찮아, 다시 걸으면 돼')),
                f'{stage:02d} {story.get("name", "")} · {mode_label}')
        panel(surface,(42,127,722,478),fill=PAPER,border=GOLD,radius=22)
        _icon(surface,assets,'badge_complete' if success else 'lantern_off',(73,151,76,76))
        text(surface,assets,'연습을 마쳤어요' if practice and success else
             ('연결등이 켜졌어요' if success else ('이번엔 조금 일찍 돌아왔어요' if low else '이번 온기는 남기지 못했어요')),
             (164,163),size=27,color=INK,bold=True)
        score=outcome.get('score') or reward.get('score') or {}
        if success and not practice:
            text(surface,assets,_stars(score.get('stars',0)),(520,171),size=34,color=GOLD,bold=True)
            text(surface,assets,f'{score.get("total",0)}점 / 100점',(520,210),size=23,color=INK,bold=True)
        text(surface,assets,'귀환해서 확정한 온기',(78,253),size=20,color=(134,109,102))
        text(surface,assets,str(outcome.get('banked',0)),(80,291),size=72,color=(174,116,43),bold=True)
        text(surface,assets,'점',(183,334),size=26,color=INK)
        
        tick_seconds = float(level.get("tick_seconds", 0.3))

        used_seconds = ctx.get("display_elapsed_seconds")
        if used_seconds is None:
            used_seconds = outcome.get("ticks_used", 0) * tick_seconds

        rows = [
            ('탐사 중 모은 온기', f'{outcome.get("carried", 0)}점'),
            ('잔상에 놀란 횟수', f'{outcome.get("damage_count", 0)}회'),
            ('걸어간 시간', f'{used_seconds:.1f}초'),
        ]
        for i,(lab,val) in enumerate(rows):
            y=264+i*43
            text(surface,assets,lab,(346,y),size=18,color=(132,111,109))
            text(surface,assets,val,(721,y),size=22,color=INK,bold=True,anchor='topright')
        pygame.draw.line(surface,(202,181,153),(79,397),(726,397),1)
        if success and not practice:
            text(surface,assets,'별 점수 구성',(79,417),size=20,color=INK,bold=True)
            rows=[('되찾은 온기',score.get('warmth',0),60),('귀환 속도',score.get('speed',0),25),('남은 용기',score.get('courage',0),15)]
            for i,(label,value,maximum) in enumerate(rows):
                x=79+i*210
                text(surface,assets,label,(x,456),size=17,color=(132,111,109))
                text(surface,assets,f'{value} / {maximum}',(x,483),size=21,color=INK,bold=True)
            msg='새 기억을 앨범에 담았어요.' if reward.get('new_photo') else '다시 걸은 밤도 기록에 남겼어요.'
            if reward.get('new_stars',0): msg+=f' 별 {reward["new_stars"]}개가 늘었어요!'
            if reward.get('new_cosmetics'):msg+=' 새 꾸미기도 열렸어요!'
            next_points=int(score.get('next_star_points',0))
            star_note='세 개의 별을 모두 얻었어요.' if score.get('stars',0)>=3 else f'다음 별까지 {next_points}점'
            text(surface,assets,star_note+' · '+msg,(79,551),size=17,color=(135,99,98))
        elif success:
            wrapped(surface,assets,
                    '여유 모드는 연습 전용이에요. 이번 결과는 별, 꾸미기, 호감도, 이야기 진행에 반영되지 않아요.',
                    pygame.Rect(79,422,644,116),size=21,color=INK,line_gap=10)
        else:
            msg=('완료 목표는 한 번에 온기 4점 이상 귀환이에요.\n가져온 온기는 기록되고, 다음 방은 아직 닫혀 있어요.' if low else
                 '용기를 모두 잃었어요. 보송과 쉼터에서 쉬어 가요.' if outcome.get('reason')=='courage' else
                 '종이 울려 차가운 안개가 찼어요. 다음엔 조금 일찍 돌아와요.')
            wrapped(surface,assets,msg,pygame.Rect(79,422,644,116),size=21,color=INK,line_gap=10)
        char=(assets.cat((300,350),data.get('cosmetics',{}).get('scarf','default')) if hasattr(assets,'cat')
              else assets.icon('art/kkamang_character.png',(300,350)))
        surface.blit(char,(863,169+int(math.sin(elapsed*2)*4)))
        if success and practice:
            _chip(surface,assets,'연습을 마쳤어요',(831,522,376,49),size=16)
        elif success:
            note=('이번 기억을 대표 기억으로 남겼어요.' if reward.get('friend_memory_replaced') else
                  '기존 대표 기억을 그대로 간직해요.' if reward.get('friend_memory_kept') else
                  '첫 번째 대표 기억을 남겼어요.')
            _chip(surface,assets,note,(831,522,376,49),size=18)
        else:
            _chip(surface,assets,'온기보다 네가 먼저야.',(831,522,376,49),size=22)
        ui.control('retry',(42,624,206,44),'같은 방 다시 걷기')
        ui.control('home',(264,624,173,44),'쉼터로')
        if success and practice:
            ui.control('rooms',(964,624,243,44),'방 선택으로',primary=True)
        elif success:
            label='새 기억 펼치기' if reward.get('new_photo') else ('정원으로' if stage==12 else '다음 방으로')
            ui.control('view_memory' if reward.get('new_photo') or stage==12 else 'next_stage',
                       (964,624,243,44),label,primary=True)
        else:
            ui.control('rooms',(964,624,243,44),'다른 방 살펴보기',primary=True)
        _footer(surface,assets)

    elif screen == 'ending':
        content = ctx.get('ending_content') or {}
        _background(surface,assets,content.get('art', 'ending_scene.png'),ctx=ctx)
        if content.get('art') and (assets.root / 'art' / content['art']).is_file():
            _art(surface, assets, content['art'], (0, 0, W, H))
        _header(surface,assets,content.get('title', '함께 남긴 밤'),content.get('subtitle', '열두 방의 기억 · 평가 전 공통 이야기'),'기억 12 / 12')
        if ctx.get('ending_art_only'):
            has_epilogue = bool(content.get('lines'))
            ui.control('ending_continue',(1000,630,240,50),
                       '후일담 보기' if has_epilogue else '쉼터로 돌아가기',primary=True)
            if has_epilogue:
                ui.control('story_skip',(40,630,200,50),'쉼터로 돌아가기')
            return ui.controls
        panel(surface,(38,488,1204,179),fill=PAPER,border=GOLD,radius=18)
        speaker,line=ctx.get('dialogue') or ('이야기','다음 비 오는 날, 까망이는 다시 저택을 찾아왔어요.')
        text(surface,assets,speaker,(64,508),size=21,color=(145,99,104),bold=True)
        wrapped(surface,assets,line,pygame.Rect(64,546,933,105),size=23,color=INK,line_gap=8)
        index=int(ctx.get('dialogue_index',0))+1
        total=max(1,int(ctx.get('total_dialogue',1)))
        text(surface,assets,f'{min(index,total)} / {total}',(1214,510),size=16,color=(138,119,113),anchor='topright')
        ui.control('ending_continue',(1032,579,184,57),'다음 이야기' if index<total else '쉼터로 돌아가기',primary=True)
        ui.control('story_skip',(1041,123,199,38),'쉼터로 돌아가기')
        _footer(surface,assets,'완료한 방과 기억 앨범은 언제든 다시 찾아갈 수 있어요')

    elif screen == 'story':
        ending=False
        _background(surface,assets,ctx=ctx,actors=False)
        speaker0=(ctx.get('dialogue') or ('보송',''))[0]
        mood='surprised' if ctx.get('story_kind')=='prologue' else ('happy' if int(ctx.get('dialogue_index',0))%2 else 'idle')
        draw_cat(surface,(355,468),240,time=elapsed,mood=mood,
            scarf=data.get('cosmetics',{}).get('scarf','default'))
        ghost(surface,(899,440),183,time=elapsed,resident=True,variant={'이파리':1,'도레':2}.get(speaker0,2 if stage in (7,11) else 0))
        if ctx.get('story_kind')!='prologue':
            prop(surface,story.get('emblem','book'),(594,319,110,110))
        _overlay(surface,(0,0,W,66),alpha=162)
        title='비가 그친 뒤에도' if ending else story.get('title','처음 만난 밤')
        if ctx.get('story_kind')=='prologue':title='비가 몹시 내리던 밤'
        text(surface,assets,title,(36,17),size=26,bold=True)
        if not ending and ctx.get('story_kind')!='prologue':
            text(surface,assets,f'기억 {stage:02d} · {story.get("photo", "")}',(1243,24),size=18,anchor='topright')
        panel(surface,(38,488,1204,179),fill=PAPER,border=GOLD,radius=18)
        dialogue=ctx.get('dialogue') or ('보송','이 집에 작은 불빛이 하나 더 켜졌어.')
        if isinstance(dialogue,dict):speaker,line=dialogue.get('speaker',''),dialogue.get('line','')
        else:speaker,line=dialogue
        text(surface,assets,speaker or '이야기',(64,508),size=21,color=(145,99,104),bold=True)
        wrapped(surface,assets,line,pygame.Rect(64,546,933,105),size=23,color=INK,line_gap=8)
        index=int(ctx.get('dialogue_index',0))+1
        total=int(ctx.get('total_dialogue',1))
        text(surface,assets,f'{min(index,total)} / {max(total,1)}',(1214,510),size=16,color=(138,119,113),anchor='topright')
        # Once a newly earned room memory has fully appeared, let the player
        # choose whether to return home or continue straight to the next room.
        source_lines=story.get('lines', []) if isinstance(story, dict) else []
        final_line_complete=(index == total and len(source_lines) >= total
                             and len(line) >= len(source_lines[total-1][1]))
        can_continue=(ctx.get('story_kind') == 'memory' and not ctx.get('replay')
                      and stage < 12 and final_line_complete)
        if can_continue:
            ui.control('story_finish_next',(836,579,190,58),'다음 방으로')
            ui.control('story_finish_home',(1037,579,177,58),'기억 간직하기',primary=True)
        else:
            ui.control('ending_continue' if ending else 'story_next',(1037,579,177,58),'다음 이야기' if index<total else ('쉼터로' if ending else '기억 간직하기'),primary=True)
        ui.control('story_skip',(1041,81,199,38),'이야기 건너뛰기')
        _footer(surface,assets,'Enter / Space 다음 이야기 · 사진과 보상은 이미 안전하게 반영됐어요' if ctx.get('story_kind')!='prologue' else 'Enter / Space 다음 이야기 · Esc 건너뛰기')

    elif screen == 'album':
        _background(surface,assets,_home_art(completed),155,ctx=ctx)
        _header(surface,assets,'까망이의 기억 앨범','작은 일이 모여, 돌아오고 싶은 집이 되었어요.',f'사진 {len(photos)} / 12')
        panel(surface,(32,119,1216,496),fill=PAPER,border=(170,143,115),radius=21)
        pygame.draw.line(surface,(214,194,163),(634,137),(634,594),3)
        text(surface,assets,'처음부터 열두 번째 밤까지',(57,139),size=20,color=INK,bold=True)
        selected=int(ctx.get('photo_selected',max(photos) if photos else 1))
        for n in range(1,13):
            col,row=(n-1)%4,(n-1)//4
            r=pygame.Rect(55+col*142,183+row*129,128,112)
            acquired=n in photos
            st=_story(ctx,n)
            _photo(surface,assets,st,(r.x,r.y,r.w,83),locked=not acquired,cosmetics=data.get('cosmetics',{}))
            text(surface,assets,f'{n:02d}',(r.x+4,r.y+88),size=17,color=INK,bold=True)
            text(surface,assets,'담은 기억' if acquired else '아직 빈자리',(r.right-4,r.y+90),size=13,
                 color=(139,118,108),anchor='topright')
            if n==selected:
                pygame.draw.rect(surface,(158,124,176),r.inflate(7,7),width=3,border_radius=7)
            ui.control(f'photo:{n}',r,f'{n}번째 사진',custom=True)
        st=_story(ctx,selected)
        acquired=selected in photos
        _fit_text(surface,assets,f'{selected:02d} {st.get("photo", "아직 빈자리")}',(661,137,557,35),size=27,color=INK,bold=True)
        _photo(surface,assets,st,(665,190,553,286),locked=not acquired,large=True,
               frame=data.get('cosmetics',{}).get('frame','default'),cosmetics=data.get('cosmetics',{}))
        caption=st.get('restoration','함께한 작은 순간이 사진에 담겼어요.') if acquired else '물건과 온기를 함께 가져오면, 이 방의 기억이 여기에 담겨요.'
        wrapped(surface,assets,caption,pygame.Rect(665,492,553,66),size=18,color=INK)
        ui.control('replay_memory',(665,559,264,42),'기억 다시 보기',primary=True,disabled=not acquired)
        ui.control('visit_photo',(943,559,275,42),'그 방으로 가기',disabled=not acquired)
        ui.control('back',(36,630,181,39),'쉼터로')
        text(surface,assets,'각 구역을 처음 완료하면 사진 한 장을 간직해요.',(246,638),size=17)
        ui.control('favorite_photo',(992,630,246,39),'대표 기억으로 간직하기' if ctx.get('preferred_photo')!=selected else '대표 기억으로 간직 중',disabled=not acquired)
        _footer(surface,assets)

    elif screen == 'appearance':
        _background(surface,assets,_home_art(completed),130,ctx=ctx)
        cosmetic_stars=max(int(ctx.get('total_stars',0)),int(data.get('legacy_cosmetic_stars',0)))
        _header(surface,assets,'까망이의 작은 취향','꾸미기는 모습만 바꿔요. 탐사의 능력과 기록 조건은 같아요.',f'모은 별 {cosmetic_stars} / 36')
        panel(surface,(37,125,474,485),fill=(70,57,82),border=GOLD,radius=20)
        # The selected rug and scarf are live shapes, not baked into the art.
        cosmetic=data.get('cosmetics',{})
        rug=cosmetic.get('rug','default')
        rugcolor={'default':(146,126,163),'stars':(124,117,160),'leaves':(131,156,135)}.get(rug,(146,126,163))
        pygame.draw.ellipse(surface,rugcolor,(77,439,394,100))
        pygame.draw.ellipse(surface,(223,195,145),(85,447,378,84),3)
        if rug!='default':
            for i in range(9):
                x=111+i*37;y=477+math.sin(i*2)*17
                if rug=='stars':
                    pygame.draw.line(surface,GOLD,(x-4,y),(x+4,y),2)
                    pygame.draw.line(surface,GOLD,(x,y-4),(x,y+4),2)
                else:pygame.draw.ellipse(surface,(189,209,163),(x-6,y-3,12,7))
        cat=(assets.cat((287,332),cosmetic.get('scarf','default')) if hasattr(assets,'cat')
             else assets.icon('art/kkamang_character.png',(287,332)))
        surface.blit(cat,(132,160))
        scarf=cosmetic.get('scarf','default')
        colors={'default':SAGE,'peach':(222,157,144),'lavender':(179,156,206)}
        _chip(surface,assets,{'default':'세이지 목도리','peach':'복숭아 목도리','lavender':'라벤더 목도리'}.get(scarf,''),
              (135,544,279,39),fill=colors.get(scarf,SAGE),color=INK)
        text(surface,assets,'현재 선택한 모습',(275,142),size=21,anchor='midtop')
        kind=ctx.get('appearance_kind','scarf')
        tabs=[('scarf','목도리'),('rug','러그'),('frame','액자')]
        for i,(k,label) in enumerate(tabs):
            ui.control(f'appearance_tab:{k}',(539+i*235,126,222,46),label,primary=kind==k)
        panel(surface,(537,193,704,417),fill=PAPER,border=GOLD,radius=18)
        choices={'scarf':[('default','세이지'),('peach','복숭아'),('lavender','라벤더')],
                 'rug':[('default','첫 러그'),('stars','별빛 무늬'),('leaves','잎사귀 무늬')],
                 'frame':[('default','나무 액자'),('leaves','잎사귀 액자'),('stars','별빛 액자')]}
        threshold={'scarf':6,'rug':12,'frame':18}.get(kind,6)
        current=cosmetic.get(kind,'default')
        text(surface,assets,dict(tabs).get(kind,'목도리')+' 고르기',(564,213),size=25,color=INK,bold=True)
        for i,(value,label) in enumerate(choices.get(kind,choices['scarf'])):
            r=pygame.Rect(562+i*220,268,201,219)
            unlocked=value=='default' or cosmetic_stars>=threshold
            panel(surface,r,fill=(245,232,205) if unlocked else (221,210,193),border=GOLD if current==value else (192,171,149),radius=12)
            if kind=='scarf':
                c=colors.get(value,SAGE)
                pygame.draw.rect(surface,c,(r.x+39,r.y+38,122,45),border_radius=15)
                pygame.draw.polygon(surface,c,[(r.x+113,r.y+60),(r.x+158,r.y+118),(r.x+120,r.y+119),(r.x+94,r.y+65)])
            elif kind=='rug':
                c={'default':(164,145,178),'stars':(124,117,160),'leaves':(131,156,135)}[value]
                pygame.draw.ellipse(surface,c,(r.x+21,r.y+32,160,88))
                pygame.draw.ellipse(surface,GOLD,(r.x+29,r.y+40,144,72),2)
                text(surface,assets,{'default':'','stars':'✦','leaves':'잎'}.get(value,''),
                     (r.centerx,r.y+74),size=32,color=CREAM,anchor='center')
            else:
                pygame.draw.rect(surface,{'default':(181,142,106),'leaves':SAGE,'stars':LAVENDER}[value],(r.x+42,r.y+25,117,110),border_radius=7)
                pygame.draw.rect(surface,(94,73,112),(r.x+52,r.y+35,97,90),border_radius=3)
                _icon(surface,assets,'paw_cursor',(r.x+70,r.y+51,61,61))
            text(surface,assets,label,(r.centerx,r.y+146),size=21,color=INK,anchor='midtop',bold=True)
            text(surface,assets,'사용 중' if current==value else ('선택하기' if unlocked else f'별 {threshold}개 필요'),
                 (r.centerx,r.y+180),size=16,color=(134,107,99),anchor='midtop')
            if not unlocked:_icon(surface,assets,'badge_locked',(r.right-40,r.y+8,30,30))
            ui.control(f'cosmetic:{kind}:{value}',r,label,disabled=not unlocked,custom=True)
        wrapped(surface,assets,'별은 소비하지 않아요. 방마다 최고 별만 합산되며,\n6개·12개·18개에서 새로운 꾸미기가 열려요.',pygame.Rect(566,519,639,67),size=18,color=INK)
        ui.control('back',(38,630,181,39),'쉼터로')
        _footer(surface,assets)

    elif screen == 'settings':
        paused=bool(ctx.get('from_game'))
        snapshot=ctx.get('background_snapshot') or ctx.get('snapshot')
        if paused and snapshot:
            from .field import draw_game
            draw_game(surface,assets,snapshot,story,elapsed=elapsed,cosmetics=data.get('cosmetics'))
            _overlay(surface,(0,0,W,H),alpha=195)
        else:_background(surface,assets,_home_art(completed),165,ctx=ctx)
        panel(surface,(278,51,724,609),fill=PAPER,border=GOLD,radius=25)
        text(surface,assets,'잠깐 쉬어 갈까요?' if paused else '나에게 편안한 밤',(640,82),size=32,color=INK,bold=True,anchor='midtop')
        text(surface,assets,'쉬는 동안 탐사 시간은 흐르지 않아요.' if paused else '설정은 다음에 돌아와도 기억해 둘게요.',
             (640,130),size=18,color=(129,106,105),anchor='midtop')
        settings=data.get('settings',{})
        for i,(key,label) in enumerate([('music','배경 음악'),('sfx','효과음')]):
            y=187+i*72
            text(surface,assets,label,(318,y+13),size=21,color=INK,bold=True)
            value=float(settings.get(key,.5))
            panel(surface,(504,y+12,279,25),fill=(219,205,184),border=(194,173,143),radius=12)
            pygame.draw.rect(surface,(181,147,100),(508,y+16,max(1,int(271*value)),17),border_radius=9)
            text(surface,assets,f'{round(value*100)}%',(836,y+13),size=20,color=INK,anchor='midleft')
            ui.control(key+'_down',(452,y,43,49),'−')
            ui.control(key+'_up',(913,y,43,49),'+')
        text(surface,assets,'화면 흔들림',(318,348),size=21,color=INK,bold=True)
        ui.control('shake',(749,331,207,50),'켜짐' if settings.get('shake',True) else '꺼짐')
        text(surface,assets,'탐사 시간',(318,418),size=21,color=INK,bold=True)
        ui.control('mode',(749,401,207,50),mode_label,disabled=paused)
        if ctx.get('save_error'):
            text(surface,assets,'진행이 아직 저장되지 않았어요.',(318,474),size=16,color=(155,101,100))
            ui.control('save_retry',(749,465,207,39),'저장 다시 시도')
        else:
            text(surface,assets,'기본 / 여유 모드의 기록은 따로 남겨요.',(318,470),size=16,color=(138,113,107))
        if paused:
            ui.control('resume',(318,520,639,55),'계속 걷기',primary=True)
            ui.control('restart',(318,591,309,44),'같은 방 다시 시작')
            ui.control('home',(644,591,313,44),'탐사 그만두기')
        else:
            ui.control('back',(318,520,639,55),'설정 닫기',primary=True)
            ui.control('tutorial',(318,591,309,44),'조작 안내')
            ui.control('title',(644,591,313,44),'시작 화면으로')
        _footer(surface,assets)

    elif screen == 'tutorial':
        _background(surface,assets,'shelter_before.png',160,ctx=ctx)
        _header(surface,assets,'까망이의 첫 탐사','천천히 읽고 출발해도 괜찮아요. 지금은 시간이 흐르지 않아요.')
        cards=[('01','작은 발걸음','방향키 / WASD로 한 칸씩 이동해요.\n움직이지 않아도 시간은 흘러요.','paw_cursor'),
               ('02','온기는 자동 획득','온기는 자동 획득!\n물건·장치·귀환은 같은 칸에서 E!','warmth_3'),
               ('03','물건과 온기 함께','방의 필수 물건 + 온기 4점.\n둘을 모두 챙겨 귀환해야 해요.','lantern_on'),
               ('04','돌아올 자리','목표를 채운 뒤 귀환점에서 E!\n조건 부족 시 탐사는 계속돼요.','return_portal'),
               ('05','순찰 잔상을 조심','다음 칸과 이동 예고를 살펴요.\n닿으면 용기가 한 칸 줄어요.','echo_body'),
               ('06','종이 울리기 전에','시간 초과나 용기 0이면 실패.\nEsc로 언제든 잠깐 쉴 수 있어요.','bell')]
        for i,(num,title,desc,icon) in enumerate(cards):
            r=pygame.Rect(39+(i%3)*407,126+(i//3)*231,389,213)
            panel(surface,r,fill=PAPER,border=GOLD,radius=17)
            _icon(surface,assets,icon,(r.x+19,r.y+20,60,60))
            text(surface,assets,num,(r.right-23,r.y+20),size=18,color=(177,146,113),anchor='topright')
            text(surface,assets,title,(r.x+95,r.y+40),size=23,color=INK,bold=True)
            wrapped(surface,assets,desc,pygame.Rect(r.x+24,r.y+108,343,86),size=18,color=INK,line_gap=9)
        ui.control('tutorial_done',(880,615,357,49),'알겠어요, 준비됐어요',primary=True)
        _footer(surface,assets,'탐사 중에는 하단 조작 안내도 확인할 수 있어요')

    elif screen in ('confirm','save_error'):
        _background(surface,assets,_home_art(completed),195,ctx=ctx)
        is_save=screen=='save_error'
        title='기억을 파일에 담지 못했어요' if is_save else ctx.get('confirm_title','잠깐, 확인할까요?')
        message=('이번 진행은 지금 열린 게임 안에 남아 있어요.\n다시 저장하거나, 저장 없이 계속할 수 있어요.' if is_save else
                 ctx.get('confirm_text','현재 탐사를 마칠까요? 이번에 모은 온기는 확정되지 않아요.'))
        panel(surface,(285,167,710,379),fill=PAPER,border=GOLD,radius=23)
        _icon(surface,assets,'lantern_off' if is_save else 'paw_cursor',(320,195,66,66))
        _fit_text(surface,assets,title,(407,205,548,47),size=28,color=INK,bold=True)
        wrapped(surface,assets,message,pygame.Rect(326,294,627,125),size=23,color=INK,line_gap=10)
        if is_save:
            ui.control('save_retry',(325,454,308,55),'다시 저장하기',primary=True)
            ui.control('save_continue',(650,454,306,55),'이번 실행에서 계속')
        else:
            ui.control('confirm_no',(325,454,308,55),'아니요, 돌아갈래요',primary=True)
            ui.control('confirm_yes',(650,454,306,55),ctx.get('confirm_yes_label','네, 진행할게요'))
        _footer(surface,assets,'Esc를 누르면 돌아가요' if not is_save else '게임을 닫기 전에 저장을 다시 시도해 주세요')
    else:
        _background(surface,assets,'title_background.png',180,ctx=ctx)
        _header(surface,assets,'잠깐 쉬어 가요',ctx.get('message','쉼터로 돌아갈 수 있어요.'))
        ui.control('home',(465,565,350,58),'쉼터로',primary=True)
        _footer(surface,assets)

    # In-page toast never intercepts controls or changes state.
    if ctx.get('toast') and screen not in ('save_error','confirm'):
        _chip(surface,assets,str(ctx['toast']),(340,110,600,40),fill=NIGHT,color=CREAM,size=17)
    return ui.controls
