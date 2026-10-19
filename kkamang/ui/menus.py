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
from .map_sprites import map_image
from .ghost_sprites import ghost_frame
from .memories import draw_memory, draw_cinematic

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
                {'stage': n, 'name': f'{n}번째 방', 'title': '기억',
                 'photo': '', 'before': '아이템 + 온기 4점 이상',
                 'lines': [], 'restoration': '', 'teaser': ''})


def _level(ctx, number=None):
    n = int(number or ctx.get('stage', 1))
    return next((l for l in ctx.get('levels', []) if int(l.get('stage', 0)) == n), {})


def _regions(ctx, memory_id):
    completed = set(_data(ctx).get('completed', []))
    return [stage in completed for stage in range((memory_id-1)*4+1, memory_id*4+1)]


def _memory_label(memory_id, regions):
    return f'기억 {memory_id} · {sum(regions)} / 4'


def _memory_art(surface, memory_id, regions, rect, **kwargs):
    """Fit the whole fixed picture inside the UI without cropping or stretching."""
    bounds = pygame.Rect(rect)
    scale = min(bounds.w / 1280, bounds.h / 720)
    picture = pygame.Rect(0, 0, max(1, round(1280*scale)), max(1, round(720*scale)))
    picture.center = bounds.center
    draw_memory(surface, memory_id, regions, picture, **kwargs)


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
            float(ctx.get('elapsed', 0)),
            before=ctx.get('scene')=='home' and ctx.get('home_view')=='before',
            show_actors=actors, released=ctx.get('released', False),
            story_time=float(ctx.get('scene_elapsed', 0)))
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
    """스테이지 클리어 여부에 따라 켜진 랜턴과 꺼진 랜턴을 표시합니다."""
    r = pygame.Rect(rect)

    # PNG 테두리는 크기별로 한 번만 준비해서 재사용합니다.
    frame_key = ('lantern_frame', id(assets), r.size)

    if frame_key not in _COVER_CACHE:
        source = assets.image('art/map/lantern_frame.png')

        # PNG 바깥의 투명 여백만 제외합니다.
        # 테두리 안쪽의 투명 영역은 그대로 유지됩니다.
        bounds = source.get_bounding_rect(min_alpha=10)
        frame = source.subsurface(bounds)

        # 도트 느낌을 유지하면서 기존 테두리 크기에 맞춥니다.
        _COVER_CACHE[frame_key] = pygame.transform.scale(
            frame,
            r.size,
        )

    surface.blit(_COVER_CACHE[frame_key], r.topleft)

    # 랜턴 이미지 영역의 기존 너비는 유지합니다.
    lantern_width = int(r.w / 12 - 12)

    # 랜턴 중심 사이의 거리
    cell = 38

    # 랜턴 12개를 테두리 안에서 가운데 정렬합니다.
    total_width = cell * 11 + lantern_width
    start_x = r.centerx - total_width / 2

    for i in range(12):
        stage_number = i + 1
        is_completed = stage_number in completed

        lantern_rect = pygame.Rect(
            start_x + i * cell,
            r.y + 7,
            lantern_width,
            r.h - 20,
        )

        if is_completed:
            # 클리어한 스테이지: map.png의 켜진 랜턴을 사용합니다.
            image = map_image('lantern', lantern_rect.size)
        else:
            # 미클리어 스테이지: 별도로 추가한 꺼진 랜턴을 사용합니다.
            # icon()은 투명 여백을 제외하고 이미지 비율을 유지합니다.
            image = assets.icon(
                'art/map/lantern_off.png',
                lantern_rect.size,
            )
        # 랜턴 이미지 화면에 표시
        surface.blit(image, lantern_rect.topleft)



def _home_keepsakes(surface, assets, ctx, data):
    r = pygame.Rect(988, 409, 250, 134)
    _overlay(surface, r, alpha=201, radius=15)
    text(surface, assets, '기억 앨범', (1005, 422), size=16, color=CREAM)
    for memory_id in range(1, 4):
        regions = _regions(ctx, memory_id)
        x = 1006 + (memory_id-1)*74
        _memory_art(surface, memory_id, regions, (x, 453, 66, 48))
        text(surface, assets, f'{sum(regions)}/4', (x+33, 515), size=14,
             color=CREAM, anchor='center')


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
        # 시작 화면도 앉아 있는 까망이 PNG를 사용하고 발 위치를 유지합니다.
        title_cat = assets.icon('art/kkamang_wait.png', (280, 315))
        surface.blit(title_cat, title_cat.get_rect(midbottom=(877, 595)))
        ghost(surface,(1110,515),125,time=elapsed,resident=True)
        # The art was composed with quiet space at left for the live title.
        _overlay(surface, (0, 0, 660, H), alpha=115)
        text(surface, assets, 'KKAMANG MANOR', (68, 110), size=19,
             color=(224, 204, 177))
        surface.blit(assets.title_font(54).render('까망이와',True,CREAM),(64,154))
        surface.blit(assets.title_font(51).render('잠 못 드는 저택',True,CREAM),(64,224))
        pygame.draw.line(surface, GOLD, (70, 312), (410, 312), 2)
        wrapped(surface, assets, '아이템 1개 + 온기 4점 이상\n귀환점에서 E',
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
            _chip(surface, assets, f'완료한 방  {len(completed)} / 12', (900, 43, 320, 43))
        _footer(surface, assets, '방향키 선택 · Enter 확인 · 마우스로도 선택할 수 있어요')

    elif screen == 'home':
        old_view = ctx.get('home_view') == 'before'
        _background(surface, assets, ctx=ctx)
        _header(surface, assets, '쉼터', '',
                f'완료 {len(completed)} / 12 · 별 {int(ctx.get("total_stars", 0))} / 36')
        _lanterns(surface, assets, completed, (662, 124, 575, 65))
        for memory_id in range(1, 4):
            regions = _regions(ctx, memory_id)
            r = pygame.Rect(38+(memory_id-1)*184, 126, 170, 108)
            panel(surface, r, fill=NIGHT, border=GOLD, radius=12)
            _memory_art(surface, memory_id, regions, (r.x+8, r.y+8, 154, 65))
            text(surface, assets, _memory_label(memory_id, regions),
                 (r.centerx, r.bottom-19), size=15, anchor='center')
            ui.control(f'memory:{memory_id}', r, f'기억 {memory_id}', custom=True)
        if completed:
            ui.control('toggle_home_view', (1004,205,233,38),
                       '현재 쉼터' if old_view else '처음 쉼터')
        if len(completed) == 12:
            _chip(surface, assets, f'온기 회수 {ctx.get("warmth_count",0)} / 12',
                  (662,205,322,38), size=17)
            ui.control('ending',(1004,253,233,40),'후일담 다시 보기')
        _overlay(surface,(0,560,W,160),alpha=190)
        note = ('온기 회수: 각 방에서 아이템 + 온기 10점 이상 성공 귀환 · 12개 방 달성 시 진엔딩'
                if len(completed)==12 else '귀환 조건: 아이템 1개 + 온기 4점 이상 · 귀환점에서 E')
        _fit_text(surface,assets,note,(40,567,1198,35),size=17,color=CREAM)
        ui.control('explore',(38,613,300,52),'방 다시 탐험' if len(completed)==12 else '탐험',primary=True)
        ui.control('album',(352,613,215,52),'기억 앨범')
        ui.control('appearance',(581,613,215,52),'꾸미기')
        ui.control('likes',(810,613,215,52),'호감도 기록')
        ui.control('settings',(1039,613,199,52),'설정')
        _footer(surface,assets)
    elif screen in ('likes', 'ending_conditions'):
        _background(surface, assets, ctx=ctx, actors=False)
        _header(surface, assets, '호감도 기록', '대표 기록 기준')
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
        _fit_text(surface,assets,'다시 도전해 별과 최고 점수를 높여요. 온기 10점 × 열두 방은 온기 회복 후일담을 열어요.',
                  (240,628,998,33),size=17,color=CREAM)
        _footer(surface, assets)

    elif screen == 'briefing':
        _background(surface, assets, _home_art(completed), 175,ctx=ctx)
        _header(surface, assets, f'{stage:02d}  {story.get("name", "탐사 준비")}', '아이템 + 온기 4점 이상 · 귀환점 E', mode_label)
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
              ('순찰 유령',f'{len(level.get("ghosts",[]))}명'),('선택 샛문',f'{len(level.get("shortcuts",[]))}개')]
        for i,(lab,val) in enumerate(rows):
            y=311+i*39
            text(surface,assets,lab,(889,y),size=17,color=(131,112,112))
            _fit_text(surface,assets,val,(986,y,230,28),size=18,color=INK,anchor='topright',bold=True)
        badge_notes = [
            ('badge_warmth', '온기 10점 이상 귀환 · 수집 배지'),
            ('badge_unhurt', '유령에게 잡히지 않고 귀환 · 무피격 배지'),
        ]
        for i, (icon, label) in enumerate(badge_notes):
            y = 481 + i * 34
            _icon(surface,assets,icon,(889,y,26,26))
            _fit_text(surface,assets,label,(922,y+2,292,24),size=15,color=INK)
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
            ('유령에 놀란 횟수', f'{outcome.get("damage_count", 0)}회'),
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
            msg='기억 영역 +1' if reward.get('first_complete') else '탐사 기록 저장'
            if reward.get('new_cosmetics'):msg+=' 새 꾸미기도 열렸어요!'
            if reward.get('new_ending'):
                msg='온기 회수 12 / 12 · 진엔딩 해금'
            next_points=int(score.get('next_star_points',0))
            stars = int(score.get('stars', 0))
            star_note = (f'이번 탐험에서 별 {stars}개를 얻었어요.' if stars else
                         f'다음 별까지 {next_points}점')
            text(surface,assets,star_note,(79,536),size=16,color=(135,99,98))
            _fit_text(surface,assets,msg,(79,559,645,32),size=17,color=(135,99,98))
        elif success:
            wrapped(surface,assets,
                    '여유 모드는 연습 전용이에요. 이번 결과는 별, 꾸미기, 호감도, 이야기 진행에 반영되지 않아요.',
                    pygame.Rect(79,422,644,116),size=21,color=INK,line_gap=10)
        else:
            msg=('완료 목표는 한 번에 온기 4점 이상 귀환이에요.\n가져온 온기는 기록되고, 다음 방은 아직 닫혀 있어요.' if low else
                 '용기를 모두 잃었어요. 보송과 쉼터에서 쉬어 가요.' if outcome.get('reason')=='courage' else
                 '종이 울려 차가운 안개가 찼어요. 다음엔 조금 일찍 돌아와요.')
            wrapped(surface,assets,msg,pygame.Rect(79,422,644,116),size=21,color=INK,line_gap=10)
        # 탐사 결과 화면의 까망이는 앉아 있는 PNG를 사용합니다.
        char = assets.icon('art/kkamang_wait.png', (300, 350))
        surface.blit(char,(863,169+int(math.sin(elapsed*2)*4)))
        if success and practice:
            _chip(surface,assets,'연습을 마쳤어요',(831,522,376,49),size=16)
        elif success:
            note=('이번 기억을 대표 기억으로 남겼어요.' if reward.get('friend_memory_replaced') else
                  '기존 대표 기억을 그대로 간직해요.' if reward.get('friend_memory_kept') else
                  '첫 번째 대표 기억을 남겼어요.')
            _chip(surface,assets,note,(831,522,376,49),size=18)
        else:
            _chip(surface,assets,'기록 유지 · 다시 도전 가능',(831,522,376,49),size=22)
        ui.control('retry',(42,624,206,44),'같은 방 다시 걷기')
        ui.control('home',(264,624,173,44),'쉼터로')
        if success and practice:
            ui.control('rooms',(964,624,243,44),'방 선택으로',primary=True)
        elif success:
            if reward.get('first_complete'):
                ident,label='view_memory','기억 공개'
            elif reward.get('new_ending'):
                ident,label='ending','진엔딩 보기'
            elif stage==12:
                ident,label='ending','정원으로'
            else:
                ident,label='next_stage','다음 방으로'
            ui.control(ident,(964,624,243,44),label,primary=True)
        else:
            ui.control('rooms',(964,624,243,44),'다른 방 살펴보기',primary=True)
        _footer(surface,assets)

    elif screen == 'ending':
        extended = ctx.get('ending_variant') == 'extended'
        draw_cinematic(surface, 'true_end' if extended else 'basic_end',
                       float(ctx.get('scene_elapsed', elapsed)), completed=completed,
                       cosmetics=data.get('cosmetics'), released=extended)
        # Spoken words appear only here, after the silent resolution.
        _header(surface,assets,'후일담', '', '진엔딩' if extended else '기본 엔딩')
        panel(surface,(38,583,1204,84),fill=PAPER,border=GOLD,radius=18)
        speaker,line=ctx.get('dialogue') or ('','')
        text(surface,assets,speaker,(64,592),size=18,color=(145,99,104),bold=True)
        wrapped(surface,assets,line,pygame.Rect(64,622,933,34),size=23,color=INK,line_gap=5)
        index=int(ctx.get('dialogue_index',0))+1
        total=max(1,int(ctx.get('total_dialogue',3)))
        text(surface,assets,f'{min(index,total)} / {total}',(1011,594),size=16,
             color=(138,119,113),anchor='topright')
        ui.control('ending_continue',(1032,600,184,49),'다음' if index<total else '쉼터로',primary=True)
        ui.control('story_skip',(1041,123,199,38),'넘기기')
        _footer(surface,assets,'Enter / Space 다음 · Esc 넘기기')

    elif screen == 'story':
        kind=ctx.get('cinematic_kind') or ctx.get('story_kind') or 'memory'
        scene_time=float(ctx.get('scene_elapsed',0))
        if kind == 'memory':
            memory_id=int(ctx.get('memory_id',(stage-1)//4+1))
            regions=ctx.get('regions',_regions(ctx,memory_id))
            surface.fill((26,25,39))
            panel(surface,(28,80,1224,548),fill=(61,49,64),border=GOLD,radius=18)
            _memory_art(surface,memory_id,regions,(44,96,1192,516),
                        previous_regions=ctx.get('previous_regions'),
                        reveal_region=ctx.get('reveal_region'),
                        progress=float(ctx.get('reveal_progress',1)),
                        item_kind=ctx.get('item_kind'),elapsed=scene_time)
            text(surface,assets,_memory_label(memory_id,regions),(38,27),size=25,bold=True)
            ready=float(ctx.get('reveal_progress',1))>=1
            if ready and not ctx.get('replay') and stage<12:
                ui.control('story_finish_next',(826,636,190,40),'다음 방으로')
                ui.control('story_finish_home',(1034,636,204,40),'쉼터로',primary=True)
            else:
                ui.control('story_next',(1034,636,204,40),'계속',primary=True)
        else:
            draw_cinematic(surface,kind,scene_time,completed=completed,
                           cosmetics=data.get('cosmetics'),released=ctx.get('released',False))
            ui.control('story_next',(1034,619,204,45),'계속',primary=True)
        ui.control('story_skip',(1041,20,199,38),'넘기기')

    elif screen == 'album':
        _background(surface,assets,shade=140,ctx=ctx,actors=False)
        selected=max(1,min(3,int(ctx.get('memory_selected',1))))
        full_count=sum(all(_regions(ctx,m)) for m in range(1,4))
        _header(surface,assets,'기억 앨범','',f'완성 {full_count} / 3')
        panel(surface,(32,120,1216,490),fill=PAPER,border=GOLD,radius=18)
        for memory_id in range(1,4):
            regions=_regions(ctx,memory_id)
            r=pygame.Rect(53,142+(memory_id-1)*144,288,128)
            _memory_art(surface,memory_id,regions,(r.x,r.y,184,104))
            text(surface,assets,f'기억 {memory_id}',(r.x+201,r.y+22),size=19,color=INK,bold=True)
            text(surface,assets,f'{sum(regions)} / 4',(r.x+201,r.y+62),size=18,color=INK)
            if selected==memory_id:
                pygame.draw.rect(surface,(158,124,176),r.inflate(8,8),3,border_radius=8)
            ui.control(f'memory:{memory_id}',r,f'기억 {memory_id}',custom=True)
        regions=_regions(ctx,selected)
        _memory_art(surface,selected,regions,(365,143,852,395))
        text(surface,assets,_memory_label(selected,regions),(382,554),size=22,color=INK,bold=True)
        ui.control('replay_memory',(818,552,190,40),'다시 보기',primary=True,disabled=not any(regions))
        ui.control('visit_memory',(1024,552,190,40),'방 선택',disabled=not any(regions))
        ui.control('back',(38,630,181,39),'쉼터로')
        _footer(surface,assets,'완료한 방만 그림에 공개됩니다 · 같은 방 재도전은 기억 영역을 늘리지 않습니다')

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
            text(surface,assets,'여유 모드는 연습 전용 · 진행 기록을 남기지 않아요.',(318,470),size=16,color=(138,113,107))
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
        # file은 개별 PNG, map은 맵 PNG의 그림, ghost는 유령 PNG의 한 프레임입니다.
        cards = [
            (
                '01', '조작키',
                'WASD / 방향키로 이동해요.\n'
                'E로 필수 물건 획득·상호작용!',
                (('file', 'art/control.png'),),
            ),
            (
                '02', '스테이지 클리어',
                '온기 4점 이상 + 필수 물건 획득!\n'
                '귀환점에서 E를 누르면 클리어해요.',
                (('map', 'warmth_1'), ('map', 'return_open')),
            ),
            (
                '03', '게임 오버',
                '제한 시간이 끝나거나\n'
                '목숨 3개를 모두 잃으면\n'
                '탐사가 종료돼요.',
                (('map', 'heart_empty'),),
            ),
            (
                '04', '유령을 조심해요',
                '유령과 부딪히면 목숨 1개 감소!\n'
                '유령을 피해 무사히 돌아오세요.',
                (('ghost', 'A'),),
            ),
            (
                '05', '자석과 무적 아이템',
                '자석: 약 4초간 주변 1점 온기 자동 획득\n'
                '무적: 3초간 유령과 닿아도 피해 없음\n'
                '두 아이템 모두 획득 즉시 사용',
                (('file', 'icons/items/magnet_item.png'),
                 ('file', 'icons/items/invincibility_item.png')),
            ),
            (
                '06', '엔딩 조건',
                '12개 스테이지 클리어 후,\n'
                '방마다 온기 10점 이상 귀환하면\n'
                '진엔딩이 열려요.',
                (('map', 'heart_full'),),
            ),
        ]
        for i, (num, title, desc, icons) in enumerate(cards):
            r = pygame.Rect(39 + (i % 3) * 407, 126 + (i // 3) * 231, 389, 213)
            panel(surface, r, fill=PAPER, border=GOLD, radius=17)

            # 여러 PNG를 쓰는 카드는 같은 영역에 나란히 표시합니다.
            icon_gap = 4
            icon_width = (88 - icon_gap * (len(icons) - 1)) // len(icons)
            for j, (kind, name) in enumerate(icons):
                icon_rect = pygame.Rect(r.x + 19 + j * (icon_width + icon_gap),
                                        r.y + 18, icon_width, 60)
                if kind == 'map':
                    image = map_image(name, icon_rect.size)
                elif kind == 'ghost':
                    source = ghost_frame(60, name)
                    scale = min(icon_rect.w / source.get_width(),
                                icon_rect.h / source.get_height())
                    image = pygame.transform.scale(source, (
                        max(1, round(source.get_width() * scale)),
                        max(1, round(source.get_height() * scale)),
                    ))
                else:
                    image = assets.icon(name, icon_rect.size)
                surface.blit(image, image.get_rect(center=icon_rect.center))

            text(surface, assets, num, (r.right - 23, r.y + 20), size=18,
                 color=(177,146,113), anchor='topright')
            _fit_text(surface, assets, title, (r.x + 119, r.y + 37, 245, 35),
                      size=23, color=INK, bold=True)
            wrapped(surface, assets, desc, pygame.Rect(r.x + 24, r.y + 100, 343, 96),
                    size=17, color=INK, line_gap=6)
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
