"""탐사 맵·캐릭터·수집 효과와 상태 안내를 겹쳐 그린다.

표시 좌표만 보간하며 충돌·이동·시간 판정은 코어 상태를 따른다.
"""

import math
import pygame
from .theme import text, fit_text, wrapped, panel, CREAM, INK, NIGHT, PAPER, GOLD, SAGE
from .illustration import cat, ghost, prop, glow, quest_prop
from .ghost_sprites import movement_direction
from .map_sprites import (
    FRAME_SHEETS, SHEET_PATH as MAP_SHEET, WARMTH_SHEET, map_image,
    wall_variant_image,
)

# 왼쪽 세 열을 오른쪽에 대칭 배치한다. v는 덩굴, f는 꽃이다.
SIDE_WALL_LAYOUTS = {
    1: ('vvv', 'vvv', 'ffv', 'ffv', 'vvv', 'vvv', 'ffv', 'vvv', 'vvv'),
    2: ('vvv', 'ffv', 'vvv', 'vvv', 'vvv', 'ffv', 'ffv', 'vvv', 'vvv'),
}

# 경로는 assets 기준이다. 시트는 이름별 영역을 쓰며 None은 별도 PNG가 없는 표시다.
FIELD_IMAGES = {
    'floor': MAP_SHEET,
    'wall': MAP_SHEET,
    'window': MAP_SHEET,
    'lantern': MAP_SHEET,
    'rug': MAP_SHEET,

    'warmth_1': WARMTH_SHEET,
    'warmth_3': WARMTH_SHEET,
    'magnet': 'icons/items/magnet_item.png',
    'invincibility': 'icons/items/invincibility_item.png',

    'return_base': None,
    'return_closed': MAP_SHEET,
    'return_open': MAP_SHEET,
    'gate_closed': MAP_SHEET,
    'gate_open': MAP_SHEET,
    'terminal_a': MAP_SHEET,
    'terminal_b': MAP_SHEET,

    'quest_frame': MAP_SHEET,
    'quest_badge': None,
    'protection': None,
    'interaction_background': None,
    'interaction_key': MAP_SHEET,

    'game_background': None,
    'heart_full': MAP_SHEET,
    'heart_empty': MAP_SHEET,
    'info_divider': MAP_SHEET,
}

def _draw_png(surface, assets, key, rect):
    """PNG 경로가 있을 때만 표시하고 성공 여부를 반환한다."""

    path = FIELD_IMAGES[key]

    if path is None:
        return False

    rect = pygame.Rect(rect)

    image = (map_image(key, rect.size)
             if path == FRAME_SHEETS.get(key)
             else assets.image(path, rect.size))
    surface.blit(image, rect.topleft)

    return True

def _draw_rug(surface, assets, rr):
    if _draw_png(surface, assets, 'rug', rr):
        return

def _draw_warmth(surface, assets, center, tile, value, show_value=True):
    cx, cy = center

    glow(surface, (cx, cy), int(tile * .6))

    key = f'warmth_{value}'

    radius = int(tile * (.16 if value == 1 else .22))
    size = max(1, radius * 2)

    image_rect = pygame.Rect(0, 0, size, size)
    image_rect.center = (int(cx), int(cy))

    _draw_png(surface, assets, key, image_rect)

    if value == 3 and show_value:
        text(
            surface,
            assets,
            '+3',
            (cx, cy + 12),
            14,
            CREAM,
            True,
            'midtop',
        )

def draw_field(
    surface,
    assets,
    snapshot,
    rect,
    elapsed=0.,
    previous=None,
    interpolation=1.,
    cosmetics=None,
    facing='down',
    mood='idle',
    player_interpolation=None,
    magnet_flights=(),
    ghost_facings=None,
):
    s = snapshot
    r = pygame.Rect(rect)

    tile = min(r.w / 16, r.h / 9)
    ox = r.x + (r.w - tile * 16) / 2
    oy = r.y + (r.h - tile * 9) / 2

    cosmetics = cosmetics or {}
    d = pygame.draw

    def box(p):
        return pygame.Rect(
            round(ox + p[0] * tile),
            round(oy + p[1] * tile),
            math.ceil(tile),
            math.ceil(tile),
        )

    def foot(p):
        return (
            ox + (p[0] + .5) * tile,
            oy + (p[1] + .95) * tile,
        )

    # 가구와 캐릭터를 발의 세로 좌표 순서로 그릴 작업에 모은다.
    tasks = []
    side_wall_layout = SIDE_WALL_LAYOUTS.get(s['stage'])

    # 장식과 무관하게 실제 이동 가능 여부는 코어의 grid가 결정한다.
    for y, row in enumerate(s['grid']):
        for x, c in enumerate(row):
            rr = box((x, y))

            if c == '.':
                _draw_png(surface, assets, 'floor', rr)

            else:
                side_column = min(x, len(row) - 1 - x)
                wall_style = (
                    side_wall_layout[y][side_column]
                    if side_wall_layout and side_column < 3 and y < len(side_wall_layout)
                    else '.'
                )
                if wall_style == '.':
                    _draw_png(surface, assets, 'wall', rr)
                else:
                    kind = 'flowers' if wall_style == 'f' else 'vine'
                    image = wall_variant_image(
                        kind, rr.size, (side_column % 2, y % 2),
                        mirrored=x >= len(row) - 3,
                    )
                    surface.blit(image, rr.topleft)

                # 1·2스테이지의 2·3·14·15열은 가구 없이 벽만 표시한다.
                if (
                    y > 0 and y < 8 and x not in (0, 15)
                    and not (s['stage'] in (1, 2) and x in (1, 2, 13, 14))
                ):
                    kind = (
                        'shelf',
                        'plant',
                        'table',
                    )[(x + y + s['stage']) % 3]

                    # 기본 인자로 현재 칸을 고정해 지연 실행 시 다른 가구가 그려지지 않게 한다.
                    tasks.append(
                        (
                            rr.bottom,
                            lambda rr=rr, kind=kind: prop(
                                surface,
                                kind,
                                rr.inflate(-4, -4) if kind == 'plant' else rr.inflate(-2, 8),
                            ),
                        )
                    )

                elif y == 0 and x in (3, 8, 12):
                    _draw_png(surface, assets, 'window', rr)

                elif y == 0 and x in (5, 10):
                    _draw_png(surface, assets, 'lantern', rr)

                    glow(surface, rr.center, 65)

    for y, row in enumerate(s['grid']):
        for x, c in enumerate(row):
            if c == '.' and (x + y * 3 + s['stage']) % 23 == 0:
                rr = box((x, y)).inflate(-8, -12)
                _draw_rug(surface, assets, rr)

    ex = box(s['exit'])
    return_rect = ex.inflate(-7, -7)

    return_ready = s['quest']['collected'] and s['carried'] >= s['target']
    return_key = 'return_open' if return_ready else 'return_closed'
    _draw_png(surface, assets, return_key, ex)

    for sc in s['shortcuts']:
        gate = box(sc['gate'])
        term = box(sc['terminal'])

        # 문을 열어도 위치가 이동하지 않도록 열린 모습과 닫힌 모습의 영역을 맞춘다.
        gate_rect = gate.inflate(-2, -2)
        if not sc['opened']:
            _draw_png(surface, assets, 'gate_closed', gate_rect)

        else:
            _draw_png(surface, assets, 'gate_open', gate_rect)

        terminal_rect = pygame.Rect(0, 0, round(tile * .8), round(tile * .8))
        terminal_rect.center = term.center

        key = f"terminal_{sc['id'].lower()}"

        _draw_png(surface, assets, key, terminal_rect)

        label_pos = (term.right - 8, term.bottom - 9)

    magnet = s.get('magnet')
    invincibility = s.get('invincibility')
    for name, item in (('magnet', magnet), ('invincibility', invincibility)):
        if item is None or item['collected']:
            continue
        rr = box(item['position'])
        icon = pygame.Rect(0, 0, round(tile * .85), round(tile * .85))
        icon.center = (rr.centerx, round(rr.centery + math.sin(elapsed * 3) * 2))
        glow(surface, icon.center, round(tile * .5))
        _draw_png(surface, assets, name, icon)

    for item in s['items']:
        if item['active']:
            rr = box(item['position'])
            cx, cy = rr.center

            cy += math.sin(elapsed * 3 + cx) * 2

            _draw_warmth(
                surface,
                assets,
                (cx, cy),
                tile,
                item['value'],
            )

    q = s['quest']

    if not q['collected']:
        rr = box(q['position'])

        quest_rect = rr.inflate(-4, -4)

        _draw_png(surface, assets, 'quest_frame', quest_rect)

        atlas_frame = FIELD_IMAGES['quest_frame'] == MAP_SHEET
        quest_prop(surface, q, rr.inflate(-24, -24) if atlas_frame else rr.inflate(-10, -10))
        glow(surface, rr.center, 60)

    ghost_facings = ghost_facings or {}
    motion = s.get('motion')
    for g in s['ghosts']:
        pos = list(g['position'])
        direction = ghost_facings.get(g['id'], movement_direction(pos, g['next_position']))

        # 유령의 화면 위치만 보간하며 코어의 위치는 바꾸지 않는다.
        previous_ghost = next((old for old in previous['ghosts']
                               if old['id'] == g['id']), None) if previous else None
        if motion is not None:
            pos = list(motion['ghost_positions'][g['id']])
            direction = movement_direction(pos, g['position'], direction)
        elif previous_ghost is not None:
            old = previous_ghost['position']
            direction = movement_direction(old, g['position'], direction)
            pos = [
                old[k] + (pos[k] - old[k]) * interpolation
                for k in (0, 1)
            ]

        pt = foot(pos)

        # 좌표·ID·방향을 고정해 나중에 그릴 때도 같은 유령을 사용한다.
        def draw_patrol(pt=pt, ghost_id=g['id'], direction=direction):
            size = int(tile * .75)
            if ghost_id == 'B':
                size += 10
            ghost(surface, pt, size, elapsed,
                  ghost_id=ghost_id, direction=direction)

        tasks.append((pt[1], draw_patrol))

    chaser = s.get('chaser')

    if chaser is not None and chaser['active']:
        pos = list(chaser['position'])
        direction = ghost_facings.get(chaser['id'], 'down')

        previous_chaser = (
            previous.get('chaser') if previous else None
        )

        # 이미 활성화되어 있던 같은 추격형 유령만 이전 위치에서 보간한다.
        if motion is not None:
            pos = list(motion['chaser_position'])
            direction = movement_direction(pos, chaser['position'], direction)
        elif (
            previous_chaser is not None
            and previous_chaser['active']
            and previous_chaser['id'] == chaser['id']
        ):
            old = previous_chaser['position']
            direction = movement_direction(old, chaser['position'], direction)
            pos = [
                old[k] + (pos[k] - old[k]) * interpolation
                for k in (0, 1)
            ]

        pt = foot(pos)

        def draw_chaser(pt=pt, ghost_id=chaser['id'], direction=direction):
            ghost(surface, pt, int(tile * .75), elapsed,
                  ghost_id=ghost_id, direction=direction)

        tasks.append((pt[1], draw_chaser))

    pos = list(s['position'])
    moving = False

    player_alpha = (
        interpolation
        if player_interpolation is None
        else player_interpolation
    )

    motion = s.get('motion')
    if motion is not None:
        # 코어의 연속 좌표를 다시 보간하면 입력 지연과 모퉁이 가로지르기가 생긴다.
        pos = list(motion['position'])
        moving = motion['moving']
    elif previous:
        old = previous['position']
        moving = old != pos and player_alpha < 1.

        pos = [
            old[k] + (pos[k] - old[k]) * player_alpha
            for k in (0, 1)
        ]

    pt = foot(pos)

    # 피격 표정과 걷기는 독립적이므로 보호 중에도 이동하면 걸음을 표시한다.
    cm = (
        'surprised'
        if s['protection']
        else 'walk'
        if moving
        else mood
    )

    tasks.append(
        (
            pt[1],
            lambda: cat(
                surface,
                pt,
                int(tile * .98),
                facing,
                elapsed,
                cm,
                cosmetics.get('scarf', 'default'),
                walking=moving,
                walk_progress=(motion['distance'] * .5 if motion is not None
                               else ((s['tick'] - 1) % 2 + player_alpha) * .5),
            ),
        )
    )

    # 발이 화면 아래에 있을수록 나중에 그려 가구와 캐릭터의 앞뒤를 맞춘다.
    for _, draw in sorted(tasks, key=lambda t: t[0]):
        draw()

    if s['protection'] or (invincibility and invincibility['remaining_ticks'] > 0):
        protection_rect = box(pos).inflate(-4, -5)

        if not _draw_png(
            surface, assets, 'protection', protection_rect
        ):
            d.ellipse(
                surface,
                (225, 210, 240),
                protection_rect,
                2,
            )

    # 수집 결과는 이미 코어에 반영되었으므로 비행 효과만 현재 고양이를 따라간다.
    target = box(pos).center
    for flight in magnet_flights:
        progress = min(1., max(0., flight['age'] / flight['duration']))
        if progress >= 1.:
            continue
        origin = box(flight['origin']).center
        eased = progress * progress
        center = (
            origin[0] + (target[0] - origin[0]) * eased,
            origin[1] + (target[1] - origin[1]) * eased
            - math.sin(progress * math.pi) * tile * .12,
        )
        _draw_warmth(surface, assets, center, tile * (1. - .65 * progress),
                     flight['value'], show_value=False)

    label = None

    if s['position'] == s['exit']:
        label = (
            'E 귀환'
            if q['collected'] and s['carried'] >= s['target']
            else '물건 + 온기 필요'
        )

    elif s['position'] == q['position'] and not q['collected']:
        label = 'E ' + q['name']

    elif any(
        not g['opened'] and g['terminal'] == s['position']
        for g in s['shortcuts']
    ):
        label = 'E 샛문 열기'

    if label:
        p = foot(s['position'])

        show_key = label.startswith('E ')
        caption = label[2:] if show_key else label
        ww = assets.font(14).size(caption)[0] + (49 if show_key else 20)
        rr = pygame.Rect(
            p[0] - ww / 2,
            p[1] - tile - 23,
            ww,
            26,
        )

        # 상호작용 안내가 탐사 화면 밖으로 나가지 않게 맞춘다.
        rr.clamp_ip(r)

        if not _draw_png(
            surface, assets, 'interaction_background', rr
        ):
            d.rect(surface, NIGHT, rr, border_radius=9)
        if show_key:
            key_rect = pygame.Rect(rr.x + 2, rr.y + 1, 24, 24)
            text(surface, assets, 'E', key_rect.center, 13, CREAM, True, 'center' )

        text(
            surface,
            assets,
            caption,
            (rr.centerx + (13 if show_key else 0), rr.centery),
            14,
            CREAM,
            False,
            'center',
        )

def draw_game(
    surface,
    assets,
    snapshot,
    stage_story,
    elapsed=0.,
    previous=None,
    interpolation=1.,
    toast='',
    cosmetics=None,
    facing='down',
    mood='idle',
    player_interpolation=None,
    magnet_flights=(),
    ghost_facings=None,
    remaining_seconds=None,
):
    s = snapshot
    if not _draw_png(
        surface, assets, 'game_background', surface.get_rect()
    ):
        surface.fill((36, 34, 49))

    text(
        surface,
        assets,
        'THE SLEEPLESS MANOR',
        (26, 16),
        12,
        (177, 160, 177),
    )
    text(
        surface,
        assets,
        f'{s["stage"]:02d}  {s["name"]}',
        (25, 38),
        27,
        CREAM,
        True,
    )

    for i in range(3):
        key = 'heart_full' if i < s['courage'] else 'heart_empty'
        heart_rect = pygame.Rect(341 + i * 31, 43, 31, 29)

        _draw_png(surface, assets, key, heart_rect)

    q = s['quest']

    panel(
        surface,
        (492, 21, 405, 58),
        fill=(69, 61, 78),
        border=(118, 102, 113),
        radius=14,
    )
    quest_prop(surface, q, (505, 29, 42, 42))
    fit_text(
        surface,
        assets,
        q['name'] + ('  회수 완료' if q['collected'] else '  찾아오기'),
        (558, 30, 324, 39),
        19,
        CREAM,
        anchor='midleft',
    )

    text(
        surface,
        assets,
        f'온기  {s["carried"]} / {s["target"]}',
        (925, 40),
        22,
        GOLD,
        True,
    )

    # 한 틱은 0.3초이며 화면에는 틱 사이 경과 시간까지 반영한다.
    if remaining_seconds is None:
        remaining = s['remaining_ticks'] * .3
    else:
        remaining = max(0., remaining_seconds)

    text(
        surface,
        assets,
        f'{remaining:04.1f} 초',
        (1245, 40),
        25,
        ROSE if remaining < 6 else CREAM,
        True,
        'topright',
    )

    panel(
        surface,
        (19, 94, 1002, 575),
        fill=(77, 63, 73),
        border=(149, 118, 103),
        radius=18,
    )
    draw_field(
        surface,
        assets,
        s,
        (31, 106, 978, 550),
        elapsed,
        previous,
        interpolation,
        cosmetics,
        facing,
        mood,
        player_interpolation=player_interpolation,
        magnet_flights=magnet_flights,
        ghost_facings=ghost_facings,
    )
    if s['protection']:
        # 피격 표시만 그리며 충돌·보호 시간 판정은 코어가 담당한다.
        pygame.draw.rect(surface, (170, 211, 225), (29, 104, 982, 554), 3,
               border_radius=9)

    panel(
        surface,
        (1039, 94, 221, 575),
        fill=PAPER,
        border=(182, 148, 116),
        radius=18,
    )
    text(
        surface,
        assets,
        '필수 아이템',
        (1058, 115),
        18,
        INK,
        True,
    )
    quest_prop(
        surface,
        q,
        (1098, 154, 102, 102),
    )
    fit_text(
        surface,
        assets,
        q['name'],
        (1056, 256, 188, 38),
        21,
        INK,
        True,
        'center',
    )
    wrapped(
        surface,
        assets,
        q['restoration'],
        (1060, 303, 178, 104),
        15,
        INK,
        4,
    )

    divider_rect = pygame.Rect(1059, 409, 182, 14)

    _draw_png(surface, assets, 'info_divider', divider_rect)

    text(
        surface,
        assets,
        '탐험 완료를 위해서는...',
        (1060, 433),
        17,
        INK,
        True,
    )
    wrapped(
        surface,
        assets,
        '아이템 + 온기 4점\n귀환점에서 E\n\n유령을 피해 용기를 지켜요',
        (1060, 465, 175, 139),
        15,
        INK,
        4,
    )
    magnet = s.get('magnet')
    if magnet is not None:
        status = (f'자석 {magnet["remaining_ticks"]}틱 · 범위 3×3'
                  if magnet['remaining_ticks'] > 0
                  else '자석: 사용 완료' if magnet['collected']
                  else '자석: 밟으면 자동 사용')
        text(surface, assets, status, (1060, 609), 13, INK, True)
    invincibility = s.get('invincibility')
    if invincibility is not None:
        status = (f'무적 {invincibility["remaining_ticks"]}틱'
                  if invincibility['remaining_ticks'] > 0
                  else '무적: 사용 완료' if invincibility['collected']
                  else '무적: 밟으면 자동 사용')
        text(surface, assets, status, (1060, 632 if magnet else 609), 13, INK, True)

    text(
        surface,
        assets,
        '유령 조심!',
        (1060, 631),
        16,
        (131, 91, 100),
    )

    text(
        surface,
        assets,
        toast or '방향키 / WASD 이동   ·   E 아이템 회수/ 상호작용 ·    Esc 일시정지',
        (28, 689),
        16,
        CREAM,
    )

ROSE = (235, 151, 156)
