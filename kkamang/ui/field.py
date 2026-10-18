"""Readable, layered room rendering. Art never changes collision or timing."""

import math
import pygame
from .theme import text, wrapped, panel, CREAM, INK, NIGHT, PAPER, GOLD, SAGE
from .illustration import cat, ghost, prop, glow, quest_prop
from .ghost_sprites import movement_direction
from .map_sprites import FRAME_SHEETS, SHEET_PATH as MAP_SHEET, WARMTH_SHEET, map_image

# Paths are relative to assets. Atlas entries select their named frame below;
# None retains the existing drawing for artwork that has not been supplied.

FIELD_IMAGES = {
    'floor': MAP_SHEET,         # 바닥 타일
    'wall': MAP_SHEET,          # 벽 타일
    'window': MAP_SHEET,
    'lantern': MAP_SHEET,
    'rug': MAP_SHEET,           # 러그

    'warmth_1': WARMTH_SHEET,   # 1점 온기
    'warmth_3': WARMTH_SHEET,   # 3점 온기
    'magnet': 'icons/items/magnet_item.png',
    'invincibility': 'icons/items/invincibility_item.png',

    'return_base': None,        # 귀환 지점의 바닥 표시
    'return_closed': MAP_SHEET,
    'return_open': MAP_SHEET,
    'gate_closed': MAP_SHEET,   # 닫힌 샛문
    'gate_open': MAP_SHEET,     # 열린 샛문
    'terminal_a': MAP_SHEET,    # A 조작 장치: 해
    'terminal_b': MAP_SHEET,    # B 조작 장치: 달

    'quest_frame': MAP_SHEET,   # 필수 물건 주변 테두리 (느낌표 포함)
    'quest_badge': None,        # 느낌표 뒤의 원형 배경
    'protection': None,         # 플레이어 보호 표시
    'interaction_background': None,  # 상호작용 안내 배경
    'interaction_key': MAP_SHEET,

    'game_background': None,    # 게임 전체 화면 배경
    'heart_full': MAP_SHEET,    # 남아 있는 용기
    'heart_empty': MAP_SHEET,   # 소진된 용기
    'info_divider': MAP_SHEET,  # 오른쪽 설명 패널의 구분선
}

def _draw_png(surface, assets, key, rect):
    """PNG를 표시했으면 True, 경로가 없으면 False를 반환합니다."""

    path = FIELD_IMAGES[key]

    if path is None:
        return False

    rect = pygame.Rect(rect)

    # 기존 이미지 로더가 파일 읽기와 크기별 캐시를 담당합니다.
    image = (map_image(key, rect.size)
             if path == FRAME_SHEETS.get(key)
             else assets.image(path, rect.size))
    surface.blit(image, rect.topleft)

    return True

def _draw_rug(surface, assets, rr):
    """PNG 경로가 있으면 이미지로, 없으면 기존 도형으로 러그를 그립니다."""

    # PNG를 그렸다면 여기서 함수를 끝냅니다.
    if _draw_png(surface, assets, 'rug', rr):
        return

    # PNG가 없을 때 사용하는 기존 러그 그림입니다.
    d = pygame.draw

    d.rect(
        surface,
        (149, 117, 139),
        rr,
        border_radius=9,
    )
    d.rect(
        surface,
        (191, 158, 163),
        rr.inflate(-5, -5),
        1,
        border_radius=6,
    )

    # 러그 위아래의 장식 선입니다.
    for dx in range(5, rr.w - 4, 6):
        d.line(
            surface,
            (191, 158, 163),
            (rr.x + dx, rr.y - 3),
            (rr.x + dx, rr.y + 2),
            1,
        )
        d.line(
            surface,
            (191, 158, 163),
            (rr.x + dx, rr.bottom - 2),
            (rr.x + dx, rr.bottom + 3),
            1,
        )


def _draw_warmth(surface, assets, center, tile, value, show_value=True):
    """온기의 그림과 빛 효과, +3 표시를 그립니다."""

    cx, cy = center

    # 빛 효과는 PNG를 사용해도 유지합니다.
    glow(surface, (cx, cy), int(tile * .6))

    # value가 1이면 warmth_1, 3이면 warmth_3을 사용합니다.
    key = f'warmth_{value}'

    # 기존 온기 원의 지름에 맞춰 표시합니다.
    radius = int(tile * (.16 if value == 1 else .22))
    size = max(1, radius * 2)

    image_rect = pygame.Rect(0, 0, size, size)
    image_rect.center = (int(cx), int(cy))

    if _draw_png(surface, assets, key, image_rect):
        pass

    else:
        # PNG가 없을 때 사용하는 기존 온기 그림입니다.
        pygame.draw.circle(
            surface,
            GOLD,
            (int(cx), int(cy)),
            int(tile * (.16 if value == 1 else .22)),
        )
        pygame.draw.circle(
            surface,
            CREAM,
            (int(cx - 2), int(cy - 3)),
            int(tile * .09),
        )

    # 점수 안내는 이미지와 별도로 표시합니다.
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
    # 게임 상태와 탐사 화면 영역을 준비
    s = snapshot
    r = pygame.Rect(rect)

    # 16열 × 9행 맵을 화면 영역 안에 맞춥니다.
    # tile: 한 칸의 크기
    # ox, oy: 맵이 화면 중앙에 놓이도록 계산한 시작 좌표
    tile = min(r.w / 16, r.h / 9)
    ox = r.x + (r.w - tile * 16) / 2
    oy = r.y + (r.h - tile * 9) / 2

    cosmetics = cosmetics or {}
    d = pygame.draw

    # 맵 좌표 [x, y]를 해당 칸의 화면 사각형으로 바꿉니다.
    def box(p):
        return pygame.Rect(
            round(ox + p[0] * tile),
            round(oy + p[1] * tile),
            math.ceil(tile),
            math.ceil(tile),
        )

    # 캐릭터의 발이 놓일 화면 좌표를 계산
    def foot(p):
        return (
            ox + (p[0] + .5) * tile,
            oy + (p[1] + .95) * tile,
        )

    # 나중에 높이 순서대로 그릴 가구와 캐릭터를 보관합니다.
    # 각 항목은 (화면의 y좌표, 그리기 함수) 형태입니다.
    tasks = []

    # ── 바닥과 벽 ──
    # 장식만 그리며, 실제 이동 가능 여부는 게임의 grid가 결정합니다.
    for y, row in enumerate(s['grid']):
        for x, c in enumerate(row):
            rr = box((x, y))

            # 먼저 현재 칸이 바닥인지 확인합니다.
            if c == '.':
                if not _draw_png(surface, assets, 'floor', rr):
                    color = (
                        130 + (x + y) % 3 * 3,
                        107 + (x + y) % 3 * 2,
                        103 + (x + y) % 3 * 2,
                    )

                    d.rect(surface, color, rr)
                    d.line(
                        surface,
                        (145, 121, 113),
                        rr.topleft,
                        rr.topright,
                        1,
                    )

                    if (x + y) % 4 == 0:
                        d.line(
                            surface,
                            (121, 99, 98),
                            (rr.x + 8, rr.bottom - 12),
                            (rr.right - 7, rr.bottom - 12),
                            1,
                        )

            else:
                if not _draw_png(surface, assets, 'wall', rr):
                    d.rect(surface, (78, 69, 87), rr)

                    if y in (0, 8):
                        d.line(
                            surface,
                            (100, 82, 98),
                            (rr.x + tile / 2, rr.y + 6),
                            (rr.x + tile / 2, rr.bottom - 6),
                            1,
                        )
                        d.line(
                            surface,
                            (167, 127, 106),
                            (rr.x, rr.bottom - 5),
                            (rr.right, rr.bottom - 5),
                            3,
                        )

                # 가구는 벽 칸에만 배치합니다.
                if y > 0 and y < 8 and x not in (0, 15):
                    kind = (
                        'shelf',
                        'plant',
                        'table',
                    )[(x + y + s['stage']) % 3]

                    # 기본 인자에 현재 rr, kind를 보관하여
                    # 나중에 함수를 실행해도 이 칸의 가구를 그립니다.
                    tasks.append(
                        (
                            rr.bottom,
                            lambda rr=rr, kind=kind: prop(
                                surface,
                                kind,
                                rr.inflate(-2, 8),
                            ),
                        )
                    )

                elif y == 0 and x in (3, 8, 12):
                    if not _draw_png(surface, assets, 'window', rr):
                        prop(surface, 'window', rr.inflate(-5, -4))

                elif y == 0 and x in (5, 10):
                    if not _draw_png(surface, assets, 'lantern', rr):
                        prop(surface, 'lantern', rr.inflate(-10, -9))
                    glow(surface, rr.center, 65)

    # ── 바닥의 러그 장식 ──
    # 바닥과 벽을 모두 그린 뒤 러그를 그립니다.
    for y, row in enumerate(s['grid']):
        for x, c in enumerate(row):
            if c == '.' and (x + y * 3 + s['stage']) % 23 == 0:
                rr = box((x, y)).inflate(-8, -12)
                _draw_rug(surface, assets, rr)

    # ── 귀환 지점 ──
    ex = box(s['exit'])
    return_rect = ex.inflate(-7, -7)

    if not _draw_png(surface, assets, 'return_base', return_rect):
        d.rect(
            surface,
            (145, 174, 151),
            return_rect,
            border_radius=12,
        )

    return_ready = s['quest']['collected'] and s['carried'] >= s['target']
    return_key = 'return_open' if return_ready else 'return_closed'
    if not _draw_png(surface, assets, return_key, ex):
        prop(surface, 'garden', ex.inflate(-9, -4))

    # ── 샛문과 조작 장치 ──
    for sc in s['shortcuts']:
        gate = box(sc['gate'])
        term = box(sc['terminal'])

        # Use the same destination in both states so opening never shifts a gate.
        gate_rect = gate.inflate(-2, -2)
        if not sc['opened']:

            if not _draw_png(
                surface, assets, 'gate_closed', gate_rect
            ):
                d.rect(
                    surface,
                    (91, 72, 82),
                    gate_rect,
                    border_radius=10,
                )

                for k in range(3):
                    d.line(
                        surface,
                        GOLD,
                        (gate.x + 15 + k * 12, gate.y + 8),
                        (gate.x + 15 + k * 12, gate.bottom - 8),
                        2,
                    )

        else:
            if not _draw_png(
                surface, assets, 'gate_open', gate_rect
            ):
                d.rect(
                    surface,
                    SAGE,
                    gate_rect,
                    2,
                    border_radius=10,
                )

        # Keep A/B readable beside the sun/moon, without covering the symbols.
        radius = int(tile * .29)
        terminal_rect = pygame.Rect(0, 0, round(tile * .8), round(tile * .8))
        terminal_rect.center = term.center

        key = f"terminal_{sc['id'].lower()}"

        if not _draw_png(surface, assets, key, terminal_rect):
            d.circle(surface, SAGE, term.center, radius)

        label_pos = (term.right - 8, term.bottom - 9)
        d.circle(surface, NIGHT, label_pos, 9)
        text(
            surface,
            assets,
            sc['id'],
            label_pos,
            13,
            CREAM,
            True,
            'center',
        )

    # ── 배경 위에 반투명 색상 덧씌우기 ──
    # veil = pygame.Surface(r.size, pygame.SRCALPHA)
    # veil.fill((28, 25, 42, 20))
    # surface.blit(veil, r)

    # 밟으면 바로 사용하는 자석입니다. 온기/물건과는 별도로 그립니다.
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

    # ── 수집할 온기 ──
    # 유령 반복문 밖에서 온기를 그립니다.
    for item in s['items']:
        if item['active']:
            rr = box(item['position'])
            cx, cy = rr.center

            # PNG를 사용해도 위아래 움직임은 유지합니다.
            cy += math.sin(elapsed * 3 + cx) * 2

            _draw_warmth(
                surface,
                assets,
                (cx, cy),
                tile,
                item['value'],
            )
                
    # ── 필수 물건 ──
    q = s['quest']

    if not q['collected']:
        rr = box(q['position'])

        quest_rect = rr.inflate(-4, -4)

        if not _draw_png(surface, assets, 'quest_frame', quest_rect):
            d.rect(
                surface,
                (238, 213, 164),
                quest_rect,
                2,
                border_radius=12,
            )

        atlas_frame = FIELD_IMAGES['quest_frame'] == MAP_SHEET
        quest_prop(surface, q, rr.inflate(-24, -24) if atlas_frame else rr.inflate(-10, -10))
        glow(surface, rr.center, 60)

        badge_rect = pygame.Rect(0, 0, 14, 14)
        badge_rect.center = (rr.right - 9, rr.y + 8)

        if not atlas_frame and not _draw_png(surface, assets, 'quest_badge', badge_rect):
            d.circle(
                surface,
                GOLD,
                (rr.right - 9, rr.y + 8),
                7,
            )

        if not atlas_frame:
            text(
                surface, assets, '!', (rr.right - 9, rr.y + 7),
                11, INK, True, 'center',
            )

    # ── 유령 캐릭터 ──
    ghost_facings = ghost_facings or {}
    for g in s['ghosts']:
        pos = list(g['position'])
        direction = ghost_facings.get(g['id'], movement_direction(pos, g['next_position']))

        # 이전 위치와 현재 위치 사이를 보간하여 부드럽게 보이게 합니다.
        # 화면에 그릴 좌표만 계산하며 게임 상태의 위치는 바꾸지 않습니다.
        previous_ghost = next((old for old in previous['ghosts']
                               if old['id'] == g['id']), None) if previous else None
        if previous_ghost is not None:
            old = previous_ghost['position']
            direction = movement_direction(old, g['position'], direction)
            pos = [
                old[k] + (pos[k] - old[k]) * interpolation
                for k in (0, 1)
            ]

        pt = foot(pos)

        # 현재 유령의 좌표와 ID를 기본 인자에 보관합니다.

        # 나중에 그릴 순찰형의 화면 좌표를 보관합니다.
        def draw_patrol(pt=pt, ghost_id=g['id'], direction=direction):
            ghost(surface, pt, int(tile * .75), elapsed,
                  ghost_id=ghost_id, direction=direction)

        tasks.append((pt[1], draw_patrol))

    # ── 추격형 유령 ──
    chaser = s.get('chaser')

    # 설정이 있고, 필수 물건 획득 후 활성화된 경우에만 그립니다.
    if chaser is not None and chaser['active']:
        pos = list(chaser['position'])
        direction = ghost_facings.get(chaser['id'], 'down')

        previous_chaser = (
            previous.get('chaser') if previous else None
        )

        # 이전 화면에서도 활성화되어 있었을 때만 이동을 보간합니다.
        if (
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

        # 나중에 실행할 때도 추격형의 좌표와 ID를 사용하도록 보관합니다.
        def draw_chaser(pt=pt, ghost_id=chaser['id'], direction=direction):
            ghost(surface, pt, int(tile * .75), elapsed,
                  ghost_id=ghost_id, direction=direction)

        tasks.append((pt[1], draw_chaser))

    # ── 플레이어 캐릭터 ──
    pos = list(s['position'])
    moving = False

    player_alpha = (
        interpolation
        if player_interpolation is None
        else player_interpolation
    )

    motion = s.get('motion')
    if motion is not None:
        # The simulation already supplies a continuous body center. Interpolating
        # it again would add input delay and can cut across a turning corner.
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

    if magnet is not None and magnet['remaining_ticks'] > 0:
        # 사각형 범위는 고양이와 함께 이동하며 벽 위에도 표시됩니다.
        extent = math.ceil(tile * 3)
        aura = pygame.Surface((extent, extent), pygame.SRCALPHA)
        pygame.draw.rect(aura, (225, 187, 112, 14), aura.get_rect(), border_radius=10)
        pygame.draw.rect(aura, (250, 212, 143, 110), aura.get_rect(), 2, border_radius=10)
        surface.blit(aura, aura.get_rect(center=box(pos).center))

    # 표정과 발걸음은 독립적입니다. 피격 중에도 이동하면 걷습니다.
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

    # y좌표가 작은 대상부터 그립니다.
    # 화면 아래쪽 대상이 나중에 그려져 앞에 있는 것처럼 보입니다.
    for _, draw in sorted(tasks, key=lambda t: t[0]):
        draw()

    # 피해를 받은 뒤 보호 상태임을 나타내는 표시입니다.
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

    # 수집 결과는 이미 코어에 반영됐습니다. 그림만 현재 고양이를 따라갑니다.
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

    # ── 현재 위치에서 가능한 상호작용 안내 ──
    label = None

    if s['position'] == s['exit']:
        label = (
            'E 귀환'
            if q['collected'] and s['carried'] >= s['target']
            else '물건 + 온기 필요'
        )

    elif s['position'] == q['position'] and not q['collected']:
        label = 'E ' + q['name']

    # elif any(
    #     i['active'] and i['position'] == s['position']
    #     for i in s['items']
    # ):
    #     label = 'E 온기 회수'

    elif any(
        not g['opened'] and g['terminal'] == s['position']
        for g in s['shortcuts']
    ):
        label = 'E 샛문 열기'

    if label:
        p = foot(s['position'])

        # 글자 너비에 여백을 더해 안내 상자의 크기를 정합니다.
        show_key = label.startswith('E ')
        caption = label[2:] if show_key else label
        ww = assets.font(14).size(caption)[0] + (49 if show_key else 20)
        rr = pygame.Rect(
            p[0] - ww / 2,
            p[1] - tile - 23,
            ww,
            26,
        )

        # 안내 상자가 탐사 화면 밖으로 나가지 않도록 맞춥니다.
        rr.clamp_ip(r)

        if not _draw_png(
            surface, assets, 'interaction_background', rr
        ):
            d.rect(surface, NIGHT, rr, border_radius=9)
        if show_key:
            key_rect = pygame.Rect(rr.x + 2, rr.y + 1, 24, 24)
            _draw_png(surface, assets, 'interaction_key', key_rect)
            text(surface, assets, 'E', key_rect.center, 13, INK, True, 'center')
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
):
    # 게임 전체 화면의 배경을 채웁니다.
    s = snapshot
    if not _draw_png(
        surface, assets, 'game_background', surface.get_rect()
    ):
        surface.fill((36, 34, 49))
    d = pygame.draw

    # ── 상단: 게임 이름과 스테이지 ──
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

    # ── 상단: 용기 하트 3칸 ──
    for i in range(3):
        key = 'heart_full' if i < s['courage'] else 'heart_empty'
        heart_rect = pygame.Rect(341 + i * 31, 43, 31, 29)

        if not _draw_png(surface, assets, key, heart_rect):
            c = (
                (208, 148, 157)
                if i < s['courage']
                else (81, 71, 88)
            )

            d.circle(surface, c, (350 + i * 31, 52), 9)
            d.circle(surface, c, (362 + i * 31, 52), 9)
            d.polygon(
                surface,
                c,
                [
                    (341 + i * 31, 53),
                    (371 + i * 31, 53),
                    (356 + i * 31, 71),
                ],
            )

    # ── 상단: 필수 물건의 회수 상태 ──
    q = s['quest']

    panel(
        surface,
        (492, 21, 405, 58),
        fill=(69, 61, 78),
        border=(118, 102, 113),
        radius=14,
    )
    quest_prop(surface, q, (505, 29, 42, 42))
    text(
        surface,
        assets,
        q['name'] + ('  회수 완료' if q['collected'] else '  찾아오기'),
        (558, 39),
        19,
        CREAM,
    )

    # ── 상단: 현재 온기와 목표 온기 ──
    text(
        surface,
        assets,
        f'온기  {s["carried"]} / {s["target"]}',
        (925, 40),
        22,
        GOLD,
        True,
    )

    # ── 상단: 남은 시간 ──
    # 남은 틱에 0.3초를 곱하여 초 단위로 표시합니다.
    remaining = s['remaining_ticks'] * .3

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

    # ── 왼쪽: 탐사 화면의 테두리와 실제 맵 ──
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

    # ── 오른쪽: 필수 물건과 복원 설명 ──
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
        '오늘 되찾을 기억',
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
    text(
        surface,
        assets,
        q['name'],
        (1149, 269),
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
        17,
        INK,
        5,
    )

    # ── 오른쪽: 추가 목표와 보상 안내 ──
    divider_rect = pygame.Rect(1059, 409, 182, 14)

    if not _draw_png(surface, assets, 'info_divider', divider_rect):
        d.line(
            surface,
            (203, 177, 144),
            (1059, 416),
            (1240, 416),
        )
    text(
        surface,
        assets,
        '조금 더 탐험하면',
        (1060, 433),
        17,
        INK,
        True,
    )
    wrapped(
        surface,
        assets,
        '온기 10점 귀환\n수집 배지와 꾸미기 해금\n\n무피격 완료\n조용한 발걸음 배지',
        (1060, 465, 175, 139),
        15,
        INK,
        5,
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
        '큰 온기는 순찰을 조심!',
        (1060, 631),
        13,
        (131, 91, 100),
    )

    # ── 하단: 알림 또는 기본 조작 안내 ──
    text(
        surface,
        assets,
        toast or '방향키 / WASD 이동   ·   E 아이템 회수/ 상호작용 ·    Esc 일시정지',
        (28, 689),
        16,
        CREAM,
    )


# 남은 시간이 6초 미만일 때 사용하는 글자 색입니다.
ROSE = (235, 151, 156)
