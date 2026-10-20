"""맵·온기 PNG 시트에서 타일·문·물건 표시를 추출하고 크기별로 캐시한다."""
from functools import lru_cache

import pygame

from ..paths import ASSET_DIR

SHEET_PATH = 'art/map/map.png'
SHEET_SIZE = (1122, 1402)
WARMTH_SHEET = 'art/map/warmth.png'
SHEET_SIZES = {SHEET_PATH: SHEET_SIZE, WARMTH_SHEET: (1278, 1230)}
WALL_VARIANT_SHEETS = {
    'vine': 'art/map/wall_vine.png',
    'flowers': 'art/map/wall_flowers.png',
}
SHEET_SIZES.update({path: (256, 256) for path in WALL_VARIANT_SHEETS.values()})
# 가구는 illustration에서 별도로 자르고 기본 타일은 테두리 장식을 제외한 내부를 쓴다.
FRAMES = {
    'floor': (39, 80, 240, 240),
    'wall': (310, 80, 240, 240),
    'window': (585, 122, 222, 200),
    'lantern': (886, 75, 159, 254),
    'rug': (37, 354, 245, 249),
    'return_closed': (53, 631, 210, 237),
    'return_open': (314, 631, 232, 246),
    'gate_closed': (585, 632, 215, 235),
    'gate_open': (853, 631, 221, 236),
    'terminal_a': (45, 897, 228, 228),
    'terminal_b': (314, 897, 228, 228),
    'quest_frame': (577, 881, 240, 241),
    'info_divider': (831, 985, 265, 66),
    'interaction_key': (90, 1184, 135, 136),
    'heart_full': (318, 1154, 220, 198),
    'heart_empty': (583, 1154, 220, 197),
}
FRAME_SHEETS = {key: SHEET_PATH for key in FRAMES}
# 온기 구슬만 추출해 흡수 효과에서는 별도의 +3 글자를 숨길 수 있게 한다.
WARMTH_FRAMES = {
    'warmth_1': (131, 730, 88, 87),
    'warmth_3': (395, 666, 158, 161),
}
FRAMES.update(WARMTH_FRAMES)
FRAME_SHEETS.update({key: WARMTH_SHEET for key in WARMTH_FRAMES})


@lru_cache(maxsize=len(SHEET_SIZES))
def _sheet(path=SHEET_PATH):
    image = pygame.image.load(str(ASSET_DIR / path))
    if image.get_size() != SHEET_SIZES[path]:
        raise ValueError(f'Atlas {path} must be {SHEET_SIZES[path]}; update frame rectangles for a new layout')
    return image.convert_alpha() if pygame.display.get_surface() else image

@lru_cache(maxsize=64)
def wall_variant_image(kind, size, quadrant, mirrored=False):
    """2×2 타일 질감의 돌 크기와 선명도를 유지한다."""
    sheet = _sheet(WALL_VARIANT_SHEETS[kind])
    column, row = quadrant
    frame = sheet.subsurface((column * 128, row * 128, 128, 128))
    image = pygame.transform.scale(frame, size)
    return pygame.transform.flip(image, True, False) if mirrored else image

@lru_cache(maxsize=len(FRAMES))
def _frame(key):
    image = _sheet(FRAME_SHEETS[key]).subsurface(FRAMES[key]).copy()
    if key in ('floor', 'wall'):
        # 원본 내부의 알파값이 249~254여도 기본 타일은 한 칸을 불투명하게 덮어야 한다.
        image.set_alpha(None)
        opaque = pygame.Surface(image.get_size())
        opaque.blit(image, (0, 0))
        return opaque
    # 열린 문과 닫힌 문은 같은 배율과 왼쪽 위 기준점을 사용한다.
    if key.startswith('return_') or key.startswith('gate_'):
        size = (248, 264) if key.startswith('return_') else (240, 252)
        canvas = pygame.Surface(size, pygame.SRCALPHA)
        canvas.blit(image, (8, 8))
        return canvas
    return image

@lru_cache(maxsize=256)
def map_image(key, size):
    """최근접 배율로 픽셀 원화의 비율을 맞추고 크기별로 캐시한다."""
    source = _frame(key)
    width, height = size
    if key in ('floor', 'wall', 'info_divider'):
        return pygame.transform.scale(source, size)
    scale = min(width / source.get_width(), height / source.get_height())
    image = pygame.transform.scale(source, (
        max(1, round(source.get_width() * scale)),
        max(1, round(source.get_height() * scale)),
    ))
    canvas = pygame.Surface(size, pygame.SRCALPHA)
    canvas.blit(image, image.get_rect(center=(width // 2, height // 2)))
    return canvas
