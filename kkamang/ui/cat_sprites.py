"""Read the irregular cat atlas without changing its original PNG.

All frames share a body/ground anchor. Tail bounds must not determine the pivot:
their width changes throughout the walk cycle. Expressions and scarf variants
are intentionally absent from this first sprite set.
"""
from functools import lru_cache
import pygame

from ..paths import ASSET_DIR

SHEET_PATH = ASSET_DIR / 'art' / 'kkamang.png'
SHEET_SIZE = (1374, 1145)
CANVAS_SIZE = (256, 288)
GROUND_ANCHOR = (128, 272)
WALK_FRAMES = 6

# Source rectangle, then the body's ground anchor in original sheet coordinates.
IDLE_FRAMES = {
    'down': ((46, 42, 164, 228), (128, 270)),
    'left': ((458, 51, 231, 213), (568, 264)),
    'up': ((697, 44, 163, 227), (792, 256)),
    'right': ((879, 51, 230, 212), (1001, 263)),
}
WALK_ATLAS = {
    'down': (
        ((51, 293, 146, 222), (124, 515)),
        ((277, 294, 148, 218), (350, 512)),
        ((507, 293, 147, 219), (580, 512)),
        ((728, 293, 146, 220), (802, 513)),
        ((951, 295, 147, 220), (1024, 515)),
        ((1181, 293, 147, 220), (1254, 513)),
    ),
    'left': (
        ((15, 523, 226, 188), (126, 711)),
        ((242, 527, 226, 186), (352, 713)),
        ((470, 527, 198, 186), (577, 713)),
        ((693, 526, 215, 187), (802, 713)),
        ((922, 526, 215, 187), (1027, 713)),
        ((1143, 527, 213, 186), (1252, 713)),
    ),
    'up': (
        ((56, 732, 148, 189), (131, 921)),
        ((261, 732, 154, 191), (354, 923)),
        ((500, 732, 162, 191), (581, 923)),
        ((724, 732, 166, 193), (805, 925)),
        ((955, 732, 163, 191), (1030, 923)),
        ((1180, 732, 153, 192), (1259, 924)),
    ),
    'right': (
        ((19, 938, 205, 173), (128, 1111)),
        ((237, 940, 217, 171), (350, 1111)),
        ((469, 933, 209, 178), (577, 1111)),
        ((698, 933, 207, 178), (802, 1111)),
        ((927, 935, 205, 176), (1027, 1111)),
        ((1152, 936, 207, 175), (1252, 1111)),
    ),
}
IDLE_SCALE = {'down': 1.05, 'left': 1.05, 'up': 1.05, 'right': 1.05}
WALK_SCALE = {'down': 1.05, 'left': 1.05, 'up': 1.05, 'right': 1.12}


@lru_cache(maxsize=1)
def _sheet():
    image = pygame.image.load(str(SHEET_PATH))
    if image.get_size() != SHEET_SIZE:
        raise ValueError(f'Cat atlas must be {SHEET_SIZE}; update frame rectangles for a new layout')
    return image.convert_alpha() if pygame.display.get_surface() else image


@lru_cache(maxsize=28)
def _normalized_frame(direction, walking, index):
    rect, anchor = WALK_ATLAS[direction][index] if walking else IDLE_FRAMES[direction]
    scale = (WALK_SCALE if walking else IDLE_SCALE)[direction]
    source = _sheet().subsurface(rect)
    scaled = pygame.transform.smoothscale(source, (round(rect[2] * scale), round(rect[3] * scale)))
    destination = (
        GROUND_ANCHOR[0] - round((anchor[0] - rect[0]) * scale),
        GROUND_ANCHOR[1] - round((anchor[1] - rect[1]) * scale),
    )
    canvas = pygame.Surface(CANVAS_SIZE, pygame.SRCALPHA)
    if not canvas.get_rect().contains(scaled.get_rect(topleft=destination)):
        raise ValueError(f'Cat frame exceeds its canvas: {direction}, {walking}, {index}')
    canvas.blit(scaled, destination)
    return canvas


@lru_cache(maxsize=256)
def sprite_frame(size, direction='down', walking=False, index=0):
    """Return a cached fixed-canvas frame; never crop each animation independently."""
    if direction not in IDLE_FRAMES:
        raise ValueError(f'Unknown cat direction: {direction}')
    index = index % WALK_FRAMES if walking else 0
    width = max(1, int(size))
    return pygame.transform.smoothscale(
        _normalized_frame(direction, walking, index),
        (width, max(1, int(width * CANVAS_SIZE[1] / CANVAS_SIZE[0]))),
    )
