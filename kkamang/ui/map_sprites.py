"""Selected map/warmth atlas art, independent of game rules."""
from functools import lru_cache

import pygame

from ..paths import ASSET_DIR

SHEET_PATH = 'art/map/map.png'
SHEET_SIZE = (1122, 1402)
WARMTH_SHEET = 'art/map/warmth.png'
SHEET_SIZES = {SHEET_PATH: SHEET_SIZE, WARMTH_SHEET: (1278, 1230)}
# Furniture is intentionally absent: keep the existing shelf/table/plant PNGs.
# Base tiles use their opaque interiors, excluding the atlas's outer bevels.
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
# Read only the orbs. The existing live +3 label can then hide during suction.
WARMTH_FRAMES = {
    'warmth_1': (131, 730, 88, 87),
    'warmth_3': (395, 666, 158, 161),
}
FRAMES.update(WARMTH_FRAMES)
FRAME_SHEETS.update({key: WARMTH_SHEET for key in WARMTH_FRAMES})


@lru_cache(maxsize=2)
def _sheet(path=SHEET_PATH):
    image = pygame.image.load(str(ASSET_DIR / path))
    if image.get_size() != SHEET_SIZES[path]:
        raise ValueError(f'Atlas {path} must be {SHEET_SIZES[path]}; update frame rectangles for a new layout')
    return image.convert_alpha() if pygame.display.get_surface() else image


@lru_cache(maxsize=len(FRAMES))
def _frame(key):
    image = _sheet(FRAME_SHEETS[key]).subsurface(FRAMES[key]).copy()
    if key in ('floor', 'wall'):
        # Exported interiors have alpha 249..254; base tiles must cover a cell.
        image.set_alpha(None)
        opaque = pygame.Surface(image.get_size())
        opaque.blit(image, (0, 0))
        return opaque
    # Paired doors keep one scale and a shared top-left hinge/frame anchor.
    if key.startswith('return_') or key.startswith('gate_'):
        size = (248, 264) if key.startswith('return_') else (240, 252)
        canvas = pygame.Surface(size, pygame.SRCALPHA)
        canvas.blit(image, (8, 8))
        return canvas
    return image


@lru_cache(maxsize=256)
def map_image(key, size):
    """Fit pixel art with nearest-neighbor scaling and cache each target size."""
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
