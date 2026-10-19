"""Progressive lobby artwork, derived only from completed campaign stages."""
from dataclasses import dataclass, replace
from functools import lru_cache

import pygame

from ..paths import ASSET_DIR
from .illustration import resident_cat, _released_residents


@dataclass(frozen=True)
class LobbySprite:
    name: str
    crop: tuple[int, int, int, int]
    size: tuple[int, int]
    foot: tuple[int, int]


# Positions use the original 1672-by-941 lobby artwork as their coordinate system.
# Each item includes its painted furniture; the curtain tie excludes the stool.
ITEMS = {
    1: LobbySprite('수건 선반', (22, 3, 330, 355), (210, 226), (153, 360)),
    2: LobbySprite('안내 지도', (387, 51, 341, 289), (185, 157), (466, 385)),
    3: LobbySprite('발자국 사진', (790, 54, 256, 282), (90, 100), (604, 257)),
    4: LobbySprite('정원 안내서', (1090, 3, 344, 354), (255, 263), (1040, 586)),
    5: LobbySprite('스카프 고리', (121, 341, 157, 385), (100, 245), (1400, 373)),
    6: LobbySprite('씨앗 주머니', (360, 375, 364, 323), (300, 267), (335, 695)),
    7: LobbySprite('잃어버린 악보', (751, 377, 342, 290), (180, 153), (1550, 410)),
    8: LobbySprite('찻잔 받침', (1111, 379, 317, 307), (330, 320), (1225, 817)),
    9: LobbySprite('커튼 끈', (68, 720, 227, 177), (100, 78), (1185, 301)),
    10: LobbySprite('담요', (384, 750, 339, 300), (310, 275), (415, 835)),
    11: LobbySprite('마지막 악보', (784, 677, 270, 382), (185, 262), (1348, 678)),
}
CAT = LobbySprite('까망이', (69, 15, 493, 607), (240, 296), (836, 723))
GHOSTS = {
    1: LobbySprite('A', (675, 54, 525, 567), (208, 225), (632, 554)),
    4: LobbySprite('C', (666, 646, 546, 591), (235, 255), (1488, 657)),
    7: LobbySprite('B', (61, 629, 573, 612), (250, 268), (355, 558)),
}
# Dialogue shares all lobby items and also shows the final garden door.
STORY_ITEMS = {
    **ITEMS,
    12: LobbySprite('정원문', (1080, 697, 335, 367), (170, 170), (0, 0)),
}
CHARACTERS = {'cat': CAT, 'A': GHOSTS[1], 'B': GHOSTS[7], 'C': GHOSTS[4]}
RESIDENT_VARIANTS = {'A': 0, 'B': 1, 'C': 2}
FINAL_STAGES = frozenset(range(1, 13))


def lobby_layers(completed, *, show_actors=True):
    """Return visible sprites in floor-depth order, without changing progress."""
    done = frozenset(completed)
    layers = [('item.png', sprite) for stage, sprite in ITEMS.items() if stage in done]
    if show_actors:
        layers.append(('character.png', CAT))
        layers.extend(('character.png', sprite)
                      for stage, sprite in GHOSTS.items() if stage in done)
    return tuple(sorted(layers, key=lambda layer: layer[1].foot[1]))


@lru_cache(maxsize=4)
def _image(filename):
    image = pygame.image.load(str(ASSET_DIR / 'art' / 'lobby' / filename))
    # Keep opaque room art in RGB. Adding alpha before smoothscale can round
    # opacity below 255 and mix the final picture with the previous frame.
    if image.get_flags() & pygame.SRCALPHA and pygame.display.get_surface():
        return image.convert_alpha()
    return image


@lru_cache(maxsize=48)
def _sprite(filename, spec):
    image = _image(filename).subsurface(spec.crop).copy()
    # A few neighbouring sprites extend across sheet cells. Keep only this object.
    component = pygame.mask.from_surface(image, 10).connected_component()
    if not component.count():
        raise ValueError(f'Empty lobby sprite: {spec.name}')
    isolation = component.to_surface(setcolor=(255, 255, 255, 255),
                                     unsetcolor=(255, 255, 255, 0))
    image.blit(isolation, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    scale = min(spec.size[0] / image.get_width(), spec.size[1] / image.get_height())
    return pygame.transform.smoothscale(image, (
        max(1, round(image.get_width() * scale)),
        max(1, round(image.get_height() * scale)),
    ))


def _draw_in_rect(surface, filename, spec, rect):
    rect = pygame.Rect(rect)
    image = _sprite(filename, replace(spec, size=rect.size))
    surface.blit(image, image.get_rect(midbottom=rect.midbottom))


def draw_story_character(surface, character_id, rect):
    """Fit a lobby portrait into the dialogue layout, preserving transparency."""
    _draw_in_rect(surface, 'character.png', CHARACTERS[character_id], rect)


def draw_story_item(surface, stage, rect):
    """Show that stage's painted item, independently of its emblem kind."""
    _draw_in_rect(surface, 'item.png', STORY_ITEMS[stage], rect)


@lru_cache(maxsize=32)
def _scene(completed, size, show_actors, released_variants=(), show_residents=True):
    # The final picture already contains everyone; never draw duplicates over it.
    if (FINAL_STAGES.issubset(completed) and show_actors
            and not released_variants and show_residents):
        return pygame.transform.smoothscale(_image('after.png'), size)
    scene = _image('before.png').copy()
    for filename, spec in lobby_layers(completed, show_actors=show_actors):
        if filename == 'character.png' and spec.name in RESIDENT_VARIANTS:
            if not show_residents:
                continue
            variant = RESIDENT_VARIANTS[spec.name]
            if variant in released_variants:
                # Keep this actor's original foot and depth amongst the PNG
                # furniture. Only its released body replaces the ghost sprite.
                resident_cat(scene, spec.foot, spec.size[0], 0., variant)
                continue
        image = _sprite(filename, spec)
        scene.blit(image, image.get_rect(midbottom=spec.foot))
    return pygame.transform.smoothscale(scene, size)


def draw_lobby(surface, completed=(), *, before=False, show_actors=True,
               released_variants=(), show_residents=True):
    """Paint saved progress, with optional active-ending resident bodies.

    The default completed room remains exactly the supplied after picture.
    Ending variants are composited from the same room, items and character
    sheet, without changing saved progress or the stage-based arrivals.
    """
    done = () if before else tuple(sorted(set(completed)))
    living = tuple(sorted(_released_residents(False, released_variants, before=before)))
    surface.blit(_scene(done, surface.get_size(), show_actors, living, bool(show_residents)), (0, 0))
