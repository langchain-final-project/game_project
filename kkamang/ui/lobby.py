"""완료한 방의 PNG 소품·캐릭터를 쌓아 쉼터를 구성한다.

엔딩별 고양이 교체·친구 숨김은 저장된 진행을 바꾸지 않는 표시 처리다.
"""
from dataclasses import dataclass, replace
from functools import lru_cache

import pygame

from ..paths import ASSET_DIR
from .illustration import _released_residents


@dataclass(frozen=True)
class LobbySprite:
    name: str
    crop: tuple[int, int, int, int]
    size: tuple[int, int]
    foot: tuple[int, int]

# 위치는 원본 쉼터 그림의 1672×941 좌표 기준이며 소품에는 해당 가구도 포함된다.
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
# 대화용 소품에는 쉼터 공용 물건과 마지막 정원문을 포함한다.
STORY_ITEMS = {
    **ITEMS,
    12: LobbySprite('정원문', (1080, 697, 335, 367), (170, 170), (0, 0)),
}
CHARACTERS = {'cat': CAT, 'A': GHOSTS[1], 'B': GHOSTS[7], 'C': GHOSTS[4]}
RESIDENT_VARIANTS = {'A': 0, 'B': 1, 'C': 2}
RESIDENT_IMAGES = {
    0: 'resident_bosong.png',
    1: 'resident_ipari.png',
    2: 'resident_dore.png',
}
FINAL_STAGES = frozenset(range(1, 13))


def lobby_layers(completed, *, show_actors=True):
    """진행 상태를 바꾸지 않고 표시할 PNG를 발의 깊이 순서로 반환한다."""
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
    # 불투명 배경은 RGB로 유지해 축소 시 알파 반올림으로 이전 프레임이 비치지 않게 한다.
    if image.get_flags() & pygame.SRCALPHA and pygame.display.get_surface():
        return image.convert_alpha()
    return image

@lru_cache(maxsize=48)
def _sprite(filename, spec):
    image = _image(filename).subsurface(spec.crop).copy()
    # 시트의 이웃 그림이 영역에 걸치므로 연결된 본체만 분리한다.
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

@lru_cache(maxsize=3)
def _resident_image(variant):
    """고양이 PNG의 투명한 바깥 여백만 제거해 캐시한다."""
    filename = RESIDENT_IMAGES[variant]
    image = pygame.image.load(str(ASSET_DIR / 'art' / 'character' / filename))
    if pygame.display.get_surface():
        image = image.convert_alpha()
    bounds = image.get_bounding_rect(min_alpha=10)
    if not bounds.width or not bounds.height:
        raise ValueError(f'Empty resident PNG: {filename}')
    return image.subsurface(bounds).copy()

@lru_cache(maxsize=12)
def _resident_sprite(variant, size):
    image = _resident_image(variant)
    scale = min(size[0] / image.get_width(), size[1] / image.get_height())
    return pygame.transform.smoothscale(image, (
        max(1, round(image.get_width() * scale)),
        max(1, round(image.get_height() * scale)),
    ))

def _draw_in_rect(surface, filename, spec, rect):
    rect = pygame.Rect(rect)
    image = _sprite(filename, replace(spec, size=rect.size))
    surface.blit(image, image.get_rect(midbottom=rect.midbottom))

def draw_story_character(surface, character_id, rect):
    """투명도를 보존하며 대화 영역에 쉼터 캐릭터 PNG를 맞춘다."""
    _draw_in_rect(surface, 'character.png', CHARACTERS[character_id], rect)

def draw_story_item(surface, stage, rect):
    """문양 종류와 관계없이 해당 스테이지의 물건 원화를 표시한다."""
    _draw_in_rect(surface, 'item.png', STORY_ITEMS[stage], rect)

@lru_cache(maxsize=32)
def _scene(completed, size, show_actors, released_variants=(), show_residents=True):
    # 완성 원화에 캐릭터가 이미 있으므로 중복해서 그리지 않는다.
    if (FINAL_STAGES.issubset(completed) and show_actors
            and not released_variants and show_residents):
        return pygame.transform.smoothscale(_image('after.png'), size)
    scene = _image('before.png').copy()
    for filename, spec in lobby_layers(completed, show_actors=show_actors):
        image = None
        if filename == 'character.png' and spec.name in RESIDENT_VARIANTS:
            if not show_residents:
                continue
            variant = RESIDENT_VARIANTS[spec.name]
            if variant in released_variants:
                # 유령을 고양이 PNG로 바꿔도 발 위치와 가구 사이의 앞뒤 순서는 유지한다.
                image = _resident_sprite(variant, spec.size)
        if image is None:
            image = _sprite(filename, spec)
        scene.blit(image, image.get_rect(midbottom=spec.foot))
    return pygame.transform.smoothscale(scene, size)

def draw_lobby(surface, completed=(), *, before=False, show_actors=True,
               released_variants=(), show_residents=True):
    """저장된 완료 상태와 현재 엔딩으로 쉼터를 그린다.

    완성 기본 화면은 제공된 원화를 쓰고 엔딩별 모습만 같은 발 위치에 합성한다.
    """
    done = () if before else tuple(sorted(set(completed)))
    living = tuple(sorted(_released_residents(False, released_variants, before=before)))
    surface.blit(_scene(done, surface.get_size(), show_actors, living, bool(show_residents)), (0, 0))
