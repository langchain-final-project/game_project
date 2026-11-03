"""유령 PNG의 방향별 자세를 읽고 발 기준점과 크기를 고정해 캐시한다."""
from functools import lru_cache
import pygame

from ..paths import ASSET_DIR

GHOST_FILES = {'A': 'a.png', 'B': 'b.png', 'C': 'c.png'}
RESIDENT_IDS = {0: 'A', 1: 'B', 2: 'C'}  # 렌더러의 변형 번호는 보송·이파리·도레 순서다.
DIRECTION_INDEX = {
    'down': 0, 'down_left': 1, 'left': 2, 'up_left': 3,
    'up': 4, 'up_right': 5, 'right': 6, 'down_right': 7,
}
SHEET_SIZE = (1774, 887)
CANVAS_SIZE = (256, 282)
GROUND_ANCHOR = (128, 268)


def movement_direction(origin, target, default='down'):
    """보간한 화면 좌표 대신 논리적 이동 시작점과 끝점으로 방향을 판정한다."""
    dx, dy = target[0] - origin[0], target[1] - origin[1]
    if dx and abs(dx) >= abs(dy):
        return 'right' if dx > 0 else 'left'
    if dy:
        return 'down' if dy > 0 else 'up'
    return default

@lru_cache(maxsize=3)
def _atlas(ghost_id):
    image = pygame.image.load(str(ASSET_DIR / 'art' / 'character' / GHOST_FILES[ghost_id]))
    if image.get_size() != SHEET_SIZE:
        raise ValueError(f'Ghost {ghost_id} atlas must be {SHEET_SIZE}')
    if pygame.display.get_surface():
        image = image.convert_alpha()
    columns = (0, 443, 887, 1330, 1774)
    rows = (0, 443, 887)
    frames = []
    for y in range(2):
        for x in range(4):
            cell = image.subsurface((columns[x], rows[y], columns[x + 1] - columns[x],
                                      rows[y + 1] - rows[y]))
            bounds = cell.get_bounding_rect(min_alpha=10)
            if not bounds.width or not bounds.height:
                raise ValueError(f'Ghost {ghost_id} has an empty pose: {y * 4 + x}')
            frame = cell.subsurface(bounds).copy()
            # 모자와 꽃다발이 중심을 밀지 않도록 몸 아래쪽으로 발 기준점을 잡는다.
            top = frame.get_height() * 2 // 3
            lower = frame.subsurface((0, top, frame.get_width(), frame.get_height() - top))
            pivot = (lower.get_bounding_rect(min_alpha=10).centerx, frame.get_height())
            frames.append((frame, pivot))
    height = max(frame.get_height() for frame, _ in frames)
    half_width = max(max(pivot[0], frame.get_width() - pivot[0]) for frame, pivot in frames)
    # 자세별로 따로 맞추지 않고 한 캐릭터에 같은 배율을 적용한다.
    return tuple(frames), min(236 / height, 120 / half_width)

@lru_cache(maxsize=24)
def _normalized_frame(ghost_id, direction):
    frames, scale = _atlas(ghost_id)
    source, pivot = frames[DIRECTION_INDEX[direction]]
    image = pygame.transform.smoothscale(source, (round(source.get_width() * scale),
                                                 round(source.get_height() * scale)))
    destination = (GROUND_ANCHOR[0] - round(pivot[0] * scale),
                   GROUND_ANCHOR[1] - round(pivot[1] * scale))
    canvas = pygame.Surface(CANVAS_SIZE, pygame.SRCALPHA)
    if not canvas.get_rect().contains(image.get_rect(topleft=destination)):
        raise ValueError(f'Ghost frame exceeds its canvas: {ghost_id}, {direction}')
    canvas.blit(image, destination)
    return canvas

@lru_cache(maxsize=256)
def ghost_frame(size=50, ghost_id='A', direction='down'):
    if ghost_id not in GHOST_FILES:
        raise ValueError(f'Unknown ghost ID: {ghost_id}')
    if direction not in DIRECTION_INDEX:
        raise ValueError(f'Unknown ghost direction: {direction}')
    width = max(1, int(size))
    return pygame.transform.smoothscale(
        _normalized_frame(ghost_id, direction),
        (width, max(1, int(width * CANVAS_SIZE[1] / CANVAS_SIZE[0]))),
    )
