"""Small, cached drawing vocabulary for the mansion's live Pygame UI.

Text is always rendered from live strings. Artwork is decorative and never owns
game state or collision coordinates.
"""
from __future__ import annotations

import math
from pathlib import Path
import pygame

from kkamang.paths import ASSET_DIR

CREAM = (255, 246, 230)
INK = (59, 48, 71)
NIGHT = (40, 32, 55)
PAPER = (249, 238, 214)
GOLD = (225, 187, 112)
LAVENDER = (165, 138, 197)
SAGE = (142, 169, 145)
MUTED = (140, 119, 151)
PEACH = (232, 160, 160)


class Assets:
    def __init__(self, root: Path = ASSET_DIR):
        self.root = Path(root)
        self._originals = {}
        self._scaled = {}
        self._fonts = {}
        self._covers = {}
        self._icons = {}
        self._cats = {}

    def cat(self, size: tuple, scarf="default") -> pygame.Surface:
        """Use the exact same rig as field animation and story portraits."""
        from .illustration import cat_frame
        source = cat_frame(int(size[0]), scarf=scarf)
        canvas = pygame.Surface(tuple(map(int,size)), pygame.SRCALPHA)
        factor = min(size[0]/source.get_width(), size[1]/source.get_height())
        image = pygame.transform.smoothscale(source,(int(source.get_width()*factor),int(source.get_height()*factor)))
        canvas.blit(image,image.get_rect(midbottom=(size[0]//2,size[1])))
        return canvas

    def image(self, name: str, size: tuple | None = None) -> pygame.Surface:
        name = str(name)
        if name not in self._originals:
            path = self.root / name
            try:
                im = pygame.image.load(str(path))
                # convert_alpha needs a video mode, but headless test surfaces do not.
                self._originals[name] = im.convert_alpha() if pygame.display.get_surface() else im
            except (FileNotFoundError, pygame.error):
                im = pygame.Surface((64, 64), pygame.SRCALPHA)
                pygame.draw.circle(im, GOLD, (32, 32), 25, 2)
                pygame.draw.circle(im, LAVENDER, (32, 32), 13)
                self._originals[name] = im
        if size is None:
            return self._originals[name]
        size = (max(1, int(size[0])), max(1, int(size[1])))
        key = (name, size)
        if key not in self._scaled:
            self._scaled[key] = pygame.transform.smoothscale(self._originals[name], size)
        return self._scaled[key]

    def icon(self, name: str, size: tuple) -> pygame.Surface:
        """Fit a transparent icon's painted bounds, ignoring export padding."""
        key = (name, tuple(map(int, size)))
        if key not in self._icons:
            source = self.image(name)
            box = source.get_bounding_rect(min_alpha=10)
            if not box.width or not box.height:
                return self.image(name, size)
            factor = min(size[0] / box.width, size[1] / box.height)
            target = (max(1, round(box.width * factor)), max(1, round(box.height * factor)))
            cut = pygame.transform.smoothscale(source.subsurface(box), target)
            result = pygame.Surface(tuple(map(int, size)), pygame.SRCALPHA)
            result.blit(cut, cut.get_rect(center=(size[0] / 2, size[1] / 2)))
            self._icons[key] = result
        return self._icons[key]

    def font(self, size: int, bold=False) -> pygame.font.Font:
        if not pygame.font.get_init():
            pygame.font.init()
        key = (max(8, int(size)), bool(bold))
        if key not in self._fonts:
            filename = self.root / "fonts" / ("NotoSansKR-Bold.ttf" if bold else "NotoSansKR-Regular.ttf")
            if not filename.exists():
                filename = self.root / "fonts" / "NotoSansKR.ttf"
            f = pygame.font.Font(str(filename) if filename.exists() else None, key[0])
            # SDL selects the minimum axis of the source variable font. Static
            # 400/700 instances avoid a thin, unreadable Korean UI on Windows.
            if "Regular" not in filename.name and "Bold" not in filename.name:
                f.set_bold(bool(bold))
            self._fonts[key] = f
        return self._fonts[key]

    def title_font(self, size: int, bold=True) -> pygame.font.Font:
        key = ("title", int(size), bool(bold))
        if key not in self._fonts:
            filename = self.root / "fonts" / ("NanumMyeongjo-Bold.ttf" if bold else "NanumMyeongjo-Regular.ttf")
            self._fonts[key] = pygame.font.Font(str(filename), int(size)) if filename.exists() else self.font(size, bold)
        return self._fonts[key]

    def background(self, surface, name, shade=0):
        size = surface.get_size()
        key = (name, size, int(shade))
        if key not in self._covers:
            original = self.image(name)
            scale = max(size[0] / original.get_width(), size[1] / original.get_height())
            scaled = pygame.transform.smoothscale(original, (round(original.get_width() * scale), round(original.get_height() * scale)))
            image = pygame.Surface(size, pygame.SRCALPHA)
            image.blit(scaled, scaled.get_rect(center=(size[0] // 2, size[1] // 2)))
            if shade:
                veil = pygame.Surface(size, pygame.SRCALPHA)
                veil.fill((*NIGHT, max(0, min(255, int(shade)))))
                image.blit(veil, (0, 0))
            self._covers[key] = image
        surface.blit(self._covers[key], (0, 0))


def text(surface, assets, content, pos, size=22, color=CREAM, bold=False, anchor="topleft"):
    label = assets.font(size, bold).render(str(content), True, color)
    rect = label.get_rect()
    setattr(rect, anchor, (round(pos[0]), round(pos[1])))
    surface.blit(label, rect)
    return rect


def fit_text(surface, assets, content, rect, size=22, color=CREAM, bold=False, anchor="center"):
    rect = pygame.Rect(rect)
    while size > 12 and assets.font(size, bold).size(str(content))[0] > rect.width:
        size -= 1
    return text(surface, assets, content, getattr(rect, anchor), size, color, bold, anchor)


def wrapped(surface, assets, content, rect, size=20, color=INK, line_gap=5):
    """Character-aware wrapping works with Korean, Latin and explicit newlines."""
    rect = pygame.Rect(rect)
    f = assets.font(size)
    y = rect.y
    line_height = f.get_height() + line_gap
    for paragraph in str(content).split("\n"):
        line = ""
        lines = []
        for char in paragraph:
            trial = line + char
            if line and f.size(trial)[0] > rect.width:
                lines.append(line.rstrip())
                line = char.lstrip()
            else:
                line = trial
        lines.append(line)
        for line in lines:
            if rect.height > 0 and y + f.get_height() > rect.bottom:
                return y
            text(surface, assets, line, (rect.x, y), size, color)
            y += line_height
    return y


def panel(surface, rect, fill=PAPER, border=GOLD, radius=16):
    r = pygame.Rect(rect)
    shadow = pygame.Surface((r.width + 8, r.height + 10), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (12, 9, 25, 65), shadow.get_rect().inflate(-5, -5), border_radius=radius)
    surface.blit(shadow, (r.x, r.y + 4))
    pygame.draw.rect(surface, fill, r, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, r, 1, border_radius=radius)
        pygame.draw.rect(surface, (*border[:3],), r.inflate(-9, -9), 1, border_radius=max(4, radius - 4))


def dark_panel(surface, rect, radius=18):
    panel(surface, rect, fill=(53, 43, 69), border=(168, 145, 112), radius=radius)


def rounded_box(surface, rect, fill=PAPER, border=None, radius=10):
    r = pygame.Rect(rect)
    pygame.draw.rect(surface, fill, r, border_radius=radius)
    if border:
        pygame.draw.rect(surface, border, r, 1, border_radius=radius)


def button(surface, assets, rect, label, focused=False, disabled=False, primary=False):
    r = pygame.Rect(rect)
    if disabled:
        bg, edge, fg = (178, 166, 160), (155, 140, 132), (100, 88, 88)
    elif primary:
        bg, edge, fg = (235, 194, 124), (253, 226, 173), INK
    else:
        bg, edge, fg = (230, 215, 190), (175, 145, 113), INK
    if focused and not disabled:
        pygame.draw.rect(surface, CREAM, r.inflate(8, 8), 2, border_radius=18)
        bg = tuple(min(255, n + 12) for n in bg)
    pygame.draw.rect(surface, (28, 22, 38), r.move(0, 3), border_radius=13)
    pygame.draw.rect(surface, bg, r, border_radius=13)
    pygame.draw.rect(surface, edge, r, 1, border_radius=13)
    pygame.draw.rect(surface, edge, r.inflate(-8, -8), 1, border_radius=9)
    fit_text(surface, assets, label, r.inflate(-22, -4), 21, fg, bold=primary)


def soft_glow(surface, center, radius=24, color=GOLD, strength=50):
    # Tiny local particle glows, not full-frame filters.
    radius = int(radius)
    s = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
    for i in range(6, 0, -1):
        pygame.draw.circle(s, (*color[:3], max(1, int(strength * (7 - i) / 12))), (radius, radius), max(1, radius * i // 6))
    surface.blit(s, (center[0] - radius, center[1] - radius))


def draw_sparkles(surface, elapsed=0.0, rect=None, color=GOLD, count=16):
    r = pygame.Rect(rect or surface.get_rect())
    for i in range(count):
        x = r.x + ((i * 173 + 37) % max(1, r.width))
        y = r.y + ((i * 97 + 17) % max(1, r.height))
        t = (math.sin(elapsed * .7 + i * 2.4) + 1) / 2
        if t > .45:
            c = tuple(int(n * (.32 + .4 * t)) for n in color)
            pygame.draw.line(surface, c, (x - 2, y), (x + 2, y))
            pygame.draw.line(surface, c, (x, y - 2), (x, y + 2))
