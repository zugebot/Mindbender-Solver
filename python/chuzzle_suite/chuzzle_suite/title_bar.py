# python/chuzzle_suite/chuzzle_suite/title_bar.py
"""The app's own title bar, drawn with pygame like a dark Windows 10 one:
title on the left; on the right pin, minimize, maximize (or expand) and
close buttons, 46 px wide each. The window itself is pygame.NOFRAME;
window_frame.py does the Win32 part."""

from __future__ import annotations

import math
import time

import pygame as pg

HEIGHT = 32
BUTTON_W = 46

FILL = (18, 20, 25)
LINE = (46, 50, 62)
ICON = (220, 222, 228)
ICON_OFF = (92, 96, 108)        # disabled (maximize: the window has a fixed size)
TITLE_ON = (226, 228, 235)
TITLE_OFF = (128, 132, 146)     # window not focused
HOVER = (52, 56, 68)
PRESS = (66, 71, 86)
CLOSE_HOVER = (232, 17, 35)     # Windows' close red
CLOSE_PRESS = (241, 112, 122)
PIN_ON = (90, 170, 255)         # the app's ACCENT

# Upright pushpin outline (head, body, collar), centre near (0, 0); the needle is drawn as a line.
_PIN = [(-3.5, -7.5), (3.5, -7.5), (3.5, -6), (2, -6), (2, -1.5), (4.5, -1.5), (4.5, 1),
        (-4.5, 1), (-4.5, -1.5), (-2, -1.5), (-2, -6), (-3.5, -6)]
_NEEDLE = [(0, 1), (0, 7.5)]
TOOLTIPS = {"pin": ("Keep on top", "Stop keeping on top"), "min": "Minimize", "max": "Maximize (fixed size)",
            "expand": "Full view", "close": "Close"}


class TitleBar:
    def __init__(self, title: str, buttons: tuple[str, ...]):
        self.title = title
        self.buttons = buttons          # left to right, from: pin, min, max, expand, close
        self.disabled: set[str] = set()
        self.pressed: str | None = None  # button under a held mouse press
        self._hover: tuple[str | None, float] = (None, 0.0)

    def rects(self, width: int) -> dict[str, pg.Rect]:
        x = width - BUTTON_W * len(self.buttons)
        return {name: pg.Rect(x + i * BUTTON_W, 0, BUTTON_W, HEIGHT) for i, name in enumerate(self.buttons)}

    def hit(self, pos, width: int) -> str | None:
        """The button at pos, "drag" for the rest of the bar, None below it."""
        if not (0 <= pos[1] < HEIGHT and 0 <= pos[0] < width):
            return None
        for name, rect in self.rects(width).items():
            if rect.collidepoint(pos):
                return name
        return "drag"

    def draw(self, surface: pg.Surface, font: pg.font.Font, mouse, pinned: bool, focused: bool = True) -> None:
        width = surface.get_width()
        bar = pg.Rect(0, 0, width, HEIGHT)
        pg.draw.rect(surface, FILL, bar)
        pg.draw.line(surface, LINE, (0, HEIGHT - 1), (width, HEIGHT - 1))
        _app_icon(surface, (16, HEIGHT // 2))
        title = font.render(self.title, True, TITLE_ON if focused else TITLE_OFF)
        surface.blit(title, title.get_rect(midleft=(32, HEIGHT // 2)))
        over = self.hit(mouse, width) if mouse is not None else None
        for name, rect in self.rects(width).items():
            enabled = name not in self.disabled
            hover = enabled and over == name and (self.pressed in (None, name))
            held = hover and self.pressed == name
            if hover:
                if name == "close":
                    color = CLOSE_PRESS if held else CLOSE_HOVER
                else:
                    color = PRESS if held else HOVER
                pg.draw.rect(surface, color, rect)
            icon = ICON if enabled else ICON_OFF
            if name == "close" and hover:
                icon = (255, 255, 255)
            elif not focused and enabled:
                icon = TITLE_OFF
            _ICONS[name](surface, rect.center, icon, pinned)
        self._tooltip(surface, font, over, pinned)

    def _tooltip(self, surface, font, over, pinned) -> None:
        """Windows-style hint under a button after hovering it for a moment."""
        if over != self._hover[0]:
            self._hover = (over, time.monotonic())
        if over in (None, "drag") or self.pressed or time.monotonic() - self._hover[1] < 0.6:
            return
        tip = TOOLTIPS.get(over, "")
        if isinstance(tip, tuple):
            tip = tip[1] if pinned else tip[0]
        text = font.render(tip, True, TITLE_ON)
        box = text.get_rect().inflate(12, 8)
        rect = self.rects(surface.get_width())[over]
        box.midtop = (rect.centerx, HEIGHT + 4)
        box.clamp_ip(surface.get_rect().inflate(-4, -4))
        pg.draw.rect(surface, (30, 33, 41), box)
        pg.draw.rect(surface, LINE, box, 1)
        surface.blit(text, text.get_rect(center=box.center))


def _app_icon(surface, center) -> None:
    """Four chuzzle dots, like the boards in the app."""
    cx, cy = center
    for (dx, dy), color in zip(((-4, -4), (4, -4), (-4, 4), (4, 4)),
                               ((230, 90, 90), (80, 200, 120), (90, 170, 255), (240, 190, 70))):
        pg.draw.circle(surface, color, (cx + dx, cy + dy), 3)


def _min(surface, c, color, _pinned) -> None:
    pg.draw.line(surface, color, (c[0] - 5, c[1]), (c[0] + 5, c[1]))


def _max(surface, c, color, _pinned) -> None:
    pg.draw.rect(surface, color, pg.Rect(c[0] - 5, c[1] - 5, 11, 11), 1)


def _expand(surface, c, color, _pinned) -> None:
    """A box with an arrow out of its top-right corner: back to the full view."""
    x, y = c[0] - 6, c[1] - 4
    pg.draw.lines(surface, color, False, [(x + 5, y), (x, y), (x, y + 10), (x + 10, y + 10), (x + 10, y + 5)])
    pg.draw.line(surface, color, (x + 4, y + 6), (x + 12, y - 2))
    pg.draw.lines(surface, color, False, [(x + 7, y - 2), (x + 12, y - 2), (x + 12, y + 3)])


def _close(surface, c, color, _pinned) -> None:
    pg.draw.aaline(surface, color, (c[0] - 5, c[1] - 5), (c[0] + 5, c[1] + 5))
    pg.draw.aaline(surface, color, (c[0] - 5, c[1] + 5), (c[0] + 5, c[1] - 5))


_PIN_SIZE, _PIN_SS = 20, 4      # icon box (px) and supersampling factor
_pin_cache: dict = {}


def _pin(surface, c, color, pinned) -> None:
    """Pinned: upright and filled in the accent colour. Unpinned: an outline tilted 45 degrees.
    Drawn 4x larger and scaled down, so the slanted edges come out smooth."""
    if pinned:
        color = PIN_ON
    key = (pinned, color)
    icon = _pin_cache.get(key)
    if icon is None:
        k, half = _PIN_SS, _PIN_SIZE * _PIN_SS / 2
        angle = 0.0 if pinned else math.radians(45)
        cos, sin = math.cos(angle), math.sin(angle)

        def place(points):
            return [(half + k * (x * cos - y * sin), half + k * (x * sin + y * cos)) for x, y in points]

        big = pg.Surface((_PIN_SIZE * k, _PIN_SIZE * k), pg.SRCALPHA)
        outline, needle = place(_PIN), place(_NEEDLE)
        pg.draw.polygon(big, color, outline, 0 if pinned else k)
        pg.draw.line(big, color, needle[0], needle[1], k + (k // 2 if pinned else 0))
        icon = pg.transform.smoothscale(big, (_PIN_SIZE, _PIN_SIZE))
        _pin_cache[key] = icon
    surface.blit(icon, icon.get_rect(center=(c[0], c[1] + 1)))


_ICONS = {"pin": _pin, "min": _min, "max": _max, "expand": _expand, "close": _close}
