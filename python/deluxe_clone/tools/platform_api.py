# Jerrin Shirks
# tools/platform_api.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class WindowRect:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


class PlatformAPI:
    """
    Thin platform abstraction.

    This isolates all OS and environment specific behavior:
    - global mouse position relative to window client area
    - window rect
    - window movement / minimize
    - global mouse button state
    - DPI setup

    The game code should not know whether the source is:
    - Win32 ctypes
    - browser / emscripten JS bridge
    - X11 / Wayland / SDL fallback
    """

    def has_focus(self) -> bool:
        return True

    def setup_process(self) -> None:
        pass

    def get_window_client_mouse_pos(self, pygame_instance) -> Tuple[int, int]:
        return pygame_instance.mouse.get_pos()

    def get_window_rect(self, pygame_instance) -> Optional[WindowRect]:
        return None

    def set_window_pos(self, pygame_instance, x: int, y: int) -> None:
        pass

    def minimize_window(self, pygame_instance) -> None:
        pass

    def get_global_mouse_buttons(self, pygame_instance) -> Tuple[bool, bool]:
        pressed = pygame_instance.mouse.get_pressed(3)
        return bool(pressed[0]), bool(pressed[2])