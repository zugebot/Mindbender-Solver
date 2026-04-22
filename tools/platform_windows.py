# Jerrin Shirks
# tools/platform_windows.py

from __future__ import annotations

import ctypes
from typing import Optional, Tuple

from tools.platform_api import PlatformAPI, WindowRect


class WindowsPlatformAPI(PlatformAPI):
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def setup_process(self) -> None:
        try:
            dpi_awareness_context_per_monitor_aware_v2 = ctypes.c_void_p(-4)
            ctypes.windll.user32.SetProcessDpiAwarenessContext(
                dpi_awareness_context_per_monitor_aware_v2
            )
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass

    def _get_hwnd(self, pygame_instance):
        try:
            return pygame_instance.display.get_wm_info().get("window")
        except Exception:
            return None

    def get_window_client_mouse_pos(self, pygame_instance) -> Tuple[int, int]:
        """
        Returns mouse position relative to the window client space,
        even when the cursor is outside the window.
        """
        try:
            hwnd = self._get_hwnd(pygame_instance)
            if not hwnd:
                return pygame_instance.mouse.get_pos()

            pt = self._POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            ctypes.windll.user32.ScreenToClient(hwnd, ctypes.byref(pt))
            return int(pt.x), int(pt.y)
        except Exception:
            return pygame_instance.mouse.get_pos()

    def get_window_rect(self, pygame_instance) -> Optional[WindowRect]:
        try:
            hwnd = self._get_hwnd(pygame_instance)
            if not hwnd:
                return None

            rect = self._RECT()
            ok = ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None

            return WindowRect(
                left=int(rect.left),
                top=int(rect.top),
                right=int(rect.right),
                bottom=int(rect.bottom),
            )
        except Exception:
            return None

    def set_window_pos(self, pygame_instance, x: int, y: int) -> None:
        try:
            hwnd = self._get_hwnd(pygame_instance)
            if not hwnd:
                return

            swp_nosize = 0x0001
            swp_nozorder = 0x0004
            swp_noactivate = 0x0010

            ctypes.windll.user32.SetWindowPos(
                hwnd,
                None,
                int(x),
                int(y),
                0,
                0,
                swp_nosize | swp_nozorder | swp_noactivate,
            )
        except Exception:
            pass

    def minimize_window(self, pygame_instance) -> None:
        try:
            hwnd = self._get_hwnd(pygame_instance)
            if not hwnd:
                return

            sw_minimize = 6
            ctypes.windll.user32.ShowWindow(hwnd, sw_minimize)
        except Exception:
            pass

    def get_global_mouse_buttons(self, pygame_instance) -> Tuple[bool, bool]:
        try:
            left_button_down = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
            right_button_down = (ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000) != 0
            return bool(left_button_down), bool(right_button_down)
        except Exception:
            pressed = pygame_instance.mouse.get_pressed(3)
            return bool(pressed[0]), bool(pressed[2])