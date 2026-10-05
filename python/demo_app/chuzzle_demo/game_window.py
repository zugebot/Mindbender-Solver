# python/demo_app/chuzzle_demo/game_window.py
"""Finds the real Chuzzle window and sends mouse input to it (Windows only,
ctypes, no pywin32/pyautogui). Coordinates are physical screen pixels: call
set_dpi_aware() first so scaled displays don't shift clicks."""

from __future__ import annotations

import ctypes
import json
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from . import paths

user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Window titles of the Chuzzle 1 builds (python/deluxe_clone uses the same list).
KNOWN_TITLES = ("Chuzzle Deluxe 1.01", "Chuzzle Deluxe 1.0", "Chuzzle Christmas Edition 1.0")
OWN_TITLE = "Chuzzle Demo"

VK_ESCAPE = 0x1B
VK_F9 = 0x78
VK_MENU = 0x12
VK_SHIFT = 0x10
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Logical layout of the Mindbender screen at 800x600 (matches
# python/deluxe_clone: board at (255, 60), 80 px chuzzles). Board points are
# in cells with (0, 0) = centre of the top-left chuzzle, as in the mouse
# library and the route guide.
LOGICAL_W, LOGICAL_H = 800, 600
BOARD_ORIGIN = (255 + 40, 60 + 40)
CELL = 80.0

# UI targets in board-cell coordinates (route guide / studio measurements;
# PREV_WORLD derived from the old macro's window fractions).
LEVEL_BUTTONS = [(-3.18, 3.0475), (-2.67, 3.4205), (-2.11, 3.5125), (-1.52, 3.4205), (-0.98, 3.1125)]
NEXT_WORLD = (-1.22, 4.73)
PREV_WORLD = (-2.98, 4.73)
NEXT_PUZZLE_PROMPT = (2.0, 3.0)   # any of the studio prompt targets (1..4, 3.0)


def set_dpi_aware() -> None:
    try:
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def key_down(vk: int) -> bool:
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def _window_text(hwnd) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


def _process_exe(hwnd) -> Path | None:
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value)
    if not handle:
        return None
    try:
        size = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(size.value)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return Path(buf.value)
    finally:
        kernel32.CloseHandle(handle)
    return None


def _is_popcap_wrapper(exe: Path) -> bool:
    """PopCap's DRM wrapper runs the game as popcapgame1.exe or as a temp
    file it unpacks next to the real exe (e.g. pop41E2.tmp)."""
    name = exe.name.lower()
    return name.startswith("popcapgame") or (name.startswith("pop") and name.endswith(".tmp"))


def game_exe_for(exe: Path | None) -> Path | None:
    """The Chuzzle exe a game process belongs to: wrapper processes map back
    to the Chuzzle*.exe in the same folder, so saves and calibration stay
    tied to that install."""
    if exe is None or "chuzzle" in exe.name.lower():
        return exe
    if _is_popcap_wrapper(exe):
        for candidate in sorted(exe.parent.glob("Chuzzle*.exe")):
            return candidate
    return exe


@dataclass
class GameWindow:
    hwnd: int
    title: str
    exe: Path | None

    def alive(self) -> bool:
        return bool(user32.IsWindow(self.hwnd))

    def client_rect(self) -> tuple[int, int, int, int]:
        """(left, top, width, height) of the client area in screen pixels."""
        rect = wintypes.RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        pt = wintypes.POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y, rect.right - rect.left, rect.bottom - rect.top

    def is_foreground(self) -> bool:
        return user32.GetForegroundWindow() == self.hwnd

    def bring_to_front(self) -> bool:
        if user32.IsIconic(self.hwnd):
            user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
        # Windows only lets the foreground process change focus; a synthetic
        # Alt press lifts that lock.
        user32.keybd_event(VK_MENU, 0, 0, 0)
        user32.keybd_event(VK_MENU, 0, 2, 0)
        user32.SetForegroundWindow(self.hwnd)
        time.sleep(0.05)
        return self.is_foreground()


def find_game_windows() -> list[GameWindow]:
    found: list[GameWindow] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def callback(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        title = _window_text(hwnd)
        if not title or title.startswith(OWN_TITLE) or "chuzzle" not in title.lower():
            return True
        exe = _process_exe(hwnd)
        # Only the game itself: skips editors, consoles and Explorer windows
        # whose title merely mentions Chuzzle.
        is_game = exe is not None and ("chuzzle" in exe.name.lower() or _is_popcap_wrapper(exe))
        if title in KNOWN_TITLES or is_game:
            found.append(GameWindow(int(hwnd), title, game_exe_for(exe)))
        return True

    user32.EnumWindows(callback, 0)
    found.sort(key=lambda w: (w.title not in KNOWN_TITLES, w.title))
    return found


# --------------------------------------------------------------------------
# Geometry: board points -> screen pixels, with optional user calibration
# --------------------------------------------------------------------------

CALIBRATION_FILE = paths.USER_DATA_DIR / "calibration.json"


@dataclass
class Geometry:
    """Board-cell point -> client fraction is origin + point * cell (both in
    client fractions). Default comes from the 800x600 layout; calibration
    replaces it with two measured chuzzle centres."""
    origin: tuple[float, float] = (BOARD_ORIGIN[0] / LOGICAL_W, BOARD_ORIGIN[1] / LOGICAL_H)
    cell: tuple[float, float] = (CELL / LOGICAL_W, CELL / LOGICAL_H)
    calibrated: bool = False

    def to_screen(self, window: GameWindow, point: tuple[float, float]) -> tuple[int, int]:
        left, top, width, height = window.client_rect()
        fx = self.origin[0] + point[0] * self.cell[0]
        fy = self.origin[1] + point[1] * self.cell[1]
        return round(left + fx * width), round(top + fy * height)

    @staticmethod
    def from_corners(top_left: tuple[float, float], bottom_right: tuple[float, float]) -> "Geometry":
        cell = ((bottom_right[0] - top_left[0]) / 5.0, (bottom_right[1] - top_left[1]) / 5.0)
        return Geometry(top_left, cell, True)


def client_fraction(window: GameWindow, screen_pt: tuple[int, int]) -> tuple[float, float]:
    left, top, width, height = window.client_rect()
    return (screen_pt[0] - left) / max(1, width), (screen_pt[1] - top) / max(1, height)


def cursor_pos() -> tuple[int, int]:
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _calibration_key(window: GameWindow) -> str:
    return str(window.exe) if window.exe else window.title


def load_geometry(window: GameWindow) -> Geometry:
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
        entry = data[_calibration_key(window)]
        return Geometry(tuple(entry["origin"]), tuple(entry["cell"]), True)
    except Exception:
        return Geometry()


def save_geometry(window: GameWindow, geometry: Geometry) -> None:
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
    except Exception:
        data = {}
    data[_calibration_key(window)] = {"origin": list(geometry.origin), "cell": list(geometry.cell),
                                      "title": window.title}
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


def clear_geometry(window: GameWindow) -> None:
    try:
        data = json.loads(CALIBRATION_FILE.read_text())
        data.pop(_calibration_key(window), None)
        CALIBRATION_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


# --------------------------------------------------------------------------
# Raw mouse
# --------------------------------------------------------------------------

def move_cursor(x: int, y: int) -> None:
    user32.SetCursorPos(int(x), int(y))


def key_tap(vk: int) -> None:
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, 2, 0)   # KEYEVENTF_KEYUP


def mouse_down() -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def mouse_up() -> None:
    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
