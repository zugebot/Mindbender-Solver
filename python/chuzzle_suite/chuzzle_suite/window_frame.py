# python/chuzzle_suite/chuzzle_suite/window_frame.py
"""Win32 side of the app's own title bar (the window is pygame.NOFRAME):
always on top, native dragging, minimize from the taskbar, and placing the
window when it changes size. Coordinates are physical pixels (the app is
DPI aware). Every call does nothing when there is no window handle (e.g.
SDL's dummy video driver in tests)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

import pygame as pg

# Own function objects, so these argtypes don't touch other modules' user32 calls.
user32 = ctypes.WinDLL("user32", use_last_error=True)

GWL_STYLE = -16
GWL_EXSTYLE = -20
WS_MAXIMIZEBOX = 0x00010000
WS_MINIMIZEBOX = 0x00020000
WS_SYSMENU = 0x00080000
WS_EX_TOPMOST = 0x00000008
HWND_TOPMOST, HWND_NOTOPMOST = -1, -2
SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x1, 0x2, 0x4, 0x10, 0x20
WM_NCLBUTTONDOWN, HTCAPTION = 0xA1, 2
MONITOR_DEFAULTTONEAREST = 2


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


LONG_PTR = ctypes.c_ssize_t
_get_long = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
_set_long = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
_get_long.argtypes = [wintypes.HWND, ctypes.c_int]
_get_long.restype = LONG_PTR
_set_long.argtypes = [wintypes.HWND, ctypes.c_int, LONG_PTR]
_set_long.restype = LONG_PTR
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, wintypes.UINT]
user32.SetWindowPos.restype = wintypes.BOOL
user32.ReleaseCapture.restype = wintypes.BOOL
user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.SendMessageW.restype = LONG_PTR
user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
user32.IsIconic.argtypes = [wintypes.HWND]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]


def set_app_id(app_id: str) -> None:
    """Groups this process's windows under `app_id` on the taskbar (call before the first
    window opens). The Start menu shortcut carries the same id, so its pinned icon and the
    running window are one taskbar button with the shortcut's icon."""
    try:
        ctypes.WinDLL("shell32").SetCurrentProcessExplicitAppUserModelID(ctypes.c_wchar_p(app_id))
    except (AttributeError, OSError):
        pass


class _GUID(ctypes.Structure):
    _fields_ = [("d1", wintypes.DWORD), ("d2", wintypes.WORD), ("d3", wintypes.WORD), ("d4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, text: str) -> "_GUID":
        h = text.strip("{}").replace("-", "")
        return cls(int(h[:8], 16), int(h[8:12], 16), int(h[12:16], 16),
                   (ctypes.c_ubyte * 8)(*bytes.fromhex(h[16:])))


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT(ctypes.Structure):   # only VT_LPWSTR is used; 24 bytes on x64 like the real one
    _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort),
                ("r3", ctypes.c_ushort), ("pwsz", ctypes.c_wchar_p), ("pad", ctypes.c_void_p)]


_IID_PROPERTY_STORE = "{886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99}"
_APP_USER_MODEL = "{9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3}"
# System.AppUserModel.* property ids
_RELAUNCH_COMMAND, _RELAUNCH_ICON, _RELAUNCH_NAME, _ID = 2, 3, 4, 5
VT_LPWSTR = 31


def set_relaunch(hwnd: int | None, app_id: str, command: str, name: str, icon: str) -> bool:
    """Tells the taskbar how to start the app again, under what name and with which icon. Without
    this, pinning the running window pins plain pythonw.exe (no script, Python's icon)."""
    if not hwnd:
        return False
    try:
        shell32 = ctypes.WinDLL("shell32")
        store = ctypes.c_void_p()
        iid = _GUID.parse(_IID_PROPERTY_STORE)
        if shell32.SHGetPropertyStoreForWindow(wintypes.HWND(hwnd), ctypes.byref(iid), ctypes.byref(store)) != 0:
            return False
        vtbl = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
        # IPropertyStore: 0 QueryInterface, 1 AddRef, 2 Release, ..., 6 SetValue, 7 Commit
        release = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(vtbl[2])
        set_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_PROPERTYKEY),
                                       ctypes.POINTER(_PROPVARIANT))(vtbl[6])
        commit = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)(vtbl[7])
        ok = True
        fmtid = _GUID.parse(_APP_USER_MODEL)
        # the id first; the command and its display name always together (the shell's rule)
        for pid, value in ((_ID, app_id), (_RELAUNCH_COMMAND, command), (_RELAUNCH_NAME, name),
                           (_RELAUNCH_ICON, icon)):
            key = _PROPERTYKEY(fmtid, pid)
            var = _PROPVARIANT(VT_LPWSTR, 0, 0, 0, value, None)
            ok &= set_value(store, ctypes.byref(key), ctypes.byref(var)) == 0
        ok &= commit(store) == 0
        release(store)
        return ok
    except (AttributeError, OSError, ValueError):
        return False


def window_handle() -> int | None:
    """The pygame window's HWND, or None (no window, or not Windows)."""
    try:
        return pg.display.get_wm_info().get("window") or None
    except pg.error:
        return None


def add_taskbar_styles(hwnd: int | None) -> None:
    """WS_MINIMIZEBOX | WS_SYSMENU on the borderless window: clicking its taskbar
    button minimizes and restores it, and Alt+Space opens the system menu. No
    WS_MAXIMIZEBOX: the layout has a fixed size."""
    if not hwnd:
        return
    style = _get_long(hwnd, GWL_STYLE)
    want = (style | WS_MINIMIZEBOX | WS_SYSMENU) & ~WS_MAXIMIZEBOX
    if want != style:
        _set_long(hwnd, GWL_STYLE, want)
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)


def style(hwnd: int | None) -> int:
    return _get_long(hwnd, GWL_STYLE) if hwnd else 0


def set_topmost(hwnd: int | None, on: bool) -> None:
    if hwnd:
        user32.SetWindowPos(hwnd, HWND_TOPMOST if on else HWND_NOTOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)


def is_topmost(hwnd: int | None) -> bool:
    return bool(hwnd) and bool(_get_long(hwnd, GWL_EXSTYLE) & WS_EX_TOPMOST)


def is_minimized(hwnd: int | None) -> bool:
    return bool(hwnd) and bool(user32.IsIconic(hwnd))


def start_drag(hwnd: int | None) -> None:
    """Hand a title-bar press to Windows: it moves the window (Aero snap
    included) until the button is released. Returns when the drag ends."""
    if hwnd:
        user32.ReleaseCapture()
        user32.SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0)


def window_rect(hwnd: int | None) -> tuple[int, int, int, int] | None:
    """(left, top, right, bottom) in screen pixels."""
    if not hwnd:
        return None
    r = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(r)):
        return None
    return r.left, r.top, r.right, r.bottom


def work_area(hwnd: int | None) -> tuple[int, int, int, int] | None:
    """The work area (screen minus taskbar) of the monitor the window is on."""
    if not hwnd:
        return None
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    monitor = user32.MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST)
    if not monitor or not user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
        return None
    w = info.rcWork
    return w.left, w.top, w.right, w.bottom


def move_window(hwnd: int | None, x: int, y: int) -> None:
    if hwnd:
        user32.SetWindowPos(hwnd, None, int(x), int(y), 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE)


def keep_top_right(old: tuple[int, int, int, int], size: tuple[int, int],
                   work: tuple[int, int, int, int] | None) -> tuple[int, int]:
    """Top-left for a window of `size` whose top-right corner stays where `old`'s
    was, moved inside the work area when it would stick out."""
    w, h = size
    x, y = old[2] - w, old[1]
    return clamp_into(x, y, size, work)


def clamp_into(x: int, y: int, size: tuple[int, int], work: tuple[int, int, int, int] | None) -> tuple[int, int]:
    if work is None:
        return x, y
    left, top, right, bottom = work
    w, h = size
    x = max(left, min(x, right - w))
    y = max(top, min(y, bottom - h))
    return x, y
