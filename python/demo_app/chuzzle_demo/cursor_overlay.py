# python/demo_app/chuzzle_demo/cursor_overlay.py
"""A mouse-pointer sprite that shows where background playback "is", so
recordings can follow it.

Background input posts mouse messages to the game; the real cursor never
moves, so a recording shows nothing. This draws a pointer in its own small
window: always on top, click-through (WS_EX_TRANSPARENT, so the real mouse
and the game are unaffected), never focused, absent from the taskbar, with
per-pixel alpha (UpdateLayeredWindow). The player sets the position in screen
pixels; a thread that owns the window applies it every few ms. While the
button is held it shows a "pressed" pointer with a ring at the tip.

Screen recorders capture it in display / region capture (it is a normal
top-level window); a capture of the game window alone does not include it."""

from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes

import pygame as pg

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32
kernel32 = ctypes.windll.kernel32

WS_POPUP = 0x80000000
WS_EX_LAYERED, WS_EX_TRANSPARENT, WS_EX_TOPMOST = 0x00080000, 0x00000020, 0x00000008
WS_EX_TOOLWINDOW, WS_EX_NOACTIVATE = 0x00000080, 0x08000000
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA = 0x2, 0x0, 0x1
PM_REMOVE = 0x1
HWND_TOPMOST = wintypes.HWND(-1)
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x1, 0x2, 0x10

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASSEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int), ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON), ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR), ("hIconSm", wintypes.HICON)]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", ctypes.c_long), ("biHeight", ctypes.c_long),
                ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", ctypes.c_long),
                ("biYPelsPerMeter", ctypes.c_long), ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte),
                ("SourceConstantAlpha", ctypes.c_ubyte), ("AlphaFormat", ctypes.c_ubyte)]


user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
user32.DefWindowProcW.restype = LRESULT
user32.RegisterClassExW.argtypes = [ctypes.POINTER(WNDCLASSEXW)]
user32.RegisterClassExW.restype = wintypes.ATOM
user32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                   wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID]
user32.CreateWindowExW.restype = wintypes.HWND
user32.UpdateLayeredWindow.argtypes = [wintypes.HWND, wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                       ctypes.POINTER(wintypes.SIZE), wintypes.HDC, ctypes.POINTER(wintypes.POINT),
                                       wintypes.COLORREF, ctypes.POINTER(BLENDFUNCTION), wintypes.DWORD]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, wintypes.UINT]
user32.PeekMessageW.argtypes = [ctypes.POINTER(wintypes.MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT,
                                wintypes.UINT]
user32.DispatchMessageW.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.TranslateMessage.argtypes = [ctypes.POINTER(wintypes.MSG)]
user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC
user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC
gdi32.CreateDIBSection.argtypes = [wintypes.HDC, ctypes.POINTER(BITMAPINFOHEADER), wintypes.UINT,
                                   ctypes.POINTER(ctypes.c_void_p), wintypes.HANDLE, wintypes.DWORD]
gdi32.CreateDIBSection.restype = wintypes.HBITMAP
gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ
kernel32.GetModuleHandleW.restype = wintypes.HMODULE

SIZE = 56                 # sprite canvas (px)
HOTSPOT = (12, 12)        # the pointer's tip within the canvas (room for the press ring)
# classic arrow outline, tip at (0, 0), in "cursor pixels"; scaled up so it reads on video
ARROW = [(0, 0), (0, 21), (5, 16), (9, 25), (12, 24), (8, 15), (15, 15)]
ARROW_SCALE = 1.45


def render_sprite(pressed: bool) -> bytes:
    """Premultiplied BGRA pixels (top-down) of the pointer sprite."""
    surf = pg.Surface((SIZE, SIZE), pg.SRCALPHA)
    hx, hy = HOTSPOT
    pts = [(hx + x * ARROW_SCALE, hy + y * ARROW_SCALE) for x, y in ARROW]
    if pressed:
        pg.draw.circle(surf, (90, 170, 255, 110), HOTSPOT, 11)
        pg.draw.circle(surf, (40, 110, 220, 230), HOTSPOT, 11, 2)
    shadow = [(x + 2, y + 2) for x, y in pts]
    pg.draw.polygon(surf, (0, 0, 0, 70), shadow)
    pg.draw.polygon(surf, (255, 225, 120, 255) if pressed else (255, 255, 255, 255), pts)
    pg.draw.polygon(surf, (0, 0, 0, 255), pts, 2)
    rgba = pg.image.tostring(surf, "RGBA")
    out = bytearray(len(rgba))
    for i in range(0, len(rgba), 4):
        r, g, b, a = rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]
        out[i] = b * a // 255
        out[i + 1] = g * a // 255
        out[i + 2] = r * a // 255
        out[i + 3] = a
    return bytes(out)


class CursorOverlay:
    """Thread-safe: move(), hide() may be called from any thread."""

    def __init__(self):
        self._images = {False: render_sprite(False), True: render_sprite(True)}
        self._state = (0, 0, False, False)     # x, y (screen px of the tip), pressed, visible
        self._lock = threading.Lock()
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(2.0)

    def move(self, x: float, y: float, pressed: bool) -> None:
        with self._lock:
            self._state = (int(round(x)), int(round(y)), pressed, True)

    def hide(self) -> None:
        with self._lock:
            self._state = self._state[:3] + (False,)

    # ------------------------------------------------------------------

    def _run(self) -> None:
        hinst = kernel32.GetModuleHandleW(None)
        self._proc = WNDPROC(lambda h, m, w, l: user32.DefWindowProcW(h, m, w, l))   # keep a reference
        wc = WNDCLASSEXW(cbSize=ctypes.sizeof(WNDCLASSEXW), lpfnWndProc=self._proc, hInstance=hinst,
                         lpszClassName="ChuzzleDemoCursorSprite")
        user32.RegisterClassExW(ctypes.byref(wc))
        hwnd = user32.CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
                                      | WS_EX_NOACTIVATE, "ChuzzleDemoCursorSprite", "Chuzzle Demo cursor",
                                      WS_POPUP, 0, 0, SIZE, SIZE, None, None, hinst, None)
        screen = user32.GetDC(None)
        dcs = {}
        for pressed, pixels in self._images.items():
            header = BITMAPINFOHEADER(biSize=ctypes.sizeof(BITMAPINFOHEADER), biWidth=SIZE, biHeight=-SIZE,
                                      biPlanes=1, biBitCount=32, biCompression=0)
            bits = ctypes.c_void_p()
            bmp = gdi32.CreateDIBSection(screen, ctypes.byref(header), 0, ctypes.byref(bits), None, 0)
            ctypes.memmove(bits, pixels, len(pixels))
            dc = gdi32.CreateCompatibleDC(screen)
            gdi32.SelectObject(dc, bmp)
            dcs[pressed] = dc
        blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        size, src = wintypes.SIZE(SIZE, SIZE), wintypes.POINT(0, 0)
        self._ready.set()

        shown = None
        visible = False
        last_raise = 0.0
        msg = wintypes.MSG()
        while True:
            while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, PM_REMOVE):
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            with self._lock:
                state = self._state
            x, y, pressed, want = state
            if want and state != shown:
                dst = wintypes.POINT(x - HOTSPOT[0], y - HOTSPOT[1])
                user32.UpdateLayeredWindow(hwnd, screen, ctypes.byref(dst), ctypes.byref(size), dcs[pressed],
                                           ctypes.byref(src), 0, ctypes.byref(blend), ULW_ALPHA)
                if not visible:
                    user32.ShowWindow(hwnd, SW_SHOWNOACTIVATE)
                    visible = True
                now = time.perf_counter()
                if now - last_raise > 0.5:   # stay above the game even after it is brought to the front
                    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
                    last_raise = now
                shown = state
            elif not want and visible:
                user32.ShowWindow(hwnd, SW_HIDE)
                visible = False
                shown = None
            time.sleep(0.003)
