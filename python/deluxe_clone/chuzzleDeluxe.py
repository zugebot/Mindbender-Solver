# Jerrin Shirks
# chuzzleDeluxe.py

# native imports
import ctypes
import sys
import pygame as p

# custom imports
from engine.constants import *
from tools.game import Game

# engines
from engine.textures import Textures
from engine.sound import Sounds
from engine.settings import Settings

# menus
from menus.mindbender import MindBender


class ChuzzleDeluxe(Game):
    TARGET_WINDOW_TITLES = {
        "Chuzzle Christmas Edition 1.0",
        "Chuzzle Deluxe 1.01",
    }

    MOVE_TARGET_WINDOWS_KEY = p.K_F8

    if sys.platform == "win32":
        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        class POINT(ctypes.Structure):
            _fields_ = [
                ("x", ctypes.c_long),
                ("y", ctypes.c_long),
            ]

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.textures = Textures(self)
        self.sounds = Sounds(self)

        self.updateLastImage = False

        self.cursorIndex = 0
        self._lastCursorIndex = -1
        self._useHardwareCursor = False
        self._hardwareCursors = []

        self.fpsFont = p.font.Font(None, 24)
        self.fpsPos = (8, 6)
        self.fpsColor = (255, 255, 255)
        self.fpsShadowColor = (0, 0, 0)

        self._moveTargetsKeyLatched = False

        self._initCursorMode()
        self.addMenu(MindBender)

    def _initCursorMode(self):
        if not self.settings.customCursor:
            self.instance.mouse.set_visible(True)
            return

        try:
            hotspot = (32, 32)
            self._hardwareCursors = [
                p.cursors.Cursor(hotspot, surf)
                for surf in self.textures.customCursors
            ]
            self._useHardwareCursor = len(self._hardwareCursors) > 0
        except Exception:
            self._useHardwareCursor = False
            self._hardwareCursors = []

        self.instance.mouse.set_visible(self._useHardwareCursor)

    def _get_python_window_hwnd(self):
        try:
            return self.instance.display.get_wm_info().get("window")
        except Exception:
            return None

    def _find_windows_by_exact_titles(self, titles):
        if sys.platform != "win32":
            return []

        user32 = ctypes.windll.user32
        found = []

        EnumWindowsProc = ctypes.WINFUNCTYPE(
            ctypes.c_bool,
            ctypes.c_void_p,
            ctypes.c_void_p,
        )

        def enum_proc(hwnd, lparam):
            try:
                if not user32.IsWindowVisible(hwnd):
                    return True

                length = user32.GetWindowTextLengthW(hwnd)
                if length <= 0:
                    return True

                buffer = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buffer, length + 1)

                title = buffer.value
                if title in titles:
                    found.append(hwnd)

            except Exception:
                pass

            return True

        user32.EnumWindows(EnumWindowsProc(enum_proc), None)
        return found

    def _get_window_rect_by_hwnd(self, hwnd):
        if sys.platform != "win32" or not hwnd:
            return None

        rect = self.RECT()

        try:
            ok = ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None
            return rect
        except Exception:
            return None

    def _get_client_origin_screen_by_hwnd(self, hwnd):
        """
        Returns the top-left of the real Windows client area in screen coordinates.

        This is the important part. GetWindowRect() includes the real titlebar,
        border, and frame. ClientToScreen() gives the actual drawable/game area.
        """
        if sys.platform != "win32" or not hwnd:
            return None

        try:
            pt = self.POINT(0, 0)
            ok = ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(pt))
            if not ok:
                return None
            return int(pt.x), int(pt.y)
        except Exception:
            return None

    def _get_client_size_by_hwnd(self, hwnd):
        if sys.platform != "win32" or not hwnd:
            return None

        try:
            rect = self.RECT()
            ok = ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None

            return int(rect.right - rect.left), int(rect.bottom - rect.top)
        except Exception:
            return None

    def _restore_window_if_minimized(self, hwnd):
        if sys.platform != "win32" or not hwnd:
            return

        try:
            user32 = ctypes.windll.user32

            if user32.IsIconic(hwnd):
                sw_restore = 9
                user32.ShowWindow(hwnd, sw_restore)
        except Exception:
            pass

    def _move_hwnd_to(self, hwnd, x, y):
        if sys.platform != "win32" or not hwnd:
            return False

        try:
            user32 = ctypes.windll.user32

            swp_nosize = 0x0001
            swp_nozorder = 0x0004
            swp_noactivate = 0x0010
            swp_noownerzorder = 0x0200

            ok = user32.SetWindowPos(
                hwnd,
                None,
                int(round(x)),
                int(round(y)),
                0,
                0,
                swp_nosize | swp_nozorder | swp_noactivate | swp_noownerzorder,
            )

            return bool(ok)

        except Exception:
            return False

    def _get_python_fake_client_origin_screen(self):
        """
        Your pygame window is borderless, but you draw a fake Windows titlebar
        and fake borders. So the clone's visible game client starts at:

            python_window_outer + (borderLeft, borderTop)

        That is what the real Chuzzle client area must align to.
        """
        python_hwnd = self._get_python_window_hwnd()
        python_rect = self._get_window_rect_by_hwnd(python_hwnd)

        if python_rect is None:
            return None

        return (
            int(python_rect.left + self.borderLeft),
            int(python_rect.top + self.borderTop),
        )

    def _move_target_client_to_python_fake_client(self, target_hwnd):
        """
        Moves the target window so its real client area lines up exactly with
        this clone's fake client area.
        """
        self._restore_window_if_minimized(target_hwnd)

        target_window_rect = self._get_window_rect_by_hwnd(target_hwnd)
        target_client_origin = self._get_client_origin_screen_by_hwnd(target_hwnd)
        python_fake_client_origin = self._get_python_fake_client_origin_screen()

        if target_window_rect is None:
            return False, "missing target window rect"

        if target_client_origin is None:
            return False, "missing target client origin"

        if python_fake_client_origin is None:
            return False, "missing python fake client origin"

        target_client_offset_x = target_client_origin[0] - int(target_window_rect.left)
        target_client_offset_y = target_client_origin[1] - int(target_window_rect.top)

        new_window_x = python_fake_client_origin[0] - target_client_offset_x
        new_window_y = python_fake_client_origin[1] - target_client_offset_y

        ok = self._move_hwnd_to(target_hwnd, new_window_x, new_window_y)

        if not ok:
            return False, "SetWindowPos failed"

        target_client_size = self._get_client_size_by_hwnd(target_hwnd)

        return True, (
            f"target_client_offset=({target_client_offset_x}, {target_client_offset_y}) "
            f"target_client_size={target_client_size} "
            f"new_window_pos=({int(round(new_window_x))}, {int(round(new_window_y))}) "
            f"python_fake_client={python_fake_client_origin}"
        )

    def moveTargetChuzzleWindowsToPythonWindow(self):
        if sys.platform != "win32":
            print("[F8] Window moving is only implemented on Windows.")
            return

        python_hwnd = self._get_python_window_hwnd()
        if not python_hwnd:
            print("[F8] Could not get Python window handle.")
            return

        target_hwnds = self._find_windows_by_exact_titles(self.TARGET_WINDOW_TITLES)

        if not target_hwnds:
            print("[F8] No matching Chuzzle windows found.")
            return

        moved_count = 0

        for hwnd in target_hwnds:
            if hwnd == python_hwnd:
                continue

            ok, detail = self._move_target_client_to_python_fake_client(hwnd)

            if ok:
                moved_count += 1
                print(f"[F8] Moved Chuzzle hwnd={hwnd}. {detail}")
            else:
                print(f"[F8] Failed to move Chuzzle hwnd={hwnd}: {detail}")

        print(f"[F8] Moved {moved_count} matching Chuzzle window(s).")

    def _handleGlobalHotkeys(self):
        """
        Use Win32 async key state on Windows so F8 works even when the original
        Chuzzle window is focused on top of the clone.
        """
        key_down = False

        if sys.platform == "win32":
            try:
                vk_f8 = 0x77
                key_down = (ctypes.windll.user32.GetAsyncKeyState(vk_f8) & 0x8000) != 0
            except Exception:
                key_down = False
        else:
            try:
                keys = self.instance.key.get_pressed()
                key_down = bool(keys[self.MOVE_TARGET_WINDOWS_KEY])
            except Exception:
                key_down = False

        if key_down and not self._moveTargetsKeyLatched:
            self._moveTargetsKeyLatched = True
            self.moveTargetChuzzleWindowsToPythonWindow()

        elif not key_down:
            self._moveTargetsKeyLatched = False

    def run(self, running=True):
        frame_dt_ms = 1000.0 / max(1, self.fps)
        instant_fps = 0.0
        smoothed_fps = 0.0

        while running:
            self.eventLoop()
            self._handleGlobalHotkeys()

            self.screen.fill(EMPTY_SCREEN)

            self._replayCurrentMenuInput()

            if len(self.menus) != self.lastMenuSize:
                self.currentMenu().update(self.screen, default=True)
                self.lastMenuImage = self.screen.copy()
                self.lastMenuSize = len(self.menus)
            else:
                self.currentMenu().update(self.screen)

            if frame_dt_ms > 0.0:
                instant_fps = 1000.0 / frame_dt_ms

            if smoothed_fps <= 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (smoothed_fps * 0.90) + (instant_fps * 0.10)
            
            if smoothed_fps < 50:
                print(smoothed_fps)
            fps_text = f"FPS: {smoothed_fps:5.1f}"
            coords_text = f"({int(self.mousePos.x):3d}, {int(self.mousePos.y):3d})"
            hud_text = f"{fps_text}  {coords_text}"

            hud_shadow = self.fpsFont.render(hud_text, True, self.fpsShadowColor)
            hud_surface = self.fpsFont.render(hud_text, True, self.fpsColor)

            self.screen.blit(hud_shadow, (self.fpsPos[0] + 1, self.fpsPos[1] + 1))
            self.screen.blit(hud_surface, self.fpsPos)

            if self._useHardwareCursor and self._hardwareCursors:
                index = self.cursorIndex % len(self._hardwareCursors)

                if index != self._lastCursorIndex:
                    try:
                        self.instance.mouse.set_cursor(self._hardwareCursors[index])
                        self._lastCursorIndex = index
                    except Exception as e:
                        print(f"Cursor failed with index {index}")
                        print(e)

            elif self.settings.customCursor:
                mouse = self.mousePos - (32, 32)
                self.screen.blit(self.textures.customCursors[self.cursorIndex], mouse.tuple())

            self._presentScreen()

            self.instance.display.flip()

            frame_dt_ms = float(self.clock.tick(self.fps))

        p.quit()