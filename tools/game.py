# Jerrin Shirks

# native imports
import ctypes

# custom imports
import pygame as p
from engine.constants import *
from tools.point import Point
from tools.menu import Menu
from engine.settings import Settings


class Game:
    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    def __init__(self, settings: Settings):

        # settings
        self.settings = settings
        self.internalSize = Point(self.settings.internalWindowSize)
        self.outputSize = Point(self.settings.outputWindowSize)  # actual game area only, e.g. 800x600
        self.screenSize = Point(self.settings.internalWindowSize)
        self.fps = self.settings.fps
        self.debug = self.settings.debug

        # measured outer metrics from original game
        self.chromeEnabled = True
        self.borderLeft = 3
        self.borderRight = 3
        self.borderTop = 26
        self.borderBottom = 3

        self.captionTextLeftPadding = 10
        self.captionTextTopPadding = 5

        self.closeButtonWidth = 45
        self.closeButtonHeight = self.borderTop
        self.minButtonWidth = 45
        self.minButtonHeight = self.borderTop

        # pygame stuff
        self.instance = p
        self.instance.init()
        self.instance.display.set_caption(self.settings.windowName)

        self.clock = p.time.Clock()

        # borderless outer window sized to old metrics
        self.windowSize = Point(
            self.outputSize.x + self.borderLeft + self.borderRight,
            self.outputSize.y + self.borderTop + self.borderBottom
        )

        window_flags = p.NOFRAME if self.chromeEnabled else 0
        self.window = self.instance.display.set_mode(self.windowSize.tuple(), window_flags)

        # internal fixed-resolution render target
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

        self.textures = None
        self.sounds = None

        # running
        self.running = True
        self.mouseState = 0
        self.mousePos = Point()
        self._mouseButtonDown = False
        self._rightMouseButtonDown = False
        self.hasWindowFocus = True

        # menus
        self.menuIndex = -1
        self.menus = []
        self.lastMenuSize = 0
        self.lastMenuImage = None

        # window dragging
        self._draggingWindow = False
        self._dragOffsetX = 0
        self._dragOffsetY = 0

        # button hover state
        self._hoveringClose = False
        self._hoveringMinimize = False

        # title font
        self.captionFont = p.font.SysFont("Segoe UI", 9, bold=False)

    def currentMenu(self, offset=0) -> Menu:
        try:
            return self.menus[self.menuIndex + offset]
        except Exception as e:
            if self.debug:
                print(e)

    def addMenu(self, menu, _set: bool = False) -> None:
        self.menus.append(menu(self))
        self.menuIndex += 1
        if self.debug:
            print("adding a menu!")
            print("Menu Stack Length:", len(self.menus), "\n")

    def popMenu(self) -> None:
        if len(self.menus) == 1:
            print("Tried to pop menu, no parent menu")
            return

        self.menus.pop()
        self.menuIndex -= 1

        self.lastMenuImage = None
        if self.debug:
            print("Popping Menu!")
            print("New Menu Length:", len(self.menus), "\n")

    def setMenuIndex(self, index) -> None:
        self.menuIndex = index

    def _rebuildWindow(self):
        self.windowSize = Point(
            self.outputSize.x + self.borderLeft + self.borderRight,
            self.outputSize.y + self.borderTop + self.borderBottom
        )
        window_flags = p.NOFRAME if self.chromeEnabled else 0
        self.window = self.instance.display.set_mode(self.windowSize.tuple(), window_flags)

    def setOutputSize(self, size) -> None:
        self.outputSize = Point(size)
        self._rebuildWindow()

    def setInternalSize(self, size) -> None:
        self.internalSize = Point(size)
        self.screenSize = Point(size)
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

    def _client_origin(self):
        return Point(self.borderLeft, self.borderTop)

    def _client_rect(self):
        return p.Rect(
            self.borderLeft,
            self.borderTop,
            self.outputSize.x,
            self.outputSize.y
        )

    def _title_bar_rect(self):
        return p.Rect(0, 0, self.windowSize.x, self.borderTop)

    def _mapMouseToInternal(self, raw_mouse):
        x, y = raw_mouse

        if self.chromeEnabled:
            x -= self.borderLeft
            y -= self.borderTop

        if self.outputSize.x == 0 or self.outputSize.y == 0:
            return Point()

        scale_x = self.internalSize.x / self.outputSize.x
        scale_y = self.internalSize.y / self.outputSize.y

        mapped_x = int(x * scale_x)
        mapped_y = int(y * scale_y)

        return Point(mapped_x, mapped_y)

    def _get_window_client_mouse_pos(self):
        """
        Returns mouse position relative to the borderless pygame window,
        even if the cursor is outside the window. Works on Windows.
        """
        try:
            hwnd = self.instance.display.get_wm_info().get("window")
            if not hwnd:
                return self.instance.mouse.get_pos()

            pt = self._POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            ctypes.windll.user32.ScreenToClient(hwnd, ctypes.byref(pt))
            return pt.x, pt.y
        except Exception:
            return self.instance.mouse.get_pos()

    def _get_window_rect(self):
        try:
            hwnd = self.instance.display.get_wm_info().get("window")
            if not hwnd:
                return None
            rect = self._RECT()
            ok = ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect))
            if not ok:
                return None
            return rect
        except Exception:
            return None

    def _set_window_pos(self, x, y):
        try:
            hwnd = self.instance.display.get_wm_info().get("window")
            if not hwnd:
                return
            SWP_NOSIZE = 0x0001
            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            ctypes.windll.user32.SetWindowPos(
                hwnd, None, int(x), int(y), 0, 0,
                SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
            )
        except Exception as e:
            if self.debug:
                print("SetWindowPos failed:", e)

    def _minimize_window(self):
        try:
            hwnd = self.instance.display.get_wm_info().get("window")
            if not hwnd:
                return
            SW_MINIMIZE = 6
            ctypes.windll.user32.ShowWindow(hwnd, SW_MINIMIZE)
        except Exception as e:
            if self.debug:
                print("minimize failed:", e)

    def _close_button_rect(self):
        return p.Rect(
            self.windowSize.x - self.closeButtonWidth,
            0,
            self.closeButtonWidth,
            self.closeButtonHeight
        )

    def _minimize_button_rect(self):
        return p.Rect(
            self.windowSize.x - self.closeButtonWidth - self.minButtonWidth,
            0,
            self.minButtonWidth,
            self.minButtonHeight
        )

    def _point_in_title_drag_zone(self, x, y):
        if not self.chromeEnabled:
            return False

        if not self._title_bar_rect().collidepoint(x, y):
            return False

        if self._close_button_rect().collidepoint(x, y):
            return False

        if self._minimize_button_rect().collidepoint(x, y):
            return False

        return True

    def _start_window_drag(self):
        try:
            rect = self._get_window_rect()
            if rect is None:
                return

            pt = self._POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))

            self._draggingWindow = True
            self._dragOffsetX = pt.x - rect.left
            self._dragOffsetY = pt.y - rect.top
        except Exception as e:
            if self.debug:
                print("drag start failed:", e)

    def _update_window_drag(self):
        if not self._draggingWindow:
            return

        try:
            pt = self._POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            self._set_window_pos(pt.x - self._dragOffsetX, pt.y - self._dragOffsetY)
        except Exception as e:
            if self.debug:
                print("drag update failed:", e)

    def _stop_window_drag(self):
        self._draggingWindow = False

    def _draw_minimize_glyph(self, rect):
        y = rect.centery + 4
        x1 = rect.centerx - 6
        x2 = rect.centerx + 6
        p.draw.line(self.window, (0, 0, 0), (x1, y), (x2, y), 1)

    def _draw_close_glyph(self, rect):
        x1 = rect.centerx - 5
        y1 = rect.centery - 5
        x2 = rect.centerx + 5
        y2 = rect.centery + 5
        p.draw.line(self.window, (0, 0, 0), (x1, y1), (x2, y2), 1)
        p.draw.line(self.window, (0, 0, 0), (x2, y1), (x1, y2), 1)

    def _draw_fake_windows10_titlebar(self):
        if not self.chromeEnabled:
            return

        raw_mouse = self._get_window_client_mouse_pos()

        title_rect = self._title_bar_rect()
        min_rect = self._minimize_button_rect()
        close_rect = self._close_button_rect()

        self._hoveringMinimize = min_rect.collidepoint(*raw_mouse)
        self._hoveringClose = close_rect.collidepoint(*raw_mouse)

        if self.hasWindowFocus:
            title_color = (255, 255, 255)
            separator_color = (208, 208, 208)
            text_color = (0, 0, 0)
        else:
            title_color = (245, 245, 245)
            separator_color = (220, 220, 220)
            text_color = (80, 80, 80)

        p.draw.rect(self.window, title_color, title_rect)

        p.draw.line(
            self.window,
            separator_color,
            (0, self.borderTop - 1),
            (self.windowSize.x - 1, self.borderTop - 1),
            1
        )

        text_surface = self.captionFont.render(self.settings.windowName, True, text_color)
        self.window.blit(text_surface, (self.captionTextLeftPadding, self.captionTextTopPadding))

        if self._hoveringMinimize:
            p.draw.rect(self.window, (229, 229, 229), min_rect)
        self._draw_minimize_glyph(min_rect)

        if self._hoveringClose:
            p.draw.rect(self.window, (232, 17, 35), close_rect)
            x1 = close_rect.centerx - 5
            y1 = close_rect.centery - 5
            x2 = close_rect.centerx + 5
            y2 = close_rect.centery + 5
            p.draw.line(self.window, (255, 255, 255), (x1, y1), (x2, y2), 1)
            p.draw.line(self.window, (255, 255, 255), (x2, y1), (x1, y2), 1)
        else:
            self._draw_close_glyph(close_rect)

    def _blit_edge_gutters(self, scaled):
        """
        Fill the 3px side/bottom gutters by extending edge pixels from the
        scaled game image so they do not look like visible bars.
        """
        x0 = self.borderLeft
        y0 = self.borderTop
        w = self.outputSize.x
        h = self.outputSize.y

        if self.borderLeft > 0:
            left_strip = scaled.subsurface((0, 0, 1, h))
            left_fill = p.transform.scale(left_strip, (self.borderLeft, h))
            self.window.blit(left_fill, (0, y0))

        if self.borderRight > 0:
            right_strip = scaled.subsurface((w - 1, 0, 1, h))
            right_fill = p.transform.scale(right_strip, (self.borderRight, h))
            self.window.blit(right_fill, (x0 + w, y0))

        if self.borderBottom > 0:
            bottom_strip = scaled.subsurface((0, h - 1, w, 1))
            bottom_fill = p.transform.scale(bottom_strip, (w, self.borderBottom))
            self.window.blit(bottom_fill, (x0, y0 + h))

        if self.borderLeft > 0 and self.borderBottom > 0:
            px = scaled.get_at((0, h - 1))
            corner = p.Surface((self.borderLeft, self.borderBottom))
            corner.fill(px)
            self.window.blit(corner, (0, y0 + h))

        if self.borderRight > 0 and self.borderBottom > 0:
            px = scaled.get_at((w - 1, h - 1))
            corner = p.Surface((self.borderRight, self.borderBottom))
            corner.fill(px)
            self.window.blit(corner, (x0 + w, y0 + h))

    def _presentScreen(self):
        self.window.fill((255, 255, 255))

        if self.settings.smoothScale:
            scaled = p.transform.smoothscale(self.screen, self.outputSize.tuple())
        else:
            scaled = p.transform.scale(self.screen, self.outputSize.tuple())

        if self.chromeEnabled:
            client_origin = self._client_origin()
            self.window.blit(scaled, client_origin.tuple())
            self._blit_edge_gutters(scaled)
            self._draw_fake_windows10_titlebar()
        else:
            self.window.blit(scaled, (0, 0))

    def quit(self):
        self.running = False
        p.quit()
        exit()

    def eventLoop(self):
        raw_mouse = self._get_window_client_mouse_pos()

        for event in self.instance.event.get():
            if event.type == p.QUIT:
                self.quit()

            elif event.type == p.KEYDOWN:
                if not self.settings.needsWindowFocus or self.hasWindowFocus:
                    self.currentMenu().onKeyDown(event)

            elif event.type == p.MOUSEBUTTONDOWN:
                if event.button == 1 and self.chromeEnabled:
                    if self._close_button_rect().collidepoint(raw_mouse[0], raw_mouse[1]):
                        self.quit()
                    elif self._minimize_button_rect().collidepoint(raw_mouse[0], raw_mouse[1]):
                        self._minimize_window()
                    elif self._point_in_title_drag_zone(raw_mouse[0], raw_mouse[1]):
                        self._start_window_drag()

            elif event.type == p.MOUSEBUTTONUP:
                if event.button == 1:
                    self._stop_window_drag()

            elif event.type == p.WINDOWFOCUSGAINED:
                self.hasWindowFocus = True

            elif event.type == p.WINDOWFOCUSLOST:
                self.hasWindowFocus = False
                if self.settings.needsWindowFocus:
                    self._mouseButtonDown = False
                    self._rightMouseButtonDown = False
                    self.mouseState = MOUSE_HOVER

            elif event.type == p.ACTIVEEVENT:
                if hasattr(event, "state") and hasattr(event, "gain"):
                    if event.state & 2:
                        self.hasWindowFocus = bool(event.gain)
                        if self.settings.needsWindowFocus and not self.hasWindowFocus:
                            self._mouseButtonDown = False
                            self._rightMouseButtonDown = False
                            self.mouseState = MOUSE_HOVER

        self._update_window_drag()

        try:
            focused = bool(self.instance.key.get_focused())
        except Exception:
            focused = True

        if self.settings.needsWindowFocus:
            self.hasWindowFocus = self.hasWindowFocus and focused if not focused else focused

            if not self.hasWindowFocus:
                self._mouseButtonDown = False
                self._rightMouseButtonDown = False
                self.mouseState = MOUSE_HOVER
                return
        else:
            self.hasWindowFocus = True

        client_rect = self._client_rect()
        over_non_client = not client_rect.collidepoint(raw_mouse[0], raw_mouse[1])

        try:
            left_button_down = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
            right_button_down = (ctypes.windll.user32.GetAsyncKeyState(0x02) & 0x8000) != 0
        except Exception:
            pressed = p.mouse.get_pressed(3)
            left_button_down = pressed[0]
            right_button_down = pressed[2]

        mapped_mouse = self._mapMouseToInternal(raw_mouse)

        if self._draggingWindow:
            self.mouseState = MOUSE_HOVER
            self._mouseButtonDown = False
            self._rightMouseButtonDown = right_button_down
        else:
            # CRITICAL: never reintroduce edge-cutoff behavior for gameplay dragging.
            # Once a left-click drag begins inside the client area, it MUST continue to
            # report HELD_DOWN and CLICK_UP even if the cursor leaves the window, goes
            # off-screen, or moves to another monitor. Removing this breaks board drag
            # behavior and makes legal long drags impossible.
            if left_button_down and not self._mouseButtonDown:
                if over_non_client:
                    self.mouseState = MOUSE_HOVER
                    self._mouseButtonDown = False
                else:
                    self.mouseState = MOUSE_CLICK_DOWN
                    self._mouseButtonDown = True

            elif left_button_down and self._mouseButtonDown:
                self.mouseState = MOUSE_HELD_DOWN

            elif (not left_button_down) and self._mouseButtonDown:
                self.mouseState = MOUSE_CLICK_UP
                self._mouseButtonDown = False

            else:
                self.mouseState = MOUSE_HOVER
                self._mouseButtonDown = False

            right_click_down = right_button_down and not self._rightMouseButtonDown
            self._rightMouseButtonDown = right_button_down

            if right_click_down:
                current_menu = self.currentMenu()
                if current_menu is not None and hasattr(current_menu, "onRightClickDown"):
                    current_menu.onRightClickDown(mapped_mouse)

        self.mousePos = mapped_mouse

    def run(self, running=True):
        while running:
            if len(self.menus) != self.lastMenuSize:
                self.lastMenuImage = self.screen.copy()
                self.lastMenuSize = len(self.menus)

            self.screen.fill(EMPTY_SCREEN)
            self.eventLoop()

            if self.settings.needsWindowFocus:
                if self.hasWindowFocus:
                    self.currentMenu().run(self.mousePos, self.mouseState)
                else:
                    self.currentMenu().run(self.mousePos, MOUSE_HOVER)
            else:
                self.currentMenu().run(self.mousePos, self.mouseState)

            self._presentScreen()

            self.instance.display.flip()
            self.clock.tick(self.fps)

        p.quit()