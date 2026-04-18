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

    def __init__(self, settings: Settings):

        # settings
        self.settings = settings
        self.internalSize = Point(self.settings.internalWindowSize)
        self.outputSize = Point(self.settings.outputWindowSize)
        self.screenSize = Point(self.settings.internalWindowSize)
        self.fps = self.settings.fps
        self.debug = self.settings.debug

        # pygame stuff
        self.instance = p
        self.instance.init()
        self.instance.display.set_caption(self.settings.windowName)

        self.clock = p.time.Clock()

        # Actual OS window.
        self.window = self.instance.display.set_mode(self.outputSize.tuple())

        # Internal fixed-resolution render target.
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

        self.textures = None
        self.sounds = None

        # running
        self.running = True
        self.mouseState = 0
        self.mousePos = Point()
        self._mouseButtonDown = False

        # Menus
        self.menuIndex = -1
        self.menus = []
        self.lastMenuSize = 0
        self.lastMenuImage = None

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

    def setOutputSize(self, size) -> None:
        self.outputSize = Point(size)
        self.window = self.instance.display.set_mode(self.outputSize.tuple())

    def setInternalSize(self, size) -> None:
        self.internalSize = Point(size)
        self.screenSize = Point(size)
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

    def _mapMouseToInternal(self, raw_mouse):
        if self.outputSize.x == 0 or self.outputSize.y == 0:
            return Point()

        scale_x = self.internalSize.x / self.outputSize.x
        scale_y = self.internalSize.y / self.outputSize.y

        mapped_x = int(raw_mouse[0] * scale_x)
        mapped_y = int(raw_mouse[1] * scale_y)

        return Point(mapped_x, mapped_y)

    def _get_window_client_mouse_pos(self):
        """
        Returns mouse position relative to the pygame client area, even if the
        cursor is outside the window. Works on Windows.
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

    def _presentScreen(self):
        if self.settings.smoothScale:
            scaled = p.transform.smoothscale(self.screen, self.outputSize.tuple())
        else:
            scaled = p.transform.scale(self.screen, self.outputSize.tuple())
        self.window.blit(scaled, (0, 0))

    def quit(self):
        self.running = False
        p.quit()
        exit()

    def eventLoop(self):
        for event in self.instance.event.get():
            if event.type == p.QUIT:
                self.quit()

            if event.type == p.KEYDOWN:
                self.currentMenu().onKeyDown(event)

        # Poll button state each frame so drag state stays consistent even if
        # an OS-level mouse-up/down event is missed while moving off-window.
        try:
            # 0x01 = left mouse button
            button_down = (ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000) != 0
        except Exception:
            button_down = p.mouse.get_pressed(3)[0]

        if button_down and not self._mouseButtonDown:
            self.mouseState = MOUSE_CLICK_DOWN
        elif button_down and self._mouseButtonDown:
            self.mouseState = MOUSE_HELD_DOWN
        elif (not button_down) and self._mouseButtonDown:
            self.mouseState = MOUSE_CLICK_UP
        else:
            self.mouseState = MOUSE_HOVER

        self._mouseButtonDown = button_down

        # Important:
        # Do NOT grab the mouse. We want drag to keep working even after the
        # cursor leaves the window, while still allowing the cursor to leave.
        raw_mouse = self._get_window_client_mouse_pos()
        self.mousePos = self._mapMouseToInternal(raw_mouse)

    def run(self, running=True):
        while running:
            if len(self.menus) != self.lastMenuSize:
                self.lastMenuImage = self.screen.copy()
                self.lastMenuSize = len(self.menus)

            self.screen.fill(EMPTY_SCREEN)
            self.eventLoop()

            # Run the menu
            self.currentMenu().run(self.mousePos, self.mouseState)

            self._presentScreen()

            # display to screen
            self.instance.display.flip()
            self.clock.tick(self.fps)

        p.quit()