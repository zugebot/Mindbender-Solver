# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from data.config import *
from tools.game import Game
from tools.point import Point

# engines
from engine.textures import Textures
from engine.sound import Sounds
from engine.settings import Settings

# menus
from menus.mainmenu import MainMenu



class ChuzzleDeluxe(Game):
    def __init__(self, size, fps, name):
        super().__init__(size, fps, name)
        self.textures = Textures(self)
        self.sounds = Sounds(self)
        self.settings = Settings()

        self.updateLastImage = False

        self.cursorIndex = 0
        self._lastCursorIndex = -1
        self._useHardwareCursor = False
        self._hardwareCursors = []

        self._initCursorMode()
        self.addMenu(MainMenu)

    def _initCursorMode(self):
        if not CUSTOM_CURSOR:
            self.instance.mouse.set_visible(True)
            return

        # Prefer hardware cursors so the OS updates the pointer independently of frame time.
        try:
            hotspot = (32, 32)
            self._hardwareCursors = [p.cursors.Cursor(hotspot, surf) for surf in self.textures.customCursors]
            self._useHardwareCursor = len(self._hardwareCursors) > 0
        except Exception:
            self._useHardwareCursor = False
            self._hardwareCursors = []

        self.instance.mouse.set_visible(self._useHardwareCursor)



    def run(self, running=True):

        while running:
            self.eventLoop()

            self.screen.fill(EMPTY_SCREEN)
            self.currentMenu().run(self.mousePos, self.mouseState)

            # used to get snapshot of previous menu
            if len(self.menus) != self.lastMenuSize:
                self.currentMenu().update(self.screen, default=True)
                self.lastMenuImage = p.display.get_surface().copy()
                self.lastMenuSize = len(self.menus)

            # otherwise, do the normal blit operations
            else:
                self.currentMenu().update(self.screen)

            if self._useHardwareCursor and self._hardwareCursors:
                index = self.cursorIndex % len(self._hardwareCursors)
                if index != self._lastCursorIndex:
                    self.instance.mouse.set_cursor(self._hardwareCursors[index])
                    self._lastCursorIndex = index
            elif CUSTOM_CURSOR:
                mouse = Point(p.mouse.get_pos()) - (32, 32)
                self.screen.blit(self.textures.customCursors[self.cursorIndex], mouse.tuple())

            # display to screen
            self.instance.display.flip()

            self.clock.tick(self.fps)
        p.quit()
