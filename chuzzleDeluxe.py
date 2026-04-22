# Jerrin Shirks
# chuzzleDeluxe.py

# native imports
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

        self._initCursorMode()
        self.addMenu(MindBender)

    def _initCursorMode(self):
        if not self.settings.customCursor:
            self.instance.mouse.set_visible(True)
            return

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

            if len(self.menus) != self.lastMenuSize:
                self.currentMenu().update(self.screen, default=True)
                self.lastMenuImage = self.screen.copy()
                self.lastMenuSize = len(self.menus)
            else:
                self.currentMenu().update(self.screen)

            fps_text = f"FPS: {self.clock.get_fps():.1f}"
            fps_shadow = self.fpsFont.render(fps_text, True, self.fpsShadowColor)
            fps_surface = self.fpsFont.render(fps_text, True, self.fpsColor)
            self.screen.blit(fps_shadow, (self.fpsPos[0] + 1, self.fpsPos[1] + 1))
            self.screen.blit(fps_surface, self.fpsPos)

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
            self.clock.tick(self.fps)

        p.quit()