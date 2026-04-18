# Jerrin Shirks

# native imports

# custom imports
from engine.constants import *


class spriteObject:
    def __init__(self, menu):
        self.menu = menu
        self.game = self.menu.game
        self.textures = self.menu.game.textures

    def onMouseHover(self, mousePos):
        pass

    def onMouseDown(self, mousePos):
        pass

    def onMouseHeldDown(self, mousePos):
        pass

    def onMouseUp(self, mousePos):
        pass

    def testMouseStates(self, mousePos, mouseState):
        if mouseState == MOUSE_CLICK_DOWN:
            self.onMouseDown(mousePos)
        if mouseState == MOUSE_HOVER:
            self.onMouseHover(mousePos)
        if mouseState == MOUSE_HELD_DOWN:
            self.onMouseHeldDown(mousePos)
        if mouseState == MOUSE_CLICK_UP:
            self.onMouseUp(mousePos)

    def render(self, screen):
        pass
