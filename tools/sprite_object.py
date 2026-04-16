# Jerrin Shirks

# native imports
import pygame as p
import random as r

# custom imports
from data.config import *
from tools.button import Button
from tools.point import Point



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
