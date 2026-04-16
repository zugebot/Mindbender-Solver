# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.menu import Menu
from tools.point import Point
from tools.image import Image
from tools.button import Button


class AwayScreen(Menu):
    def __init__(self, game):
        super().__init__(game, display_under=True)
        self.populate()

    def populate(self):
        self.textures = self.game.textures

        bkg = p.Surface(self.game.screenSize.tuple())
        bkg.set_alpha(128)
        bkg.fill((0, 0, 50))
        self.sprites.append(
            Button(game=self.game, clickable=True,
                   data={
                       "hitbox": "all",
                       "textures": {"default": bkg},
                       "func": {
                           "clickup": self.game.popMenu
                       }
                   }))
