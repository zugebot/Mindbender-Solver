# Jerrin Shirks

# native imports
import pygame as p
from tools.button import Button

# custom imports

class Menu:
    def __init__(self, game, display_under=False):
        self.game = game
        self.textures = self.game.textures
        self.display_under = display_under
        self.sprites = []

        self.buttonSound = {
            "hover": self.game.sounds.Hilite,
            "clickdown": self.game.sounds.GameSelect
        }
        # hiliteLow for hover
        self.buttonSoundLow = {
            "hover": self.game.sounds.HiliteLow,
            "clickdown": self.game.sounds.GameSelect
        }
        self.buttonSoundLowOnlyClick = {
            "hover": None,
            "clickdown": self.game.sounds.GameSelect
        }


    def onKeyDown(self, event_key):
        pass


    def run(self, mousePos, mouseState):
        for sprite in self.sprites:
            sprite.testMouseStates(mousePos.tuple(), mouseState)



    def update(self, screen, default=False):

        # display menu behind this menu
        if self.display_under:
            screen.blit(self.game.lastMenuImage, (0, 0))

        for sprite in self.sprites:
            if isinstance(sprite, Button):
                sprite.render(screen, default=default)
            else:
                sprite.render(screen)



