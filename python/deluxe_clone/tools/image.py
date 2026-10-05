# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.point import Point
from tools.button import Button
from tools.texture_utils import rotate, convertToAlpha



class Image:
    def __init__(self, coords=Point(), image=None, default=None):

        if isinstance(coords, tuple):
            coords = Point(coords)
        self.coords = coords

        self.image = image
        self.size = Point(self.image.get_width(), self.image.get_height())
        self.rect = self.image.get_rect(center=(coords + self.size / 2).tuple())

    def setDefault(self):
        pass

    def testMouseStates(self, mousePos, mouseState):
        pass

    def render(self, screen):
        screen.blit(self.image, self.rect)







