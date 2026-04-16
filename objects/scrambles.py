# Jerrin Shirks

# custom imports
from tools.sprite_object import spriteObject
from tools.point import Point



class Scrambles(spriteObject):
    def __init__(self, menu):
        super().__init__(menu)
        self.scrambleCount = 3

    def render(self, screen):
        screen.blit(self.textures.Scrambles, (500, 1))
