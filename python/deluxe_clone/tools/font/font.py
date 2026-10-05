# Jerrin Shirks

# native imports
import pygame



class Font:
    def __init__(self):
        self.filename = ""
        self.CharList = ()
        self.WidthList = ()
        self.RectList = ()
        self.OffsetList = ()
        self.letters: dict[str, pygame.Surface] = {}

    def prepare(self):
        for index, char in enumerate(self.CharList):
            pass


class Text:
    def __init__(self):
        pass

    def createText(self, string: str, font=None) -> pygame.Surface:
        pass
