# Jerrin Shirks

# custom imports
from tools.sprite_object import spriteObject
from tools.point import Point




class puzzleButton(Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.number = kwargs["number"]
        self.coords = [(20, 325), (61, 353), (107, 363), (153, 353), (195, 330)][self.number]


