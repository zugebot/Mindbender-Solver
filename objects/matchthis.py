# Jerrin Shirks

# native imports
import random as r
import pygame as p

# custom imports
from data.config import *
from tools.point import Point
from tools.sprite_object import spriteObject



class MatchBoard(spriteObject):
    def __init__(self, menu, board):
        super().__init__(menu)

        self.board = board
        self.settings = self.game.settings

        # 120 base
        self.size = Point(20, 20) * BOARD_SIZE
        self.surface = self.game.instance.Surface(self.size.tuple()).convert_alpha()
        self.surface.fill((0, 0, 0, 0))
        self.colors = [[r.randint(0, 8) for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


    def render(self, screen):
        surface = self.surface.copy()
        for y in range(self.board.chuzCount.y):
            for x in range(self.board.chuzCount.x):
                point = Point(x, y) * 18
                # self.board[x + y * 6].color
                color = self.board[x * BOARD_SIZE + y].color
                texture = self.game.textures.MiniChuzzles[color]
                surface.blit(texture, point.tuple())

        newSize = self.size / BOARD_SIZE * 6
        surface = p.transform.smoothscale(surface, newSize.tuple())
        screen.blit(surface, (75, 203))

