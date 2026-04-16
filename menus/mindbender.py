# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.menu import Menu
from tools.button import Button
from tools.image import Image
from objects.board import Board
from objects.matchthis import MatchBoard
from engine.chuzzle_list import ChuzzleList
from puzzles.puzzles import Puzzles


class MindBender(Menu):
    def __init__(self, game):
        super().__init__(game)

        self.puzzles = Puzzles()
        self.level = 0
        self.puzzleIndex = 0
        self.world_number = 1
        self.level_number = 1

        self.board = Board(self, gamemode="mindbender")
        self.solution_board = ChuzzleList(game=self.game)
        self.solution_board.fill()
        self.matchBoard = MatchBoard(menu=self, board=self.solution_board)

        self.level_font = p.font.SysFont("arial", 26, bold=True)

        self.populate()
        self.updatePuzzle()

    def saveGame(self):
        self.game.popMenu()

    def setPuzzle(self, value):
        self.puzzleIndex = max(0, min(4, value))
        self.updatePuzzle()

    def setLevel(self, value):
        self.level = max(0, min(19, value))
        self.updatePuzzle()

    def incrementLevel(self):
        self.setLevel(self.level + 1)

    def decrementLevel(self):
        self.setLevel(self.level - 1)

    def updatePuzzle(self):
        if not self.puzzles.puzzles:
            return

        index = self.level * 5 + self.puzzleIndex
        index = max(0, min(len(self.puzzles.puzzles) - 1, index))

        self.level = index // 5
        self.puzzleIndex = index % 5
        self.world_number = self.level + 1
        self.level_number = self.puzzleIndex + 1

        self.puzzle = self.puzzles.puzzles[index]

        self.board.chuzzles.load_from_array(self.puzzle.initial_cells)
        if self.puzzle.start_fat is not None:
            self.board.chuzzles.placeFatChuzzle(*self.puzzle.start_fat)

        self.solution_board.load_from_array(self.puzzle.solved_cells)
        if self.puzzle.end_fat is not None:
            self.solution_board.placeFatChuzzle(*self.puzzle.end_fat)

    def onKeyDown(self, event):
        e = event.key

        if e == p.K_e:
            x = int(input("x: "))
            y = int(input("y: "))
            self.board.chuzzles.getPos((x, y)).locked = True

        if e == p.K_d:
            for chuzzle in self.board.chuzzles:
                chuzzle.locked = True
        if e == p.K_c:
            for chuzzle in self.board.chuzzles:
                chuzzle.locked = False

        if e == p.K_r:
            x = int(input("x: "))
            y = int(input("y: "))
            self.board.chuzzles.placeFatChuzzle(x, y)
        if e == p.K_f:
            self.board.chuzzles.placeFatChuzzle()
        if e == p.K_v:
            self.board.chuzzles.deleteFatChuzzle()
        if e == p.K_q:
            self.board.chuzzles.loadFile("puzzle.json")

        if e == p.K_1:
            self.board.printStates(0)
        if e == p.K_2:
            self.board.printStates(1)
        if e == p.K_3:
            self.board.printStates(2)

    def populate(self):
        bkg = self.game.instance.image.load("textures/Backdrops/bkg1.jpg").convert_alpha()
        self.sprites.append(Image(image=bkg))

        self.sprites.append(self.board)

        self.sprites.append(Button(game=self.game, coords=(0, 553),
                                   textures=self.textures.MainMenuButton, data={
            "hitbox": self.textures.MainMenuButton_Alpha[0],
            "func": {"clickup": self.saveGame},
            "sounds": self.buttonSoundLow
        }))

        self.sprites.append(Image((0, 125), self.textures.MindBenderGadget))

        self.sprites.append(Button(game=self.game, coords=(20, 325),
                                   textures=self.textures.MindBenderMini, data={
            "hitbox": self.textures.MindBenderMini_Alpha[0],
            "sounds": self.buttonSoundLowOnlyClick,
            "func": {"clickup": (self.setPuzzle, 0)}
        }))
        self.sprites.append(Button(game=self.game, coords=(61, 353),
                                   textures=self.textures.MindBenderMini, data={
            "hitbox": self.textures.MindBenderMini_Alpha[0],
            "sounds": self.buttonSoundLowOnlyClick,
            "func": {"clickup": (self.setPuzzle, 1)}
        }))
        self.sprites.append(Button(game=self.game, coords=(107, 363),
                                   textures=self.textures.MindBenderMini, data={
            "hitbox": self.textures.MindBenderMini_Alpha[0],
            "sounds": self.buttonSoundLowOnlyClick,
            "func": {"clickup": (self.setPuzzle, 2)}
        }))
        self.sprites.append(Button(game=self.game, coords=(153, 353),
                                   textures=self.textures.MindBenderMini, data={
            "hitbox": self.textures.MindBenderMini_Alpha[0],
            "sounds": self.buttonSoundLowOnlyClick,
            "func": {"clickup": (self.setPuzzle, 3)}
        }))
        self.sprites.append(Button(game=self.game, coords=(195, 330),
                                   textures=self.textures.MindBenderMini, data={
            "hitbox": self.textures.MindBenderMini_Alpha[0],
            "sounds": self.buttonSoundLowOnlyClick,
            "func": {"clickup": (self.setPuzzle, 4)}
        }))

        self.sprites.append(Image((630, 536), self.textures.LevelStar))
        self.sprites.append(Image(image=self.textures.MindBenderMarquee))
        self.sprites.append(self.matchBoard)

        self.sprites.append(Button(game=self.game, coords=(23, 455),
                                   textures=self.textures.MindBenderSmallStarLeft, data={
            "hitbox": self.textures.MindBenderSmallStarLeft_Alpha[0],
            "sounds": self.buttonSoundLow,
            "func": {"clickup": self.decrementLevel}
        }))

        self.sprites.append(Image((78, 410), self.textures.MindBenderBigStar))

        self.sprites.append(Button(game=self.game, coords=(170, 455),
                                   textures=self.textures.MindBenderSmallStarRight, data={
            "hitbox": self.textures.MindBenderSmallStarRight_Alpha[0],
            "sounds": self.buttonSoundLow,
            "func": {"clickup": self.incrementLevel}
        }))

    def update(self, screen, default=False):
        super().update(screen, default=default)

        level_text = f"{self.world_number}-{self.level_number}"
        text_surface = self.level_font.render(level_text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(bottomright=(self.game.screenSize.x - 18, self.game.screenSize.y - 16))
        screen.blit(text_surface, text_rect)
