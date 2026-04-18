# Jerrin Shirks

# native imports
import json
import random

# custom imports
from tools.point import Point
from tools.game import Game
from engine.chuzzle import Chuzzle




class ChuzzleList(list):
    def __init__(self, game: Game, count=None, *args):
        super().__init__(args)
        if count is None:
            count = (game.settings.boardSize, game.settings.boardSize)
        
        self.game = game
        self.chuzCount = Point(count)
        self.random = random


    def fill(self):
        self.clear()
        for x in range(self.chuzCount.x):
            for y in range(self.chuzCount.y):
                self.append(Chuzzle(game=self.game, position=Point(x, y)))


    def getPos(self, *args) -> Chuzzle:
        if type(args[0]) == Point:
            point = Point(args[0])
        else:
            point = Point(*args)
        if point.x > self.chuzCount.x:
            point.x %= self.chuzCount.x
        if point.y > self.chuzCount.y:
            point.y %= self.chuzCount.y
        for chuzzle in self:
            if chuzzle.position == point:
                return chuzzle
        return None


    # probably really slow
    def iterator(self):
        for y in range(self.chuzCount.y):
            for x in range(self.chuzCount.x):
                yield self.getPos(Point(x, y))


    def placeFatChuzzle(self, x=0, y=0):
        if x >= self.chuzCount.x or y >= self.chuzCount.y:
            return
        self.getPos((x, y)).setFatTL()

        # this should remove fats from corners of new fat?
        for i in [(0, 1), (1, 0), (1, 1)]:
            point = Point(x, y) + i
            if self.getPos(point.tuple()).fatTL:
                self.deleteFatChuzzle(point.x, point.y)

        self.getPos(x + 1, y).fatTR = True
        self.getPos(x, y + 1).fatBL = True
        self.getPos(x + 1, y + 1).fatBR = True


    def deleteFatChuzzle(self, x=0, y=0):
        if x >= self.chuzCount.x or y >= self.chuzCount.y:
            return
        # if not self.getPos((x, y)).fatTR:
        #     return

        self.getPos(x, y).delFatTL()
        self.getPos(x + 1, y).fatTR = False
        self.getPos(x, y + 1).fatBL = False
        self.getPos(x + 1, y + 1).fatBR = False


    def placeRandomLock(self):
        pass


    def delRandomLock(self):
        pass


    def saveToFile(self, filename):
        pass


    def loadFile(self, filename):
        f = open(filename)
        print(self)
        data = json.load(f)


    def loadData(self, data):
        for chuzzle, color in zip(self.iterator(), data):
            chuzzle.color = color

    def load_from_array(self, array_data):
        self.clear()
        for y, row in enumerate(array_data):
            for x, char_color in enumerate(row):
                self.append(Chuzzle(self.game, Point(x, y), color=int(char_color)))


    def copyChuzzles(self, board_to_copy):
        self.clear()
        for chuzzle in board_to_copy:
            self.append(chuzzle.copy())
            self[-1].position = chuzzle.position.copy()


    def shiftSection(self, sect, direction, amount):
        for chuzzle in self.getMultipleDirs(sects=[sect], direction=direction):
            if direction == 1:
                chuzzle.position.x = (chuzzle.position.x + amount) % self.chuzCount.x
            elif direction == 0:
                chuzzle.position.y = (chuzzle.position.y + amount) % self.chuzCount.y


    def randomMoves(self, amount):
        for count in range(amount):
            direction = self.random.randint(0, 1)  # 0 - 1
            if direction == 0:
                sect = self.random.randint(0, self.chuzCount.y - 1)  # 0 - 5
                amount = self.random.randint(1, self.chuzCount.x - 1)  # 1 - 5
                self.shiftSection(sect, direction, amount)
            else:
                sect = self.random.randint(0, self.chuzCount.x - 1)  # 0 - 5
                amount = self.random.randint(1, self.chuzCount.y - 1)  # 1 - 5
                self.shiftSection(sect, direction, amount)


    def getRow(self, point=Point()):
        if type(point) == int:
            point = Point(point, point)
        for chuzzle in self:
            if chuzzle.position.x == point.x:
                yield chuzzle


    def getCol(self, point=Point()):
        if type(point) == int:
            point = Point(point, point)
        for chuzzle in self:
            if chuzzle.position.y == point.y:
                yield chuzzle


    def getDir(self, point=Point(), direction=0):
        if type(point) in [int, float]:
            point = Point(point)
        if direction == 0:
            for chuzzle in self:
                if chuzzle.position.x == point.x:
                    yield chuzzle
        elif direction == 1:
            for chuzzle in self:
                if chuzzle.position.y == point.y:
                    yield chuzzle


    def getMultipleDirs(self, sects=None, direction=0):
        if sects is None:
            sects = []

        for sect in sects:
            for chuzzle in self.getDir(sect, direction):
                yield chuzzle


    def checkForLocks(self, direction, point=Point()):
        chuzList = []
        for chuzzle in self:
            if direction == 1:  # vertical
                if chuzzle.position.y == point.y:
                    chuzList.append(chuzzle)

            elif direction == 0:  # horizontal
                if chuzzle.position.x == point.x:
                    chuzList.append(chuzzle)

        for chuzzle in chuzList:
            if chuzzle.locked:
                return True
        return False
