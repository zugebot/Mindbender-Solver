# Jerrin Shirks

# native imports
import pygame as p
import math

# custom imports
from data.config import *

from tools.sprite_object import spriteObject
from tools.point import Point
from tools.texture_utils import blit_alpha

from engine.chuzzle_list import ChuzzleList
from engine.chuzzle import Chuzzle



class Board(spriteObject):
    def __init__(self, menu, boardOffset=(255, 60), chuzSize=(80, 80), chuzCount=(BOARD_SIZE, BOARD_SIZE), gamemode=None):
        super().__init__(menu)

        self.sectOffset = None
        self.boardOffset = Point(boardOffset)

        self.chuzSize = Point(480, 480) // BOARD_SIZE
        self.chuzCount = Point(chuzCount)

        self.surfaceSize = self.chuzSize * self.chuzCount
        self.surface = p.Surface(self.surfaceSize.tuple(), p.SRCALPHA)

        self.surface.fill((0, 0, 0, 0))

        # dealing with board states
        self.dragging = False
        self.mouseDown = False
        self.inBounds = False
        self.inBoundsDuringClickDown = False

        self.threshold1 = 7
        self.threshold2 = 5

        # used for changing direction of a locked row
        self.threshold3 = 0

        self.centerPos = Point()
        self.pointPos = Point()
        self.lastPos = Point()
        self.mouseDownPos = Point()
        self.initialMouseDown = Point()
        self.currentMouseDown = Point()
        self.lockLastDelta = 0
        self.lockCurDelta = 0
        self.lastDelta = Point()
        self.curDelta = Point()
        self.movedMouse = False

        # 0 | horizontal
        # 1 | vertical
        self.curDir = 0
        self.lastDir = 0

        self.isLocked = False
        self.isFatTL = False
        self.isFatBR = False
        self.sectsToMove = []

        self.chuzzles = ChuzzleList(game=self.game)

        self.cursorDragging = False

        # copies of rainbow, resizes it
        self.DragRainbowV = self.textures.DragRainbowV
        self.DragRainbowH = self.textures.DragRainbowH
        if BOARD_SIZE != 6:
            oldSize = Point(self.DragRainbowV.get_size())
            newSize = (oldSize * 6 / BOARD_SIZE).tuple()
            self.DragRainbowV = p.transform.smoothscale(self.DragRainbowV, newSize)
            oldSize = Point(self.DragRainbowH.get_size())
            newSize = (oldSize * 6 / BOARD_SIZE).tuple()
            self.DragRainbowH = p.transform.smoothscale(self.DragRainbowH, newSize)

        # print(self.chuzCount)



        for i in range(self.chuzCount.x * self.chuzCount.y):
            self.chuzzles.append(Chuzzle(self.game, Point(i % BOARD_SIZE, i // BOARD_SIZE)))

        self.gamemode = gamemode
        self.objects = []

        # Mindbender feels better with a smaller dead-zone before movement starts.
        if self.gamemode == "mindbender":
            self.threshold1 = 4
            self.threshold2 = 1

        self._lock_texture = None
        self._lock_texture_fat = None


    def initiateGameMode(self):
        pass

    def _get_lock_textures(self):
        if self._lock_texture is None or self._lock_texture_fat is None:
            size_normal = Point(6 / BOARD_SIZE) * 128
            self._lock_texture = p.transform.smoothscale(self.textures.Lock, size_normal.tuple())

            size_fat = Point(480 / BOARD_SIZE * 2 + 18) * 1.58
            self._lock_texture_fat = p.transform.smoothscale(self.textures.Lock, size_fat.tuple())

        return self._lock_texture, self._lock_texture_fat

    def testMouseStates(self, mousePos, mouseState):

        self.inBounds = self.boardOffset.x <= mousePos[0] <= self.boardOffset.x + self.surfaceSize.x and \
            self.boardOffset.y <= mousePos[1] <= self.boardOffset.y + self.surfaceSize.y

        if self.inBounds:
            if mouseState == MOUSE_CLICK_DOWN:
                self.onMouseDown(mousePos)
            if mouseState == MOUSE_HOVER:
                self.onMouseHover(mousePos)
        if mouseState == MOUSE_HELD_DOWN:
            self.onHeldDown(mousePos)
        if mouseState == MOUSE_CLICK_UP:
            self.onMouseUp(mousePos)

    def convertToPosition(self, other, mode="floor"):
        if type(other) == tuple:
            other = Point(other)
        if type(other) == int:
            if mode == "floor":
                return other // self.chuzSize.x
            elif mode == "nearest":
                return round(float(other) / float(self.chuzSize.x))

        elif type(other) == Point:
            if mode == "floor":
                return other // self.chuzSize
            elif mode == "nearest":
                other = other.toFloat()
                value = other / self.chuzSize.toFloat()
                if value.x % 1.0 == 0.5:
                    value.x = math.floor(value.x)
                else:
                    value.x = round(value.x)
                if value.y % 1.0 == 0.5:
                    value.y = math.floor(value.y)
                else:
                    value.y = round(value.y)
                return value

    def convertFromPosition(self, other, direction=0):
        if type(other) == tuple:
            other = Point(other)

        if type(other) == Point:
            return other * self.chuzSize
        elif type(other) in [int, float]:
            if direction == 0:
                return other * self.chuzSize.x
            else:
                return other * self.chuzSize.y
        else:
            raise Exception("you suck ass at coding")


    # If I grab it at 0, and the fat is on -5/0, it does not move the row -5
    def recurseMain(self, sectNumber, debug=False):
        def recurseRight(_sectNumber, _sects=None, _skipList=None, _recurse=False):
            if _skipList is None:
                _skipList = []
            if _sects is None:
                _sects = []
            sect = self.chuzzles.getDir(Point(_sectNumber), self.curDir)
            for chuzzle in sect:
                if chuzzle.fatTL:
                    if chuzzle.position not in _skipList:
                        _skipList.append(chuzzle.position)
                        _skipList.append((chuzzle.position + (1, 1)) % BOARD_SIZE)
                        _sects.append(chuzzle.position.get(self.curDir))
                        newSect = (chuzzle.position.get(self.curDir) + 1) % BOARD_SIZE
                        _sects.append(newSect)
                        _recurse = True
            if _recurse:
                _sects, _skipList = recurseRight(_sectNumber + 1, _sects, _skipList)
            return _sects, _skipList

        def recurseLeft(_sectNumber, _sects=None, _skipList=None, _recurse=False):
            if _skipList is None:
                _skipList = []
            if _sects is None:
                _sects = []
            sect = self.chuzzles.getDir(Point(_sectNumber), self.curDir)
            for chuzzle in sect:
                if chuzzle.fatBR:
                    if chuzzle.position not in _skipList:
                        _skipList.append(chuzzle.position)  # BR
                        newPos = chuzzle.position + (-1, -1)
                        if newPos.x < 0:
                            x = newPos.x
                            newPos.x = self.chuzCount.get(0) + newPos.x
                            print(x, newPos.x)
                        if newPos.y < 0:
                            newPos.y = self.chuzCount.get(1) + newPos.y
                        _skipList.append(newPos)  # TL
                        _sects.append(chuzzle.position.get(self.curDir))  # BR
                        newSect = chuzzle.position.get(self.curDir) - 1
                        if newSect < 0:
                            newSect = self.chuzCount.get(self.curDir) + newSect
                        _sects.append(newSect)  # TL
                        _recurse = True
            if _recurse:
                _sectNumber = (_sectNumber - 1) % self.chuzCount
                print("left", _sectNumber, _sects, _skipList)
                _sects, _skipList = recurseLeft(_sectNumber, _sects, _skipList)
            return _sects, _skipList

        sectsL, sectsR, skipListL, skipListR = [], [], [], []

        if any([chuz.fatTL for chuz in self.chuzzles.getDir(sectNumber, self.curDir)]):
            sectsR, skipListL = recurseRight(sectNumber, _sects=[], _skipList=[])

        if any([chuz.fatBR for chuz in self.chuzzles.getDir(sectNumber, self.curDir)]):
            sectsL, skipListL = recurseLeft(sectNumber, _sects=[], _skipList=[])

        sects = list(set(sectsR + sectsL))
        sects = [int(i) for i in sects]

        if debug:
            print("recursion", sectsR, sectsL)

        return sects

    def testChuzzleStates(self):
        self.isFatTL = any([chuzzle.fatTL for chuzzle in self.chuzzles.getDir(self.mouseDownPos, self.curDir)])
        self.isFatBR = any([chuzzle.fatBR for chuzzle in self.chuzzles.getDir(self.mouseDownPos, self.curDir)])

        if self.isFatTL or self.isFatBR:
            self.sectsToMove = self.recurseMain(self.mouseDownPos)
        else:
            self.sectsToMove = [self.mouseDownPos.get(self.curDir)]

        # print(self.sectsToMove)

        self.isLocked = any([chuzzle.locked for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir)])


    def onMouseHover(self, mousePos):
        self.centerPos = Point(mousePos) - self.boardOffset
        self.mouseDownPos = self.convertToPosition(self.centerPos)

        pos = self.convertToPosition(self.mouseDownPos)
        if pos == (0, 0):
            self.chuzzles.getPos(pos).setEyeBallDirection(self.centerPos)
            # print(self.centerPos, self.mouseDownPos)


    def onMouseDown(self, mousePos):
        if not self.mouseDown:
            self.mouseDown = True
            self.inBoundsDuringClickDown = self.inBounds
            self.initialMouseDown = Point(mousePos) - self.boardOffset
            self.currentMouseDown = Point(mousePos) - self.boardOffset
            self.centerPos = Point(mousePos) - self.boardOffset
            self.mouseDownPos = self.convertToPosition(self.centerPos)



    def onHeldDown(self, mousePos):
        if not self.inBoundsDuringClickDown:
            return

        self.currentMouseDown = Point(mousePos) - self.boardOffset
        self.pointPos = Point(mousePos) - self.boardOffset

        # finds delta between cur mouse position - last mouse position
        X = abs(self.centerPos.x - self.pointPos.x)
        Y = abs(self.centerPos.y - self.pointPos.y)

        # fixes glitch where row you are moving changes while locked lol
        if self.isLocked:
            if self.curDir == 1:
                X = 0
            else:
                Y = 0

        # this makes it so if dragging a chuzzle the cursor goes yellow
        if X != 0 or Y != 0:
            if not self.cursorDragging:
                self.game.cursorIndex = 4
                self.movedMouse = True
                self.cursorDragging = True

        # this changes your direction based off how far you scroll a chuzzle
        if X < self.threshold1 and Y < self.threshold1:
            if X > self.threshold2 or Y > self.threshold2:
                self.curDir = 0 if X < Y else 1
                self.testChuzzleStates()
                # updates center as to be an updating reference
                self.centerPos = self.pointPos

        # if mouse is moved so fast the other if statement never gets to run
        elif (X > self.threshold1 > Y) or (X < self.threshold1 < Y):
            self.curDir = 0 if X < Y else 1
            self.testChuzzleStates()

        # prevents you from switching directions when a locked chuzzle is in that row
        if self.isLocked:
            if self.lastDir != -1:
                self.curDir = self.lastDir

        # this deals with snapping chuzzles to the board when changing directions
        if self.curDir != self.lastDir:
            self.lastDir = self.curDir
            for chuzzle in self.chuzzles:
                chuzzle.snapToBoard()

                # prevents changing directions while locked
                if self.isLocked:

                    if self.lockLastDelta == 0:
                        self.lockLastDelta = self.pointPos.x - self.lastPos.x

                    elif self.curDir == 1:
                        self.lockCurDelta = self.pointPos.x - self.lastPos.x
                        if self.lockLastDelta > 0 and self.lockCurDelta < -self.threshold3:
                            self.centerPos.x = self.pointPos.x
                            self.lockLastDelta = self.lockCurDelta

                        elif self.lockLastDelta < 0 and self.lockCurDelta > self.threshold3:
                            self.centerPos.x = self.pointPos.x
                            self.lockLastDelta = self.lockCurDelta

                    elif self.curDir == 0:
                        self.lockCurDelta = self.pointPos.y - self.lastPos.y
                        if self.lockLastDelta > 0 and self.lockCurDelta < -self.threshold3:
                            self.centerPos.y = self.pointPos.y
                            self.lockLastDelta = self.lockCurDelta

                        elif self.lockLastDelta < 0 and self.lockCurDelta > self.threshold3:
                            self.centerPos.y = self.pointPos.y
                            self.lockLastDelta = self.lockCurDelta

        self.sectOffset = abs((self.mouseDownPos - self.pointPos).get(self.curDir))

        #  this determines the final offset of the chuzzles.
        if self.curDir == 1:  # vertical

            for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir):

                if self.lockCurDelta != 0:
                    chuzzle.endSwayAnim()

                chuzzle.offset.x = self.pointPos.x - self.centerPos.x
                # ensures sections can only move a little for lock rattling.
                if self.isLocked:
                    chuzzle.offset.x = max(min(chuzzle.offset.x, self.threshold1), -self.threshold1)

                chuzzle.calculateVelocity(self.curDir)

        elif self.curDir == 0:  # horizontal

            # for chuzzle in self.getRow(self.mouseDownPos):
            for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir):

                if self.lockCurDelta != 0:
                    chuzzle.endSwayAnim()

                chuzzle.offset.y = self.pointPos.y - self.centerPos.y

                # ensures sections can only move a little for lock rattling.
                if self.isLocked:
                    chuzzle.offset.y = max(min(chuzzle.offset.y, self.threshold1), -self.threshold1)

                chuzzle.calculateVelocity(self.curDir)



        # print(self.chuzzles.getPos(self.mouseDownPos).direction)
        # g = self.pointPos - self.centerPos
        # print(X, g.x, Y, g.y)

        # performed at end of onHeldDown()
        self.lastPos = self.pointPos



    def onMouseUp(self, mousePos):
        self.mouseDown = False
        self.cursorDragging = False
        self.game.cursorIndex = 0
        self.lastDir = -1
        self.curDir = -1
        self.lockLastDelta = 0
        self.lockCurDelta = 0
        self.sectsToMove = []
        self.isFatTL = False
        self.isFatBR = False
        self.isLocked = False
        self.inBoundsDuringClickDown = False


        # places the chuzzles where they are currently at
        if self.gamemode == "mindbender":

            if self.movedMouse:
                self.game.sounds.playSound(self.game.sounds.Click)
                self.movedMouse = False

            for chuzzle in self.chuzzles:
                new_pos = self.convertToPosition(chuzzle.getXY(), mode="nearest") % self.chuzCount
                chuzzle.offset.clear()
                chuzzle.position = new_pos
        else:
            for chuzzle in self.chuzzles:
                chuzzle.snapToBoard()



    def render(self, screen):

        # wash out the background
        self.surface.fill((0, 0, 0, 0))


        # draw drag rainbow behind the chuzzles
        if self.mouseDown:
            offset = abs(self.pointPos - self.centerPos)
            texture = [self.DragRainbowV, self.DragRainbowH][self.curDir]

            point = Point()
            opacity = 210

            if self.curDir == 0:
                point.x = self.convertFromPosition(self.convertToPosition(self.initialMouseDown.x), 0)
                point.y = self.boardOffset.y

                if offset.y < self.chuzSize.y * 2:
                    opacity *= (offset.y / (self.chuzSize.y * 2))

                # X: this moves it up, so it isn't cut off
                # Y: this centers the dragRainbow
                point += ((self.chuzSize.x - texture.get_size()[0]) / 2, -60)

                blit_alpha(self.surface, texture, point, opacity)

            elif self.curDir == 1:
                point.x = self.boardOffset.x
                point.y = self.convertFromPosition(self.convertToPosition(self.initialMouseDown.y), 0)

                if offset.x < self.chuzSize.x * 2:
                    opacity *= offset.x / (self.chuzSize.x * 2)

                # X: this moves it left, so it isn't cut off
                # Y: this centers the dragRainbow
                point += (-260, (self.chuzSize.y - texture.get_size()[1]) / 2)

                blit_alpha(self.surface, texture, point, opacity)


        # render each chuzzle to screen
        for chuzzle in self.chuzzles:
            chuzzle.incrementAnimation()

            coord = Point(chuzzle.getX(), chuzzle.getY()) % (self.chuzSize * BOARD_SIZE)

            # make it figure out which chuzzle is, "real"
            surf = chuzzle.getSurface()
            if surf is None:
                continue


            if chuzzle.fatTL:
                coord = coord - (8, 10)

            # main chuzzle?
            self.surface.blit(surf, coord.tuple())

            # might need to re-add direction specific if statements, draws the others outside
            self.surface.blit(surf, (coord.x - (BOARD_SIZE * self.chuzSize.x), coord.y - (BOARD_SIZE * self.chuzSize.y)))
            self.surface.blit(surf, (coord.x - (BOARD_SIZE * self.chuzSize.x), coord.y))
            self.surface.blit(surf, (coord.x, coord.y - (BOARD_SIZE * self.chuzSize.y)))

        screen.blit(self.surface, self.boardOffset.tuple())

        # render border frame
        screen.blit(self.textures.Border, (229, 30))


        # render locks
        lock_texture, lock_texture_fat = self._get_lock_textures()

        for chuzzle in self.chuzzles:
            if chuzzle.locked:
                coord = Point((chuzzle.getX(), chuzzle.getY()))
                if coord.x > self.chuzSize.x * BOARD_SIZE + self.threshold1:
                    coord.x = coord.x % self.chuzSize * BOARD_SIZE
                if coord.y > self.chuzSize.y * BOARD_SIZE + self.threshold1:
                    coord.y = coord.y % self.chuzSize * BOARD_SIZE

                if chuzzle.isNormal():
                    coords = coord - 25 * (lock_texture.get_size()[0] / 128)
                    screen.blit(lock_texture, (coords + self.boardOffset).tuple())

                elif chuzzle.fatTL:
                    coords = coord - 25 * (lock_texture_fat.get_size()[0] / 128) - 10
                    screen.blit(lock_texture_fat, (coords + self.boardOffset).tuple())








        # screen.blit(self.surface, self.boardOffset.tuple())
