# Jerrin Shirks

# native imports
import pygame as p
import math

# custom imports
from engine.constants import *

from tools.sprite_object import spriteObject
from tools.point import Point
from tools.texture_utils import blit_alpha

from engine.chuzzle_list import ChuzzleList
from engine.chuzzle import Chuzzle



class Board(spriteObject):
    def __init__(self, menu, boardOffset=(255, 60), chuzSize=(80, 80), chuzCount=None, gamemode=None):
        super().__init__(menu)
        if chuzCount is None:
            chuzCount = (self.game.settings.boardSize, self.game.settings.boardSize)

        self.sectOffset = None
        self.boardOffset = Point(boardOffset)

        self.chuzSize = Point(480, 480) // self.game.settings.boardSize
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
        self.directionSwitchDelta = 10

        # Lock direction after substantial movement (about 33% of one tile).
        self.directionLockDistance = max(1, int(self.chuzSize.x * 0.33))
        self.directionLocked = False
        self._dirTieFlip = False

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
        self.cursorDragging = False

        # 0 | horizontal
        # 1 | vertical
        self.curDir = -1
        self.lastDir = -1

        self.isLocked = False
        self.isFatTL = False
        self.isFatBR = False
        self.sectsToMove = []

        self.chuzzles = ChuzzleList(game=self.game)

        # copies of rainbow, resizes it
        self.DragRainbowV = self.textures.DragRainbowV
        self.DragRainbowH = self.textures.DragRainbowH
        if self.game.settings.boardSize != 6:
            oldSize = Point(self.DragRainbowV.get_size())
            newSize = (oldSize * 6 / self.game.settings.boardSize).tuple()
            self.DragRainbowV = p.transform.smoothscale(self.DragRainbowV, newSize)
            oldSize = Point(self.DragRainbowH.get_size())
            newSize = (oldSize * 6 / self.game.settings.boardSize).tuple()
            self.DragRainbowH = p.transform.smoothscale(self.DragRainbowH, newSize)

        # print(self.chuzCount)



        for i in range(self.chuzCount.x * self.chuzCount.y):
            self.chuzzles.append(Chuzzle(self.game, Point(i % self.game.settings.boardSize, i // self.game.settings.boardSize)))

        self.gamemode = gamemode
        self.objects = []

        # Mindbender feels better with a smaller dead-zone before movement starts.
        if self.gamemode == "mindbender":
            self.threshold1 = 4
            self.directionSwitchDelta = 2
            self.directionLockDistance = max(1, int(self.chuzSize.x * 0.33))

        self._lock_texture = None
        self._lock_texture_fat = None


    def initiateGameMode(self):
        pass

    def _get_lock_textures(self):
        if self._lock_texture is None or self._lock_texture_fat is None:
            size_normal = Point(6 / self.game.settings.boardSize) * 128
            self._lock_texture = p.transform.smoothscale(self.textures.Lock, size_normal.tuple())

            size_fat = Point(480 / self.game.settings.boardSize * 2 + 18) * 1.58
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
                        _skipList.append((chuzzle.position + (1, 1)) % self.game.settings.boardSize)
                        _sects.append(chuzzle.position.get(self.curDir))
                        newSect = (chuzzle.position.get(self.curDir) + 1) % self.game.settings.boardSize
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
            self.curDir = -1
            self.lastDir = -1
            self.directionLocked = False
            self._dirTieFlip = False
            self.lastPos = Point(self.centerPos)

    def onHeldDown(self, mousePos):
        if not self.inBoundsDuringClickDown:
            return

        self.currentMouseDown = Point(mousePos) - self.boardOffset
        self.pointPos = Point(mousePos) - self.boardOffset

        # Offset from current drag anchor (anchor resets on direction switches).
        delta_x = self.pointPos.x - self.centerPos.x
        delta_y = self.pointPos.y - self.centerPos.y
        X = abs(delta_x)
        Y = abs(delta_y)

        # Per-frame movement drives pre-lock direction jitter.
        frame_dx = self.pointPos.x - self.lastPos.x
        frame_dy = self.pointPos.y - self.lastPos.y
        frame_x = abs(frame_dx)
        frame_y = abs(frame_dy)

        if X != 0 or Y != 0:
            if not self.cursorDragging:
                self.game.cursorIndex = 4
                self.movedMouse = True
                self.cursorDragging = True

        # Before lock, follow the dominant axis of this frame so a slow true diagonal
        # naturally jitters/switches direction.
        candidate_dir = None
        if frame_x > frame_y:
            candidate_dir = 1
        elif frame_y > frame_x:
            candidate_dir = 0
        elif frame_x != 0:
            self._dirTieFlip = not self._dirTieFlip
            candidate_dir = 1 if self._dirTieFlip else 0

        if self.curDir == -1 and candidate_dir is not None:
            self.curDir = candidate_dir
            self.testChuzzleStates()

        # Dynamic lock: once past 33% it locks, and if dragged back below 33% it unlocks.
        if self.curDir == 1:
            active_offset = X
        elif self.curDir == 0:
            active_offset = Y
        else:
            active_offset = 0
        self.directionLocked = active_offset >= self.directionLockDistance

        if (not self.directionLocked) and candidate_dir is not None and self.curDir in [0, 1] and candidate_dir != self.curDir:
            self.curDir = candidate_dir
            # Keep lane source fixed at original click-down; only re-anchor offsets.
            self.centerPos = Point(self.pointPos)
            self.lockLastDelta = 0
            self.lockCurDelta = 0
            self.testChuzzleStates()
            delta_x = 0
            delta_y = 0
            X = 0
            Y = 0

        # fixes glitch where row you are moving changes while locked lol
        if self.isLocked:
            if self.curDir == 1:
                X = 0
            else:
                Y = 0

        if self.curDir not in [0, 1]:
            self.lastPos = self.pointPos
            return

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
        committed_move = None

        if self.mouseDown and self.movedMouse and self.curDir in [0, 1]:
            moved_chuzzles = list(self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir))
            if moved_chuzzles:
                ref = moved_chuzzles[0]

                if self.curDir == 1:
                    raw_disp = int(round(ref.offset.x / float(self.chuzSize.x)))
                    axis = "R"
                else:
                    raw_disp = int(round(ref.offset.y / float(self.chuzSize.y)))
                    axis = "C"

                if raw_disp != 0:
                    committed_move = {
                        "axis": axis,
                        "selected_line": int(self.mouseDownPos.get(self.curDir)),
                        "displacement": int(raw_disp),
                        "sects_to_move": [int(v) for v in self.sectsToMove],
                    }

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
        self.directionLocked = False
        self._dirTieFlip = False

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

        if committed_move is not None and hasattr(self.menu, "onBoardMoveCommitted"):
            self.menu.onBoardMoveCommitted(committed_move)



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

            coord = Point(chuzzle.getX(), chuzzle.getY()) % (self.chuzSize * self.game.settings.boardSize)

            # make it figure out which chuzzle is, "real"
            surf = chuzzle.getSurface()
            if surf is None:
                continue


            if chuzzle.fatTL:
                coord = coord - (8, 10)

            # main chuzzle?
            self.surface.blit(surf, coord.tuple())

            # might need to re-add direction specific if statements, draws the others outside
            self.surface.blit(surf, (coord.x - (self.game.settings.boardSize * self.chuzSize.x), coord.y - (self.game.settings.boardSize * self.chuzSize.y)))
            self.surface.blit(surf, (coord.x - (self.game.settings.boardSize * self.chuzSize.x), coord.y))
            self.surface.blit(surf, (coord.x, coord.y - (self.game.settings.boardSize * self.chuzSize.y)))

        screen.blit(self.surface, self.boardOffset.tuple())

        # render border frame
        screen.blit(self.textures.Border, (229, 30))


        # render locks
        lock_texture, lock_texture_fat = self._get_lock_textures()

        for chuzzle in self.chuzzles:
            if chuzzle.locked:
                coord = Point((chuzzle.getX(), chuzzle.getY()))
                if coord.x > self.chuzSize.x * self.game.settings.boardSize + self.threshold1:
                    coord.x = coord.x % self.chuzSize * self.game.settings.boardSize
                if coord.y > self.chuzSize.y * self.game.settings.boardSize + self.threshold1:
                    coord.y = coord.y % self.chuzSize * self.game.settings.boardSize

                if chuzzle.isNormal():
                    coords = coord - 25 * (lock_texture.get_size()[0] / 128)
                    screen.blit(lock_texture, (coords + self.boardOffset).tuple())

                elif chuzzle.fatTL:
                    coords = coord - 25 * (lock_texture_fat.get_size()[0] / 128) - 10
                    screen.blit(lock_texture_fat, (coords + self.boardOffset).tuple())








        # screen.blit(self.surface, self.boardOffset.tuple())
