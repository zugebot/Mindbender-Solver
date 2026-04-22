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

        self.boardSizePx = self.chuzSize * self.chuzCount

        # 20 total extra pixels, centered as 10 on each side
        self.surfacePad = Point(10, 10)
        self.surfaceSize = self.boardSizePx + self.surfacePad * 2
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

        for i in range(self.chuzCount.x * self.chuzCount.y):
            self.chuzzles.append(
                Chuzzle(
                    self.game,
                    Point(i % self.game.settings.boardSize, i // self.game.settings.boardSize)
                )
            )

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

    def _get_drag_span_bounds(self):
        if not self.sectsToMove:
            return None

        min_sect = min(self.sectsToMove)
        max_sect = max(self.sectsToMove)

        if self.curDir == 1:
            # moving rows horizontally
            # rows are stacked vertically, so span is in Y
            x = 0
            y = min_sect * self.chuzSize.y
            w = self.boardSizePx.x
            h = (max_sect - min_sect + 1) * self.chuzSize.y
            return p.Rect(int(x), int(y), int(w), int(h))

        elif self.curDir == 0:
            # moving cols vertically
            # cols are stacked horizontally, so span is in X
            x = min_sect * self.chuzSize.x
            y = 0
            w = (max_sect - min_sect + 1) * self.chuzSize.x
            h = self.boardSizePx.y
            return p.Rect(int(x), int(y), int(w), int(h))

        return None

    def _build_drag_rainbow_surface(self):
        span_rect = self._get_drag_span_bounds()
        if span_rect is None or self.curDir not in [0, 1]:
            return None, None

        if self.curDir == 1:
            # horizontal movement across one or more rows
            base = self.DragRainbowH
            target_size = (span_rect.width, span_rect.height)
        else:
            # vertical movement across one or more columns
            base = self.DragRainbowV
            target_size = (span_rect.width, span_rect.height)

        if target_size[0] <= 0 or target_size[1] <= 0:
            return None, None

        scaled = p.transform.smoothscale(base, target_size)
        return scaled, span_rect

    def _get_lock_textures(self):
        if self._lock_texture is None or self._lock_texture_fat is None:
            size_normal = Point(6 / self.game.settings.boardSize) * 128
            self._lock_texture = p.transform.smoothscale(self.textures.Lock, size_normal.tuple())

            size_fat = Point(480 / self.game.settings.boardSize * 2 + 18) * 1.58
            self._lock_texture_fat = p.transform.smoothscale(self.textures.Lock, size_fat.tuple())

        return self._lock_texture, self._lock_texture_fat

    def _surface_blit_pos(self, board_pos):
        return Point(board_pos) + self.surfacePad

    def testMouseStates(self, mousePos, mouseState):
        self.inBounds = (
            self.boardOffset.x <= mousePos[0] <= self.boardOffset.x + self.boardSizePx.x and
            self.boardOffset.y <= mousePos[1] <= self.boardOffset.y + self.boardSizePx.y
        )

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

        self.isLocked = any([chuzzle.locked for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir)])

    def onMouseHover(self, mousePos):
        self.centerPos = Point(mousePos) - self.boardOffset
        self.mouseDownPos = self.convertToPosition(self.centerPos)

        pos = Point(self.mouseDownPos)
        chuzzle = self.chuzzles.getPos(pos)

        if chuzzle is not None and pos == (0, 0):
            chuzzle.setEyeBallDirection(self.centerPos)

    def onRightClickDown(self, mousePos):
        local = Point(mousePos) - self.boardOffset

        if not (0 <= local.x < self.boardSizePx.x and 0 <= local.y < self.boardSizePx.y):
            print("[RIGHT CLICK] outside board")
            return

        cell_pos = self.convertToPosition(local, mode="floor")
        chuzzle = self.chuzzles.getPos(cell_pos)

        print("\n" + "=" * 70)
        print(f"[RIGHT CLICK] screen={mousePos} local={local} cell={cell_pos}")

        if chuzzle is None:
            print("No chuzzle found at that logical cell.")
            print("=" * 70)
            return

        self._print_chuzzle_debug(chuzzle, clicked_cell=cell_pos)
        print("=" * 70)

    def _print_chuzzle_debug(self, chuzzle, clicked_cell=None):
        fat_anchor = None
        if hasattr(self.chuzzles, "_fat_anchor_from_chuzzle"):
            fat_anchor = self.chuzzles._fat_anchor_from_chuzzle(chuzzle)

        print(f"id                : {id(chuzzle)}")
        print(f"clicked_cell      : {clicked_cell}")
        print(f"color             : {chuzzle.color}")
        print(f"position          : {chuzzle.position}")
        print(f"offset            : {chuzzle.offset}")
        print(f"delta             : {chuzzle.delta}")
        print(f"velocity          : {chuzzle.velocity}")
        print(f"direction         : {chuzzle.direction}")
        print(f"motionSpeed       : {chuzzle.motionSpeed}")
        print(f"motionSpeedSmooth : {chuzzle.motionSpeedSmoothed}")
        print(f"locked            : {chuzzle.locked}")
        print(f"isNormal          : {chuzzle.isNormal()}")
        print(f"fatTL             : {chuzzle.fatTL}")
        print(f"fatTR             : {chuzzle.fatTR}")
        print(f"fatBL             : {chuzzle.fatBL}")
        print(f"fatBR             : {chuzzle.fatBR}")
        print(f"fat_anchor        : {fat_anchor}")
        print(f"getX/getY         : ({chuzzle.getX()}, {chuzzle.getY()})")
        print(f"getXY             : {chuzzle.getXY()}")

        if fat_anchor is not None:
            ax = int(fat_anchor.x)
            ay = int(fat_anchor.y)
            body = [
                Point(ax, ay),
                Point(ax + 1, ay),
                Point(ax, ay + 1),
                Point(ax + 1, ay + 1),
            ]
            print(f"fat_body_cells    : {body}")

        if self.curDir in [0, 1]:
            print(f"drag curDir       : {self.curDir}")
            print(f"drag sectsToMove  : {self.sectsToMove}")
            in_drag_set = chuzzle in list(self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir))
            print(f"in dragged set    : {in_drag_set}")
    
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

        if self.curDir == 1:
            active_offset = X
        elif self.curDir == 0:
            active_offset = Y
        else:
            active_offset = 0
        self.directionLocked = active_offset >= self.directionLockDistance

        if (not self.directionLocked) and candidate_dir is not None and self.curDir in [0, 1] and candidate_dir != self.curDir:
            self.curDir = candidate_dir
            self.centerPos = Point(self.pointPos)
            self.lockLastDelta = 0
            self.lockCurDelta = 0
            self.testChuzzleStates()
            delta_x = 0
            delta_y = 0
            X = 0
            Y = 0

        if self.isLocked:
            if self.curDir == 1:
                X = 0
            else:
                Y = 0

        if self.curDir not in [0, 1]:
            self.lastPos = self.pointPos
            return

        if self.isLocked:
            if self.lastDir != -1:
                self.curDir = self.lastDir

        if self.curDir != self.lastDir:
            self.lastDir = self.curDir
            for chuzzle in self.chuzzles:
                chuzzle.snapToBoard()

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

        if self.curDir == 1:  # moving rows horizontally
            proposed_offset = self.pointPos.x - self.centerPos.x

            if self.isLocked:
                proposed_offset = max(min(proposed_offset, self.threshold1), -self.threshold1)

            clamp_info = None
            if self.chuzzles.sectionHasFat(self.sectsToMove, self.curDir):
                proposed_offset, clamp_info = self.chuzzles.clampSectionOffsetPixels(
                    sects=self.sectsToMove,
                    direction=self.curDir,
                    offset_pixels=proposed_offset
                )
                if clamp_info["clamped"]:
                    self.chuzzles.onFatEdgeClamp(clamp_info)

            for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir):
                if self.lockCurDelta != 0:
                    chuzzle.endSwayAnim()

                chuzzle.offset.x = proposed_offset

                if clamp_info is not None and clamp_info["clamped"]:
                    chuzzle.stopMotionKeepOffset()
                else:
                    chuzzle.calculateVelocity(self.curDir)



        elif self.curDir == 0:  # moving cols vertically

            proposed_offset = self.pointPos.y - self.centerPos.y

            if self.isLocked:
                proposed_offset = max(min(proposed_offset, self.threshold1), -self.threshold1)

            clamp_info = None

            if self.chuzzles.sectionHasFat(self.sectsToMove, self.curDir):

                proposed_offset, clamp_info = self.chuzzles.clampSectionOffsetPixels(
                    sects=self.sectsToMove,
                    direction=self.curDir,
                    offset_pixels=proposed_offset
                )

                if clamp_info["clamped"]:
                    self.chuzzles.onFatEdgeClamp(clamp_info)

            for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir):

                if self.lockCurDelta != 0:
                    chuzzle.endSwayAnim()

                chuzzle.offset.y = proposed_offset
                if clamp_info is not None and clamp_info["clamped"]:
                    chuzzle.stopMotionKeepOffset()
                else:
                    chuzzle.calculateVelocity(self.curDir)

        self.lastPos = self.pointPos

    def onMouseUp(self, mousePos):
        committed_move = None

        # Capture drag state BEFORE clearing it.
        active_cur_dir = self.curDir
        active_sects = list(self.sectsToMove)
        active_mouse_down_pos = self.mouseDownPos.copy() if isinstance(self.mouseDownPos, Point) else Point(
            self.mouseDownPos)
        moved_chuzzles = []

        if self.mouseDown and self.movedMouse and active_cur_dir in [0, 1]:
            moved_chuzzles = list(self.chuzzles.getMultipleDirs(active_sects, active_cur_dir))
            if moved_chuzzles:
                ref = moved_chuzzles[0]

                if active_cur_dir == 1:
                    raw_disp = int(round(ref.offset.x / float(self.chuzSize.x)))
                    axis = "R"
                else:
                    raw_disp = int(round(ref.offset.y / float(self.chuzSize.y)))
                    axis = "C"

                if raw_disp != 0:
                    committed_move = {
                        "axis": axis,
                        "selected_line": int(active_mouse_down_pos.get(active_cur_dir)),
                        "displacement": int(raw_disp),
                        "sects_to_move": [int(v) for v in active_sects],
                    }

        
        if self.movedMouse:
            self.game.sounds.playSound(self.game.sounds.Click)
            self.movedMouse = False

        if active_cur_dir in [0, 1] and moved_chuzzles:
            ref = moved_chuzzles[0]

            if active_cur_dir == 1:
                requested_amount = int(round(ref.offset.x / float(self.chuzSize.x)))
            else:
                requested_amount = int(round(ref.offset.y / float(self.chuzSize.y)))

            if self.chuzzles.sectionHasFat(active_sects, active_cur_dir):
                applied_amount, clamp_info = self.chuzzles.clampSectionShiftCells(
                    sects=active_sects,
                    direction=active_cur_dir,
                    amount=requested_amount
                )

                if clamp_info["clamped"]:
                    self.chuzzles.onFatEdgeClamp(clamp_info)

                self.chuzzles.applySectionShiftMixed(
                    sects=active_sects,
                    direction=active_cur_dir,
                    amount=applied_amount
                )
            else:
                applied_amount = requested_amount

                self.chuzzles.applySectionShiftWrap(
                    sects=active_sects,
                    direction=active_cur_dir,
                    amount=applied_amount
                )

            for chuzzle in self.chuzzles:
                chuzzle.offset.clear()
                chuzzle.last = chuzzle.offset.copy()
                chuzzle.delta.clear()
                chuzzle.velocity.clear()
                chuzzle.motionSpeed = 0.0
                chuzzle.motionSpeedSmoothed = 0.0

            if applied_amount != 0:
                committed_move = {
                    "axis": "R" if active_cur_dir == 1 else "C",
                    "selected_line": int(active_mouse_down_pos.get(active_cur_dir)),
                    "displacement": int(applied_amount),
                    "sects_to_move": [int(v) for v in active_sects],
                }
            else:
                committed_move = None
        else:
            for chuzzle in self.chuzzles:
                chuzzle.offset.clear()
                chuzzle.last = chuzzle.offset.copy()
                chuzzle.delta.clear()
                chuzzle.velocity.clear()
                chuzzle.motionSpeed = 0.0
                chuzzle.motionSpeedSmoothed = 0.0
                

        # NOW clear drag state.
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

        if committed_move is not None and hasattr(self.menu, "onBoardMoveCommitted"):
            self.menu.onBoardMoveCommitted(committed_move)

    def render(self, screen):
        # wash out the background
        self.surface.fill((0, 0, 0, 0))

        board_w = self.boardSizePx.x
        board_h = self.boardSizePx.y

        # draw drag rainbow behind the chuzzles
        if self.mouseDown and self.curDir in [0, 1] and self.sectsToMove:
            offset = abs(self.pointPos - self.centerPos)
            texture, span_rect = self._build_drag_rainbow_surface()

            if texture is not None and span_rect is not None:
                opacity = 154

                if self.curDir == 0:
                    # vertical drag on one or more columns
                    if offset.y < self.chuzSize.y * 2:
                        opacity *= (offset.y / float(self.chuzSize.y * 2))
                else:
                    # horizontal drag on one or more rows
                    if offset.x < self.chuzSize.x * 2:
                        opacity *= (offset.x / float(self.chuzSize.x * 2))

                blit_alpha(
                    self.surface,
                    texture,
                    self._surface_blit_pos(Point(span_rect.x, span_rect.y)),
                    opacity
                )

        # render each chuzzle to padded surface
        for chuzzle in self.chuzzles:
            chuzzle.incrementAnimation()

            coord = Point(chuzzle.getX(), chuzzle.getY()) % board_w
            coord.y = coord.y % board_h

            surf = chuzzle.getSurface()
            if surf is None:
                continue

            if chuzzle.fatTL:
                coord = coord - (10, 12)

            main_coord = self._surface_blit_pos(coord)

            # main chuzzle
            self.surface.blit(surf, main_coord.tuple())

            # Only normal chuzzles get wrap helper copies.
            # Fats are now bounded and should never appear on the opposite side.
            if chuzzle.isNormal():
                self.surface.blit(surf, (main_coord.x - board_w, main_coord.y))
                self.surface.blit(surf, (main_coord.x, main_coord.y - board_h))
                self.surface.blit(surf, (main_coord.x - board_w, main_coord.y - board_h))

        # shift back by pad so the visible board still starts at boardOffset
        screen.blit(self.surface, (self.boardOffset - self.surfacePad).tuple())

        # render border frame
        screen.blit(self.textures.Border, (229, 29))

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