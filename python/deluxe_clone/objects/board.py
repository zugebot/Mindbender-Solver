# Jerrin Shirks

from __future__ import annotations

import math
import pygame as p

from engine.constants import *
from tools.sprite_object import spriteObject
from tools.point import Point
from tools.texture_utils import blit_alpha
from engine.chuzzle_list import ChuzzleList
from engine.chuzzle import Chuzzle


class Board(spriteObject):
    """
    Chuzzle Deluxe board input model.

    Direction convention used by this project:

        curDir == 1:
            row drag
            horizontal movement
            selected section is the clicked Y row

        curDir == 0:
            column drag
            vertical movement
            selected section is the clicked X column

    Decompiled behavior this class is trying to preserve:

        0040ec10:
            mouse-down board entry

        0040c3b0:
            begin horizontal row drag

        0040c5f0:
            begin vertical column drag

        00414f70:
            held-drag update
            frame-to-frame mouse delta accumulation
            direction jitter/switching while active offset is near zero

        00418c70:
            release and convert visual offset to whole-cell shifts

        0040c8b0:
            clear active drag flags after release
    """

    DIR_COLUMN_VERTICAL = 0
    DIR_ROW_HORIZONTAL = 1

    BOARD_CELLS = 6

    # Decompiled constants by role.
    # _DAT_004f8db0 is the cell size in the original 6x6 board.
    ORIGINAL_CELL_PX = 80.0

    # _DAT_004f8e28 is used as the near-zero active offset gate before direction switching.
    # In practice this behaves like a small pixel epsilon.
    SWITCH_ACTIVE_OFFSET_EPSILON = 25.0

    # _DAT_004f84e0 appears in:
    #   abs(dx) < abs(dy / K)
    #   abs(dy) < abs(dx / K)
    # This biases against switching. If K is too low, diagonal jitter becomes too eager.
    # If K is too high, switching becomes too rare.
    SWITCH_PERPENDICULAR_DIVISOR = 3.0

    # Default normal drag clamp from 0040c3b0 / 0040c5f0.
    # Normal rows/columns can visually move around one full board width/height.
    NORMAL_MIN_CELLS = -6.0
    NORMAL_MAX_CELLS = 6.0

    # Locked row/column clamp from the locked branch in 0040c3b0 / 0040c5f0.
    # The exact data constants are not emitted in the Ghidra C dump, but this is no longer
    # tied to your old arbitrary threshold names or direction lock logic.
    LOCKED_MIN_CELLS = -0.25
    LOCKED_MAX_CELLS = 0.25

    def __init__(self, menu, boardOffset=(255, 60), chuzSize=(80, 80), chuzCount=None, gamemode=None):
        super().__init__(menu)

        if chuzCount is None:
            chuzCount = (self.game.settings.boardSize, self.game.settings.boardSize)

        self.boardOffset = Point(boardOffset)
        self.chuzSize = Point(480, 480) // self.game.settings.boardSize
        self.chuzCount = Point(chuzCount)
        self.boardSizePx = self.chuzSize * self.chuzCount

        self.surfacePad = Point(10, 10)
        self.surfaceSize = self.boardSizePx + self.surfacePad * 2
        self.surface = p.Surface(self.surfaceSize.tuple(), p.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))

        self.gamemode = gamemode
        self.objects = []

        self.chuzzles = ChuzzleList(game=self.game)

        for i in range(self.chuzCount.x * self.chuzCount.y):
            self.chuzzles.append(
                Chuzzle(
                    self.game,
                    Point(i % self.game.settings.boardSize, i // self.game.settings.boardSize),
                )
            )

        self.DragRainbowV = self.textures.DragRainbowV
        self.DragRainbowH = self.textures.DragRainbowH

        if self.game.settings.boardSize != 6:
            old_size = Point(self.DragRainbowV.get_size())
            new_size = (old_size * 6 / self.game.settings.boardSize).tuple()
            self.DragRainbowV = p.transform.smoothscale(self.DragRainbowV, new_size)

            old_size = Point(self.DragRainbowH.get_size())
            new_size = (old_size * 6 / self.game.settings.boardSize).tuple()
            self.DragRainbowH = p.transform.smoothscale(self.DragRainbowH, new_size)

        self._lock_texture = None
        self._lock_texture_fat = None

        self._reset_drag_state()

    def _reset_drag_state(self):
        self.sectOffset = None

        self.dragging = False
        self.mouseDown = False
        self.inBounds = False
        self.inBoundsDuringClickDown = False

        self.centerPos = Point()
        self.pointPos = Point()
        self.lastPos = Point()

        self.mouseDownPos = Point()
        self.originalMouseDownPos = Point()
        self.originalMouseDownCell = Point()
        self.initialMouseDown = Point()
        self.currentMouseDown = Point()

        self.lastDelta = Point()
        self.curDelta = Point()

        self.movedMouse = False
        self.cursorDragging = False

        self.curDir = -1
        self.lastDir = -1

        self.isLocked = False
        self.isFatTL = False
        self.isFatBR = False
        self.sectsToMove = []
        self.activeSelectedSection = -1

        self.dragMinPx = 0.0
        self.dragMaxPx = 0.0

        self._dirTieFlip = False

    def initiateGameMode(self):
        pass

    def testMouseStates(self, mousePos, mouseState):
        self.inBounds = (
                self.boardOffset.x <= mousePos[0] < self.boardOffset.x + self.boardSizePx.x and
                self.boardOffset.y <= mousePos[1] < self.boardOffset.y + self.boardSizePx.y
        )

        if self.inBounds:
            if mouseState == MOUSE_CLICK_DOWN:
                self.onMouseDown(mousePos)
            elif mouseState == MOUSE_HOVER:
                self.onMouseHover(mousePos)

        if mouseState == MOUSE_HELD_DOWN:
            self.onHeldDown(mousePos)

        elif mouseState == MOUSE_CLICK_UP:
            # Your event model is frame-polled. The original game processes WM_MOUSEMOVE
            # events before WM_LBUTTONUP. Without this, the clone can release using a stale
            # offset and commit 0 when the real game commits 1.
            if self.mouseDown:
                self.onHeldDown(mousePos)

            self.onMouseUp(mousePos)

    def convertToPosition(self, other, mode="floor"):
        if isinstance(other, tuple):
            other = Point(other)

        if isinstance(other, int):
            if mode == "floor":
                return other // self.chuzSize.x
            if mode == "nearest":
                return round(float(other) / float(self.chuzSize.x))

        if isinstance(other, Point):
            if mode == "floor":
                return other // self.chuzSize

            if mode == "nearest":
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

        return Point()

    def convertFromPosition(self, other, direction=0):
        if isinstance(other, tuple):
            other = Point(other)

        if isinstance(other, Point):
            return other * self.chuzSize

        if isinstance(other, (int, float)):
            if direction == 0:
                return other * self.chuzSize.x
            return other * self.chuzSize.y

        raise TypeError("Unsupported position type")

    def _local_from_screen(self, mousePos) -> Point:
        return Point(mousePos) - self.boardOffset

    def _in_local_board(self, local: Point) -> bool:
        return (
            0 <= local.x < self.boardSizePx.x and
            0 <= local.y < self.boardSizePx.y
        )

    def _clamp_cell(self, cell: Point) -> Point:
        return Point(
            max(0, min(int(cell.x), int(self.chuzCount.x) - 1)),
            max(0, min(int(cell.y), int(self.chuzCount.y) - 1)),
        )

    def _axis_cell_px(self, direction: int) -> float:
        if direction == self.DIR_ROW_HORIZONTAL:
            return float(self.chuzSize.x)
        return float(self.chuzSize.y)

    def _axis_board_px(self, direction: int) -> float:
        if direction == self.DIR_ROW_HORIZONTAL:
            return float(self.boardSizePx.x)
        return float(self.boardSizePx.y)

    def _section_count_for_direction(self, direction: int) -> int:
        if direction == self.DIR_ROW_HORIZONTAL:
            return int(self.chuzCount.y)
        return int(self.chuzCount.x)

    def _selected_section_from_original_click(self, direction: int) -> int:
        if direction == self.DIR_ROW_HORIZONTAL:
            return int(self.originalMouseDownCell.y)
        return int(self.originalMouseDownCell.x)

    def _normalize_section(self, section: int, direction: int) -> int:
        count = self._section_count_for_direction(direction)
        return int(section) % count

    def _clear_all_motion_offsets(self):
        for chuzzle in self.chuzzles:
            chuzzle.offset.clear()
            chuzzle.last = chuzzle.offset.copy()
            chuzzle.delta.clear()
            chuzzle.velocity.clear()
            chuzzle.motionSpeed = 0.0
            chuzzle.motionSpeedSmoothed = 0.0

    def _clear_offsets_for_dragged_sections(self):
        if self.curDir not in [0, 1]:
            return

        for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir):
            chuzzle.offset.clear()
            chuzzle.last = chuzzle.offset.copy()
            chuzzle.delta.clear()
            chuzzle.velocity.clear()
            chuzzle.motionSpeed = 0.0
            chuzzle.motionSpeedSmoothed = 0.0

    def _current_dragged_chuzzles(self):
        if self.curDir not in [0, 1] or not self.sectsToMove:
            return []
        return list(self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir))

    def _current_drag_offset_pixels(self) -> float:
        moved = self._current_dragged_chuzzles()
        if not moved:
            return 0.0

        ref = moved[0]
        if self.curDir == self.DIR_ROW_HORIZONTAL:
            return float(ref.offset.x)

        return float(ref.offset.y)

    def _set_current_drag_offset_pixels(self, value: float, clamp_info=None):
        moved = self._current_dragged_chuzzles()

        for chuzzle in moved:
            if self.curDir == self.DIR_ROW_HORIZONTAL:
                chuzzle.offset.x = value
                chuzzle.offset.y = 0
            else:
                chuzzle.offset.y = value
                chuzzle.offset.x = 0

            if clamp_info is not None and clamp_info.get("clamped"):
                chuzzle.stopMotionKeepOffset()
            else:
                chuzzle.calculateVelocity(self.curDir)

    def _choose_dir_from_frame_delta(self, dx: float, dy: float):
        ax = abs(dx)
        ay = abs(dy)

        if ax == 0 and ay == 0:
            return None

        if ax > ay:
            return self.DIR_ROW_HORIZONTAL

        if ay > ax:
            return self.DIR_COLUMN_VERTICAL

        self._dirTieFlip = not self._dirTieFlip
        return self.DIR_ROW_HORIZONTAL if self._dirTieFlip else self.DIR_COLUMN_VERTICAL

    def _should_switch_direction(self, frame_dx: float, frame_dy: float) -> bool:
        if self.curDir not in [0, 1]:
            return False

        active_offset = abs(self._current_drag_offset_pixels())

        if active_offset >= self.SWITCH_ACTIVE_OFFSET_EPSILON:
            return False

        ax = abs(frame_dx)
        ay = abs(frame_dy)

        if self.curDir == self.DIR_ROW_HORIZONTAL:
            return ax < abs(ay / self.SWITCH_PERPENDICULAR_DIVISOR)

        if self.curDir == self.DIR_COLUMN_VERTICAL:
            return ay < abs(ax / self.SWITCH_PERPENDICULAR_DIVISOR)

        return False

    def _configure_default_drag_limits(self, direction: int):
        cell_px = self._axis_cell_px(direction)
        self.dragMinPx = self.NORMAL_MIN_CELLS * cell_px
        self.dragMaxPx = self.NORMAL_MAX_CELLS * cell_px

    def _configure_locked_drag_limits(self, direction: int):
        cell_px = self._axis_cell_px(direction)
        self.dragMinPx = self.LOCKED_MIN_CELLS * cell_px
        self.dragMaxPx = self.LOCKED_MAX_CELLS * cell_px

    def _clamp_offset_to_drag_limits(self, offset_pixels: float) -> float:
        return max(min(float(offset_pixels), float(self.dragMaxPx)), float(self.dragMinPx))

    def _begin_drag_from_mouse_down_cell(self, direction: int):
        """
        Begin a row/column drag from the original clicked cell.

        Row drag uses original click Y.
        Column drag uses original click X.

        This prevents diagonal jitter from changing which row/column is being moved.
        """

        self.curDir = direction
        self.lastDir = direction
        self.dragging = True

        self.mouseDownPos = self.originalMouseDownCell.copy()

        if direction == self.DIR_ROW_HORIZONTAL:
            self.activeSelectedSection = int(self.originalMouseDownCell.y)
        elif direction == self.DIR_COLUMN_VERTICAL:
            self.activeSelectedSection = int(self.originalMouseDownCell.x)
        else:
            self.activeSelectedSection = -1

        self._configure_default_drag_limits(direction)
        self._select_sections_for_drag(self.activeSelectedSection)

    def _switch_drag_direction_from_mouse_down_cell(self, new_direction: int):
        """
        Switch row/column orientation without changing the selected row/column anchor.

        This keeps the drag tied to the cell where the mouse was originally pressed.
        """

        if new_direction not in [0, 1] or new_direction == self.curDir:
            return

        self._clear_offsets_for_dragged_sections()
        self._begin_drag_from_mouse_down_cell(new_direction)

    def _select_sections_for_drag(self, selected_section: int):
        selected_section = self._normalize_section(selected_section, self.curDir)

        self.isFatTL = any(
            chuzzle.fatTL
            for chuzzle in self.chuzzles.getDir(selected_section, self.curDir)
        )

        self.isFatBR = any(
            chuzzle.fatBR
            for chuzzle in self.chuzzles.getDir(selected_section, self.curDir)
        )

        if self.isFatTL or self.isFatBR:
            self.sectsToMove = self.recurseMain(selected_section)
            if selected_section not in self.sectsToMove:
                self.sectsToMove.append(selected_section)
        else:
            self.sectsToMove = [selected_section]

        self.sectsToMove = sorted(
            set(self._normalize_section(v, self.curDir) for v in self.sectsToMove)
        )

        self.isLocked = any(
            chuzzle.locked
            for chuzzle in self.chuzzles.getMultipleDirs(self.sectsToMove, self.curDir)
        )

        if self.isLocked:
            self._configure_locked_drag_limits(self.curDir)

    def recurseMain(self, sectNumber, debug=False):
        section_count = self._section_count_for_direction(self.curDir)

        def recurse_right(_sectNumber, _sects=None, _skipList=None):
            if _skipList is None:
                _skipList = []
            if _sects is None:
                _sects = []

            should_recurse = False

            for chuzzle in self.chuzzles.getDir(Point(_sectNumber), self.curDir):
                if not chuzzle.fatTL:
                    continue

                if chuzzle.position in _skipList:
                    continue

                _skipList.append(chuzzle.position)
                _skipList.append((chuzzle.position + (1, 1)) % self.game.settings.boardSize)

                sect_a = self._normalize_section(chuzzle.position.get(self.curDir), self.curDir)
                sect_b = self._normalize_section(sect_a + 1, self.curDir)

                _sects.append(sect_a)
                _sects.append(sect_b)
                should_recurse = True

            if should_recurse:
                return recurse_right((_sectNumber + 1) % section_count, _sects, _skipList)

            return _sects, _skipList

        def recurse_left(_sectNumber, _sects=None, _skipList=None):
            if _skipList is None:
                _skipList = []
            if _sects is None:
                _sects = []

            should_recurse = False

            for chuzzle in self.chuzzles.getDir(Point(_sectNumber), self.curDir):
                if not chuzzle.fatBR:
                    continue

                if chuzzle.position in _skipList:
                    continue

                _skipList.append(chuzzle.position)

                new_pos = chuzzle.position + (-1, -1)
                new_pos.x %= self.chuzCount.x
                new_pos.y %= self.chuzCount.y
                _skipList.append(new_pos)

                sect_a = self._normalize_section(chuzzle.position.get(self.curDir), self.curDir)
                sect_b = self._normalize_section(sect_a - 1, self.curDir)

                _sects.append(sect_a)
                _sects.append(sect_b)
                should_recurse = True

            if should_recurse:
                return recurse_left((_sectNumber - 1) % section_count, _sects, _skipList)

            return _sects, _skipList

        sects_r, _ = recurse_right(sectNumber, [], [])
        sects_l, _ = recurse_left(sectNumber, [], [])

        sects = sorted(set(int(i) for i in (sects_r + sects_l)))

        if debug:
            print("recursion", sects_r, sects_l, sects)

        return sects

    def testChuzzleStates(self):
        if self.curDir not in [0, 1]:
            self.isFatTL = False
            self.isFatBR = False
            self.isLocked = False
            self.sectsToMove = []
            self.activeSelectedSection = -1
            return

        if self.activeSelectedSection < 0:
            self.activeSelectedSection = self._selected_section_from_local(
                self.originalMouseDownPos,
                self.curDir
            )

        self._select_sections_for_drag(self.activeSelectedSection)

    def onMouseHover(self, mousePos):
        local = self._local_from_screen(mousePos)
        self.centerPos = Point(local)
        self.mouseDownPos = self._clamp_cell(self.convertToPosition(local))

        pos = Point(self.mouseDownPos)
        chuzzle = self.chuzzles.getPos(pos)

        if chuzzle is not None and pos == (0, 0):
            chuzzle.setEyeBallDirection(self.centerPos)

    def onRightClickDown(self, mousePos):
        local = self._local_from_screen(mousePos)

        if not self._in_local_board(local):
            print("[RIGHT CLICK] outside board")
            return

        cell_pos = self._clamp_cell(self.convertToPosition(local, mode="floor"))
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
        if self.mouseDown:
            return

        local = self._local_from_screen(mousePos)

        self.mouseDown = True
        self.dragging = False
        self.inBoundsDuringClickDown = self.inBounds

        self.initialMouseDown = Point(local)
        self.currentMouseDown = Point(local)
        self.originalMouseDownPos = Point(local)
        self.originalMouseDownCell = self._clamp_cell(self.convertToPosition(local, mode="floor"))

        self.mouseDownPos = self.originalMouseDownCell.copy()

        self.centerPos = Point(local)
        self.pointPos = Point(local)
        self.lastPos = Point(local)

        self.curDir = -1
        self.lastDir = -1

        self.isLocked = False
        self.isFatTL = False
        self.isFatBR = False
        self.sectsToMove = []
        self.activeSelectedSection = -1

        self.movedMouse = False
        self.cursorDragging = False
        self._dirTieFlip = False

        self.dragMinPx = 0.0
        self.dragMaxPx = 0.0

        self._clear_all_motion_offsets()

    def onHeldDown(self, mousePos):
        if not self.mouseDown:
            return

        if not self.inBoundsDuringClickDown:
            return

        self.currentMouseDown = self._local_from_screen(mousePos)
        self.pointPos = Point(self.currentMouseDown)

        frame_dx = self.pointPos.x - self.lastPos.x
        frame_dy = self.pointPos.y - self.lastPos.y

        if frame_dx != 0 or frame_dy != 0:
            self.movedMouse = True

            if not self.cursorDragging:
                self.game.cursorIndex = 4
                self.cursorDragging = True

        candidate_dir = self._choose_dir_from_frame_delta(frame_dx, frame_dy)

        if self.curDir == -1:
            if candidate_dir is None:
                self.centerPos = Point(self.pointPos)
                self.lastPos = Point(self.pointPos)
                return

            # Initial drag begins from the mouse-down coordinate, same as 0040ec10
            # calling 0040c3b0/0040c5f0 with the click coordinates.
            self._begin_drag_from_mouse_down_cell(candidate_dir)

        elif candidate_dir is not None and candidate_dir != self.curDir:
            if self._should_switch_direction(frame_dx, frame_dy):
                # Direction switch uses this+0x6dc / this+0x6e0 in the decomp.
                # Those are current board-local mouse coordinates, not original click.
                self._switch_drag_direction_from_mouse_down_cell(candidate_dir)

        if self.curDir not in [0, 1]:
            self.centerPos = Point(self.pointPos)
            self.lastPos = Point(self.pointPos)
            return

        if self.curDir == self.DIR_ROW_HORIZONTAL:
            frame_axis_delta = frame_dx
        else:
            frame_axis_delta = frame_dy

        old_offset = self._current_drag_offset_pixels()
        proposed_offset = old_offset + frame_axis_delta
        proposed_offset = self._clamp_offset_to_drag_limits(proposed_offset)

        clamp_info = None

        if self.chuzzles.sectionHasFat(self.sectsToMove, self.curDir):
            proposed_offset, clamp_info = self.chuzzles.clampSectionOffsetPixels(
                sects=self.sectsToMove,
                direction=self.curDir,
                offset_pixels=proposed_offset,
            )

            if clamp_info["clamped"]:
                self.chuzzles.onFatEdgeClamp(clamp_info)

        self._set_current_drag_offset_pixels(proposed_offset, clamp_info=clamp_info)

        self.centerPos = Point(self.pointPos)
        self.lastPos = Point(self.pointPos)

    def _release_amount_from_offset(self, offset_pixels: float, cell_pixels: float) -> int:
        """
        Matches the release math in 00418c70 more closely:

            positive:
                shift = (round(offset) + half_cell) / cell

            negative:
                shift = (round(offset) - half_cell) / cell

        The important difference from the old clone code is that the original rounds
        the pixel offset to an integer before applying the half-cell bias.
        """

        if cell_pixels == 0:
            return 0

        half_cell = cell_pixels * 0.5

        if offset_pixels > 0:
            rounded_offset = int(math.floor(offset_pixels + 0.5))
            return int((rounded_offset + half_cell) / cell_pixels)

        if offset_pixels < 0:
            rounded_offset = int(math.ceil(offset_pixels - 0.5))
            return int((rounded_offset - half_cell) / cell_pixels)

        return 0

    def onMouseUp(self, mousePos):
        committed_move = None

        active_cur_dir = self.curDir
        active_sects = list(self.sectsToMove)
        active_selected_section = int(self.activeSelectedSection)

        moved_chuzzles = []

        if self.mouseDown and self.movedMouse and active_cur_dir in [0, 1]:
            moved_chuzzles = list(self.chuzzles.getMultipleDirs(active_sects, active_cur_dir))

        requested_amount = 0

        if moved_chuzzles:
            ref = moved_chuzzles[0]

            if active_cur_dir == self.DIR_ROW_HORIZONTAL:
                requested_amount = self._release_amount_from_offset(
                    float(ref.offset.x),
                    float(self.chuzSize.x),
                )
            elif active_cur_dir == self.DIR_COLUMN_VERTICAL:
                requested_amount = self._release_amount_from_offset(
                    float(ref.offset.y),
                    float(self.chuzSize.y),
                )

        applied_amount = 0

        if active_cur_dir in [0, 1] and moved_chuzzles and requested_amount != 0:
            if self.chuzzles.sectionHasFat(active_sects, active_cur_dir):
                applied_amount, clamp_info = self.chuzzles.clampSectionShiftCells(
                    sects=active_sects,
                    direction=active_cur_dir,
                    amount=requested_amount,
                )

                if clamp_info["clamped"]:
                    self.chuzzles.onFatEdgeClamp(clamp_info)

                if applied_amount != 0:
                    self.chuzzles.applySectionShiftMixed(
                        sects=active_sects,
                        direction=active_cur_dir,
                        amount=applied_amount,
                    )
            else:
                applied_amount = requested_amount
                self.chuzzles.applySectionShiftWrap(
                    sects=active_sects,
                    direction=active_cur_dir,
                    amount=applied_amount,
                )

        self._clear_all_motion_offsets()

        if self.movedMouse:
            self.game.sounds.playSound(self.game.sounds.Click)

        if applied_amount != 0:
            committed_move = {
                "axis": "R" if active_cur_dir == self.DIR_ROW_HORIZONTAL else "C",
                "selected_line": active_selected_section,
                "displacement": int(applied_amount),
                "sects_to_move": [int(v) for v in active_sects],
            }

        self.mouseDown = False
        self.dragging = False
        self.inBoundsDuringClickDown = False

        self.cursorDragging = False
        self.game.cursorIndex = 0

        self.curDir = -1
        self.lastDir = -1

        self.isLocked = False
        self.isFatTL = False
        self.isFatBR = False
        self.sectsToMove = []
        self.activeSelectedSection = -1

        self.dragMinPx = 0.0
        self.dragMaxPx = 0.0

        self.movedMouse = False
        self._dirTieFlip = False

        if committed_move is not None and hasattr(self.menu, "onBoardMoveCommitted"):
            self.menu.onBoardMoveCommitted(committed_move)

    def _get_drag_span_bounds(self):
        if not self.sectsToMove:
            return None

        min_sect = min(self.sectsToMove)
        max_sect = max(self.sectsToMove)

        if self.curDir == self.DIR_ROW_HORIZONTAL:
            x = 0
            y = min_sect * self.chuzSize.y
            w = self.boardSizePx.x
            h = (max_sect - min_sect + 1) * self.chuzSize.y
            return p.Rect(int(x), int(y), int(w), int(h))

        if self.curDir == self.DIR_COLUMN_VERTICAL:
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

        if self.curDir == self.DIR_ROW_HORIZONTAL:
            base = self.DragRainbowH
            target_size = (span_rect.width, span_rect.height)
        else:
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

    def _active_drag_opacity_offset(self) -> float:
        if self.curDir not in [0, 1]:
            return 0.0

        return abs(self._current_drag_offset_pixels())

    def render(self, screen):
        self.surface.fill((0, 0, 0, 0))

        board_w = self.boardSizePx.x
        board_h = self.boardSizePx.y

        if self.mouseDown and self.curDir in [0, 1] and self.sectsToMove:
            texture, span_rect = self._build_drag_rainbow_surface()

            if texture is not None and span_rect is not None:
                opacity = 154
                active_offset = self._active_drag_opacity_offset()
                cell_px = self._axis_cell_px(self.curDir)

                if active_offset < cell_px * 2:
                    opacity *= active_offset / float(cell_px * 2)

                blit_alpha(
                    self.surface,
                    texture,
                    self._surface_blit_pos(Point(span_rect.x, span_rect.y)),
                    opacity,
                )

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
            self.surface.blit(surf, main_coord.tuple())

            if chuzzle.isNormal():
                self.surface.blit(surf, (main_coord.x - board_w, main_coord.y))
                self.surface.blit(surf, (main_coord.x + board_w, main_coord.y))
                self.surface.blit(surf, (main_coord.x, main_coord.y - board_h))
                self.surface.blit(surf, (main_coord.x, main_coord.y + board_h))

        screen.blit(self.surface, (self.boardOffset - self.surfacePad).tuple())
        screen.blit(self.textures.Border, (229, 29))

        lock_texture, lock_texture_fat = self._get_lock_textures()

        for chuzzle in self.chuzzles:
            if not chuzzle.locked:
                continue

            coord = Point((chuzzle.getX(), chuzzle.getY()))

            if chuzzle.isNormal():
                coords = coord - 25 * (lock_texture.get_size()[0] / 128)
                screen.blit(lock_texture, (coords + self.boardOffset).tuple())

            elif chuzzle.fatTL:
                coords = coord - 25 * (lock_texture_fat.get_size()[0] / 128) - 10
                screen.blit(lock_texture_fat, (coords + self.boardOffset).tuple())