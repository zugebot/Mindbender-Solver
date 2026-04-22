# Jerrin Shirks

# native imports
import json
import random
from typing import List

# custom imports
from tools.point import Point
from tools.game import Game
from engine.chuzzle import Chuzzle


class ChuzzleList(List[Chuzzle]):
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
        if isinstance(args[0], Point):
            point = Point(args[0])
        else:
            point = Point(*args)

        point.x %= self.chuzCount.x
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

    def applySectionShiftWrap(self, sects=None, direction=0, amount=0):
        if sects is None:
            sects = []

        targets = list(self.getMultipleDirs(sects=sects, direction=direction))

        for chuzzle in targets:
            if direction == 1:
                chuzzle.position.x = (chuzzle.position.x + amount) % self.chuzCount.x
            elif direction == 0:
                chuzzle.position.y = (chuzzle.position.y + amount) % self.chuzCount.y

    def placeFatChuzzle(self, x=0, y=0):
        if x >= self.chuzCount.x or y >= self.chuzCount.y:
            return
        chuzzle: Chuzzle = self.getPos((x, y))
        if chuzzle is None:
            return

        chuzzle.setFatTL()

        # this should remove fats from corners of new fat?
        for i in [(0, 1), (1, 0), (1, 1)]:
            point = Point(x, y) + i
            _chuzzle = self.getPos(point.tuple())
            if _chuzzle is None:
                continue
            if _chuzzle.fatTL:
                self.deleteFatChuzzle(point.x, point.y)

        tr = self.getPos(x + 1, y)
        if tr is not None:
            tr.fatTR = True
        bl = self.getPos(x, y + 1)
        if bl is not None:
            bl.fatBL = True
        br = self.getPos(x + 1, y + 1)
        if br is not None:
            br.fatBR = True

    def deleteFatChuzzle(self, x=0, y=0):
        if x >= self.chuzCount.x or y >= self.chuzCount.y:
            return
        tl = self.getPos(x, y)
        if tl is not None:
            tl.delFatTL()
        tr = self.getPos(x + 1, y)
        if tr is not None:
            tr.fatTR = False
        bl = self.getPos(x, y + 1)
        if bl is not None:
            bl.fatBL = False
        br = self.getPos(x + 1, y + 1)
        if br is not None:
            br.fatBR = False
        
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
        targets = list(self.getMultipleDirs(sects=[sect], direction=direction))

        for chuzzle in targets:
            if direction == 1:
                chuzzle.position.x = (chuzzle.position.x + amount) % self.chuzCount.x
            elif direction == 0:
                chuzzle.position.y = (chuzzle.position.y + amount) % self.chuzCount.y

    def randomMoves(self, amount):
        for count in range(amount):
            direction = self.random.randint(0, 1)
            if direction == 0:
                sect = self.random.randint(0, self.chuzCount.y - 1)
                amount = self.random.randint(1, self.chuzCount.x - 1)
                self.shiftSection(sect, direction, amount)
            else:
                sect = self.random.randint(0, self.chuzCount.x - 1)
                amount = self.random.randint(1, self.chuzCount.y - 1)
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

        seen = set()

        for sect in sects:
            for chuzzle in self.getDir(sect, direction):
                key = id(chuzzle)
                if key in seen:
                    continue
                seen.add(key)
                yield chuzzle

    def checkForLocks(self, direction, point=Point()):
        chuzList = []
        for chuzzle in self:
            if direction == 1:
                if chuzzle.position.y == point.y:
                    chuzList.append(chuzzle)
            elif direction == 0:
                if chuzzle.position.x == point.x:
                    chuzList.append(chuzzle)

        for chuzzle in chuzList:
            if chuzzle.locked:
                return True
        return False

    def _fat_anchor_from_chuzzle(self, chuzzle: Chuzzle):
        if chuzzle is None:
            return None
        if chuzzle.fatTL:
            return chuzzle.position.copy()
        if chuzzle.fatTR:
            return chuzzle.position + (-1, 0)
        if chuzzle.fatBL:
            return chuzzle.position + (0, -1)
        if chuzzle.fatBR:
            return chuzzle.position + (-1, -1)
        return None

    def getFatAnchorsInMultipleDirs(self, sects=None, direction=0):
        if sects is None:
            sects = []

        anchors = []
        seen = set()

        for chuzzle in self.getMultipleDirs(sects, direction):
            anchor = self._fat_anchor_from_chuzzle(chuzzle)
            if anchor is None:
                continue

            key = (int(anchor.x), int(anchor.y))
            if key not in seen:
                seen.add(key)
                anchors.append(anchor)

        return anchors

    def sectionHasFat(self, sects=None, direction=0):
        return len(self.getFatAnchorsInMultipleDirs(sects, direction)) > 0

    def clampSectionOffsetPixels(self, sects=None, direction=0, offset_pixels=0):
        """
        direction == 1 -> moving rows horizontally -> clamp x offset
        direction == 0 -> moving cols vertically   -> clamp y offset
        """
        if sects is None:
            sects = []

        anchors = self.getFatAnchorsInMultipleDirs(sects, direction)
        if not anchors:
            return offset_pixels, {
                "clamped": False,
                "requested": offset_pixels,
                "applied": offset_pixels,
                "direction": direction,
                "sects": list(sects),
                "anchors": [],
            }

        if direction == 1:
            cell_px = self.game.currentMenu().board.chuzSize.x
            min_allowed = max(-anchor.x * cell_px for anchor in anchors)
            max_allowed = min((self.chuzCount.x - 2 - anchor.x) * cell_px for anchor in anchors)
        else:
            cell_px = self.game.currentMenu().board.chuzSize.y
            min_allowed = max(-anchor.y * cell_px for anchor in anchors)
            max_allowed = min((self.chuzCount.y - 2 - anchor.y) * cell_px for anchor in anchors)

        applied = max(min(offset_pixels, max_allowed), min_allowed)

        return applied, {
            "clamped": applied != offset_pixels,
            "requested": offset_pixels,
            "applied": applied,
            "direction": direction,
            "sects": list(sects),
            "anchors": [(int(a.x), int(a.y)) for a in anchors],
            "min_allowed": min_allowed,
            "max_allowed": max_allowed,
        }

    def clampSectionShiftCells(self, sects=None, direction=0, amount=0):
        """
        direction == 1 -> moving rows horizontally -> clamp x cell shift for fat pieces only
        direction == 0 -> moving cols vertically   -> clamp y cell shift for fat pieces only
        """
        if sects is None:
            sects = []

        anchors = self.getFatAnchorsInMultipleDirs(sects, direction)
        if not anchors:
            return amount, {
                "clamped": False,
                "requested": amount,
                "applied": amount,
                "direction": direction,
                "sects": list(sects),
                "anchors": [],
            }

        if direction == 1:
            min_allowed = max(-anchor.x for anchor in anchors)
            max_allowed = min((self.chuzCount.x - 2 - anchor.x) for anchor in anchors)
        else:
            min_allowed = max(-anchor.y for anchor in anchors)
            max_allowed = min((self.chuzCount.y - 2 - anchor.y) for anchor in anchors)

        applied = max(min(amount, max_allowed), min_allowed)

        return applied, {
            "clamped": applied != amount,
            "requested": amount,
            "applied": applied,
            "direction": direction,
            "sects": list(sects),
            "anchors": [(int(a.x), int(a.y)) for a in anchors],
            "min_allowed": min_allowed,
            "max_allowed": max_allowed,
        }

    def applySectionShiftNoWrap(self, sects=None, direction=0, amount=0):
        if sects is None:
            sects = []

        targets = list(self.getMultipleDirs(sects=sects, direction=direction))

        for chuzzle in targets:
            if direction == 1:
                chuzzle.position.x += amount
            elif direction == 0:
                chuzzle.position.y += amount

    def applySectionShiftMixed(self, sects=None, direction=0, amount=0):
        """
        Fat pieces move bounded.
        Normal chuzzles in the same moved row/column still wrap.
        This is the piece your previous code was missing.
        """
        if sects is None:
            sects = []

        targets = list(self.getMultipleDirs(sects=sects, direction=direction))

        for chuzzle in targets:
            is_fat_piece = chuzzle.fatTL or chuzzle.fatTR or chuzzle.fatBL or chuzzle.fatBR

            if direction == 1:
                if is_fat_piece:
                    chuzzle.position.x += amount
                else:
                    chuzzle.position.x = (chuzzle.position.x + amount) % self.chuzCount.x
            elif direction == 0:
                if is_fat_piece:
                    chuzzle.position.y += amount
                else:
                    chuzzle.position.y = (chuzzle.position.y + amount) % self.chuzCount.y

    def onFatEdgeClamp(self, clamp_info):
        pass