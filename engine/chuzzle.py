# Jerrin Shirks

# native imports
import random as r
import math
from copy import copy
import pygame as p

# custom imports
from tools.point import Point
from tools.game import Game

class Chuzzle:
    def __init__(self, game: Game, position, version="normal", color=None, mini=False):

        self.game = game
        self.textures = game.textures
        self.fps = game.fps

        # position
        self.direction = 0
        self.position = position

        self.last = Point()
        self.offset = Point()
        self.delta = Point()
        self.velocity = Point()
        self.motionSpeed = 0.0
        self.motionSpeedSmoothed = 0.0

        # properties
        self.version = version
        self.super = False
        self.naked = False
        self.fatTL = False
        self.fatTR = False
        self.fatBL = False
        self.fatBR = False
        self.locked = False

        if color is None:
            self.color = r.randint(0, 8)
        else:
            self.color = color

        # facial features
        self.eyeballShape = r.randint(0, 3)
        self.eyeballColor = r.randint(0, 5)
        self.leftEyeDir = Point((0.0, 0.0))
        self.rightEyeDir = Point((0.0, 0.0))
        self.mouthType = r.randint(0, 15)
        self.superPulse = []

        # acceleration frames

        # swaying animation, at 60 fps
        self.swayFrameCap = 20
        self.swayFrame = r.randint(0, self.swayFrameCap)
        self.swaySubFrameCap = 4
        self.swaySubFrame = r.randint(0, self.swaySubFrameCap)

        # deals with blinking
        self.waitToBlinkFrame = 0
        self.waitToBlinkFrameCap = r.randint(1 * self.fps, 25 * self.fps)
        self.blinkingFrame = 0
        self.blinkingFrameCap = r.randint(3, 10)

        # deals with hovering over the chuzzle motionless
        self.fallAsleepTimer = 0

    def copy(self):
        return copy(self)

    def isNormal(self):
        if not self.fatTL and not self.fatTR and not self.fatBR and not self.fatBL:
            return True
        else:
            return False

    def getX(self):
        x = self.game.currentMenu().board.convertFromPosition(self.position).x + self.offset.x
        return x

    def getY(self):
        y = self.game.currentMenu().board.convertFromPosition(self.position).y + self.offset.y
        return y

    def getXY(self):
        x = self.getX()
        y = self.getY()
        return Point((x, y))

    def changeEyeDirections(self):
        pass

    def endSwayAnim(self):
        self.swayFrame = 0
        self.swaySubFrame = 0

    def calculateVelocity(self, direction):
        self.delta = self.offset - self.last
        self.last = self.offset.copy()
        self.velocity = self.delta.copy()

        raw_speed = 0.0
        if direction == 0:
            raw_speed = abs(self.delta.y)
            if self.delta.y > 0:
                self.direction = 0  # down
            elif self.delta.y < 0:
                self.direction = 1  # up

        elif direction == 1:
            raw_speed = abs(self.delta.x)
            if self.delta.x > 0:
                self.direction = 2  # right
            elif self.delta.x < 0:
                self.direction = 3  # left

        self.motionSpeed = raw_speed
        # Smoothing reduces flicker between neighboring velocity frames.
        self.motionSpeedSmoothed = (self.motionSpeedSmoothed * 0.65) + (raw_speed * 0.35)

    def isFatCorner(self):
        return any([self.fatBL, self.fatTR, self.fatBR])

    def setFatTL(self):
        self.fatTL = True
        self.swaySubFrame = 0
        self.swaySubFrameCap = 12
        self.swayFrame = 0
        self.swayFrameCap = 10

    def delFatTL(self):
        self.fatTL = False
        self.swaySubFrame = 0
        self.swaySubFrameCap = 6
        self.swayFrame = 0
        self.swayFrameCap = 20

    # def setEyeBallDirection(self, mousePos):
    #     dx = self.getX() + 0.5 * self.game.currentMenu().board.chuzSize.x - mousePos.x
    #     dy = self.getY() + 0.5 * self.game.currentMenu().board.chuzSize.y - mousePos.y
    #     rad_angle = math.atan2(dy, dx)
    #     deg_angle = math.degrees(rad_angle)
    #     if deg_angle < 0:
    #         deg_angle = 180 + (180 + deg_angle)
# 
    #     # print(mousePos, dx, dy, deg_angle)
    #     # starts at W goes clockwise from 0 to 360

    def setEyeBallDirection(self, mousePos):
        board = self.game.currentMenu().board

        # Mouse position relative to the board surface, not the whole screen.
        local_mouse = Point(mousePos) - board.boardOffset

        cell_top_left = self.getXY()
        cell_size = board.chuzSize.copy()

        # Fat main occupies a 2x2 area.
        if self.fatTL:
            cell_size = cell_size * 2

        cell_rect = p.Rect(
            int(cell_top_left.x),
            int(cell_top_left.y),
            int(cell_size.x),
            int(cell_size.y)
        )

        # If the mouse is outside this chuzzle's cell area, keep the previous direction.
        if not cell_rect.collidepoint(int(local_mouse.x), int(local_mouse.y)):
            return self.leftEyeDir

        center_x = cell_top_left.x + cell_size.x * 0.5
        center_y = cell_top_left.y + cell_size.y * 0.5

        half_w = max(1.0, cell_size.x * 0.5)
        half_h = max(1.0, cell_size.y * 0.5)

        # Convert mouse position inside this cell to [-1, 1] range.
        dx = (local_mouse.x - center_x) / half_w
        dy = (local_mouse.y - center_y) / half_h

        dx = max(-1.0, min(1.0, dx))
        dy = max(-1.0, min(1.0, dy))

        direction = Point((dx, dy))
        self.leftEyeDir = direction
        self.rightEyeDir = direction
        return direction

    def _blit_accessories(self, surface, swayOffset):
        if surface is None:
            return

        eyes = self.textures.Eyes[self.eyeballShape]
        pupils = self.textures.Pupils[self.eyeballColor]

        mouse_pos = Point(p.mouse.get_pos())
        self.setEyeBallDirection(mouse_pos)

        if self.fatTL:
            eyes_pos = Point((42 + 5, 64 + 14 + swayOffset))

            left_center = Point((31 + 5, 59 + 4))
            right_center = Point((48 + 5, 59 + 4))

            pupil_radius_x = 4.0
            pupil_radius_y = 4.0
        else:
            eyes_pos = Point((-2, -3 + swayOffset))

            left_center = Point((31 + 5, 59 + 4))
            right_center = Point((48 + 5, 59 + 4))

            pupil_radius_x = 4.0
            pupil_radius_y = 4.0

        left_pupil_pos = Point((
            eyes_pos.x + left_center.x + self.leftEyeDir.x * pupil_radius_x - pupils.get_width() / 2,
            eyes_pos.y + left_center.y + self.leftEyeDir.y * pupil_radius_y - pupils.get_height() / 2
        ))
        right_pupil_pos = Point((
            eyes_pos.x + right_center.x + self.rightEyeDir.x * pupil_radius_x - pupils.get_width() / 2,
            eyes_pos.y + right_center.y + self.rightEyeDir.y * pupil_radius_y - pupils.get_height() / 2
        ))

        surface.blit(eyes, eyes_pos.tuple())
        surface.blit(pupils, left_pupil_pos.tuple())
        surface.blit(pupils, right_pupil_pos.tuple())

        if self.game.settings.colorblindMode:
            if self.fatTL:
                index = self.textures.colorsMainBlind.get(str(self.color))
                if index is not None:
                    symbol = self.textures.ColorblindSymbols[index]
                    symbol_rect = symbol.get_rect(center=(90, 107))
                    surface.blit(symbol, symbol_rect.topleft)
            elif self.isNormal():
                index = self.textures.colorsMainBlind.get(str(self.color))
                if index is not None:
                    surface.blit(self.textures.ColorblindSymbols[index], (16, 4 + swayOffset))

    def getSurface(self):

        # sway loop
        self.swaySubFrame = (self.swaySubFrame + 1) % self.swaySubFrameCap
        if self.swaySubFrame == 0:
            self.swayFrame = (self.swayFrame + 1) % self.swayFrameCap

        is_fat_main = self.fatTL
        if not (self.fatTL or self.fatTR or self.fatBL or self.fatBR):
            sway_count = 20
            dir_count = 10
            frame_bank = self.textures.ChuzzleBase[self.color]
        elif is_fat_main:
            sway_count = 10
            dir_count = 5
            frame_bank = self.textures.ChuzzleHuge[self.color]
        else:
            return None

        # Direction index into sheet order: down, left, right, up.
        dir_to_segment = {
            0: 0,  # down
            3: 1,  # left
            2: 2,  # right
            1: 3,  # up
        }

        moving = (abs(self.offset.x) > 0.0) or (abs(self.offset.y) > 0.0)
        if moving and self.direction in dir_to_segment:
            board = self.game.currentMenu().board
            # Lower normalization target so upper cells are reachable in normal fast drags.
            max_speed = max(1.0, float(board.chuzSize.x) * 0.10)
            speed = float(self.motionSpeedSmoothed)

            # Fat movement progresses through directional frames at half rate.
            if is_fat_main:
                speed *= 0.5

            t = min(max(speed / max_speed, 0.0), 1.0)
            # Ease-out response: keeps low speed nuanced but reaches high cells sooner.
            t = math.sqrt(t)
            speed_index = int(round(t * (dir_count - 1)))
            segment = dir_to_segment[self.direction]
            frame_index = sway_count + segment * dir_count + speed_index
        else:
            self.motionSpeed = 0.0
            self.motionSpeedSmoothed = 0.0
            frame_index = self.swayFrame % sway_count

        surface = frame_bank[frame_index].copy()

        # deals with the chuzzle swaying moving the eyes etc.
        swayOffset = 0
        if self.isNormal() and frame_index < sway_count:
            if self.swayFrame in [10, 11, 12, 13, 14]:
                swayOffset = 0.5

        self._blit_accessories(surface, swayOffset)

        return surface

    def snapToBoard(self):
        self.offset.clear()
        self.delta.clear()
        self.velocity.clear()
        self.motionSpeed = 0.0
        self.motionSpeedSmoothed = 0.0
        self.last = self.offset.copy()

    def incrementAnimation(self):
        pass
