# Jerrin Shirks

# native imports
import math
import random as r
import time
from copy import copy

import pygame as p

# custom imports
from tools.point import Point
from tools.game import Game


class Chuzzle:
    """
    Chuzzle eye/render behavior modeled from the decomp.

    Relevant original functions:
        00404560  normal draw path with eyes and pupils
        00404a70  mini/simple draw path with eyes and pupils
        0041f290  stateful draw path, sleepy/angry/special eye branches
        00422390  per-frame state update, eye timer countdown
        00455f60  random integer in range
        00467ae0  angle -> vector, x = cos(angle), y = -sin(angle)

    Important decomp-backed behavior:
        - normal eye timer is random(60000) + 30000 ms
        - when timer reaches 200, sleepy/closed-eye state is armed
        - sleepy/closed-eye state uses SleepyEyes texture, not squashed Eyes texture
        - pupil travel and center tables come from DAT_00529628 through DAT_00529687
        - mouse tracking only occurs while the mouse is inside this chuzzle's own cell rect
    """

    EYE_PUPIL_TRAVEL = [
        (4.0, 5.0),
        (5.0, 4.0),
        (4.0, 4.0),
        (5.0, 4.0),
    ]

    EYE_PUPIL_CENTERS = [
        ((34.0, 60.0), (53.0, 62.0)),
        ((33.0, 62.0), (53.0, 63.0)),
        ((34.0, 63.0), (52.0, 62.0)),
        ((35.0, 62.0), (54.0, 62.0)),
    ]

    NORMAL_EYES_POS = Point((-2.0, -3.0))

    FAT_EYES_POS = Point((42.0, 64.0))
    FAT_LEFT_PUPIL_CENTER = Point((70.0, 95.0))
    FAT_RIGHT_PUPIL_CENTER = Point((115.0, 95.0))
    FAT_PUPIL_TRAVEL = Point((9.0, 9.0))

    NORMAL_BLINK_MIN_MS = 3000
    NORMAL_BLINK_RANGE_MS = 7000
    NORMAL_BLINK_FRAMES = 5

    EXCITED_BLINK_MIN_MS = 180
    EXCITED_BLINK_RANGE_MS = 420
    EXCITED_BLINK_FRAMES = 4

    NORMAL_IDLE_LOOK_MIN_MS = 5000
    NORMAL_IDLE_LOOK_RANGE_MS = 9000

    EXCITED_IDLE_LOOK_MIN_MS = 250
    EXCITED_IDLE_LOOK_RANGE_MS = 650

    MOUSE_TRACK_LERP_PER_SECOND = 32.0
    IDLE_TRACK_LERP_PER_SECOND = 4.5
    EXCITED_TRACK_LERP_PER_SECOND = 16.0

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
        self.eyeTargetDir = Point((0.0, 0.0))

        # Idle look direction.
        # This is only changed by the idle look timer, not immediately when the mouse leaves.
        self.idleEyeDir = self._random_eye_direction()

        # Tracks whether the cursor was inside this exact chuzzle cell last frame.
        # Used so mouse exit freezes the eyes at their last mouse-facing direction.
        self.mouseWasInsideOwnCell = False

        # Clone-only special mode hook. Normal chuzzles should leave this False.
        self.excitedEyes = False

        self.mouthType = r.randint(0, 15)
        self.superPulse = []

        # swaying animation
        self.swayFrameCap = 20
        self.swayFrame = r.randint(0, self.swayFrameCap - 1)
        self.swaySubFrameCap = 4
        self.swaySubFrame = r.randint(0, self.swaySubFrameCap - 1)

        # Normal blink state.
        # Uses generated ChuzzleBlinkEyes[color][shape], not AngryEyes and not SleepyEyes.
        self.blinkTimerMs = self._random_blink_wait_ms()
        self.blinkFramesLeft = 0
        self.isBlinking = False

        # Slow idle direction state.
        # When mouse leaves the cell, this is reset so the eyes hold their last direction first.
        self.idleLookTimerMs = self._random_idle_look_wait_ms()

        # deals with hovering over the chuzzle motionless
        self.fallAsleepTimer = 0

        now = self._now_ms()
        self._lastUpdateMs = now

    def copy(self):
        return copy(self)

    def isNormal(self):
        return not self.fatTL and not self.fatTR and not self.fatBR and not self.fatBL

    def getX(self):
        x = self.game.currentMenu().board.convertFromPosition(self.position).x + self.offset.x
        return x

    def getY(self):
        y = self.game.currentMenu().board.convertFromPosition(self.position).y + self.offset.y
        return y

    def getXY(self):
        return Point((self.getX(), self.getY()))

    def _now_ms(self) -> int:
        try:
            return int(p.time.get_ticks())
        except Exception:
            return int(time.monotonic() * 1000.0)
        
    def _now_ms(self):
        try:
            return int(p.time.get_ticks())
        except Exception:
            return int(time.monotonic() * 1000.0)

    def _random_original_range(self, amount):
        if amount <= 0:
            return 0
        return r.randrange(int(amount))

    def _random_blink_wait_ms(self):
        if self.excitedEyes:
            return self.EXCITED_BLINK_MIN_MS + self._random_original_range(self.EXCITED_BLINK_RANGE_MS)

        return self.NORMAL_BLINK_MIN_MS + self._random_original_range(self.NORMAL_BLINK_RANGE_MS)

    def _blink_frame_count(self):
        if self.excitedEyes:
            return self.EXCITED_BLINK_FRAMES

        return self.NORMAL_BLINK_FRAMES

    def _random_idle_look_wait_ms(self):
        if self.excitedEyes:
            return self.EXCITED_IDLE_LOOK_MIN_MS + self._random_original_range(self.EXCITED_IDLE_LOOK_RANGE_MS)

        return self.NORMAL_IDLE_LOOK_MIN_MS + self._random_original_range(self.NORMAL_IDLE_LOOK_RANGE_MS)

    def _random_eye_direction(self):
        if r.random() < 0.25:
            return Point((0.0, 0.0))

        angle = r.uniform(0.0, math.tau)
        radius = r.uniform(0.35, 1.0)

        return Point((math.cos(angle) * radius, -math.sin(angle) * radius))

    def _normalize_vector(self, dx, dy):
        length = math.hypot(dx, dy)

        if length <= 0.0001:
            return Point((0.0, 0.0))

        return Point((dx / length, dy / length))

    def _exp_lerp_amount(self, rate_per_second, dt_ms):
        dt_seconds = max(0.0, float(dt_ms) / 1000.0)
        return 1.0 - math.exp(-float(rate_per_second) * dt_seconds)

    def _lerp_point(self, current, target, amount):
        amount = max(0.0, min(1.0, float(amount)))

        return Point((
            current.x + (target.x - current.x) * amount,
            current.y + (target.y - current.y) * amount,
        ))

    def _get_board_local_mouse(self):
        board = self.game.currentMenu().board
        return Point(self.game.mousePos) - board.boardOffset

    def _get_cell_rect_local(self):
        board = self.game.currentMenu().board

        x = self.getX()
        y = self.getY()

        if self.fatTL:
            w = board.chuzSize.x * 2
            h = board.chuzSize.y * 2
        else:
            w = board.chuzSize.x
            h = board.chuzSize.y

        return p.Rect(int(x), int(y), int(w), int(h))

    def _mouse_is_inside_own_cell(self):
        local_mouse = self._get_board_local_mouse()
        return self._get_cell_rect_local().collidepoint(int(local_mouse.x), int(local_mouse.y))

    def _direction_to_mouse(self):
        board = self.game.currentMenu().board
        local_mouse = self._get_board_local_mouse()

        if self.fatTL:
            center_x = self.getX() + board.chuzSize.x
            center_y = self.getY() + board.chuzSize.y
        else:
            center_x = self.getX() + board.chuzSize.x * 0.5
            center_y = self.getY() + board.chuzSize.y * 0.5

        dx = float(local_mouse.x) - float(center_x)
        dy = float(local_mouse.y) - float(center_y)

        return self._normalize_vector(dx, dy)

    def _random_original_range(self, amount: int) -> int:
        if amount <= 0:
            return 0
        return r.randrange(amount)

    def _random_sleep_wait_ms(self) -> int:
        if self.excitedEyes:
            return self.EXCITED_SLEEP_MIN_MS + self._random_original_range(self.EXCITED_SLEEP_RANGE_MS)

        return self.NORMAL_SLEEP_MIN_MS + self._random_original_range(self.NORMAL_SLEEP_RANGE_MS)

    def _random_idle_direction_wait_ms(self) -> int:
        if self.excitedEyes:
            return self.EXCITED_IDLE_DIR_MIN_MS + self._random_original_range(self.EXCITED_IDLE_DIR_RANGE_MS)

        return self.IDLE_DIR_MIN_MS + self._random_original_range(self.IDLE_DIR_RANGE_MS)

    def _angle_to_original_eye_vector(self, angle: float) -> Point:
        # FUN_00467ae0:
        #   x = cos(angle)
        #   y = -sin(angle)
        return Point((math.cos(angle), -math.sin(angle)))

    def _normalize_vector(self, dx: float, dy: float) -> Point:
        length = math.hypot(dx, dy)
        if length <= 0.0001:
            return Point((0.0, 0.0))
        return Point((dx / length, dy / length))

    def _exp_lerp_amount(self, rate_per_second: float, dt_ms: int) -> float:
        dt_seconds = max(0.0, float(dt_ms) / 1000.0)
        return 1.0 - math.exp(-float(rate_per_second) * dt_seconds)

    def _lerp_point(self, current: Point, target: Point, amount: float) -> Point:
        amount = max(0.0, min(1.0, float(amount)))
        return Point((
            current.x + (target.x - current.x) * amount,
            current.y + (target.y - current.y) * amount,
        ))

    def setExcitedEyes(self, enabled=True):
        self.excitedEyes = bool(enabled)

        self.blinkTimerMs = self._random_blink_wait_ms()
        self.blinkFramesLeft = 0
        self.isBlinking = False

        self.idleLookTimerMs = self._random_idle_look_wait_ms()

        if self.excitedEyes:
            self.idleEyeDir = Point((0.0, 0.0))
            self.eyeTargetDir = Point((0.0, 0.0))

    def changeEyeDirections(self):
        self.idleEyeDir = self._random_eye_direction()
        self.eyeTargetDir = self.idleEyeDir.copy()

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
        self.motionSpeedSmoothed = (self.motionSpeedSmoothed * 0.65) + (raw_speed * 0.35)

    def isFatCorner(self):
        return self.fatBL or self.fatTR or self.fatBR

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

    def stopMotionKeepOffset(self, decay=0.50):
        self.delta.clear()
        self.velocity.clear()
        self.motionSpeed = 0.0
        self.last = self.offset.copy()

        self.motionSpeedSmoothed *= decay
        if self.motionSpeedSmoothed < 0.05:
            self.motionSpeedSmoothed = 0.0

    def _get_board_local_mouse(self) -> Point:
        board = self.game.currentMenu().board
        return Point(self.game.mousePos) - board.boardOffset

    def _get_cell_rect_local(self) -> p.Rect:
        board = self.game.currentMenu().board

        x = self.getX()
        y = self.getY()

        if self.fatTL:
            w = board.chuzSize.x * 2
            h = board.chuzSize.y * 2
        else:
            w = board.chuzSize.x
            h = board.chuzSize.y

        return p.Rect(int(x), int(y), int(w), int(h))

    def _mouse_is_inside_own_cell(self) -> bool:
        local_mouse = self._get_board_local_mouse()
        return self._get_cell_rect_local().collidepoint(int(local_mouse.x), int(local_mouse.y))

    def _direction_to_mouse(self) -> Point:
        board = self.game.currentMenu().board
        local_mouse = self._get_board_local_mouse()

        if self.fatTL:
            center_x = self.getX() + board.chuzSize.x
            center_y = self.getY() + board.chuzSize.y
        else:
            center_x = self.getX() + board.chuzSize.x * 0.5
            center_y = self.getY() + board.chuzSize.y * 0.5

        dx = float(local_mouse.x) - float(center_x)
        dy = float(local_mouse.y) - float(center_y)

        return self._normalize_vector(dx, dy)

    def setEyeBallDirection(self, mousePos=None, dt_ms=16):
        mouse_inside = self._mouse_is_inside_own_cell()

        if mouse_inside:
            self.eyeTargetDir = self._direction_to_mouse()
            self.mouseWasInsideOwnCell = True
            rate = self.MOUSE_TRACK_LERP_PER_SECOND

        else:
            if self.mouseWasInsideOwnCell:
                # This is the key behavior you asked for:
                # on mouse exit, freeze the eyes at the last mouse-facing direction.
                # Do not immediately pick a new idle direction.
                self.idleEyeDir = self.leftEyeDir.copy()
                self.eyeTargetDir = self.idleEyeDir.copy()
                self.idleLookTimerMs = self._random_idle_look_wait_ms()
                self.mouseWasInsideOwnCell = False

            else:
                self.eyeTargetDir = self.idleEyeDir

            rate = self.EXCITED_TRACK_LERP_PER_SECOND if self.excitedEyes else self.IDLE_TRACK_LERP_PER_SECOND

        lerp = self._exp_lerp_amount(rate, dt_ms)

        self.leftEyeDir = self._lerp_point(self.leftEyeDir, self.eyeTargetDir, lerp)
        self.rightEyeDir = self.leftEyeDir.copy()

        return self.leftEyeDir

    def _update_eye_timers(self, dt_ms):
        if dt_ms <= 0:
            return

        if self.blinkFramesLeft > 0:
            self.blinkFramesLeft -= 1
            self.isBlinking = True

            if self.blinkFramesLeft <= 0:
                self.blinkFramesLeft = 0
                self.isBlinking = False
                self.blinkTimerMs = self._random_blink_wait_ms()

            return

        self.isBlinking = False
        self.blinkTimerMs = max(0, self.blinkTimerMs - int(dt_ms))

        if self.blinkTimerMs <= 0:
            self.blinkFramesLeft = self._blink_frame_count()
            self.isBlinking = True
            return

        if not self._mouse_is_inside_own_cell():
            self.idleLookTimerMs = max(0, self.idleLookTimerMs - int(dt_ms))

            if self.idleLookTimerMs <= 0:
                self.changeEyeDirections()
                self.idleLookTimerMs = self._random_idle_look_wait_ms()

    def _surface_from_texture(self, texture, index=0):
        if isinstance(texture, (list, tuple)):
            if not texture:
                return None
            return texture[index % len(texture)]
        return texture

    def _get_eye_surface(self):
        # if self.isSleepyBlink and hasattr(self.textures, "SleepyEyes"):
        #     sleepy = self._surface_from_texture(self.textures.SleepyEyes, self.eyeballShape)
        #     if sleepy is not None:
        #         return sleepy

        # if self.excitedEyes and hasattr(self.textures, "AngryEyes"):
        #     angry = self._surface_from_texture(self.textures.AngryEyes, self.eyeballShape)
        #     if angry is not None:
        #         return angry

        return self.textures.Eyes[self.eyeballShape]

    def _get_pupil_surface(self):
        return self.textures.Pupils[self.eyeballColor]

    def _blit_accessories(self, surface, swayOffset, dt_ms):
        if surface is None:
            return

        self.setEyeBallDirection(dt_ms=dt_ms)

        eyes = self._get_eye_surface()
        if eyes is None:
            return

        shape = int(self.eyeballShape) % 4

        if self.fatTL:
            eyes_pos = Point((
                self.FAT_EYES_POS.x,
                self.FAT_EYES_POS.y,
            ))

            left_center = self.FAT_LEFT_PUPIL_CENTER
            right_center = self.FAT_RIGHT_PUPIL_CENTER

            travel_x = self.FAT_PUPIL_TRAVEL.x
            travel_y = self.FAT_PUPIL_TRAVEL.y

        else:
            eyes_pos = Point((
                self.NORMAL_EYES_POS.x,
                self.NORMAL_EYES_POS.y,
            ))

            left_raw, right_raw = self.EYE_PUPIL_CENTERS[shape]
            left_center = Point(left_raw)
            right_center = Point(right_raw)

            travel_x, travel_y = self.EYE_PUPIL_TRAVEL[shape]

        surface.blit(eyes, eyes_pos.tuple())

        if not self.isBlinking:
            pupils = self._get_pupil_surface()
            pupil_half_w = pupils.get_width() * 0.5
            pupil_half_h = pupils.get_height() * 0.5

            left_pupil_pos = Point((
                eyes_pos.x + left_center.x + self.leftEyeDir.x * travel_x - pupil_half_w,
                eyes_pos.y + left_center.y + swayOffset + self.leftEyeDir.y * travel_y - pupil_half_h,
            ))

            right_pupil_pos = Point((
                eyes_pos.x + right_center.x + self.rightEyeDir.x * travel_x - pupil_half_w,
                eyes_pos.y + right_center.y + swayOffset + self.rightEyeDir.y * travel_y - pupil_half_h,
            ))

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

    def _update_sway(self):
        self.swaySubFrame = (self.swaySubFrame + 1) % self.swaySubFrameCap
        if self.swaySubFrame == 0:
            self.swayFrame = (self.swayFrame + 1) % self.swayFrameCap

    def _get_frame_surface(self):
        is_fat_main = self.fatTL

        if self.isNormal():
            sway_count = 20
            dir_count = 10
            frame_bank = self.textures.ChuzzleBase[self.color]
        elif is_fat_main:
            sway_count = 10
            dir_count = 5
            frame_bank = self.textures.ChuzzleHuge[self.color]
        else:
            return None, 0, 0

        dir_to_segment = {
            0: 0,  # down
            3: 1,  # left
            2: 2,  # right
            1: 3,  # up
        }

        speed_animating = self.motionSpeedSmoothed > 0.05

        if speed_animating and self.direction in dir_to_segment:
            board = self.game.currentMenu().board

            max_speed = max(1.0, float(board.chuzSize.x) * 0.10)
            speed = float(self.motionSpeedSmoothed)

            if is_fat_main:
                speed *= 0.5

            t = min(max(speed / max_speed, 0.0), 1.0)
            t = math.sqrt(t)

            speed_index = int(round(t * (dir_count - 1)))
            segment = dir_to_segment[self.direction]
            frame_index = sway_count + segment * dir_count + speed_index
        else:
            self.motionSpeed = 0.0
            self.motionSpeedSmoothed = 0.0
            frame_index = self.swayFrame % sway_count

        return frame_bank[frame_index].copy(), frame_index, sway_count

    def getSurface(self):
        now = self._now_ms()
        dt_ms = max(0, min(250, now - self._lastUpdateMs))
        self._lastUpdateMs = now

        self.swaySubFrame = (self.swaySubFrame + 1) % self.swaySubFrameCap
        if self.swaySubFrame == 0:
            self.swayFrame = (self.swayFrame + 1) % self.swayFrameCap

        is_fat_main = self.fatTL

        if self.isNormal():
            sway_count = 20
            dir_count = 10
            frame_bank = self.textures.ChuzzleBase[self.color]
        elif is_fat_main:
            sway_count = 10
            dir_count = 5
            frame_bank = self.textures.ChuzzleHuge[self.color]
        else:
            return None

        dir_to_segment = {
            0: 0,
            3: 1,
            2: 2,
            1: 3,
        }

        speed_animating = self.motionSpeedSmoothed > 0.05

        if speed_animating and self.direction in dir_to_segment:
            board = self.game.currentMenu().board

            max_speed = max(1.0, float(board.chuzSize.x) * 0.10)
            speed = float(self.motionSpeedSmoothed)

            if is_fat_main:
                speed *= 0.5

            t = min(max(speed / max_speed, 0.0), 1.0)
            t = math.sqrt(t)

            speed_index = int(round(t * (dir_count - 1)))
            segment = dir_to_segment[self.direction]
            frame_index = sway_count + segment * dir_count + speed_index

        else:
            self.motionSpeed = 0.0
            self.motionSpeedSmoothed = 0.0
            frame_index = self.swayFrame % sway_count

        surface = frame_bank[frame_index].copy()

        swayOffset = 0
        if self.isNormal() and frame_index < sway_count:
            if self.swayFrame in [10, 11, 12, 13, 14]:
                swayOffset = 0.5

        self._blit_accessories(surface, swayOffset, dt_ms)

        return surface

    def snapToBoard(self):
        self.offset.clear()
        self.delta.clear()
        self.velocity.clear()
        self.motionSpeed = 0.0
        self.motionSpeedSmoothed = 0.0
        self.last = self.offset.copy()

    def incrementAnimation(self):
        now = self._now_ms()
        dt_ms = max(0, min(250, now - self._lastUpdateMs))

        # Do not update _lastUpdateMs here. getSurface owns that timestamp.
        # Board.render calls incrementAnimation() immediately before getSurface().
        self._update_eye_timers(dt_ms)