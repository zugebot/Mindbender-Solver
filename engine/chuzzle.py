# Jerrin Shirks

# native imports
import random as r
import math
from copy import copy

# custom imports
from tools.point import Point


class Chuzzle:
    def __init__(self, game, position, version="normal", color=None, mini=False):

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
        self.leftEyeDir = 0
        self.rightEyeDir = 0
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
        self.last = self.offset

        if direction == 0:
            # if self.offset.y != 0:
            #     print(direction, self.offset, self.last, self.delta, self.direction)


            if self.delta.y > 0:
                self.direction = 0
            elif self.delta.y < 0:
                self.direction = 1


        elif direction == 1:
            # if self.offset.x != 0:
            #     print(direction, self.offset, self.last, self.delta, self.direction)

            if self.delta.x > 0:
                self.direction = 2
            elif self.delta.x < 0:
                self.direction = 3



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

    def setEyeBallDirection(self, mousePos):
        dx = self.getX() + 0.5 * self.game.currentMenu().board.chuzSize.x - mousePos.x
        dy = self.getY() + 0.5 * self.game.currentMenu().board.chuzSize.y - mousePos.y
        rad_angle = math.atan2(dy, dx)
        deg_angle = math.degrees(rad_angle)
        if deg_angle < 0:
            deg_angle = 180 + (180 + deg_angle)

        # print(mousePos, dx, dy, deg_angle)
        # starts at W goes clockwise from 0 to 360

    def getSurface(self):

        # sway loop
        self.swaySubFrame = (self.swaySubFrame + 1) % self.swaySubFrameCap
        if self.swaySubFrame == 0:
            self.swayFrame = (self.swayFrame + 1) % self.swayFrameCap

        # deals with the chuzzle swaying moving the eyes etc.
        swayOffset = 0
        if self.isNormal():
            if self.swayFrame in [10, 11, 12, 13, 14]:
                swayOffset = 0.5
            # swayOffset = math.sin(math.pi * self.swayFrame / 20)

        if not (self.fatTL or self.fatTR or self.fatBL or self.fatBR):
            surface = self.textures.ChuzzleBase[self.color][self.swayFrame].copy()
        elif self.fatTL:
            surface = self.textures.ChuzzleHuge[self.color][self.swayFrame].copy()
        else:
            surface = None

        # blink loop
        if self.waitToBlinkFrame < self.waitToBlinkFrameCap:
            self.waitToBlinkFrame += 1
        else:
            if self.blinkingFrame < self.blinkingFrameCap:
                self.blinkingFrame += 1
            else:
                self.waitToBlinkFrame = 0
                self.waitToBlinkFrameCap = r.randint(5 * self.fps, 25 * self.fps)
                self.blinkingFrame = 0
                self.blinkingFrameCap = r.randint(3, 10)
                # change eyeball direction for next frame

        """
        if surface is not None:
            mouth = self.textures.ChuzzleMouths[self.mouthType]

            mouth = p.transform.scale(mouth, (mouth.get_width()*2, mouth.get_height()*2))

            if self.fatTL:
                surface.blit(mouth, (55, 140))
            else:
                surface.blit(mouth, (4, 55))

        eyeballShape = self.textures.Eyes[self.eyeballShape]
        eyeballPupil = self.textures.Pupils[self.eyeballColor]
        offsetBase = [(-2, -4), (-2, -4), (-2, -4), (-2, -4)]
        offsetHuge = [(-2+49, -4+80), (-2+49, -4+80), (-2+49, -4+80), (-2+49, -4+80)]


        # eyes are not done yet smh
        # eyeballShape = self.textures.Eyes[self.eyeballShape]
        eyeball1 = Point()# Point((30, 50))
        # eyeball1.x += 22 - 11 * m.sin(self.leftEyeDir)
        # eyeball1.y += 22 - 11 * m.cos(self.leftEyeDir)

        eyeball2 = Point((50, 50))
        eyeball2.x += 22 - 11 * math.sin(self.rightEyeDir)
        eyeball2.y += 22 - 11 * math.cos(self.rightEyeDir)

        if surface is not None:
            if self.fatTL:
                surface.blit(eyeballShape, offsetHuge[self.eyeballShape])
                surface.blit(self.textures.LevelFont_Color[self.eyeballShape], (0, 0))

            elif not self.fatTL:
                surface.blit(eyeballShape, offsetBase[self.eyeballShape])
                surface.blit(self.textures.LevelFont_Color[self.eyeballShape], (0, 0))

                surface.blit(eyeballPupil, eyeball1.tuple())
                surface.blit(eyeballPupil, eyeball2.tuple())


        # to rotate the pupils:
        # step 1: get current mouse position
        # step 2: find the
        """

        if self.game.settings.colorblindMode:
            color = self.textures.colorsMainBlind[str(self.color)]
            if color is not None:
                index = self.textures.colorsMainBlind[str(color)]

                if self.isNormal():
                    surface.blit(self.textures.ColorblindSymbols[index], (16, 4 + swayOffset))

        return surface

    def snapToBoard(self):
        self.offset.clear()

    def incrementAnimation(self):
        pass
