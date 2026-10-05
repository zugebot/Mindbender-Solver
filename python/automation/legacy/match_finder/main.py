# Jerrin Shirks

import pygame as p
import random as r
import numpy as np
from PIL import Image
import cv2
import math


black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)


class Point:
    def __init__(self, *args):
        if args == tuple():
            args = (0, 0)
        if type(args[0]) == tuple:
            self.x = args[0][0]
            self.y = args[0][1]
        else:
            self.x = args[0]
            self.y = args[1]

    def getTuple(self):
        return self.x, self.y

    def __str__(self):
        string = f"({self.x}, {self.y})"
        return string




def createAlpha(filename, new_filename):
    im = Image.open(filename)
    im = im.convert("RGBA")
    pix = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            c = pix[x, y]
            pix[x, y] = (c[0], c[0], c[0], c[0])
    im.save(new_filename)


class ChuzzleRenderer:
    def __init__(self, parent):

        createAlpha("Chuzzle-Base_Alpha.png", "Base_Alpha.png")
        createAlpha("Chuzzle-Hilite_Alpha.png", "Hilite_Alpha.png")
        colorsMain = {
            "0": (158, 29, 29), # red
            "1": (255, 143, 12), # orange
            "2": (255, 255, 0), # yellow
            "3": (74, 178, 74), # green
            "4": (81, 186, 210), # cyan
            "5": (57, 57, 255), # blue
            "6": (185, 60, 185), # purple
            "7": (255, 255, 255), # white
        }
        self.parent = parent
        self.BaseImage = p.image.load("Chuzzle-Base.png").convert_alpha()
        self.AlphaImage = p.image.load("Base_Alpha.png").convert_alpha()
        self.HiliteImage = p.image.load("Hilite_Alpha.png").convert_alpha()

        surface = p.Surface((4860, 84), p.SRCALPHA)
        surface.fill((160, 0, 0, 255))

        self.colors = [None for _ in range(8)]
        for n in range(8):

            base = self.BaseImage.copy()
            color1 = surface.copy()
            color1.fill(colorsMain[str(n)])

            color2 = surface.copy()
            color2.fill((70, 70, 70, 70))
            color2.blit(self.HiliteImage, (0, 0), None, p.BLEND_RGBA_MULT)

            base.blit(color1, (0, 0), None, p.BLEND_RGBA_MULT)
            base.blit(color2, (0, 0), None, p.BLEND_RGBA_ADD)
            base.blit(self.AlphaImage, (0, 0), None, p.BLEND_RGBA_MULT)

            surfaces = []
            for i in range(60):
                surf = p.Surface((81, 81), p.SRCALPHA)
                surf.fill((0, 0, 0, 0))
                surfaces.append(surf)
                surfaces[-1].blit(base, (0, 0), (81 * i, 1, 81, 81), p.BLEND_RGBA_ADD)

            self.colors[n] = surfaces

    def getChuzzle(self, frame, color):
        return self.colors[color][frame]





class Chuzzle:
    def __init__(self, parent, posX=0, posY=0):

        self.pos = Point(posX, posY)
        self.offset = Point(0, 0)

        self.velocity = Point(0, 0)
        self.acceleration = Point(0, 0)

        self.snapFrame = 0
        self.snapRow = False

        self.parent = parent
        self.board = parent.parent

        self.colorInt = r.randint(0, 7)

        self.frameCap = 220
        self.frame = r.randint(0, self.frameCap)

    def setPos(self, posX, posY):
        self.pos = Point(posX, posY)

    def incrementPhysics(self):

        if self.snapRow:
            if self.snapFrame < 30:
                self.snapFrame += 1
                self.velocity.x += self.acceleration.x
                self.velocity.y += self.acceleration.y
                self.offset.x += self.velocity.x
                self.offset.y += self.velocity.y

                # print(self.velocityX, self.accelerationX, self.offsetX, self.velocityY, self.accelerationY, self.offsetY)

            else:
                self.snapFrame = 0
                self.offset.x = 0
                self.offset.y = 0
                self.acceleration.x = 0
                self.acceleration.y = 0
                self.velocity.x = 0
                self.velocity.y = 0

                self.snapRow = False

    def snap(self, immediate=False):

        if immediate:
            self.offset = Point()

        elif self.board.mindBenderMode:
            self.snapRow = True
            self.snapFrame = 0
            self.acceleration.x = 0 if self.offset.x == 0 else - 1
            self.acceleration.y = 0 if self.offset.y == 0 else - 1

            time = 29
            self.velocity.x = (-self.offset.x - 0.5 * self.acceleration.x * time ** 2) / time
            self.velocity.y = (-self.offset.y - 0.5 * self.acceleration.y * time ** 2) / time

        else:
            self.offset = Point()

            # print(self.velocityX, self.accelerationX, self.offsetX, self.velocityY, self.accelerationY, self.offsetY)

    def getX(self):
        x = self.pos.x * self.board.chuzW + (self.pos.x + 1) * self.board.chuzSpacing + self.offset.x
        return x

    def getY(self):
        y = self.pos.y * self.board.chuzW + (self.pos.y + 1) * self.board.chuzSpacing + self.offset.y
        return y

    def getSurface(self):
        frame = self.frame // 11
        self.frame = (self.frame + 1) % self.frameCap
        return self.parent.getChuzzle(frame=frame, color=self.colorInt)










class Board:
    def __init__(self, parent, offsetCoords=(0, 0), chuzW=81, chuzH=81, chuzSpacing=0, chuzCountW=6, chuzCountH=6):
        self.parent = parent
        self.offsetCoords = Point(offsetCoords)
        self.chuzzleRenderer = ChuzzleRenderer(self)
        self.surfaceWidth = chuzW * chuzCountW + chuzSpacing * (chuzCountW + 1)
        self.surfaceHeight = chuzH * chuzCountH + chuzSpacing * (chuzCountH + 1)
        self.chuzCountW = chuzCountW
        self.chuzCountH = chuzCountH
        self.chuzW = chuzW
        self.chuzH = chuzH
        self.chuzSpacing = chuzSpacing
        self.surface = p.Surface((self.surfaceWidth, self.surfaceHeight), p.SRCALPHA)
        self.surface.fill((0, 0, 0, 0))

        self.mindBenderMode = True

        # dealing with dragging rows
        self.dragging = False
        self.mouseDown = False

        self.centerCoords = Point()
        self.pointCoords = Point()
        self.lastCoords = Point()
        self.mouseDownPos = Point()

        self.curDir = 0
        self.lastDir = 0




        # makes chuzzles
        self.chuzzles = [[0 for _ in range(self.chuzCountH)] for _ in range(self.chuzCountW)]
        for y in range(self.chuzCountH):
            for x in range(self.chuzCountW):
                # xVal = x * chuzW + (x + 1) * chuzSpacing
                # yVal = y * chuzH + (y + 1) * chuzSpacing
                self.chuzzles[y][x] = Chuzzle(self.chuzzleRenderer, chuzW, x, y)

    def chuzIterator(self):
        for y in range(self.chuzCountH):
            for x in range(self.chuzCountW):
                yield self.chuzzles[y][x]

    # position means index
    def getCoordsFromPos(self, *args):
        args = list(args)
        for i in range(len(args)):
            args[i] = args[i] * self.chuzW + (args[i] + 1) * self.chuzSpacing
        return tuple(args)


    # coords means where it is on the screen
    def getPosFromCoords(self, coord, direction=0):
        if 0 < coord < self.chuzSpacing:
            coord = 0
        elif 6.5 * self.chuzSpacing + 6 * self.chuzW <= coord <= 7 * self.chuzSpacing + 6 * self.chuzW:
            coord = self.chuzCountW
        else:
            coord = (coord - 0.5 * self.chuzSpacing) // (self.chuzW + self.chuzSpacing)


        return int(coord)



    def onMouseDown(self):
        # 0 means horizontal, 1 means vertical

        # only happens when mouse is first clicked down
        if not self.mouseDown:
            self.mouseDown = True
            self.centerCoords = Point(*p.mouse.get_pos())
            x = self.getPosFromCoords(self.centerCoords.x - self.offsetCoords.x, 0)
            y = self.getPosFromCoords(self.centerCoords.y - self.offsetCoords.y, 1)
            self.mouseDownPos = Point(x, y)

        self.pointCoords = Point(*p.mouse.get_pos())

        threshold1 = 10
        threshold2 = 2


        X = abs(self.centerCoords.x - self.pointCoords.x)
        Y = abs(self.centerCoords.y - self.pointCoords.y)

        if X < threshold1 and Y < threshold1:
            if X > threshold2 or Y > threshold2:
                self.curDir = 0 if X < Y else 1
                if self.curDir != self.lastDir:
                    for chuz in self.chuzIterator():
                        chuz.snap(immediate=True)

                self.centerCoords = self.pointCoords

        elif self.pointCoords.x > threshold1 > self.pointCoords.x and self.pointCoords.y > threshold1 > self.pointCoords.y:
            self.curDir = 0 if X < Y else 1

        self.lastCoords = self.pointCoords

        dir = ["horizontal", "vertical"][self.curDir]

        if self.curDir == 1:  # vertical
            for chuz in self.chuzIterator():
                if chuz.pos.y == self.mouseDownPos.y:
                    chuz.offset.x = self.pointCoords.x - self.centerCoords.x

        elif self.curDir == 0:  # horizontal]
            for chuz in self.chuzIterator():
                if chuz.pos.x == self.mouseDownPos.x:
                    chuz.offset.y = self.pointCoords.y - self.centerCoords.y





    def onMouseUp(self):
        self.mouseDown = False


        now = Point(*p.mouse.get_pos())
        now.x = self.getPosFromCoords(now.x - self.offsetCoords.x, 0) % 6
        now.y = self.getPosFromCoords(now.y - self.offsetCoords.y, 0) % 6

        change = Point()
        change.x = now.x - self.mouseDownPos.x
        change.y = now.y - self.mouseDownPos.y

        print("change, ", change)

        if True:
            for chuz in self.chuzIterator():

                x = self.getPosFromCoords(chuz.getX())
                y = self.getPosFromCoords(chuz.getY())
                # print("old:", chuz.pos, end="")
                chuz.setPos(x, y)
                # print("new:", chuz.pos)
                chuz.offset = Point()
                chuz.snap(immediate=True)

        print(self.mouseDownPos, now)


        for chuz in self.chuzIterator():
            chuz.snap()
        print("clicked up")
        pass


    def render(self):

        # wash out the background
        self.surface.fill((0, 0, 0, 0))

        # render each chuzzle to screen
        for y, row in enumerate(self.chuzzles):
            for x, chuzzle in enumerate(row):
                chuzzle.incrementPhysics()
                coordX = chuzzle.getX() % (6 * self.chuzW)
                coordY = chuzzle.getY() % (6 * self.chuzH)
                self.surface.blit(chuzzle.getSurface(), (coordX, coordY))
                if self.curDir == 0:
                    self.surface.blit(chuzzle.getSurface(), (coordX, coordY - (6 * self.chuzH)))
                elif self.curDir == 1:
                    self.surface.blit(chuzzle.getSurface(), (coordX - (6 * self.chuzW), coordY))





class Game:
    def __init__(self, width, height, fps):
        self.instance = p
        self.instance.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.clock = p.time.Clock()
        self.screen = self.instance.display.set_mode((self.width, self.height))
        # objects
        self.board = Board(self, offsetCoords=(255, 51))
        self.mouseDown = -1


    def run(self, running=True):
        background = p.image.load("bkg1.jpg").convert_alpha()

        while running:
            for event in self.instance.event.get():
                if event.type == p.QUIT:
                    running = False

                if event.type == p.KEYDOWN:
                    if event.key == p.K_RIGHT:
                        print("clicked down")

                if event.type == p.MOUSEBUTTONDOWN:
                    self.mouseDown = 1

                if event.type == p.MOUSEBUTTONUP:
                    self.mouseDown = 0


            if self.mouseDown == 1:
                self.board.onMouseDown()
            if self.mouseDown == 0:
                self.board.onMouseUp()
                self.mouseDown = -1

            self.screen.fill(white)
            self.screen.blit(background, (0, 0))
            self.screen.blit(self.board.surface, self.board.offsetCoords.getTuple())
            self.board.render()
            self.instance.display.flip()
            self.clock.tick(self.fps)

        p.quit()


if __name__ == "__main__":
    game = Game(800, 600, fps=60)
    game.run()

