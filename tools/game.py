# Jerrin Shirks

# native imports
import pygame as p
import sys

# custom imports
from data.config import *
from tools.point import Point
from tools.menu import Menu




class Game:
    def __init__(self, size, fps, name="", debug=True):
        # pygame stuff
        self.instance = p
        self.instance.init()
        self.instance.display.set_caption(name)

        self.debug = debug
        self.screenSize = Point(size)
        self.fps = fps
        self.clock = p.time.Clock()
        self.screen = self.instance.display.set_mode(size)

        self.textures = None
        self.sounds = None

        # running
        self.running = True
        self.mouseState = 0
        self.mousePos = Point()

        # Menus
        self.menuIndex = -1
        self.menus = []
        self.lastMenuSize = 0
        self.lastMenuImage = None


    def currentMenu(self, offset=0) -> Menu:
        return self.menus[self.menuIndex + offset]


    def addMenu(self, menu, _set: bool = False) -> None:
        self.menus.append(menu(self))
        self.menuIndex += 1
        if self.debug:
            print("adding a menu!")
            print("Menu Stack Length:", len(self.menus), "\n")


    def popMenu(self) -> None:
        self.menus.pop()
        self.menuIndex -= 1

        # average comment enjoyer
        self.lastMenuImage = None
        if self.debug:
            print("Popping Menu!")
            print("New Menu Length:", len(self.menus), "\n")



    def setMenuIndex(self, index) -> None:
        self.menuIndex = index


    def quit(self):
        self.running = False
        p.quit()
        exit()

    def eventLoop(self):
        if self.mouseState == MOUSE_CLICK_DOWN: # click down
            self.mouseState = MOUSE_HELD_DOWN # held down
        if self.mouseState == MOUSE_CLICK_UP: # click up
            self.mouseState = MOUSE_HOVER # nothing

        self.mousePos = Point(p.mouse.get_pos())

        for event in self.instance.event.get():
            if event.type == p.QUIT:
                self.quit()

            if event.type == p.MOUSEBUTTONDOWN:
                self.mouseState = MOUSE_CLICK_DOWN
            if event.type == p.MOUSEBUTTONUP:
                self.mouseState = MOUSE_CLICK_UP

            if event.type == p.KEYDOWN:
                self.currentMenu().onKeyDown(event)


    def run(self, running=True):
        while running:

            if len(self.menus) != self.lastMenuSize:
                self.lastMenuImage = p.display.get_surface().copy()
                self.lastMenuSize = len(self.menus)

            self.screen.fill(EMPTY_SCREEN)
            self.eventLoop()

            # Run the menu
            self.currentMenu().run(self.mousePos, self.mouseState)

            # display to screen
            self.instance.display.flip()
            self.clock.tick(self.fps)

        p.quit()
