# Jerrin Shirks

import win32gui as w32
import pyautogui
import keyboard
from pynput.keyboard import Key, Listener
import time
import os



class Window:
    def __init__(self, name):
        self.ID = 0
        self.Name = name
        self.turnOff = False
        self.x, self.y, self.w, self.h = 0, 0, 0, 0
        w32.EnumWindows(self.callback, None)
        self.getSize()

    def callback(*args):
        self, hwnd, extra = args
        text = w32.GetWindowText(hwnd)
        if self.Name in text:
            self.ID = hwnd

    def getSize(self, debug=False):
        rect = w32.GetWindowRect(self.ID)
        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2] - rect[0]
        self.h = rect[3] - rect[1]
        if debug:
            print(f"Window ({w32.GetWindowText(self.ID)}):")
            print(f"\tLocation: ({self.x}, {self.y})")
            print(f"\tSize: ({self.w}, {self.h})")

    def pos(self):
        return pyautogui.position()

    def relativePos(self):
        self.getSize()
        mx, my = pyautogui.position()
        rx = (mx - self.x) / self.w
        ry = (my - self.y) / self.h
        return rx, ry

    def getCoordinates(self, ix, iy):
        rx = ix * self.w + self.x
        ry = iy * self.h + self.y
        return rx, ry

    def resize(self, x, y, w, h):
        w32.MoveWindow(self.ID, x, my, w, h, True)


version = "christmas"
window = Window("Chuzzle ")
resetButton = "space"
failsafeButton = "alt_gr"
retimes1 = [0.0, 0.0, 0.0, 0.05, 0.1]
retimes2 = [0.1, 0.1, 0.1, 0.15, 0.1]

if version == "christmas":
    timeDelay = retimes1
else:
    timeDelay = retimes2

#filename = r"C:\ProgramData\PopCap Games\Chuzzle\Profiles\Jerrin\SAVEGAME-SPEEDCHUZZLE.DAT"
filename = r"C:\Users\jerrin\Documents\my games\Chuzzle\Profiles\Jerrin\SAVEGAME-SPEEDCHUZZLE.DAT"
# filename = r"C:\ProgramData\PopCap Games\Chuzzle\Profiles\Kvadrat0\SAVEGAME-SPEEDCHUZZLE.DAT"




def getKey(key):
    try:
        k = key.char
    except AttributeError:
        k = key.name
    return k

def on_press(key):
    k = getKey(key)

    if k == failsafeButton:
        keyboard.press(resetButton)

    if k == resetButton:
        if window.turnOff:
            window.turnOff = False

        else:
            window.getSize()
            reset = [

                [0.07692307692307693, 0.9634340222575517],
                [0.4987593052109181, 0.712241653418124],
                [0.5037220843672456, 0.6804451510333863],
                [0.41935483870967744, 0.6661367249602543],
                [0.41935483870967744, 0.6661367249602543]
            ]
            try:
                os.remove(filename)
            except:
                ""

            for n in range(len(reset)):
                x, y = reset[n][0], reset[n][1]
                x, y = window.getCoordinates(x, y)

                time.sleep(timeDelay[n])

                pyautogui.moveTo(x, y)




                if n == 2:
                    try:
                        os.remove(filename)
                    except:
                        ""


            time.sleep(0.1)
            window.turnOff = True
            keyboard.press(resetButton)




def on_release(key):
    if key == Key.esc:
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()






"""
import time
while True:
    x = window.relativePos()
    print(x)
    time.sleep(1)
"""










