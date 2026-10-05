# Jerrin Shirks

# native imports
import win32gui as w32
import pyautogui as p
import time

# custom imports
from window import Window
from chuzzle import ChuzzleWindow
from support import *










if __name__ == "__main__":

    Chuzzle = ChuzzleWindow(p, "Chuzzle Christmas")
    p.PAUSE = 1 / 5
    # Chuzzle.moveTo(1400, 200)
    Chuzzle.update()

    time.sleep(2)

    p.press("F11")
    time.sleep(0.5)
    Chuzzle.solve("5-3")
    time.sleep(0.5)
    p.press("F11")

    """
    Chuzzle.makeNewAccount("tasbot", 0.5, 0.3)
    time.sleep(0.5)
    Chuzzle.clickPercent(Chuzzle.mindbender_button)
    time.sleep(2)

    p.PAUSE = 1 / 200
    Chuzzle.playMindbender()
    """
    """
    while True:
        x, y = Chuzzle.screenToPercentage(p.position())
        print(f"{x}, {y}")
        time.sleep(1)
    """
    # Chuzzle.playMindbender()


