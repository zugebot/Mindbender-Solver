# Jerrin Shirks

# native imports
import win32gui as w32
import pyautogui as p
import time

# custom imports
from window import Window
from support import *



class ChuzzleWindow(Window):
    def __init__(self, _p, *args, **kwargs):
        super().__init__(_p, *args, **kwargs)
        self.p = _p

        self.sizes = [[640, 480], [800, 600], [1024, 768], [1280, 1024]]
        self.size = 1
        self.puzzles = {}
        self.moves = {}
        self.locs = {}

        # MINDBENDER
        self.puzzle_left = (0.07125, 0.80573)
        self.puzzle_right = (0.25, 0.80732)
        self.puzzle_buttons = [
            (0.055, 0.5828),
            (0.10375, 0.6242),
            (0.16375, 0.63854),
            (0.2225, 0.6242),
            (0.27375, 0.58758)
        ]
        self.next_puzzle = (0.62, 0.56369)
        self.chuz_x = [0.37000, 0.47875, 0.57125, 0.66750, 0.77375, 0.87500]
        self.chuz_y = [0.19427, 0.32166, 0.44745, 0.58599, 0.70701, 0.83121]


        # MAIN SCREEN
        self.change_profile_button = (0.50293, 0.53769)
        self.new_profile_button = (0.42285, 0.6608)
        self.accept_profile_button = (0.43066, 0.73869)
        # type
        self.accept_new_profile_button = (0.41895, 0.71608)


        self.mindbender_button = (0.82129, 0.52387)







        self.loadPuzzles()


    def loadPuzzles(self):
        files = os.listdir("levels")
        for file in files:
            if file.endswith(".txt"):
                name = file.split("_")[0]
                with open("levels/" + file, "r") as f:
                    self.puzzles[name] = f.read().splitlines()[0]
                    self.moves[name] = solveToTuples(self.puzzles[name])
        self.convert()

    def convert(self):
        for key in self.moves:
            moves = self.moves[key]

            for n, sub in enumerate(moves):
                chuz1 = self.percentageToScreen((self.chuz_x[sub[0][0]], self.chuz_y[sub[0][1]]))
                chuz2 = self.percentageToScreen((self.chuz_x[sub[1][0]], self.chuz_y[sub[1][1]]))
                moves[n] = (chuz1, chuz2)

            self.locs[key] = moves

    def moveTo(self, _x: int, _y: int) -> None:
        size = self.sizes[self.size]
        w32.MoveWindow(self.ID, _x, _y, size[0], size[1] + 28, True)

    def drag(self, _start: tuple[int], _end: tuple[int]) -> None:
        self.p.moveTo(*_start)
        self.p.mouseDown()
        self.p.moveRel(3, 3)
        self.p.moveTo(*_end)
        self.p.mouseUp()




    def makeNewAccount(self, name: str,
                       click_speed: float = 0,
                       type_speed: float = 0):
        self.clickPercent(self.change_profile_button)
        time.sleep(click_speed)
        self.clickPercent(self.new_profile_button)
        time.sleep(click_speed)
        time.sleep(type_speed)
        for letter in name:
            self.p.press(letter)
            time.sleep(type_speed)
        time.sleep(click_speed)
        self.clickPercent(self.accept_new_profile_button)
        time.sleep(click_speed)
        self.clickPercent(self.accept_profile_button)





    def solve(self, puzzle: str) -> bool:
        for move in self.locs[puzzle]:
            self.drag(*move)

    def playSection(self, *levels):
        last_level = 0
        for n, level in enumerate(levels):
            sect, cur_lev = level.split("-")
            cur_lev = int(cur_lev)
            if cur_lev == last_level + 1:
                if n != 0:
                    self.clickPercent(self.next_puzzle)
            else:
                self.clickPercent(self.puzzle_buttons[cur_lev - 1])

            last_level = cur_lev
            self.solve(level) # this is what actually does each solve
            if n == len(levels) - 1:
                self.clickPercent(self.puzzle_right) # clicks to next level

    def playMindbender(self, livesplit=True):
        if livesplit:
            p.press("num1")
        self.playSection("1-1", "1-2", "1-3")
        self.playSection("2-1", "2-4", "2-5")
        self.playSection("3-1", "3-2", "3-3")
        self.playSection("4-1", "4-3", "4-4")
        self.playSection("5-2", "5-3", "5-4")

        self.playSection("6-1", "6-2", "6-5")
        self.playSection("7-1", "7-2", "7-3")
        self.playSection("8-1", "8-3", "8-5")
        self.playSection("9-2", "9-4", "9-5")
        if livesplit:
            p.press("num1")












