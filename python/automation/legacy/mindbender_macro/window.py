# Jerrin Shirks'

from win32 import win32gui as wg



class Window:
    def __init__(self, p, windowName):
        self.p = p
        self.windowName = windowName
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.ID = 0
        self.running = False
        self.debug = True
        self.decimal = 5

        self.startUp()


    def callback(*args):
        self, hwnd, extra = args
        text = wg.GetWindowText(hwnd)
        if self.windowName in str(text):
            self.ID = hwnd


    def update(self):
        rect = wg.GetWindowRect(self.ID)
        self.x = rect[0]
        self.y = rect[1]
        self.w = rect[2] - rect[0]
        self.h = rect[3] - rect[1]
        if self.debug:
            print(f"Window ({wg.GetWindowText(self.ID)}):")
            print(f"\tLocation: ({self.x}, {self.y})")
            print(f"\tSize: ({self.w}, {self.h})")


    def checkWindowIsOpen(self):
        try:
            self.update()
            return True
        except:
            return False

    def startUp(self):
        while True: # gets the window id
            try:
                wg.EnumWindows(self.callback, None)
                break
            except:
                print("Failed to get ID")
        self.update()

    def round(self, x, y):
        x = round(x, self.decimal)
        y = round(y, self.decimal)
        return x, y

    def windowToScreen(self, x, y):
        return self.x + x, self.y + y

    def screenToPercentage(self, xy: tuple) -> tuple:
        x = (xy[0] - self.x) / self.w
        y = (xy[1] - self.y) / self.h
        return self.round(x, y)

    def percentageToScreen(self, xy: tuple) -> tuple:
        x = self.x + xy[0] * self.w
        y = self.y + xy[1] * self.h
        return int(x), int(y)

    def click(self, _point: tuple[int]) -> None:
        self.p.moveTo(*_point)
        self.p.mouseDown()
        self.p.mouseUp()


    def clickPercent(self, _point: tuple[int]) -> None:
        _point = self.percentageToScreen(_point)
        self.p.moveTo(*_point)
        self.p.mouseDown()
        self.p.mouseUp()




































