# Jerrin Shirks







class Point:
    """
    def __init__(self, other=(0, 0)):
        if type(other) in [tuple, list]:
            self.x = other[0]
            self.y = other[1]
        elif type(other) == Point:
            self.x = other.x
            self.y = other.y
        elif type(other) == int:
            self.x = other
            self.y = other
    """

    def __init__(self, *args):
        if len(args) == 0:
            self.x = 0
            self.y = 0

        elif len(args) == 1:
            args = args[0]

            if type(args) in [tuple, list]:
                if len(args) < 2:
                    raise Exception("Passed tuple/list is too smalL!")
                self.x = args[0]
                self.y = args[1]

            elif type(args) == Point:
                self.x = args.x
                self.y = args.y
            elif type(args) in [int, float]:
                self.x = args
                self.y = args


        elif len(args) == 2:
            self.x = args[0]
            self.y = args[1]



    def copy(self):
        _point = Point()
        _point.x = self.x
        _point.y = self.y
        return _point

    def tuple(self):
        return self.x, self.y

    def clear(self):
        self.x = 0
        self.y = 0

    def get(self, _index):
        if _index == 0:
            return self.x
        elif _index == 1:
            return self.y
        else:
            return 0

    def swap(self):
        self.x, self.y = self.y, self.x

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __eq__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            if _point.x == other[0] and _point.y == other[1]:
                return True
        elif type(other) == tuple:
            other = list(other)
            if _point.x == other[0] and _point.y == other[1]:
                return True
        elif type(other) == Point:
            if _point.x == other.x and _point.y == other.y:
                return True
        elif type(other) in [int, float]:
            if _point.x == other and _point.y == other:
                return True
        return False

    # bloat code remove later lol!
    def eq(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            if _point.x == other[0] and _point.y == other[1]:
                return True
        elif type(other) == tuple:
            other = list(other)
            if _point.x == other[0] and _point.y == other[1]:
                return True
        elif type(other) == Point:
            if _point.x == other.x and _point.y == other.y:
                return True
        elif type(other) in [int, float]:
            if _point.x == other and _point.y == other:
                return True
        return False

    def __abs__(self):
        _point = Point((self.x, self.y))
        _point.x = abs(_point.x)
        _point.y = abs(_point.y)
        return _point

    def __add__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x += other[0]
            _point.y += other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x += other[0]
            _point.y += other[1]
        elif type(other) == Point:
            _point.x += other.x
            _point.y += other.y
        elif type(other) in [int, float]:
            _point.x += other
            _point.y += other
        return _point

    def __sub__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x -= other[0]
            _point.y -= other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x -= other[0]
            _point.y -= other[1]
        elif type(other) == Point:
            _point.x -= other.x
            _point.y -= other.y
        elif type(other) in [int, float]:
            _point.x -= other
            _point.y -= other
        return _point

    def __mul__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x *= other[0]
            _point.y *= other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x *= other[0]
            _point.y *= other[1]
        elif type(other) == Point:
            _point.x *= other.x
            _point.y *= other.y
        elif type(other) in [int, float]:
            _point.x *= other
            _point.y *= other
        return _point

    def __truediv__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x /= other[0]
            _point.y /= other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x /= other[0]
            _point.y /= other[1]
        elif type(other) == Point:
            _point.x /= other.x
            _point.y /= other.y
        elif type(other) in [int, float]:
            _point.x /= other
            _point.y /= other
        return _point



    def __mod__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x %= other[0]
            _point.y %= other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x %= other[0]
            _point.y %= other[1]
        elif type(other) == Point:
            _point.x %= other.x
            _point.y %= other.y
        elif type(other) == int:
            _point.x %= other
            _point.y %= other
        return _point

    def __floordiv__(self, other):
        _point = Point((self.x, self.y))
        if type(other) == list:
            _point.x //= other[0]
            _point.y //= other[1]
        elif type(other) == tuple:
            other = list(other)
            _point.x //= other[0]
            _point.y //= other[1]
        elif type(other) == Point:
            _point.x //= other.x
            _point.y //= other.y
        elif type(other) in [int, float]:
            _point.x //= other
            _point.y //= other
        return _point

    def __round__(self):
        x = round(self.x)
        y = round(self.y)
        return Point((x, y))

    def toFloat(self):
        _point = Point((self.x, self.y))
        _point.x = float(_point.x)
        _point.y = float(_point.y)
        return _point

    def multiply(self):
        return self.x * self.y

