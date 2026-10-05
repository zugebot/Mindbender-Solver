import os

def solveToTuples(_text: str, backwards=None) -> list[tuple[int]]:
    if backwards is None:
        backwards = []
    _moves = []
    _text = _text.strip()
    if " " in _text:
        moveset = _text.split(" ")
    else:
        moveset = [_text]
    for n, _move in enumerate(moveset):
        _line = [i for i in _move]
        _line[1] = int(_line[1])
        _line[2] = int(_line[2])

        if _line[0] == "R":
            _moves.append(((0, _line[1]), (_line[2], _line[1])))
        elif _line[0] == "C":
            _moves.append(((_line[1], 0), (_line[1], _line[2])))
        elif _line[0] == "L":
            _moves.append(((5, _line[1]), (5 - _line[2], _line[1])))
        elif _line[0] == "U":
            _moves.append(((_line[1], 5), (_line[1], 5 - _line[2])))

    return _moves
