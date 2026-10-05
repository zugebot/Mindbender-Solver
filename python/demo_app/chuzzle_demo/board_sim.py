# python/demo_app/chuzzle_demo/board_sim.py
"""Applies solver move strings to a 6x6 board.

Tokens (stored, 0-indexed): `R{y}{n}` shifts row y right by n, `C{x}{n}`
shifts column x down by n. On fat boards the two rows/columns under the 2x2
piece move together and are written with both lines, e.g. `R342`. The 2x2
piece cannot straddle the edge, so such a move is only legal when the piece
lands fully inside the board (same rule as mouse/src/chuzzle_mouse.c).
"""

from __future__ import annotations

from dataclasses import dataclass

GRID = 6


class IllegalMove(ValueError):
    pass


@dataclass
class BoardState:
    cells: list[int]
    fat: tuple[int, int] | None = None

    def copy(self) -> "BoardState":
        return BoardState(list(self.cells), self.fat)

    def get(self, x: int, y: int) -> int:
        return self.cells[y * GRID + x]


def parse_token(token: str) -> tuple[str, list[int], int]:
    token = token.strip().upper()
    if len(token) not in (3, 4) or token[0] not in "RC" or not token[1:].isdigit():
        raise IllegalMove(f"bad move token {token!r}")
    lines = [int(c) for c in token[1:-1]]
    amount = int(token[-1]) % GRID
    if any(not 0 <= line < GRID for line in lines):
        raise IllegalMove(f"line out of range in {token!r}")
    return token[0], lines, amount


def _fat_lines(axis: str, fat: tuple[int, int]) -> set[int]:
    fx, fy = fat
    return {fy, fy + 1} if axis == "R" else {fx, fx + 1}


def apply_move(state: BoardState, token: str) -> BoardState:
    axis, lines, amount = parse_token(token)
    out = state.copy()
    fat = state.fat

    if fat is not None:
        touched = set(lines) & _fat_lines(axis, fat)
        if touched and set(lines) != _fat_lines(axis, fat):
            raise IllegalMove(f"{token} moves only part of the 2x2 piece at {fat}")
        if set(lines) == _fat_lines(axis, fat):
            fx, fy = fat
            if axis == "R":
                fx = (fx + amount) % GRID
            else:
                fy = (fy + amount) % GRID
            if fx > GRID - 2 or fy > GRID - 2:
                raise IllegalMove(f"{token} would push the 2x2 piece across the edge")
            out.fat = (fx, fy)

    for line in lines:
        if axis == "R":
            row = [state.get(x, line) for x in range(GRID)]
            for x in range(GRID):
                out.cells[line * GRID + (x + amount) % GRID] = row[x]
        else:
            col = [state.get(line, y) for y in range(GRID)]
            for y in range(GRID):
                out.cells[((y + amount) % GRID) * GRID + line] = col[y]
    return out


def play(state: BoardState, move_string: str) -> list[BoardState]:
    """All states from the start through each move."""
    states = [state.copy()]
    for token in move_string.split():
        states.append(apply_move(states[-1], token))
    return states
