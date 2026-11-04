# python/chuzzle_suite/chuzzle_suite/levels.py
"""Mindbender boards (from src/chuzzles) and their known optimal solutions
(from build/levels_final).

The game counts a board as solved when every cell shows the goal's colour, so where the goal has
another 2x2 block in the 2x2 piece's colour, the piece may end there too. Each such placement is
a Level of its own ("4-4@r2c3": the piece's top-left cell at row 2, column 3, 1-indexed), kept
in its puzzle's `placements`, with solutions from build/levels_final/game_accepts. They are not
the official goal."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

GRID = 6

# Same patterns as python/deluxe_clone/puzzles/puzzles.py, pointed at the
# solver's own level tables so the app always matches the solver.
CELL_RE = re.compile(r"INIT\s+([bs]\d+_\d+)\s*\[36]\s*=\s*{([^}]*)};")
LEVEL_DEF_RE = re.compile(
    r"{\s*\"(\d+-\d+)\"\s*,\s*LevelCells::(b\d+_\d+)\s*,\s*LevelCells::(s\d+_\d+)\s*,\s*({[^}]*})\s*,\s*({[^}]*})\s*}"
)
FAT_RE = re.compile(r"{\s*true\s*,\s*(\d+)\s*,\s*(\d+)\s*}")
SOLUTION_FILE_RE = re.compile(r"^(\d+-\d+(?:@r\d+c\d+)?)_c(\d+)_(\d+)\.txt$")
BOUND_RE = re.compile(r"^\s*(\d+-\d+):\s*at least\s+(\d+)\s+moves", re.M)
PLACEMENT_BOUND_RE = re.compile(r"^\s*(\d+-\d+@r\d+c\d+):\s*at least\s+(\d+)\s+moves", re.M)
DEFAULT_LOWER_BOUND = 13   # every unsolved puzzle has no solution at 12 (2026-10-05)

# levels.hpp: RED 0, GREEN 1, BLUE 2, ORANGE 3, YELLOW 4, PURPLE 5, WHITE 6, CYAN 7
COLOR_NAMES = ["red", "green", "blue", "orange", "yellow", "purple", "white", "cyan"]
COLOR_RGB = [
    (226, 52, 52), (70, 190, 60), (50, 110, 230), (245, 140, 30),
    (240, 215, 40), (160, 70, 210), (235, 235, 235), (60, 210, 220),
]


@dataclass
class Level:
    name: str                      # "world-puzzle", e.g. "6-2"
    start: list[int]               # 36 colors, row-major
    goal: list[int]
    start_fat: tuple[int, int] | None
    goal_fat: tuple[int, int] | None
    solutions: list[str] = field(default_factory=list)   # stored (0-indexed) move strings
    optimum: int | None = None     # move count of the stored solutions
    solution_file: Path | None = None
    lower_bound: int | None = None   # unsolved: proven minimum move count (None: not searched)
    official: str | None = None      # a placement the game also accepts: its puzzle ("4-4")
    placements: list["Level"] = field(default_factory=list)   # the official level's game-accepted ones

    @property
    def world(self) -> int:
        return int(self.name.split("@")[0].split("-")[0])

    @property
    def puzzle(self) -> int:
        return int(self.name.split("@")[0].split("-")[1])

    @property
    def is_placement(self) -> bool:
        return self.official is not None

    @property
    def piece_text(self) -> str:
        """Where the goal's 2x2 piece sits, 1-indexed: 'rows 2-3, columns 3-4'."""
        if self.goal_fat is None:
            return ""
        x, y = self.goal_fat
        return f"rows {y + 1}-{y + 2}, columns {x + 1}-{x + 2}"

    @property
    def solved(self) -> bool:
        return bool(self.solutions)

    @property
    def has_fat(self) -> bool:
        return self.start_fat is not None


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _parse_fat(text: str) -> tuple[int, int] | None:
    match = FAT_RE.match(text.strip())
    return (int(match.group(1)), int(match.group(2))) if match else None


def game_accepted_fats(goal: list[int], goal_fat: tuple[int, int] | None) -> list[tuple[int, int]]:
    """Every other 2x2 block (not across an edge) in the piece's colour: the piece may end
    there and the game still counts the board as solved (as BoardLookup::gameAcceptedFats)."""
    if goal_fat is None:
        return []
    colour = goal[GRID * goal_fat[1] + goal_fat[0]]
    return [(x, y) for y in range(GRID - 1) for x in range(GRID - 1)
            if (x, y) != tuple(goal_fat)
            and all(goal[GRID * (y + dy) + x + dx] == colour for dx in (0, 1) for dy in (0, 1))]


def placement_name(puzzle: str, fat: tuple[int, int]) -> str:
    return f"{puzzle}@r{fat[1] + 1}c{fat[0] + 1}"


def find_level(levels: dict[str, Level], name: str) -> Level | None:
    """A puzzle ("4-4") or one of its game-accepted placements ("4-4@r2c3")."""
    level = levels.get(name.split("@")[0])
    if level is None or "@" not in name:
        return level
    return next((p for p in level.placements if p.name == name), None)


def all_levels(levels: dict[str, Level]):
    """Every puzzle, each followed by its game-accepted placements."""
    for level in levels.values():
        yield level
        yield from level.placements


def load_levels(levels_hpp: Path = paths.LEVELS_HPP,
                cells_hpp: Path = paths.LEVELS_CELLS_HPP,
                solutions_dir: Path = paths.LEVELS_FINAL_DIR,
                bounds_file: Path = paths.BUILD_README,
                placements_dir: Path | None = None) -> dict[str, Level]:
    cells = {}
    cells_text = strip_comments(cells_hpp.read_text())
    for symbol, values in CELL_RE.findall(cells_text):
        nums = [int(v) for v in values.split(",") if v.strip()]
        if len(nums) == GRID * GRID:
            cells[symbol] = nums

    levels: dict[str, Level] = {}
    levels_text = strip_comments(levels_hpp.read_text())
    for name, b, s, start_fat, goal_fat in LEVEL_DEF_RE.findall(levels_text):
        if b in cells and s in cells:
            levels[name] = Level(name, cells[b], cells[s], _parse_fat(start_fat), _parse_fat(goal_fat))

    for level in levels.values():
        level.placements = [Level(placement_name(level.name, fat), level.start, level.goal, level.start_fat,
                                  fat, official=level.name)
                            for fat in game_accepted_fats(level.goal, level.goal_fat)]
    attach_solutions(levels, solutions_dir)
    placements = {p.name: p for level in levels.values() for p in level.placements}
    attach_solutions(placements, placements_dir if placements_dir is not None else solutions_dir / "game_accepts")
    attach_bounds(levels, bounds_file)
    return levels


def attach_bounds(levels: dict[str, Level], bounds_file: Path) -> None:
    """Lower bounds of unsolved puzzles from build/README.md ("6-4: at least
    15 moves"), which the solver runs keep up to date."""
    bounds = {}
    if bounds_file.is_file():
        text = bounds_file.read_text(errors="replace")
        bounds = {name: int(n) for name, n in BOUND_RE.findall(text) + PLACEMENT_BOUND_RE.findall(text)}
    for level in levels.values():
        if not level.solved:
            level.lower_bound = bounds.get(level.name, DEFAULT_LOWER_BOUND)
        for p in level.placements:
            if not p.solved:
                p.lower_bound = bounds.get(p.name)   # None: not searched yet


def attach_solutions(levels: dict[str, Level], solutions_dir: Path) -> None:
    """Use the shortest stored file per puzzle (levels_final may also hold
    longer, non-optimal lists such as 13-4's c11 file)."""
    best: dict[str, tuple[int, Path]] = {}
    if solutions_dir.is_dir():
        for path in solutions_dir.iterdir():
            match = SOLUTION_FILE_RE.match(path.name)
            if not match:
                continue
            name = match.group(1)
            moves = int(match.group(2))
            if name not in best or moves < best[name][0]:
                best[name] = (moves, path)
    for name, (moves, path) in best.items():
        level = levels.get(name)
        if level is None:
            continue
        lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
        level.solutions = list(dict.fromkeys(lines))
        level.optimum = moves
        level.solution_file = path


def to_display(move_string: str) -> str:
    """Stored moves use 0-indexed lines (R04 = row 1 shifted 4). Show them
    1-indexed, e.g. 'R04 C342' -> 'R14 C453'."""
    out = []
    for token in move_string.split():
        lines = "".join(str(int(d) + 1) for d in token[1:-1])
        out.append(f"{token[0]}{lines}{token[-1]}")
    return " ".join(out)
