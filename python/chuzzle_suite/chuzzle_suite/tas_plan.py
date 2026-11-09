# python/chuzzle_suite/chuzzle_suite/tas_plan.py
"""Plans a tool-assisted speedrun of Mind Bender for its trophies (no input, no Windows calls:
the TAS tab shows the plan, tas_runner.py plays it).

Game rules the plan follows (from the decomp of the 2008 Christmas exe, X:/chuzzle-decomp):
- Solving 3 puzzles of a level opens the next level (GameInterface_Mindbender::OnSolved).
- Trophies: 21 PUZZLER when the slide to another level ends on level 10 or later; 23 MENTALIST
  MASTER the same at level 20; 22 PSYCHIC when all 5 puzzles of levels 1-10 are solved; 24
  MENTALIST SUPREMO when all 100 are. 25 TROPHY COLLECTOR follows once every other trophy is won.
  A trophy the profile already has does not pop up again.
- Entering a level (or Mind Bender on a new profile) loads its puzzle 1, scrambled; any other
  puzzle needs a click on its star (or on the "play the next puzzle" band for the next one).

The run starts on the Mind Bender screen of a profile that has solved nothing (level 1 shows,
puzzle 1 loaded). Each level plays 3 or all 5 puzzles; with 3, the planner picks the 3 that
take the least time (fewest moves, then puzzle 1 because it needs no click). Moves can also end
with the 2x2 piece on another 2x2 block of its colour (the game accepts it; levels.Level
placements), where that is shorter: 4-4 takes 6 moves that way instead of 7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from . import levels as lv
from .board_sim import GRID, BoardState, apply_move, parse_token

PUZZLER, PSYCHIC, MENTALIST_MASTER, MENTALIST_SUPREMO, TROPHY_COLLECTOR = 21, 22, 23, 24, 25
MB_TROPHIES = (PUZZLER, PSYCHIC, MENTALIST_MASTER, MENTALIST_SUPREMO)
TROPHY_NAMES = {
    PUZZLER: "PUZZLER",
    PSYCHIC: "PSYCHIC",
    MENTALIST_MASTER: "MENTALIST MASTER",
    MENTALIST_SUPREMO: "MENTALIST SUPREMO",
    TROPHY_COLLECTOR: "TROPHY COLLECTOR!",
}
TROPHY_GOALS = {
    PUZZLER: "reach level 10",
    PSYCHIC: "solve all 5 puzzles of levels 1-10",
    MENTALIST_MASTER: "reach level 20",
    MENTALIST_SUPREMO: "solve all 100 puzzles",
}
LEVELS = 20
PUZZLES = 5
UNLOCK = 3                      # solved puzzles of a level that open the next one


@dataclass(frozen=True)
class TasOptions:
    target: int = PUZZLER                   # the trophy the run is for (21..24)
    per_level: int = 3                      # puzzles played per level: 3 or 5 (22 and 24 need 5)
    last_level: int | None = None           # last level whose puzzles are played; None: the target's
    owned: frozenset = frozenset()          # trophies the profile already has (no pop-up for them)
    others_owned: bool = False              # the profile has every trophy outside Mind Bender

    def resolved(self) -> "TasOptions":
        """Fill in the target's defaults: 22/24 need every puzzle, levels up to the target's."""
        per = PUZZLES if self.target in (PSYCHIC, MENTALIST_SUPREMO) else self.per_level
        last = self.last_level if self.last_level is not None else default_last_level(self.target)
        return TasOptions(self.target, per, max(1, min(LEVELS, last)), frozenset(self.owned), self.others_owned)

    @property
    def finish_arrow(self) -> bool:
        """21/23 end with the slide into level 10/20, after the last played level."""
        o = self.resolved()
        return o.target in (PUZZLER, MENTALIST_MASTER) and o.last_level + 1 == trophy_level(o.target)


def trophy_level(trophy: int) -> int:
    return {PUZZLER: 10, MENTALIST_MASTER: 20}.get(trophy, 0)


def default_last_level(target: int) -> int:
    return {PUZZLER: 9, PSYCHIC: 10, MENTALIST_MASTER: 19, MENTALIST_SUPREMO: 20}[target]


@dataclass(frozen=True)
class CostModel:
    """Seconds per action; only used to choose puzzles and to estimate the run's length.
    tas_runner.TasTiming.cost_model() gives the measured ones."""
    move_s: float = 0.03        # one drag (press, nudge, jump, release)
    click_s: float = 0.003      # one click on a star, the banner or the arrow
    puzzle_s: float = 0.0       # per puzzle (the same for every pick; the game has no scramble wait)
    slide_s: float = 0.3        # the level slide, when the run ends with it (21/23)
    trophy_s: float = 4.8       # a trophy pop-up, the trophy room and back


@dataclass(frozen=True)
class Pick:
    """One puzzle the run plays and the solution it uses."""
    puzzle: str                 # "4-4"
    goal: str                   # "4-4" or a placement the game also accepts, "4-4@r3c4"
    moves: str                  # stored (0-indexed) move string

    @property
    def move_count(self) -> int:
        return len(self.moves.split())

    @property
    def level(self) -> int:
        return int(self.puzzle.split("-")[0])

    @property
    def number(self) -> int:
        return int(self.puzzle.split("-")[1])


@dataclass(frozen=True)
class Step:
    kind: str                   # "split" | "open" | "solve" | "arrow" | "trophy"
    level: int = 0
    puzzle: int = 0
    how: str = ""               # open: "auto" (loaded by the game), "star", "next" (the band)
    pick: Pick | None = None    # solve
    trophy: int = 0             # trophy: the pop-up to click away; arrow/solve: the trophy it earns
    final: bool = False         # the step that earns the target trophy (the run's end)

    def describe(self) -> str:
        if self.kind == "split":
            return "LiveSplit key"
        if self.kind == "open":
            return {"auto": f"{self.level}-{self.puzzle} (loads by itself)",
                    "star": f"click star {self.puzzle} -> {self.level}-{self.puzzle}",
                    "next": f"click 'next puzzle' -> {self.level}-{self.puzzle}"}[self.how]
        if self.kind == "solve":
            goal = f" (2x2 at {self.pick.goal.split('@')[1]})" if "@" in self.pick.goal else ""
            return f"solve {self.pick.puzzle}{goal}: {lv.to_display(self.pick.moves)}"
        if self.kind == "arrow":
            return f"arrow to level {self.level}"
        return f"trophy pop-up: {TROPHY_NAMES.get(self.trophy, self.trophy)}"


@dataclass
class TasPlan:
    options: TasOptions
    steps: list[Step]
    picks: dict[int, list[Pick]]            # level -> puzzles in play order
    earned: list[int]                       # trophies earned, in order (pop-up or not)
    popups: list[int]                       # trophies whose pop-up the run clicks away
    missing: list[str] = field(default_factory=list)    # puzzles without a known solution
    cost: CostModel = field(default_factory=CostModel)

    @property
    def moves(self) -> int:
        return sum(p.move_count for ps in self.picks.values() for p in ps)

    @property
    def clicks(self) -> int:
        return sum(1 for s in self.steps if s.kind == "open" and s.how != "auto") + \
            sum(1 for s in self.steps if s.kind in ("arrow", "trophy"))

    @property
    def puzzles(self) -> int:
        return sum(len(ps) for ps in self.picks.values())

    def estimate_s(self, cost: CostModel | None = None) -> float:
        """The timer's time: from the first split to the last (or the end), so a pop-up after the
        target trophy does not count. Input goes on during a slide, so a slide only costs time
        when the run ends with it."""
        c = cost or self.cost
        total, splits = 0.0, 0
        for s in self.steps:
            if s.kind == "split":
                splits += 1
                if splits == 2:
                    break
            elif s.kind == "open":
                total += 0.0 if s.how == "auto" else c.click_s
            elif s.kind == "solve":
                total += s.pick.move_count * c.move_s + c.puzzle_s
            elif s.kind == "arrow":
                total += c.click_s + (c.slide_s if s.final else 0.0)
            elif s.kind == "trophy":
                total += c.trophy_s
        return total

    @property
    def ok(self) -> bool:
        return not self.missing


# --------------------------------------------------------------------------
# Choosing puzzles
# --------------------------------------------------------------------------

def best_pick(levels: dict[str, lv.Level], name: str) -> Pick | None:
    """The shortest way to solve puzzle `name`: its official goal or a placement of the 2x2
    piece the game also accepts (official on a tie), first solution of the stored file."""
    level = levels.get(name)
    if level is None:
        return None
    best: Pick | None = None
    for goal in [level] + list(level.placements):
        if not goal.solutions:
            continue
        pick = Pick(name, goal.name, goal.solutions[0])
        if best is None or pick.move_count < best.move_count:
            best = pick
    return best


def choose_level(levels: dict[str, lv.Level], level: int, count: int,
                 cost: CostModel = CostModel()) -> tuple[list[Pick], list[str]]:
    """The `count` puzzles of `level` that take the least time, in play order (puzzle 1 first:
    the game loads it on entering the level; then ascending). Returns (picks, missing)."""
    picks = {n: best_pick(levels, f"{level}-{n}") for n in range(1, PUZZLES + 1)}
    known = [n for n in picks if picks[n] is not None]
    missing = [f"{level}-{n}" for n in picks if picks[n] is None]
    if len(known) < count:
        return [picks[n] for n in known], missing

    def key(combo: tuple[int, ...]):
        moves = sum(picks[n].move_count for n in combo)
        clicks = len(combo) - (1 if 1 in combo else 0)
        return (moves * cost.move_s + clicks * cost.click_s, moves, combo)

    combo = min(combinations(known, count), key=key)
    return [picks[n] for n in sorted(combo)], ([] if count < PUZZLES else missing)


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------

class _Trophies:
    """Which trophies the run earns, and which of them pop up."""

    def __init__(self, options: TasOptions):
        self.owned = set(options.owned)
        self.others_owned = options.others_owned
        self.earned: list[int] = []
        self.popups: list[int] = []

    def award(self, trophy: int) -> list[int]:
        """The pop-ups this award shows (none if the profile already has it)."""
        if trophy in self.owned:
            return []
        self.owned.add(trophy)
        self.earned.append(trophy)
        shown = [trophy]
        if (trophy != TROPHY_COLLECTOR and self.others_owned and TROPHY_COLLECTOR not in self.owned
                and all(t in self.owned for t in MB_TROPHIES)):
            self.owned.add(TROPHY_COLLECTOR)
            self.earned.append(TROPHY_COLLECTOR)
            shown.append(TROPHY_COLLECTOR)
        self.popups += shown
        return shown


def build_plan(levels: dict[str, lv.Level], options: TasOptions, livesplit: bool = True,
               cost: CostModel = CostModel()) -> TasPlan:
    o = options.resolved()
    picks: dict[int, list[Pick]] = {}
    missing: list[str] = []
    for level in range(1, o.last_level + 1):
        chosen, miss = choose_level(levels, level, o.per_level, cost)
        picks[level] = chosen
        missing += miss

    trophies = _Trophies(o)
    solved: set[str] = set()
    steps: list[Step] = [Step("split")] if livesplit else []
    target_step: int | None = None

    def reached(trophies_now: list[int], kind: str, level: int, number: int = 0, pick=None) -> None:
        """Add a solve/arrow step that meets the conditions of `trophies_now`, then the pop-ups
        of those the profile does not have yet."""
        nonlocal target_step
        new = [t for t in trophies_now if t not in trophies.owned]
        steps.append(Step(kind, level, number, pick=pick, trophy=new[-1] if new else 0))
        if o.target in trophies_now and target_step is None:
            target_step = len(steps) - 1
        for t in trophies_now:
            for shown in trophies.award(t):
                steps.append(Step("trophy", trophy=shown))

    for level in range(1, o.last_level + 1):
        previous = 0
        for i, pick in enumerate(picks[level]):
            if i == 0 and pick.number == 1:
                how = "auto"
            elif previous and pick.number == previous + 1:
                how = "next"
            else:
                how = "star"
            steps.append(Step("open", level, pick.number, how))
            solved.add(pick.puzzle)
            now = ([PSYCHIC] if _all_solved(solved, 10) else []) + \
                  ([MENTALIST_SUPREMO] if _all_solved(solved, LEVELS) else [])
            reached(now, "solve", level, pick.number, pick)
            previous = pick.number
        if level < o.last_level or o.finish_arrow:
            nxt = level + 1
            now = ([PUZZLER] if nxt >= 10 else []) + ([MENTALIST_MASTER] if nxt >= 20 else [])
            reached(now, "arrow", nxt)

    if target_step is not None:
        steps[target_step] = _replace(steps[target_step], final=True)
        if livesplit:
            # the timer stops when the target trophy is earned, before its pop-up
            steps.insert(target_step + 1, Step("split"))
            # nothing after the target trophy's own pop-up is needed for the run
            end = target_step + 2
            while end < len(steps) and steps[end].kind == "trophy":
                end += 1
            del steps[end:]
    elif livesplit:
        steps.append(Step("split"))         # a shorter practice run: stop the timer at its end
    return TasPlan(o, steps, picks, trophies.earned, [s.trophy for s in steps if s.kind == "trophy"],
                   missing, cost)


def _all_solved(solved: set[str], up_to: int) -> bool:
    return all(f"{lvl}-{n}" in solved for lvl in range(1, up_to + 1) for n in range(1, PUZZLES + 1))


def _replace(step: Step, **changes) -> Step:
    from dataclasses import replace
    return replace(step, **changes)


# --------------------------------------------------------------------------
# Drags: straight lines along the lane (the cursor jumps; mouse travel costs nothing)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TasDrag:
    token: str                          # stored move, e.g. "R342"
    axis: str                           # "R" (the lane moves sideways) or "C" (up/down)
    press: tuple[int, int]              # the cell pressed (x, y), 0-indexed
    cells: int                          # signed: cells the lane moves (right / down positive)

    @property
    def release(self) -> tuple[int, int]:
        x, y = self.press
        return (x + self.cells, y) if self.axis == "R" else (x, y + self.cells)


def straight_drags(start: BoardState, move_string: str) -> list[TasDrag]:
    """Each move as one straight drag that starts and ends on the board: a normal lane moves at
    most 3 cells either way (5 right = 1 left); the 2x2 piece moves the way that keeps it on the
    board, pressed on the piece itself."""
    state = start.copy()
    out = []
    for token in move_string.split():
        axis, lines, amount = parse_token(token)
        fat = state.fat
        if fat is not None and len(lines) == 2:
            fx, fy = fat
            if axis == "R":
                d = (fx + amount) % GRID - fx
                press = (fx, fy)
            else:
                d = (fy + amount) % GRID - fy
                press = (fx, fy)
        else:
            d = amount if amount <= GRID // 2 else amount - GRID
            along = 0 if d > 0 else GRID - 1
            line = lines[0]
            press = (along, line) if axis == "R" else (line, along)
        out.append(TasDrag(token, axis, press, d))
        state = apply_move(state, token)
    return out


def drags_for(levels: dict[str, lv.Level], pick: Pick) -> list[TasDrag]:
    level = lv.find_level(levels, pick.goal)
    return straight_drags(BoardState(list(level.start), level.start_fat), pick.moves)
