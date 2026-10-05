# python/demo_app/chuzzle_demo/routes.py
"""Turns a solution into mouse drags.

Uses the mouse-path library from python/viewer (mouse/src/chuzzle_mouse.c,
built as python/viewer/libchuzzle_mouse.dll), which picks where to grab each
lane, the drag direction (the 2x2 piece cannot cross the edge) and the
shortest mouse travel. Without the DLL it falls back to plain drags.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import paths
from .board_sim import GRID, BoardState, apply_move, parse_token
from .levels import Level

if str(paths.VIEWER_DIR) not in sys.path:
    sys.path.insert(0, str(paths.VIEWER_DIR))
try:
    import chuzzle_mouse_adapter as _adapter
except Exception:  # pragma: no cover - viewer missing
    _adapter = None


def backend_available() -> bool:
    return _adapter is not None and _adapter.backend_available()


@dataclass
class Drag:
    move: str                               # stored token, e.g. "R342"
    points: list[tuple[float, float]]       # board cells: press, ..., release


@dataclass
class Route:
    move_string: str
    drags: list[Drag]
    cost: float | None                      # mouse travel in cells (library only)


def _library_route(level: Level, move_string: str, end_next_puzzle: bool) -> Route:
    solution = _adapter.solve_sequence(
        move_string,
        has_fat=level.has_fat,
        initial_fat_position=level.start_fat,
        end_next_puzzle=end_next_puzzle,
    )
    drags = []
    for move in solution.move_data:
        pts = [tuple(p) for p in (move.path_points or [move.click_down, move.lock_point, move.release])]
        drags.append(Drag(move.move, [(float(x), float(y)) for x, y in pts]))
    return Route(move_string, drags, solution.total_cost)


def _simple_route(level: Level, move_string: str) -> Route:
    """Grab the first cell of the lane and drag `n` cells; for the 2x2 piece
    pick the direction that keeps it on the board."""
    state = BoardState(list(level.start), level.start_fat)
    drags = []
    for token in move_string.split():
        axis, lines, amount = parse_token(token)
        disp = amount
        if state.fat is not None and len(lines) == 2:
            pos = state.fat[0] if axis == "R" else state.fat[1]
            if not 0 <= pos + disp <= GRID - 2:
                disp = amount - GRID
        elif amount > GRID // 2:
            disp = amount - GRID
        start = 0 if disp > 0 else GRID - 1
        line = lines[0]
        if axis == "R":
            pts = [(start, line), (start + disp, line)]
        else:
            pts = [(line, start), (line, start + disp)]
        drags.append(Drag(token, [(float(x), float(y)) for x, y in pts]))
        state = apply_move(state, token)
    return Route(move_string, drags, None)


def route_for(level: Level, move_string: str, end_next_puzzle: bool = False) -> Route:
    if backend_available():
        try:
            return _library_route(level, move_string, end_next_puzzle)
        except Exception:
            pass
    return _simple_route(level, move_string)


def rank_solutions(level: Level, workers: int = 4,
                   on_progress=None) -> list[tuple[float, str]]:
    """(mouse cost, move string), cheapest first. The library takes ~8 ms per
    solution; ctypes releases the GIL, so a few threads split big files
    (11-1 has 6877). Without the library every solution costs 0 and the stored
    order is kept."""
    sols = list(level.solutions)
    if not backend_available():
        return [(0.0, s) for s in sols]

    def score(s: str) -> tuple[float, str]:
        try:
            return (_library_route(level, s, False).cost or 0.0, s)
        except Exception:
            return (float("inf"), s)

    ranked = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, item in enumerate(pool.map(score, sols, chunksize=64)):
            ranked.append(item)
            if on_progress is not None and i % 256 == 0:
                on_progress(i, len(sols))
    ranked.sort()
    return ranked


# --------------------------------------------------------------------------
# Cache of rankings, keyed by solution file name + size + mtime
# --------------------------------------------------------------------------

RANK_CACHE = paths.USER_DATA_DIR / "route_rank_cache.json"


def _cache_key(level: Level) -> str | None:
    f = level.solution_file
    if f is None:
        return None
    st = f.stat()
    return f"{f.name}|{st.st_size}|{int(st.st_mtime)}|{int(backend_available())}"


def cached_ranking(level: Level) -> list[tuple[float, str]] | None:
    try:
        entry = json.loads(RANK_CACHE.read_text()).get(level.name)
        if entry and entry["key"] == _cache_key(level):
            return [(float(c), s) for c, s in entry["ranked"]]
    except Exception:
        pass
    return None


def store_ranking(level: Level, ranked: list[tuple[float, str]]) -> None:
    try:
        data = json.loads(RANK_CACHE.read_text())
    except Exception:
        data = {}
    data[level.name] = {"key": _cache_key(level), "ranked": [[round(c, 4), s] for c, s in ranked]}
    RANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    RANK_CACHE.write_text(json.dumps(data))
