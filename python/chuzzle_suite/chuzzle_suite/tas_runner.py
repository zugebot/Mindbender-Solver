# python/chuzzle_suite/chuzzle_suite/tas_runner.py
"""Plays a TAS plan (tas_plan.py) in the real game with the real mouse, on a worker thread.

How the game takes input (decomp of the 2008 Christmas exe, X:/chuzzle-decomp; the Mind Bender
logic is the same in the 2005 and 2007 builds):
- It updates at 100 Hz and handles every queued mouse message, in order, at the start of each
  update. Everything a run needs happens inside those handlers: a release snaps the lane and
  checks the board; a solved board makes the "next puzzle" banner live at once; a star loads and
  scrambles its puzzle at once (no animation); the level arrow sets up the next level at once and
  input is taken during its 0.29 s slide. So no input has to wait for the game, with three
  exceptions handled here:
  * a drag starts at the first move more than 2 px from the press, and its offset counts from
    that move. Windows merges cursor moves the game has not read yet, so the small first move and
    the move to the target must reach the game as two messages: either a gap of one main-loop
    pass between them, or another input between them (an F24 key press, which the game ignores).
    "Measure speed" finds which works and how short the gap can be (user_data/tas_timing.json).
  * a trophy pop-up (TrophyAward) blocks all input until it takes a press: 4.32 s after the
    arrow click into level 10/20 (21/23 pop up as the 0.30 s slide ends, and abort a drag in
    progress, so nothing is sent in between), 4.00 s after the solve that earns 22/24. The press
    opens the trophy room; its "Return to Game" button works 0.63 s later (it needs the pointer
    on it an update before the press), and the board takes drags 0.62 s after that. These were
    measured on the original exe with the decomp's replay harness (scripted input at exact
    frames), as was a whole PUZZLER, PSYCHIC and MENTALIST MASTER run of this runner's inputs.
  * buttons fire on release only when hovered (a move onto them first), and a second press within
    ~4 px of the previous one is a double-click, which buttons ignore (the board drags anyway).
    No two clicks of a run are that close.
- Every input goes through one SendInput stream (absolute moves, button and key events), so
  presses, moves and releases reach the game in the order sent. SetCursorPos is synchronous while
  button events are queued, so mixing the two can let a press land after the next move.

Safety: holding Esc, Stop, the game window closing or losing the foreground stop the run between
two inputs; the button is always released on the way out. With checks on, the screen is read
before the start (the board must be 1-1, scrambled) and before each level arrow (every puzzle the
run played on the level must show its solved star: the green check appears within 4 updates of
the solve and stays), and the run stops on a mismatch.
"""

from __future__ import annotations

import ctypes
import json
import threading
import time
from ctypes import wintypes
from dataclasses import asdict, dataclass, fields, replace
from typing import Callable

from . import game_window as gw
from . import levels as lv
from . import paths
from . import tas_plan as tp

TIMING_FILE = paths.USER_DATA_DIR / "tas_timing.json"
VK_NUMPAD1 = 0x61               # the 2023 run's LiveSplit start/split key
VK_F24 = 0x87                   # no key sends it; the game ignores it (unlike Shift: no Sticky Keys)

# Logical 800x600 positions (decomp: GameInterface_Mindbender.cpp, Board.cpp, TrophyRoom.cpp).
BOARD_X, BOARD_Y, CELL = 255, 57, 80          # board rect (255, 57, 480, 480); cell centres +40
STARS = [(40, 342), (81, 371), (127, 379), (174, 370), (216, 346)]   # 40x40 circles
NEXT_ARROW = (198, 482)                       # hit rect (169, 452, 58, 61)
BANNER = (496, 332)                           # "next puzzle" / "play again": (315, 317, 363, 31)
TROPHY_CLICK = (620, 140)                     # the pop-up takes a press anywhere
TROPHY_BACK = (413, 563)                      # the trophy room's Back button (280, 533, 267, 60)
PAUSE_CLICK = gw.PAUSE_BANNER_CLICK           # a click anywhere ends the focus-loss pause

# Game constants, measured on the original exe with the decomp's replay harness (updates of 10 ms,
# counted from the update that handles the input)
SLIDE_S = 0.30                  # arrow click -> slide ends (21/23 are awarded then)
POPUP_AFTER_ARROW_S = 4.32      # arrow click into level 10/20 -> the pop-up takes a press (432)
POPUP_AFTER_SOLVE_S = 4.00      # solving release (22/24) -> the pop-up takes a press (397-400)
ROOM_IN_S = 0.63                # that press -> the trophy room's "Return to Game" works (62)
ROOM_OUT_S = 0.62               # "Return to Game" -> the board takes drags again (60-62)
HOVER_S = 0.02                  # trophy screens: the pointer must rest a frame before the press
SOLVE_CHECK_GRACE_S = 0.6       # how long a board check waits for the screen to show the result
LEVEL_CHECK_S = 0.05            # what a level's star check costs (the newest check is drawn ~4
                                # updates after the solve, then a capture)

# Fur colours as the Christmas Edition draws them (levels.hpp order: red, green, blue, orange,
# yellow, purple, white, cyan). The reader goes by hue, since Deluxe's fur is more saturated
# (green 12,178,12) and its colour-blind option draws a symbol on each chuzzle.
PALETTE = [(167, 64, 64), (65, 166, 65), (67, 67, 250), (249, 130, 4), (249, 249, 67), (203, 69, 203),
           (247, 247, 247), (85, 192, 216)]
# hue ranges (degrees) of the coloured ones; white is any bright colour with little saturation
HUES = [(0, 15, 0), (15, 45, 3), (45, 90, 4), (90, 160, 1), (160, 215, 7), (215, 270, 2), (270, 330, 5),
        (330, 361, 0)]
# four patches on each chuzzle's fur: its sides, clear of the eyes (below the centre) and of the
# colour-blind symbol (above it)
FUR_PATCHES = [(-24, -2), (24, -2), (-20, -16), (20, -16)]


def fur_colour(rgb) -> int | None:
    import colorsys
    r, g, b = (v / 255 for v in rgb)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.3:
        return None
    if s < 0.25:
        return 6 if v > 0.6 else None
    if s < 0.4:
        return None
    deg = h * 360
    return next(colour for lo, hi, colour in HUES if lo <= deg < hi)


@dataclass
class TasTiming:
    """How inputs are paced (seconds). Only the nudge-to-jump separation depends on the PC; the
    game itself needs no waits (see the module doc)."""
    key_break: bool = False     # an F24 press between the nudge and the jump keeps them apart
    gap: float = 0.025          # else: wait this long between them (one main-loop pass or more)
    pace: float = 0.002         # after each drag or click
    nudge_px: float = 4.0       # logical px (the game starts a drag after 2 px)
    margin: float = 0.10        # added to the game's fixed waits (trophy pop-up, trophy room)
    measured: bool = False

    def drag_s(self) -> float:
        """One drag: the waits plus ~2.5 ms of SendInput and checks (a real PUZZLER run: 178 drags
        and 28 clicks in ~0.5 s)."""
        return (0.0 if self.key_break else self.gap) + self.pace + 0.0025

    def cost_model(self, checks: bool = False) -> tp.CostModel:
        return tp.CostModel(move_s=self.drag_s(), click_s=self.pace + 0.001, puzzle_s=0.0,
                            slide_s=SLIDE_S, check_s=LEVEL_CHECK_S if checks else 0.0,
                            trophy_s=POPUP_AFTER_SOLVE_S + ROOM_IN_S + ROOM_OUT_S + 3 * self.margin
                            + 2 * HOVER_S)


def _timing_key(window) -> str:
    if window is None:
        return "default"
    return str(window.exe) if window.exe else window.title


def load_timing(window) -> TasTiming:
    try:
        entry = json.loads(TIMING_FILE.read_text())[_timing_key(window)]
        names = {f.name for f in fields(TasTiming)}
        return TasTiming(**{k: v for k, v in entry.items() if k in names})
    except Exception:
        return TasTiming()


def save_timing(window, timing: TasTiming) -> None:
    try:
        data = json.loads(TIMING_FILE.read_text())
    except Exception:
        data = {}
    data[_timing_key(window)] = asdict(timing)
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)
    TIMING_FILE.write_text(json.dumps(data, indent=1))


def clear_timing(window) -> None:
    try:
        data = json.loads(TIMING_FILE.read_text())
        data.pop(_timing_key(window), None)
        TIMING_FILE.write_text(json.dumps(data, indent=1))
    except Exception:
        pass


# --------------------------------------------------------------------------
# SendInput: one ordered stream of absolute moves, button and key events
# --------------------------------------------------------------------------

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP = 0x0001, 0x0002, 0x0004
MOUSEEVENTF_VIRTUALDESK, MOUSEEVENTF_ABSOLUTE = 0x4000, 0x8000
KEYEVENTF_KEYUP = 0x0002
SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN = 76, 77, 78, 79
ULONG_PTR = ctypes.c_size_t


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ULONG_PTR)]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUTUNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]


def _to_input(event, desk) -> list[_INPUT]:
    vx, vy, vw, vh = desk
    kind = event[0]
    if kind == "move":
        x, y = event[1], event[2]
        nx = ((x - vx) * 65536 + vw - 1) // vw          # rounds back to exactly x
        ny = ((y - vy) * 65536 + vh - 1) // vh
        mi = _MOUSEINPUT(nx, ny, 0, MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, 0)
        return [_INPUT(INPUT_MOUSE, _INPUTUNION(mi=mi))]
    if kind in ("down", "up"):
        flag = MOUSEEVENTF_LEFTDOWN if kind == "down" else MOUSEEVENTF_LEFTUP
        return [_INPUT(INPUT_MOUSE, _INPUTUNION(mi=_MOUSEINPUT(0, 0, 0, flag, 0, 0)))]
    vk = event[1]
    return [_INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=_KEYBDINPUT(vk, 0, 0, 0, 0))),
            _INPUT(INPUT_KEYBOARD, _INPUTUNION(ki=_KEYBDINPUT(vk, 0, KEYEVENTF_KEYUP, 0, 0)))]


def send(events: list[tuple]) -> None:
    """Send ("move", x, y) screen px / ("down",) / ("up",) / ("key", vk) as one SendInput call."""
    if not events:
        return
    u = ctypes.windll.user32
    desk = (u.GetSystemMetrics(SM_XVIRTUALSCREEN), u.GetSystemMetrics(SM_YVIRTUALSCREEN),
            max(1, u.GetSystemMetrics(SM_CXVIRTUALSCREEN)), max(1, u.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
    items = [i for e in events for i in _to_input(e, desk)]
    array = (_INPUT * len(items))(*items)
    sent = u.SendInput(len(items), array, ctypes.sizeof(_INPUT))
    if sent != len(items):
        raise OSError(f"SendInput sent {sent} of {len(items)} inputs (blocked by another program?)")


# --------------------------------------------------------------------------
# Reading the board from the screen
# --------------------------------------------------------------------------

def _pixel(shot, lx: float, ly: float) -> tuple[int, int, int]:
    data, cw, ch = shot
    x = min(cw - 1, max(0, int(lx * cw / gw.LOGICAL_W)))
    y = min(ch - 1, max(0, int(ly * ch / gw.LOGICAL_H)))
    i = (y * cw + x) * 4
    return data[i + 2], data[i + 1], data[i]


def board_from_shot(shot) -> list[int | None]:
    """36 colour indices (None where the fur matches no colour, and for the two rows under the
    SOLVED! band while it shows: its green tint turns some of them into false greens)."""
    cells: list[int | None] = []
    band = band_from_shot(shot)
    for r in range(6):
        for c in range(6):
            if band and r in (2, 3):
                cells.append(None)
                continue
            cx, cy = BOARD_X + 40 + CELL * c, BOARD_Y + 40 + CELL * r
            votes: dict[int, int] = {}
            for px, py in FUR_PATCHES:
                pts = [_pixel(shot, cx + px + dx, cy + py + dy) for dx in (-2, 0, 2) for dy in (-2, 0, 2)]
                colour = fur_colour([sum(p[i] for p in pts) / len(pts) for i in range(3)])
                if colour is not None:
                    votes[colour] = votes.get(colour, 0) + 1
            best = max(votes, key=votes.get) if votes else None
            cells.append(best if best is not None and votes[best] >= 3 else None)
    return cells


def band_from_shot(shot) -> bool:
    """The green SOLVED! band: the dark gaps between chuzzles at y=300 turn bright green."""
    green = 0
    for k in range(5):
        for y in (296, 300, 304):
            r, g, b = _pixel(shot, 335 + 80 * k, y)
            green += g > 120 and g > r + 60 and g > b + 60
    return green >= 9


STAR_SAMPLES = [(dx, dy) for dx in (-9, -6, -3, 0, 3, 6, 9) for dy in (-9, -6, -3, 0, 3, 6, 9)
                if dx * dx + dy * dy <= 81]


def stars_from_shot(shot) -> list[bool]:
    """Which of the level's 5 stars show the green check of a solved puzzle (it appears within
    4 updates of the solving release and stays while the level is shown)."""
    out = []
    for cx, cy in STARS:
        px = [_pixel(shot, cx + dx, cy + dy) for dx, dy in STAR_SAMPLES]
        r, g, b = (sum(p[i] for p in px) / len(px) for i in range(3))
        out.append(g > 130 and g > r + 60 and g > b + 50)
    return out


def read_board(window) -> list[int | None] | None:
    shot = gw.capture_client(window)
    return None if shot is None else board_from_shot(shot)


def board_matches(cells: list[int | None] | None, expected: list[int], min_read: int = 36) -> bool:
    if cells is None:
        return False
    read = [(c, e) for c, e in zip(cells, expected) if c is not None]
    return len(read) >= min_read and all(c == e for c, e in read)


def identify(cells: list[int | None], levels: dict[str, lv.Level]) -> tuple[str, str] | None:
    """Which puzzle the board shows: (name, "start" | "goal"), or None. Cells hidden by the
    SOLVED! band are skipped (at least 24 of 36 must read)."""
    for state in ("start", "goal"):
        for name, level in levels.items():
            if board_matches(cells, getattr(level, state), min_read=24 if state == "goal" else 36):
                return name, state
    return None


class Stopped(Exception):
    pass


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

class _Driver:
    """Shared input and safety code of the run and the speed measurement."""

    def __init__(self, window, levels, timing: TasTiming, on_status=None):
        self.window = window
        self.levels = levels
        self.timing = timing
        self.on_status = on_status or (lambda s: None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._down = False
        self.started_at: float | None = None
        self._buffer: list[tuple] | None = None     # None: every input goes out at once

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        raise NotImplementedError

    # safety and waiting

    def _check(self) -> None:
        if self._stop.is_set():
            raise Stopped("stopped")
        if gw.key_down(gw.VK_ESCAPE):
            raise Stopped("Esc pressed")
        if not self.window.alive():
            raise Stopped("game window closed")
        if not self.window.is_foreground():
            raise Stopped("the game window lost the foreground")

    def _until(self, deadline: float) -> None:
        self._flush()                   # what is queued goes out before any wait
        while True:
            self._check()
            left = deadline - time.perf_counter()
            if left <= 0:
                return
            time.sleep(0.001 if left > 0.002 else 0)

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self._until(time.perf_counter() + seconds)

    # coordinates

    def _screen(self, lx: float, ly: float) -> tuple[int, int]:
        left, top, cw, ch = self.window.client_rect()
        return round(left + lx * cw / gw.LOGICAL_W), round(top + ly * ch / gw.LOGICAL_H)

    @staticmethod
    def cell_point(x: float, y: float) -> tuple[float, float]:
        return BOARD_X + 40 + CELL * x, BOARD_Y + 40 + CELL * y

    # inputs

    def _send(self, events: list[tuple]) -> None:
        self._check()
        buttons = [e[0] for e in events if e[0] in ("down", "up")]
        if buttons and buttons[-1] == "down":
            self._down = True               # set first: a failed send still gets its release
        send(events)
        if buttons and buttons[-1] == "up":
            self._down = False

    def _emit(self, events: list[tuple]) -> None:
        """Send now, or add to the burst being collected (TasRunner.burst)."""
        if self._buffer is None:
            self._send(events)
        else:
            self._buffer.extend(events)

    def _flush(self) -> None:
        if self._buffer:
            events = list(self._buffer)
            self._buffer.clear()
            self._send(events)

    def click(self, lx: float, ly: float, hover: bool = False) -> None:
        """Move there, press, release. The Mind Bender screen's buttons take all three in one
        update; the trophy room's button only counts a press after an update with the pointer
        on it (hover=True: a frame's rest first)."""
        x, y = self._screen(lx, ly)
        if hover:
            self._flush()
            self._send([("move", x, y)])
            self._wait(HOVER_S)
            self._send([("down",), ("up",)])
        else:
            self._emit([("move", x, y), ("down",), ("up",)])
        self._wait(self.timing.pace)

    def drag(self, d: tp.TasDrag, t: TasTiming | None = None) -> None:
        """Press, nudge along the lane, jump to one nudge past the target cell, release: the
        offset counts from the nudge, so the lane moves exactly d.cells cells."""
        t = t or self.timing
        px, py = self.cell_point(*d.press)
        sign = 1 if d.cells > 0 else -1
        nx, ny = (px + sign * t.nudge_px, py) if d.axis == "R" else (px, py + sign * t.nudge_px)
        tx, ty = (nx + CELL * d.cells, ny) if d.axis == "R" else (nx, ny + CELL * d.cells)
        first = [("move",) + self._screen(px, py), ("down",), ("move",) + self._screen(nx, ny)]
        second = [("move",) + self._screen(tx, ty), ("up",)]
        if t.key_break:
            self._emit(first + [("key", VK_F24)] + second)
        else:
            self._flush()
            self._send(first)
            self._wait(t.gap)
            self._send(second)
        self._wait(t.pace)

    def release_button(self) -> None:
        if self._down:
            try:
                send([("up",)])
            except Exception:
                pass
            self._down = False

    # game state

    def bring_front(self) -> None:
        if not self.window.is_foreground():
            if not self.window.bring_to_front():
                raise Stopped("could not bring the game window to the front")
            time.sleep(0.3)
        # the game pauses when it loses focus (e.g. the click on Start); a click ends the pause
        if gw.looks_paused(self.window):
            self.on_status("the game was paused: resuming it")
            self.click(*PAUSE_CLICK)
            time.sleep(0.4)
            if gw.looks_paused(self.window):
                raise Stopped("the game stays paused")

    def wait_board(self, expected: list[int], allow_band: bool, timeout: float = SOLVE_CHECK_GRACE_S) -> bool | None:
        """Poll the screen until the board shows `expected` (or the SOLVED! band, if allowed).
        None when the window cannot be captured (no check possible)."""
        give_up = time.perf_counter() + timeout
        while True:
            self._check()
            shot = gw.capture_client(self.window)
            if shot is None:
                return None
            if board_matches(board_from_shot(shot), expected) or (allow_band and band_from_shot(shot)):
                return True
            if time.perf_counter() > give_up:
                return False
            time.sleep(0.005)

    def wait_stars(self, numbers: list[int], timeout: float = SOLVE_CHECK_GRACE_S) -> list[int] | None:
        """Poll until the stars `numbers` (1-5) have each shown solved. Returns the ones that never
        did ([] = all fine), or None when the window cannot be captured. A star counts once it has
        been seen with its check: the newest one glints for 25 updates, hiding it now and then."""
        give_up = time.perf_counter() + timeout
        seen: set[int] = set()
        while True:
            self._check()
            shot = gw.capture_client(self.window)
            if shot is None:
                return None
            solved = stars_from_shot(shot)
            seen |= {n for n in numbers if solved[n - 1]}
            missing = [n for n in numbers if n not in seen]
            if not missing or time.perf_counter() > give_up:
                return missing
            time.sleep(0.005)


BURSTS = ("drag", "puzzle", "level", "run")


class TasRunner(_Driver):
    """Plays `plan`. Callbacks run on the worker thread: post them to the UI.

    burst: how many inputs go out in one SendInput call: each drag or click on its own
    ("drag"), each puzzle's drags with the click that opened it ("puzzle"), a whole level up to
    and with its arrow ("level"), or everything until something must wait ("run"). The events
    and their order are the same; anything that waits (a pop-up, the last slide, a screen check)
    sends what is queued first. Esc and Stop are checked between bursts only. Bursts need the
    F24 key break and no pause after drags (else every drag goes out on its own)."""

    def __init__(self, window, geometry, levels, plan: tp.TasPlan, timing: TasTiming,
                 check_solves: bool = True, livesplit_vk: int = VK_NUMPAD1, burst: str = "drag",
                 on_step: Callable[[int], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 on_done: Callable[[str, float | None], None] | None = None):
        super().__init__(window, levels, timing, on_status)
        self.geometry = geometry            # kept for the UI; the TAS uses the game's own layout
        self.plan = plan
        self.check_solves = check_solves
        self.livesplit_vk = livesplit_vk
        self.on_step = on_step or (lambda i: None)
        self.on_done = on_done or (lambda msg, secs: None)
        self.finished_at: float | None = None
        self._after_room = False        # the first click after a trophy room rests on its target first
        self.burst = burst if burst in BURSTS else "drag"
        self.bursts_sent = 0

    def _run(self) -> None:
        message, elapsed = "done", None
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
        try:
            self.bring_front()
            self._check_start()
            self._play()
            if self.started_at is not None and self.finished_at is not None:
                elapsed = self.finished_at - self.started_at
        except Stopped as exc:
            message = f"stopped: {exc}"
        except Exception as exc:
            message = f"error: {exc}"
        finally:
            self.release_button()
            winmm.timeEndPeriod(1)
            self.on_done(message, elapsed)

    def _check_start(self) -> None:
        """The run starts on 1-1, scrambled (a profile with no Mind Bender save)."""
        if not self.check_solves:
            return
        first = next((s for s in self.plan.steps if s.kind == "solve"), None)
        if first is None:
            return
        cells = read_board(self.window)
        if cells is None:
            return
        start = lv.find_level(self.levels, first.pick.goal).start
        if not board_matches(cells, start):
            seen = identify(cells, self.levels)
            shows = f"{seen[0]} ({'solved' if seen[1] == 'goal' else 'scrambled'})" if seen else "something else"
            raise Stopped(f"the game should show {first.pick.puzzle} scrambled, but shows {shows}")

    def _send(self, events: list[tuple]) -> None:
        super()._send(events)
        self.bursts_sent += 1

    def _play(self) -> None:
        t = self.timing
        if self.burst != "drag" and t.key_break and t.pace == 0:
            self._buffer = []
        event_at = time.perf_counter()      # when the last trophy-earning input went out
        for i, step in enumerate(self.plan.steps):
            self.on_step(i)
            if step.kind == "split":
                self._flush()
                if self.started_at is None:
                    self.started_at = time.perf_counter()
                else:
                    self.finished_at = time.perf_counter()
                if self.livesplit_vk:
                    self._send([("key", self.livesplit_vk)])
                continue
            if self.started_at is None:
                self.started_at = time.perf_counter()
            if step.kind == "open":
                if step.how == "star":
                    self.click(*STARS[step.puzzle - 1], hover=self._after_room)
                    self._after_room = False
                elif step.how == "next":
                    self.click(*BANNER, hover=self._after_room)
                    self._after_room = False
            elif step.kind == "solve":
                if self._buffer is None:
                    self.on_status(f"{step.pick.puzzle}: {step.pick.move_count} moves")
                for d in tp.drags_for(self.levels, step.pick):
                    self.drag(d)
                if self.burst == "puzzle":
                    self._flush()
                event_at = time.perf_counter()
            elif step.kind == "arrow":
                self._check_level(step.level - 1)
                self.click(*NEXT_ARROW, hover=self._after_room)
                self._after_room = False
                if self.burst in ("puzzle", "level"):
                    self._flush()
                event_at = time.perf_counter()
                if step.final:
                    self._wait(SLIDE_S)          # the trophy comes as the slide ends
            elif step.kind == "trophy":
                self.on_status(f"trophy pop-up: {tp.TROPHY_NAMES.get(step.trophy, step.trophy)}")
                ready = POPUP_AFTER_ARROW_S if self._earned_by_arrow(i) else POPUP_AFTER_SOLVE_S
                self._until(event_at + ready + t.margin - HOVER_S)
                self.click(*TROPHY_CLICK, hover=True)   # into the trophy room
                self._wait(ROOM_IN_S + t.margin - HOVER_S)
                self.click(*TROPHY_BACK, hover=True)    # and back to the game
                self._after_room = True
                event_at = time.perf_counter()          # a chained pop-up (25) counts from here
                self._wait(ROOM_OUT_S + t.margin)
        self._flush()
        if not any(s.final for s in self.plan.steps) and self.plan.picks:
            self._check_level(max(self.plan.picks))      # a practice run: its last level too
        if self.finished_at is None:
            self.finished_at = time.perf_counter()

    def _check_level(self, level: int) -> None:
        """With checks on: every puzzle the run played on `level` shows its solved star. A
        puzzle the game did not take would otherwise leave the arrow locked (or the board wrong)
        and the rest of the run would play on the wrong board."""
        if not self.check_solves or level not in self.plan.picks:
            return
        self._flush()
        numbers = [p.number for p in self.plan.picks[level]]
        missing = self.wait_stars(numbers)
        if missing:
            names = ", ".join(f"{level}-{n}" for n in missing)
            raise Stopped(f"{names} did not register as solved (inputs too fast? measure the speed)")

    def _earned_by_arrow(self, i: int) -> bool:
        """Was step i's trophy earned by a level arrow (it pops up as the slide ends)? A chained
        one (25 after another) pops up as the game comes back from the trophy room."""
        j = i - 1
        while j >= 0 and self.plan.steps[j].kind == "split":
            j -= 1
        return j >= 0 and self.plan.steps[j].kind == "arrow"


class SpeedMeasure(_Driver):
    """Finds the fastest reliable input timing on the puzzle the game shows: replays its solution
    over and over (reset = its star, then the banner, which re-scrambles a solved puzzle) and
    reads the board after each try. First tries an F24 press between nudge and jump with no
    wait; else the shortest gap that solves every try. on_done(message, TasTiming | None)."""

    GAPS = (0.0, 0.002, 0.004, 0.006, 0.008, 0.011, 0.014, 0.018, 0.024, 0.032, 0.045, 0.06)
    TRIES = 4           # every try of a candidate must solve
    CONFIRM = 6         # then this many more at the chosen setting

    def __init__(self, window, geometry, levels, timing: TasTiming,
                 on_status: Callable[[str], None] | None = None,
                 on_done: Callable[[str, TasTiming | None], None] | None = None):
        super().__init__(window, levels, timing, on_status)
        self.geometry = geometry
        self.on_done = on_done or (lambda msg, t: None)
        self.tries = 0

    def _run(self) -> None:
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)
        result, message = None, ""
        try:
            self.bring_front()
            result, message = self._measure()
        except Stopped as exc:
            message = f"measurement stopped: {exc}"
        except Exception as exc:
            message = f"measurement error: {exc}"
        finally:
            self.release_button()
            winmm.timeEndPeriod(1)
            self.on_done(message, result)

    def _find_puzzle(self) -> tuple[str, tp.Pick]:
        cells = read_board(self.window)
        if cells is None:
            raise Stopped("cannot read the game window")
        seen = identify(cells, self.levels)
        if seen is None:
            raise Stopped("open a Mind Bender puzzle in the game first (the board did not match any)")
        level = self.levels[seen[0]]
        pick = tp.Pick(level.name, level.name, level.solutions[0])
        return seen[1], pick

    def _reset(self, pick: tp.Pick) -> None:
        """Back to the puzzle's scrambled start: its star (a solved one shows "play again"), then
        the banner (re-scrambles a solved puzzle; on an unsolved one it is a plain click)."""
        start = self.levels[pick.puzzle].start
        for _ in range(2):
            self.click(*STARS[pick.number - 1])
            self._wait(0.05)
            self.click(*BANNER)
            if self.wait_board(start, allow_band=False, timeout=0.6):
                return
        raise Stopped(f"could not get {pick.puzzle} back to its start")

    def _try(self, pick: tp.Pick, t: TasTiming) -> bool:
        self._reset(pick)
        self._wait(0.05)
        for d in tp.drags_for(self.levels, pick):
            self.drag(d, t)
        self.tries += 1
        return bool(self.wait_board(self.levels[pick.puzzle].goal, allow_band=True, timeout=0.8))

    def _all_solve(self, pick, t: TasTiming, n: int, label: str) -> bool:
        for k in range(n):
            self.on_status(f"{label}: try {k + 1}/{n}")
            if not self._try(pick, t):
                return False
        return True

    def _measure(self) -> tuple[TasTiming | None, str]:
        _, pick = self._find_puzzle()
        if pick.move_count < 2:
            raise Stopped(f"{pick.puzzle} has only {pick.move_count} move; open a puzzle with 3 or more")
        base = replace(self.timing, pace=0.002)
        found, how = None, ""
        # 1) the game's own speed limit: none. With an F24 press between nudge and jump, no wait.
        keyed = replace(base, key_break=True, gap=0.0)
        if self._all_solve(pick, keyed, self.TRIES, "F24 between nudge and jump, no wait") and                 self._all_solve(pick, keyed, self.CONFIRM, "confirming F24, no wait"):
            found, how = keyed, "an F24 press keeps the moves apart, no wait needed"
        else:
            # 2) else the shortest wait between them that always works, plus a third for safety
            for gap in self.GAPS:
                t = replace(base, key_break=False, gap=gap)
                if self._all_solve(pick, t, self.TRIES, f"gap {gap * 1000:.0f} ms"):
                    safe = replace(t, gap=round(gap * 1.34 + 0.002, 4))
                    if self._all_solve(pick, safe, self.CONFIRM, f"confirming gap {safe.gap * 1000:.1f} ms"):
                        found = safe
                        how = f"{gap * 1000:.0f} ms between nudge and jump works, using {safe.gap * 1000:.1f} ms"
                        break
        if found is None:
            return None, f"no setting solved {pick.puzzle} reliably ({self.tries} tries); timing unchanged"
        # 3) no pause between drags either?
        quick = replace(found, pace=0.0)
        if self._all_solve(pick, quick, self.TRIES + self.CONFIRM, "no pause between drags"):
            found, how = quick, how + ", no pause between drags"
        else:
            how += f", {found.pace * 1000:.0f} ms between drags"
        self._reset(pick)       # leave it scrambled: measuring on 1-1 of a new profile, a run can follow
        return replace(found, measured=True), f"measured on {pick.puzzle}: {how} ({self.tries} tries)"
