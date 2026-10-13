# python/demo_app/chuzzle_demo/player.py
"""Plays drags in the real game window on a worker thread.

Two input modes:
- background (default): mouse messages are posted straight to the game
  window, so the real cursor stays yours and the game needn't be in front.
  It unpauses the game if it shows "GAME PAUSED" (checked at the start and
  after waiting), and it waits while your cursor is over the game window
  (real moves there would mix into the drag).
- cursor: moves the real cursor (SetCursorPos + mouse_event); the game must
  be in front and you must not touch the mouse.

Motion is smooth for recording: each glide and each drag is one eased
motion whose position comes from the clock (not a step count), pointer
updates go out every ~4 ms with a 1 ms system timer, and the glide to the
next press point overlaps the previous move's settle time instead of
stopping. Glides between drags follow a gentle arc (bowing to alternate
sides) with a hand-like speed profile (quick start, longer slow-down); the
drags themselves follow their routes exactly. In background mode an optional
cursor sprite (cursor_overlay.py) shows the posted pointer, since the real
cursor does not move. Each press is still held for a few frames before the drag starts,
and the pointer rests briefly before release, so the game sees every step.
Esc (held), Stop, or the game window closing stop it at once; the button is
always released on the way out.
"""

from __future__ import annotations

import ctypes
import math
import threading
import time
from typing import Callable

from . import game_window as gw
from .routes import Drag


class Player:
    BASE_SPEED = 1400.0     # average pointer speed in logical px/s at 1.0x (eased in/out)
    MIN_GLIDE_S = 0.12      # shortest glide, so short hops don't snap
    STEP_S = 0.004          # pointer update every ~4 ms (several per game frame)
    HOVER_S = 0.03          # pointer rests on the press point before pressing
    HOLD_S = 0.06           # button held still before moving (a few frames)
    RELEASE_S = 0.04        # pointer rests on the release point before letting go

    def __init__(self, window: gw.GameWindow, geometry: gw.Geometry, drags: list[Drag],
                 speed: float = 1.0, settle_s: float = 0.35, background: bool = True,
                 after: list[tuple[float, float]] | None = None,
                 on_progress: Callable[[int], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 on_done: Callable[[str], None] | None = None,
                 overlay=None):
        self.window = window
        self.geometry = geometry
        self.drags = drags
        self.speed = speed
        self.settle_s = settle_s
        self.background = background
        self.after = after or []        # board points to click once all drags are done
        self.on_progress = on_progress or (lambda i: None)
        self.on_status = on_status or (lambda msg: None)
        self.on_done = on_done or (lambda msg: None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._down = False
        self._pos: tuple[float, float] | None = None    # client px (background mode)
        self.overlay = overlay if background else None     # cursor sprite (cursor_overlay.CursorOverlay)
        self._arc_side = 1.0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------
    # input backends (client pixel coordinates)
    # ------------------------------------------------------------------

    def _current(self) -> tuple[float, float]:
        if self.background:
            return self._pos if self._pos is not None else (0.0, 0.0)
        left, top, _, _ = self.window.client_rect()
        x, y = gw.cursor_pos()
        return x - left, y - top

    def _move(self, x: float, y: float) -> None:
        if self.background:
            gw.post_move(self.window, x, y, self._down)
            self._pos = (x, y)
            self._show_sprite()
        else:
            left, top, _, _ = self.window.client_rect()
            gw.move_cursor(round(left + x), round(top + y))

    def _button(self, down: bool) -> None:
        x, y = self._current()
        if self.background:
            gw.post_button(self.window, down, x, y)
        elif down:
            gw.mouse_down()
        else:
            gw.mouse_up()
        self._down = down
        if self.background:
            self._show_sprite()

    def _show_sprite(self) -> None:
        if self.overlay is not None and self._pos is not None:
            left, top, _, _ = self.window.client_rect()
            self.overlay.move(left + self._pos[0], top + self._pos[1], self._down)

    # ------------------------------------------------------------------
    # timing
    # ------------------------------------------------------------------

    def _check(self) -> None:
        if self._stop.is_set():
            raise InterruptedError("stopped")
        if gw.key_down(gw.VK_ESCAPE):
            raise InterruptedError("Esc pressed")
        if not self.window.alive():
            raise InterruptedError("game window closed")
        if not self.background and not self.window.is_foreground():
            raise InterruptedError("game window lost focus")

    def _wait_until(self, deadline: float) -> None:
        while time.perf_counter() < deadline:
            self._check()
            time.sleep(0.002)

    def _pause(self, seconds: float) -> None:
        """A wait that scales with playback speed."""
        self._wait_until(time.perf_counter() + seconds / max(0.1, self.speed))

    def _hold(self, seconds: float) -> None:
        """A wait that must not shrink with speed (frame timing)."""
        self._wait_until(time.perf_counter() + seconds)

    # ------------------------------------------------------------------
    # motion
    # ------------------------------------------------------------------

    @staticmethod
    def _ease(u: float) -> float:
        """Smooth start and stop (smootherstep)."""
        u = max(0.0, min(1.0, u))
        return u * u * u * (u * (u * 6 - 15) + 10)

    @classmethod
    def _ease_reach(cls, u: float) -> float:
        """Like a hand reaching for a target: speed peaks early (about 40% of the
        way through the time) and the approach is slower."""
        u = max(0.0, min(1.0, u))
        return cls._ease(u ** 0.8)

    def _approach(self, target: tuple[float, float]) -> list[tuple[float, float]]:
        """Points of a gentle arc from the pointer to `target` (a quadratic curve
        bowing sideways by ~12% of the distance, at most ~60 logical px, to
        alternating sides), for glides between drags."""
        sx, sy = self._current()
        tx, ty = target
        dx, dy = tx - sx, ty - sy
        dist = math.hypot(dx, dy)
        _, _, cw, _ = self.window.client_rect()
        scale = cw / gw.LOGICAL_W
        if dist < 24 * scale:
            return [target]
        self._arc_side = -self._arc_side
        bend = min(0.12 * dist, 60 * scale) * self._arc_side
        cx, cy = (sx + tx) / 2 - dy / dist * bend, (sy + ty) / 2 + dx / dist * bend
        n = 24
        return [((1 - u) ** 2 * sx + 2 * (1 - u) * u * cx + u * u * tx,
                 (1 - u) ** 2 * sy + 2 * (1 - u) * u * cy + u * u * ty) for u in (i / n for i in range(1, n + 1))]

    def _glide_path(self, points: list[tuple[float, float]], min_duration: float = 0.0, ease=None) -> None:
        """Move from the current position through `points` as one eased
        motion; the position along the path is a function of elapsed time.
        `min_duration` stretches it (used to arrive just as the board is ready)."""
        path = [self._current()] + list(points)
        cumulative = [0.0]
        for a, b in zip(path, path[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = cumulative[-1]
        if total < 0.5:
            self._move(*path[-1])
            return
        _, _, cw, _ = self.window.client_rect()
        px_per_s = self.BASE_SPEED * (cw / gw.LOGICAL_W) * self.speed
        duration = max(self.MIN_GLIDE_S / max(0.25, self.speed), total / px_per_s, min_duration)
        t0 = time.perf_counter()
        segment = 0
        while True:
            self._check()
            u = (time.perf_counter() - t0) / duration
            if u >= 1.0:
                break
            s = (ease or self._ease)(u) * total
            while segment < len(path) - 2 and s > cumulative[segment + 1]:
                segment += 1
            a, b = path[segment], path[segment + 1]
            length = cumulative[segment + 1] - cumulative[segment]
            f = 0.0 if length <= 0 else (s - cumulative[segment]) / length
            self._move(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            time.sleep(self.STEP_S)
        self._move(*path[-1])

    def _prepare_move(self, first: bool) -> None:
        """Background mode: wait while the real cursor is over the game, and
        unpause the game. The pause check captures the window, which can cost
        the game a frame, so it only runs at the start or after a wait."""
        if not self.background:
            return
        waited = False
        if gw.cursor_over(self.window):
            waited = True
            self.on_status("waiting: move your mouse off the game window")
            while gw.cursor_over(self.window):
                self._check()
                time.sleep(0.05)
            self.on_status("")
        if (first or waited) and gw.resume_if_paused(self.window):
            self.on_status("game was paused; resumed it")
            self._pos = None

    def _drag(self, drag: Drag, first: bool = False, ready_at: float = 0.0) -> float:
        """One move: glide to the press point (overlapping the previous move's
        settle time), wait until the board is ready (`ready_at`), press, hold
        a few frames, drag along the route in one eased motion, rest, release.
        Returns when the board will be ready for the next press."""
        pts = [self.geometry.to_client(self.window, p) for p in drag.points]
        self._prepare_move(first)
        if self.background and self._pos is None:
            self._move(*self._rest_point())
        # arrive as the board becomes ready, at most 2x slower than normal
        slack = ready_at - time.perf_counter()
        self._glide_path(self._approach(pts[0]), min_duration=min(slack, 0.6 / max(0.1, self.speed)),
                         ease=self._ease_reach)
        self._wait_until(ready_at)
        self._hold(self.HOVER_S)
        self._button(True)
        self._hold(self.HOLD_S)
        route = [p for prev, p in zip(pts, pts[1:]) if p != prev]
        if route:
            self._glide_path(route)
        self._hold(self.RELEASE_S)
        self._button(False)
        return time.perf_counter() + self.settle_s / max(0.1, self.speed)

    def _rest_point(self) -> tuple[float, float]:
        """Where the background pointer starts: the real cursor's position, kept
        inside the game's client area."""
        left, top, cw, ch = self.window.client_rect()
        x, y = gw.cursor_pos()
        return min(max(x - left, 0), cw - 1), min(max(y - top, 0), ch - 1)

    def _click(self, point: tuple[float, float]) -> float:
        return self._drag(Drag("click", [point]), first=True)

    def _run(self) -> None:
        message = "done"
        winmm = ctypes.windll.winmm
        winmm.timeBeginPeriod(1)            # 1 ms timer: even pointer updates
        try:
            if self.background:
                gw.post_focus(self.window)
            elif not self.window.is_foreground():
                if not self.window.bring_to_front():
                    raise InterruptedError("could not bring the game window to the front")
                gw.key_tap(gw.VK_SHIFT)     # the game pauses on focus loss
                self._hold(0.4)
            self._hold(0.15)
            ready = 0.0
            for i, drag in enumerate(self.drags):
                if drag.move == "click":
                    self._wait_until(ready)
                    ready = self._click(drag.points[0])
                else:
                    ready = self._drag(drag, first=(i == 0), ready_at=ready)
                self.on_progress(i + 1)
            self._wait_until(ready)
            for point in self.after:
                self._pause(0.6)
                self._click(point)
        except InterruptedError as exc:
            message = f"aborted: {exc}"
        except Exception as exc:  # report, never leave the button held
            message = f"error: {exc}"
        finally:
            if self._down:
                try:
                    self._button(False)
                except Exception:
                    pass
            winmm.timeEndPeriod(1)
            if self.overlay is not None:
                time.sleep(0.4)
                self.overlay.hide()
            self.on_done(message)
