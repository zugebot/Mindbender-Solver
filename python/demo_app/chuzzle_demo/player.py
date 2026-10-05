# python/demo_app/chuzzle_demo/player.py
"""Plays drags in the real game window on a worker thread.

Two input modes:
- background (default): mouse messages are posted straight to the game
  window, so the real cursor stays yours and the game needn't be in front.
  Before each move it unpauses the game if it shows "GAME PAUSED", and it
  waits while your cursor is over the game window (real moves there would
  mix into the drag).
- cursor: moves the real cursor (SetCursorPos + mouse_event); the game must
  be in front and you must not touch the mouse.

Each press is held for a few frames before the drag starts, and the button
waits at the release point before letting go, so the game sees every step.
Esc (held), Stop, or the game window closing stop it at once; the button is
always released on the way out.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from . import game_window as gw
from .routes import Drag


class Player:
    BASE_SPEED = 1800.0     # pointer speed in logical px/s at 1.0x
    STEP_S = 0.008          # one pointer update per ~half a frame
    HOVER_S = 0.05          # pointer rests on the press point before pressing
    HOLD_S = 0.06           # button held still before moving (a few frames)
    RELEASE_S = 0.05        # pointer rests on the release point before letting go

    def __init__(self, window: gw.GameWindow, geometry: gw.Geometry, drags: list[Drag],
                 speed: float = 1.0, settle_s: float = 0.35, background: bool = True,
                 after: list[tuple[float, float]] | None = None,
                 on_progress: Callable[[int], None] | None = None,
                 on_status: Callable[[str], None] | None = None,
                 on_done: Callable[[str], None] | None = None):
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

    def _pause(self, seconds: float) -> None:
        end = time.perf_counter() + seconds / max(0.1, self.speed)
        while time.perf_counter() < end:
            self._check()
            time.sleep(0.004)

    def _hold(self, seconds: float) -> None:
        """Fixed waits that must not shrink with speed (frame timing)."""
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self._check()
            time.sleep(0.004)

    def _glide(self, target: tuple[float, float]) -> None:
        _, _, cw, _ = self.window.client_rect()
        scale = cw / gw.LOGICAL_W
        x0, y0 = self._current()
        dist = math.hypot(target[0] - x0, target[1] - y0)
        steps = max(1, int(dist / (self.BASE_SPEED * scale * self.speed) / self.STEP_S))
        for i in range(1, steps + 1):
            self._check()
            t = i / steps
            self._move(x0 + (target[0] - x0) * t, y0 + (target[1] - y0) * t)
            time.sleep(self.STEP_S)

    def _prepare_move(self) -> None:
        if not self.background:
            return
        if gw.cursor_over(self.window):
            self.on_status("waiting: move your mouse off the game window")
            while gw.cursor_over(self.window):
                self._check()
                time.sleep(0.05)
            self.on_status("")
        if gw.resume_if_paused(self.window):
            self.on_status("game was paused; resumed it")
            self._pos = None

    def _drag(self, drag: Drag) -> None:
        pts = [self.geometry.to_client(self.window, p) for p in drag.points]
        self._prepare_move()
        if self.background and self._pos is None:
            self._move(*pts[0])
        self._glide(pts[0])
        self._hold(self.HOVER_S)
        self._button(True)
        self._hold(self.HOLD_S)
        for pt in pts[1:]:
            self._glide(pt)
        self._hold(self.RELEASE_S)
        self._button(False)

    def _click(self, point: tuple[float, float]) -> None:
        self._drag(Drag("click", [point]))

    def _run(self) -> None:
        message = "done"
        try:
            if self.background:
                gw.post_focus(self.window)
            elif not self.window.is_foreground():
                if not self.window.bring_to_front():
                    raise InterruptedError("could not bring the game window to the front")
                gw.key_tap(gw.VK_SHIFT)     # the game pauses on focus loss
                self._hold(0.4)
            self._hold(0.15)
            for i, drag in enumerate(self.drags):
                if drag.move == "click":
                    self._click(drag.points[0])
                else:
                    self._drag(drag)
                self.on_progress(i + 1)
                self._pause(self.settle_s)
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
            self.on_done(message)
