# python/demo_app/chuzzle_demo/player.py
"""Plays drags in the real game window on a worker thread.

Safety: stops at once if Esc is held, the Stop button is pressed, the game
window closes, or another window takes focus (e.g. you alt-tab away). The
mouse button is always released on the way out.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Callable

from . import game_window as gw
from .routes import Drag


class Player:
    BASE_SPEED = 2200.0     # cursor speed in px/s at 1.0x
    STEP_S = 0.004

    def __init__(self, window: gw.GameWindow, geometry: gw.Geometry, drags: list[Drag],
                 speed: float = 1.0, settle_s: float = 0.35,
                 after: list[tuple[float, float]] | None = None,
                 on_progress: Callable[[int], None] | None = None,
                 on_done: Callable[[str], None] | None = None):
        self.window = window
        self.geometry = geometry
        self.drags = drags
        self.speed = speed
        self.settle_s = settle_s
        self.after = after or []        # board points to click once all drags are done
        self.on_progress = on_progress or (lambda i: None)
        self.on_done = on_done or (lambda msg: None)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.button_down = False

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    # ------------------------------------------------------------------

    def _check(self) -> None:
        if self._stop.is_set():
            raise InterruptedError("stopped")
        if gw.key_down(gw.VK_ESCAPE):
            raise InterruptedError("Esc pressed")
        if not self.window.alive():
            raise InterruptedError("game window closed")
        if not self.window.is_foreground():
            raise InterruptedError("game window lost focus")

    def _glide(self, target: tuple[int, int]) -> None:
        x0, y0 = gw.cursor_pos()
        dist = math.hypot(target[0] - x0, target[1] - y0)
        steps = max(1, int(dist / (self.BASE_SPEED * self.speed) / self.STEP_S))
        for i in range(1, steps + 1):
            self._check()
            t = i / steps
            gw.move_cursor(round(x0 + (target[0] - x0) * t), round(y0 + (target[1] - y0) * t))
            time.sleep(self.STEP_S)

    def _pause(self, seconds: float) -> None:
        end = time.perf_counter() + seconds / max(0.1, self.speed)
        while time.perf_counter() < end:
            self._check()
            time.sleep(0.005)

    def _drag(self, drag: Drag) -> None:
        pts = [self.geometry.to_screen(self.window, p) for p in drag.points]
        self._glide(pts[0])
        self._pause(0.03)
        gw.mouse_down()
        self.button_down = True
        self._pause(0.03)
        for pt in pts[1:]:
            self._glide(pt)
        self._pause(0.03)
        gw.mouse_up()
        self.button_down = False

    def _click(self, point: tuple[float, float]) -> None:
        self._glide(self.geometry.to_screen(self.window, point))
        self._pause(0.05)
        gw.mouse_down()
        time.sleep(0.03)
        gw.mouse_up()

    def _run(self) -> None:
        message = "done"
        try:
            if not self.window.bring_to_front():
                raise InterruptedError("could not bring the game window to the front")
            self._pause(0.15)
            for i, drag in enumerate(self.drags):
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
            if self.button_down:
                gw.mouse_up()
                self.button_down = False
            self.on_done(message)
