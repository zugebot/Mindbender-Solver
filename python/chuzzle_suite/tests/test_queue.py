# python/chuzzle_suite/tests/test_queue.py
"""Queue tab, title bar and mini view. Run: python -m unittest discover -s python/chuzzle_suite/tests

Everything talks to a fake scheduler on a free local port, never to the real
mbq daemon (127.0.0.1:8766)."""

import copy
import io
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")        # App tests draw off-screen
os.environ["CHUZZLE_SUITE_MBQ_URL"] = "http://127.0.0.1:9"   # never the real daemon, even by accident

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_DIR.parents[0] / "scheduler"))

from chuzzle_suite import queue_client as qc, settings, title_bar as tb, window_frame as wf  # noqa: E402

NOW = time.time()


def job(id_, puzzle, status="queued", **kw):
    j = {"id": id_, "puzzle": puzzle, "moves": 16, "until": None, "engine": "ssd", "table_depth": 7,
         "seed_depth": 5, "stream_depth": 4, "gpu": "5060", "status": status, "wait_reason": "",
         "shard": None, "merge": None, "held_since": None}
    j.update(kw)
    return j


def sample_status() -> dict:
    """Shaped like mbq's GET /status (job 206 on 2026-11-02)."""
    running = job(206, "15-3", "running", started=NOW - 19831)
    running["eta"] = {"phase": "probe", "detail": "pass 4/6 streaming", "done": 155877945, "total": 311755890,
                      "percent": 50.0, "elapsed": 19831.5, "held_since": None, "phase_remaining": 19806.9,
                      "total_remaining": 19853.3, "finish_at": NOW + 19853.3}
    return {"paused": False, "running": running, "finalizing": None,
            "pending": [job(207, "15-4"), job(208, "16-1"), job(209, "19-2", wait_reason="needs 300 GB free"),
                        job(210, "19-4", until=17)],
            "external_solvers": [], "solved": 92, "queue_estimate": {"known_seconds": 97266.4, "unknown_jobs": 0},
            "alarm": {"enabled": True, "active": None}, "finished_today": 5, "table_root_free_gb": 292.6}


class FakeScheduler:
    """A stand-in for mbq's HTTP API: serves `status`, records POSTs and applies
    pause / resume / hold / unhold to its status like mbq does."""

    def __init__(self, status: dict | None = None, port: int = 0):
        self.status = status if status is not None else sample_status()
        self.posts: list[tuple[str, dict]] = []
        self.lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/status":
                    with fake.lock:
                        return self._send(200, fake.status)
                return self._send(404, {"error": "unknown path"})

            def do_POST(self):
                if self.headers.get("X-MBQ") != "1":
                    return self._send(403, {"error": "missing X-MBQ header"})
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                cmd = self.path.strip("/")
                with fake.lock:
                    fake.posts.append((cmd, body))
                    s = fake.status
                    if cmd in ("pause", "resume"):
                        s["paused"] = cmd == "pause"
                        return self._send(200, {"paused": s["paused"]})
                    if cmd in ("hold", "unhold"):
                        if not s.get("running"):
                            return self._send(400, {"error": "nothing is running"})
                        s["running"]["held_since"] = time.time() if cmd == "hold" else None
                        return self._send(200, s["running"])
                return self._send(404, {"error": "unknown path"})

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def closed_port_url() -> str:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}"


class Formatting(unittest.TestCase):
    def test_matches_mbq_cli_wording(self):
        import mbq
        for j in sample_status()["pending"] + [sample_status()["running"],
                                               job(5, "6-4", "running", shard=[2, 8], gpu=None),
                                               job(6, "1-1", engine="inram", gpu=None)]:
            self.assertEqual(qc.describe(j), mbq.describe(j))
        for secs in (None, 0, 59, 61, 3599, 3600, 97266.4):
            self.assertEqual(qc.fmt_dur(secs), mbq.fmt_dur(secs))

    def test_status_lines_are_mbq_status(self):
        import mbq
        for status in self._variants():
            saved = mbq.api
            mbq.api = lambda *a, **k: copy.deepcopy(status)
            try:
                out = io.StringIO()
                with redirect_stdout(out):
                    mbq.cmd_status(None)
            finally:
                mbq.api = saved
            cli = out.getvalue().splitlines()[:-1]          # last line: free space on mbq's TABLE_ROOT
            self.assertEqual([t for t, _ in qc.status_lines(status, hints=True)], cli)

    def _variants(self):
        base = sample_status()
        held = sample_status()
        held["paused"] = True
        held["running"]["held_since"] = NOW - 600
        held["external_solvers"] = [["experimentDiskDepth6.exe", 4242]]
        idle = sample_status()
        idle["running"], idle["pending"] = None, []
        idle["finalizing"] = job(205, "6-4", "finalizing")
        unknown = sample_status()
        unknown["running"]["eta"].update(total_remaining=None, finish_at=None, percent=None)
        return [base, held, idle, unknown]

    def test_short_lines_for_the_mini_view(self):
        s = sample_status()
        self.assertEqual(qc.short_describe(s["running"]), "#206 15-3  16 moves  <5,4,7> gpu")
        self.assertTrue(qc.left_line(s["running"]["eta"]).startswith("about 5h 30m left, done around "))
        e = dict(s["running"]["eta"], total_remaining=None)
        self.assertEqual(qc.left_line(e), "phase ends in 5h 30m, whole job unknown")
        self.assertEqual(qc.held_line(s["running"]), None)


class Http(unittest.TestCase):
    def setUp(self):
        self.fake = FakeScheduler()

    def tearDown(self):
        self.fake.close()

    def test_fetch_and_commands(self):
        self.assertEqual(qc.fetch_status(self.fake.url)["running"]["id"], 206)
        self.assertEqual(qc.post_command("pause", self.fake.url), {"paused": True})
        self.assertTrue(qc.post_command("hold", self.fake.url)["held_since"])
        self.assertEqual([c for c, _ in self.fake.posts], ["pause", "hold"])
        self.fake.status["running"] = None
        with self.assertRaises(qc.QueueError) as err:
            qc.post_command("unhold", self.fake.url)
        self.assertEqual(str(err.exception), "nothing is running")
        with self.assertRaises(ValueError):
            qc.post_command("shutdown", self.fake.url)

    def test_unreachable_reasons(self):
        with self.assertRaises(OSError) as err:
            qc.fetch_status(closed_port_url())
        self.assertEqual(qc.unreachable_reason(err.exception), "connection refused")
        silent = socket.socket()            # accepts connections but never answers
        silent.bind(("127.0.0.1", 0))
        silent.listen(4)
        try:
            with self.assertRaises(OSError) as err:
                qc.fetch_status(f"http://127.0.0.1:{silent.getsockname()[1]}", timeout=0.3)
            self.assertEqual(qc.unreachable_reason(err.exception), "timed out")
        finally:
            silent.close()

    def test_poller_polls_only_while_active_and_sends(self):
        poller = qc.QueuePoller(self.fake.url, interval=0.05).start()
        try:
            time.sleep(0.2)
            self.assertIsNone(poller.snapshot[2])       # inactive: no requests
            poller.active = True
            poller.poke()
            self._wait(lambda: poller.snapshot[0] is not None)
            self.assertEqual(poller.snapshot[0]["running"]["puzzle"], "15-3")
            results = []
            poller.send("hold", lambda r, e: results.append((r, e)))
            self._wait(lambda: results)
            self.assertIsNone(results[0][1])
            self._wait(lambda: poller.snapshot[0]["running"]["held_since"])
        finally:
            poller.stop()
        down = qc.QueuePoller(closed_port_url())
        down.poll_once()
        self.assertEqual(down.snapshot[:2], (None, "connection refused"))

    def _wait(self, cond, timeout=5.0):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return
            time.sleep(0.02)
        self.fail("timed out waiting")


class FrameHelpers(unittest.TestCase):
    def test_keep_top_right_and_clamp(self):
        work = (0, 0, 1920, 1040)
        self.assertEqual(wf.keep_top_right((600, 100, 1800, 972), (440, 150), work), (1360, 100))
        # growing back to the full view from a mini window near the left edge: stays on screen
        self.assertEqual(wf.keep_top_right((10, 900, 450, 1050), (1200, 872), work), (0, 168))
        self.assertEqual(wf.keep_top_right((10, 20, 450, 170), (1200, 872), None), (-750, 20))

    def test_title_bar_hit_test(self):
        bar = tb.TitleBar("x", ("pin", "min", "max", "close"))
        self.assertEqual(bar.hit((1199, 5), 1200), "close")
        self.assertEqual(bar.hit((1200 - 4 * tb.BUTTON_W + 1, 5), 1200), "pin")
        self.assertEqual(bar.hit((300, 5), 1200), "drag")
        self.assertIsNone(bar.hit((300, tb.HEIGHT), 1200))

    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "settings.json"
            self.assertFalse(settings.get("pinned", False, path))
            settings.put("pinned", True, path)
            settings.put("other", 3, path)
            self.assertEqual(settings.load(path), {"pinned": True, "other": 3})
            path.write_text("not json")
            self.assertEqual(settings.load(path), {})


class AppButtons(unittest.TestCase):
    """The real App, drawn off-screen (SDL dummy driver), against the fake scheduler."""

    @classmethod
    def setUpClass(cls):
        import pygame as pg
        from chuzzle_suite import ui
        cls.pg, cls.ui = pg, ui
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved = (settings.SETTINGS_FILE, ui.App.find_game, ui.App.rescan_saves, ui.App.refresh_games)
        settings.SETTINGS_FILE = Path(cls.tmp.name) / "settings.json"
        ui.App.find_game = lambda self, quiet=False: None          # no game lookup, save or drive scans
        ui.App.rescan_saves = lambda self: None
        ui.App.refresh_games = lambda self, full=False: None
        cls.fake = FakeScheduler()
        cls.app = ui.App()
        cls.app.queue.stop()
        cls.app.queue = qc.QueuePoller(cls.fake.url)

    @classmethod
    def tearDownClass(cls):
        cls.fake.close()
        ui = cls.ui
        settings.SETTINGS_FILE, ui.App.find_game, ui.App.rescan_saves, ui.App.refresh_games = cls.saved
        cls.pg.quit()
        cls.tmp.cleanup()

    def setUp(self):
        self.fake.status = sample_status()
        self.fake.posts.clear()
        self.app.set_view("full")

    def click(self, label_start: str) -> None:
        for b in self.app.active_buttons():
            if b.text().startswith(label_start):
                self.app.handle_click(b.rect.center)
                return
        self.fail(f"no button {label_start!r}")

    def wait_posts(self, n):
        end = time.time() + 5
        while len(self.fake.posts) < n and time.time() < end:
            time.sleep(0.02)
        while self.app.queue.busy and time.time() < end:
            time.sleep(0.02)
        self.app.queue.poll_once()
        while not self.app.events.empty():
            self.app.events.get()()

    def test_pause_needs_a_second_click_then_resume(self):
        self.app.set_tab("queue")
        self.app.queue.poll_once()
        self.click("Pause queue")
        time.sleep(0.2)
        self.assertEqual(self.fake.posts, [])               # first click only asks
        self.click("Click again to pause")
        self.wait_posts(1)
        self.assertEqual([c for c, _ in self.fake.posts], ["pause"])
        self.click("Resume queue")                          # resuming needs no confirmation
        self.wait_posts(2)
        self.assertEqual([c for c, _ in self.fake.posts], ["pause", "resume"])
        self.assertFalse(self.app.queue_paused())

    def test_hold_and_unhold_from_the_mini_view(self):
        self.app.set_view("mini")
        self.assertEqual(self.app.screen.get_size(), (self.ui.MINI_W, self.ui.MINI_H - self.ui.TB))
        self.app.queue.poll_once()
        self.click("Hold")
        self.wait_posts(1)
        self.assertTrue(self.app.queue_held())
        self.app.draw()
        self.click("Unhold")
        self.wait_posts(2)
        self.assertEqual([c for c, _ in self.fake.posts], ["hold", "unhold"])
        self.assertFalse(self.app.queue_held())
        self.click("Pause")
        self.click("Click again")
        self.wait_posts(3)
        self.assertTrue(self.app.queue_paused())

    def test_buttons_disabled_when_the_scheduler_is_down(self):
        self.app.set_tab("queue")
        self.app.queue = qc.QueuePoller(closed_port_url())
        try:
            self.app.queue.poll_once()
            self.assertEqual(self.app.queue.snapshot[1], "connection refused")
            for b in self.app.buttons["queue"][:2] + self.app.buttons["mini"]:
                self.assertFalse(b.enabled())
            self.app.draw()                                 # draws "scheduler not running"
            self.app.set_view("mini")
            self.app.draw()
        finally:
            self.app.queue = qc.QueuePoller(self.fake.url)

    def test_every_view_draws(self):
        self.app.queue.poll_once()
        for tab in ("launcher", "solutions", "speed", "saves", "queue"):
            self.app.set_tab(tab)
            self.app.draw()
        self.fake.status["running"] = None
        self.app.queue.poll_once()
        self.app.draw()
        self.app.set_view("mini")
        self.app.draw()

    def test_title_bar_close_goes_through_quit_and_pin_is_saved(self):
        pg, app = self.pg, self.app
        rects = app.title_bars["full"].rects(app.window_surface.get_width())
        pinned = app.pinned
        app.title_press(rects["pin"].center)
        app.title_release(rects["pin"].center)
        self.assertEqual(settings.get("pinned"), not pinned)
        app.title_press(rects["pin"].center)
        app.title_release(rects["pin"].center)
        self.assertEqual(settings.get("pinned"), pinned)
        app.title_press(rects["max"].center)               # disabled: nothing happens
        self.assertIsNone(app.title_bars["full"].pressed)
        pg.event.clear()
        app.title_press(rects["close"].center)
        app.title_release((5, 5))                          # released elsewhere: cancelled
        self.assertFalse(any(e.type == pg.QUIT for e in pg.event.get()))
        app.title_press(rects["close"].center)
        app.title_release(rects["close"].center)
        self.assertTrue(any(e.type == pg.QUIT for e in pg.event.get()))


if __name__ == "__main__":
    unittest.main()
