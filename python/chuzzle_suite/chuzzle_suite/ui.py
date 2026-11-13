# python/chuzzle_suite/chuzzle_suite/ui.py
"""pygame control panel: level grid, solution preview, real-game auto-play,
save-file tracking, a launcher for every Chuzzle exe on the PC and the solver
queue. The window is borderless with the app's own title bar (title_bar.py,
window_frame.py), and the queue has a small "mini view" that can stay on top."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import pygame as pg

from . import board_sim, game_window as gw, launcher, levels as lv, queue_client as qc, routes, saves, settings
from . import paths, tas_plan as tp, tas_runner as tr, title_bar as tb, window_frame as wf

W, H = 1200, 840                        # content; the window is W x (TB + H)
TB = tb.HEIGHT                          # the app's own title bar above the content
MINI_W, MINI_H = 440, 150               # the queue's mini view (title bar included)
BAR = pg.Rect(12, 776, W - 24, 54)      # game status bar along the bottom
BG = (24, 26, 32)
PANEL = (34, 37, 46)
PANEL_LINE = (58, 62, 76)
TEXT = (226, 228, 235)
DIM = (140, 146, 160)
ACCENT = (90, 170, 255)
GOOD = (80, 200, 120)
WARN = (240, 190, 70)
BAD = (230, 90, 90)

# y positions in the right ("Real game") panel
RY = {"hint": 84, "play": 120, "stop": 160, "speed": 198, "sprite": 232, "nav": 280, "world": 310,
      "arrows": 338, "levels": 374, "log": 426}

MOVES_HELP = ("Moves are 1-indexed. R = row (1 = top) shifted right, C = column (1 = left) shifted "
              "down; the last digit is how many cells. Moves of the 2x2 piece name both of its "
              "lines, e.g. C453 = columns 4 and 5 down 3.")
GRID_HELP = ("Number = optimal move count. Green = solved (darker = longer). Red = unsolved; "
             ">=N means no solution shorter than N moves exists (from build/README.md). "
             "White dot = level has a 2x2 piece. Yellow dot = the game also accepts the piece "
             "on another 2x2 block of its colour (switch goals below the level's moves).")


@dataclass
class Button:
    rect: pg.Rect
    label: str | Callable[[], str]
    action: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True
    active: Callable[[], bool] = lambda: False
    small: bool = False
    visible: Callable[[], bool] = lambda: True

    def text(self) -> str:
        return self.label() if callable(self.label) else self.label


class App:
    def __init__(self):
        gw.set_dpi_aware()
        wf.set_app_id(paths.APP_ID)
        pg.init()
        pg.display.set_caption(gw.OWN_TITLE)
        if paths.ICON_PNG.is_file():
            pg.display.set_icon(pg.image.load(str(paths.ICON_PNG)))
        # borderless window with the app's own title bar; "mini" is the small queue view
        self.view = "full"
        self.pinned = bool(settings.get("pinned", False))
        self.title_bars = {"full": tb.TitleBar(gw.OWN_TITLE, ("pin", "min", "max", "close")),
                           "mini": tb.TitleBar("Solver queue", ("pin", "expand", "close"))}
        self.title_bars["full"].disabled.add("max")     # fixed layout, like the old native window
        self.hwnd = None
        self._open_window()
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("segoeui", 15)
        self.small = pg.font.SysFont("segoeui", 13)
        self.bold = pg.font.SysFont("segoeui", 17, bold=True)
        self.big = pg.font.SysFont("segoeui", 24, bold=True)
        self.mono = pg.font.SysFont("consolas", 15)
        # the Queue tab's fonts, a step larger than the other tabs'
        self.q_title = pg.font.SysFont("segoeui", 28, bold=True)
        self.q_font = pg.font.SysFont("segoeui", 17)
        self.q_bold = pg.font.SysFont("segoeui", 17, bold=True)
        self.q_small = pg.font.SysFont("segoeui", 15)
        self.q_stat = pg.font.SysFont("segoeui", 22, bold=True)
        self.q_mono = pg.font.SysFont("consolas", 16)
        self.q_log = pg.font.SysFont("consolas", 14)

        self.events: queue.Queue[Callable[[], None]] = queue.Queue()
        self.log_lines: list[tuple[float, str, tuple]] = []
        self.tab = "solutions"

        # levels / solutions
        self.levels = lv.load_levels()
        self.selected = "1-4"
        self.ranked: dict[str, list[tuple[float, str]]] = {}
        self.ranking: set[str] = set()
        self.rank_progress: dict[str, str] = {}
        self.solution_index = 0
        self.preview_step = 0
        self.animating = False
        self.anim_t = 0.0
        self.refreshing = False

        # game
        self.window: gw.GameWindow | None = None
        self.geometry = gw.Geometry()
        self.player = None
        self.played_moves = 0
        self.speed = 1.0
        self.click_next_after = False
        self.background_input = True
        self.show_sprite = True         # background mode: draw a pointer sprite where the posted mouse is
        self.cursor_overlay = None      # cursor_overlay.CursorOverlay, made on first use
        self.player_status = ""
        self.tracked_world = 1
        self.calib_step = 0
        self.calib_points: list[tuple[float, float]] = []
        self._f9_was_down = False

        # saves
        self.locations: list[saves.SaveLocation] = []
        self.active_loc: saves.SaveLocation | None = None
        self.watcher: saves.SaveWatcher | None = None
        self.scanning = False
        self.save_scroll = 0
        self.selected_save: tuple[saves.SaveLocation, saves.Profile | None] | None = None
        self.confirm_restore_until = 0.0
        self.save_rows: list[tuple[pg.Rect, saves.SaveLocation, saves.Profile | None]] = []

        # launcher
        self.games: list[launcher.GameExe] = []
        self.games_scanned_at: float | None = None
        self.games_scanning = False
        self.games_scan_status = ""
        self.launcher_scroll = 0
        self.launcher_max_scroll = 0
        self.launcher_buttons: list[Button] = []
        self.launcher_inner = pg.Rect(0, 0, 0, 0)
        self._running_cache: tuple[float, dict[str, gw.GameWindow]] = (0.0, {})

        # solver queue (python/scheduler/mbq.py); polled only while the Queue tab or mini view shows
        self.queue = qc.QueuePoller().start()
        self.confirm_pause_until = 0.0

        # Mind Bender TAS (tas_plan.py, tas_runner.py); options kept in settings.json "tas"
        saved = settings.get("tas", {}) or {}
        self.tas_target = int(saved.get("target", tp.PUZZLER))
        self.tas_per_level = int(saved.get("per_level", 3))
        self.tas_last_level = saved.get("last_level")             # None: the target's default
        self.tas_owned = set(saved.get("owned", []))
        self.tas_others = bool(saved.get("others_owned", False))
        self.tas_livesplit = bool(saved.get("livesplit", True))
        self.tas_check = bool(saved.get("check_solves", True))
        self.tas_keep_awake = bool(saved.get("keep_awake", False))
        self.tas_burst = saved.get("burst", "drag") if saved.get("burst") in tr.BURSTS else "drag"
        self.focus_keeper = gw.FocusKeeper(lambda: getattr(self, "window", None))
        self.focus_keeper.enabled = self.tas_keep_awake
        self.focus_keeper.start()
        self.tas_runner: tr.TasRunner | None = None
        self.tas_step = -1
        self.tas_status = ""
        self.tas_scroll = 0
        self.tas_max_scroll = 0
        self.tas_timing = tr.load_timing(None)
        self.tas_confirm_until = 0.0
        self._tas_confirm_what = ""
        self.tas_last_result = ""
        self.tas_plan: tp.TasPlan | None = None

        self.buttons: dict[str, list[Button]] = {"solutions": [], "saves": [], "speed": [], "launcher": [],
                                                 "queue": [], "tas": [], "mini": [], "all": []}
        self._speed_cache: tuple[float, dict] | None = None
        self._build_buttons()
        self.select_level(self.selected)
        self.find_game(quiet=True)
        self.rescan_saves()
        self.refresh_games()
        threading.Thread(target=self._watch_loop, daemon=True).start()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def log(self, message: str, color=TEXT) -> None:
        self.log_lines.append((time.time(), message, color))
        del self.log_lines[:-60]

    def post(self, fn: Callable[[], None]) -> None:
        """Run fn on the UI thread (worker threads must not touch state)."""
        self.events.put(fn)

    @property
    def level(self) -> lv.Level:
        return lv.find_level(self.levels, self.selected) or self.levels["1-4"]

    def goals(self) -> list[lv.Level]:
        """The selected puzzle's official goal, then every placement the game also accepts."""
        base = self.levels.get(self.selected.split("@")[0])
        return [base] + base.placements if base else []

    def cycle_goal(self, delta: int) -> None:
        goals = self.goals()
        names = [g.name for g in goals]
        if len(names) > 1:
            i = names.index(self.selected) if self.selected in names else 0
            self.select_level(names[(i + delta) % len(names)])

    def current_solution(self) -> str | None:
        ranked = self.ranked.get(self.level.name)
        if ranked:
            return ranked[min(self.solution_index, len(ranked) - 1)][1]
        sols = self.level.solutions
        return sols[min(self.solution_index, len(sols) - 1)] if sols else None

    def solution_count(self) -> int:
        return len(self.level.solutions)

    def playing(self) -> bool:
        """Is anything driving the game (auto-play, or a TAS run or measurement)?"""
        return (self.player is not None and self.player.running) or self.tas_running()

    # ------------------------------------------------------------------
    # buttons
    # ------------------------------------------------------------------

    def _build_buttons(self) -> None:
        b = self.buttons
        tabs = [("solutions", "Solutions"), ("tas", "TAS"), ("queue", "Queue"), ("speed", "Speed Chuzzle"),
                ("saves", "Saves"), ("launcher", "Launcher")]
        widths = {"tas": 64, "speed": 122}
        x = W - 12
        for tab, label in reversed(tabs):
            w_ = widths.get(tab, 102)
            x -= w_
            b["all"].append(Button(pg.Rect(x, 8, w_, 30), label,
                                   lambda tab=tab: self.set_tab(tab), active=lambda tab=tab: self.tab == tab))
            x -= 6
        self._build_tas_buttons()
        b["solutions"].append(
            Button(pg.Rect(12 + 410 - 10 - 96, 58, 96, 26), lambda: "Loading..." if self.refreshing else "Refresh",
                   self.refresh_levels, lambda: not self.refreshing))
        # preview controls (middle column): rows span the panel's inner width
        has_sol = lambda: self.current_solution() is not None
        x0, inner, y = 444, 338, 452
        widths = [54, 54, 0, 54, 54]
        widths[2] = inner - sum(widths) - 4 * 6
        labels = ["|<", "<", None, ">", ">|"]
        actions = [lambda: self.set_step(0), lambda: self.set_step(self.preview_step - 1), self.toggle_animate,
                   lambda: self.set_step(self.preview_step + 1), lambda: self.set_step(10 ** 6)]
        x = x0
        for w_, label, action in zip(widths, labels, actions):
            if label is None:
                b["solutions"].append(Button(pg.Rect(x, y, w_, 28), lambda: "Pause" if self.animating else "Animate",
                                             action, has_sol, lambda: self.animating))
            else:
                b["solutions"].append(Button(pg.Rect(x, y, w_, 28), label, action, has_sol))
            x += w_ + 6
        b["solutions"] += [
            Button(pg.Rect(x0, y + 34, 74, 28), "< Sol", lambda: self.set_solution(self.solution_index - 1),
                   lambda: self.solution_count() > 1),
            Button(pg.Rect(x0 + inner - 74, y + 34, 74, 28), "Sol >",
                   lambda: self.set_solution(self.solution_index + 1), lambda: self.solution_count() > 1),
        ]
        # goal switch (bottom right of the preview): official goal / placements the game accepts
        has_goals = lambda: len(self.goals()) > 1
        b["solutions"] += [
            Button(pg.Rect(612, 734, 82, 26), "< Goal", lambda: self.cycle_goal(-1),
                   lambda: not self.playing(), visible=has_goals),
            Button(pg.Rect(700, 734, 82, 26), "Goal >", lambda: self.cycle_goal(1),
                   lambda: not self.playing(), visible=has_goals),
        ]
        # game status bar (bottom, every tab)
        bx = BAR.right - 10 - 3 * 116
        b["all"] += [
            Button(pg.Rect(bx, BAR.y + 12, 110, 30), "Find game", lambda: self.find_game(),
                   lambda: not self.playing()),
            Button(pg.Rect(bx + 116, BAR.y + 12, 110, 30), lambda: "Cancel calib" if self.calib_step else "Calibrate",
                   self.toggle_calibration, lambda: self.window is not None and not self.playing(),
                   lambda: self.calib_step > 0),
            Button(pg.Rect(bx + 232, BAR.y + 12, 110, 30), "Reset calib", self.reset_calibration,
                   lambda: self.window is not None and self.geometry.calibrated and not self.playing()),
        ]
        # game controls (right column); RY is shared with draw_game_panel
        x = 816
        has_sol = lambda: self.current_solution() is not None
        game = lambda: self.window is not None and self.window.alive() and not self.playing()
        b["solutions"] += [
            Button(pg.Rect(x, RY["play"], 177, 34), "Play solution in game", self.play_all,
                   lambda: game() and has_sol()),
            Button(pg.Rect(x + 183, RY["play"], 177, 34), lambda: f"Step in game ({self.played_moves})",
                   self.play_step, lambda: game() and has_sol()),
            Button(pg.Rect(x, RY["stop"], 150, 30), "Stop (or hold Esc)", self.stop_player, self.playing),
            Button(pg.Rect(x + 156, RY["stop"], 204, 30),
                   lambda: "Input: background" if self.background_input else "Input: real cursor",
                   lambda: setattr(self, "background_input", not self.background_input),
                   lambda: not self.playing(), lambda: self.background_input),
            Button(pg.Rect(x, RY["speed"], 36, 28), "-", lambda: self.set_speed(self.speed / 1.25)),
            Button(pg.Rect(x + 136, RY["speed"], 36, 28), "+", lambda: self.set_speed(self.speed * 1.25)),
            Button(pg.Rect(x + 184, RY["speed"], 176, 28),
                   lambda: ("[x] " if self.click_next_after else "[ ] ") + "click Next after",
                   lambda: setattr(self, "click_next_after", not self.click_next_after),
                   active=lambda: self.click_next_after),
            Button(pg.Rect(x, RY["sprite"], 360, 28),
                   lambda: ("[x] " if self.show_sprite else "[ ] ") + "show a cursor sprite (background input)",
                   lambda: setattr(self, "show_sprite", not self.show_sprite),
                   lambda: self.background_input and not self.playing(), lambda: self.show_sprite),
            Button(pg.Rect(x + 184, RY["world"] - 3, 36, 26), "-", lambda: self.set_tracked_world(self.tracked_world - 1)),
            Button(pg.Rect(x + 224, RY["world"] - 3, 36, 26), "+", lambda: self.set_tracked_world(self.tracked_world + 1)),
            Button(pg.Rect(x, RY["arrows"], 112, 28), "< World", lambda: self.click_world(-1), game),
            Button(pg.Rect(x + 118, RY["arrows"], 112, 28), "World >", lambda: self.click_world(+1), game),
            Button(pg.Rect(x + 236, RY["arrows"], 124, 28), "Open selected", self.open_selected_in_game, game),
        ]
        for i in range(5):
            b["solutions"].append(Button(pg.Rect(x + i * 73, RY["levels"], 68, 28), f"Level {i + 1}",
                                         lambda i=i: self.click_level_button(i), game))
        # speed chuzzle tab
        info = lambda: self.speed_info()
        b["speed"] += [
            Button(pg.Rect(36, 300, 300, 40), "Reset Speed Chuzzle save", self.reset_speed_save,
                   lambda: info().get("save") is not None and info()["save"].is_file()),
            Button(pg.Rect(346, 300, 180, 40), "Open profile folder",
                   lambda: self._open(info()["profile"].path), lambda: info().get("profile") is not None),
            Button(pg.Rect(536, 300, 180, 40), "Open reset stash",
                   lambda: self._open(info()["stash_dir"]), lambda: info().get("stash_dir") is not None),
        ]
        # saves tab
        bx = W - 230
        b["saves"] += [
            Button(pg.Rect(bx, 60, 210, 30), lambda: "Scanning..." if self.scanning else "Rescan all locations",
                   self.rescan_saves, lambda: not self.scanning),
            Button(pg.Rect(bx, 100, 210, 30), "Back up selected", self.backup_selected,
                   lambda: self.selected_save is not None),
            Button(pg.Rect(bx, 140, 210, 30),
                   lambda: "Click again to restore" if time.time() < self.confirm_restore_until
                   else "Restore latest backup", self.restore_selected,
                   lambda: self.selected_save is not None and bool(self._backups_for_selected())),
            Button(pg.Rect(bx, 180, 210, 30), "Open folder", self.open_selected_folder,
                   lambda: self.selected_save is not None),
            Button(pg.Rect(bx, 220, 210, 30), "Open backup folder",
                   lambda: self._open(saves.paths.SAVE_BACKUP_DIR)),
        ]
        # launcher tab (per-game Launch / Open folder buttons are made while drawing)
        b["launcher"] += [
            Button(pg.Rect(bx, 60, 210, 30), lambda: "Scanning drives..." if self.games_scanning else "Scan all drives",
                   lambda: self.refresh_games(full=True), lambda: not self.games_scanning),
            Button(pg.Rect(bx, 100, 210, 30), "Re-check list", lambda: self.refresh_games(),
                   lambda: not self.games_scanning),
        ]
        # queue tab and the mini view (Pause asks for a second click)
        up = lambda: self.queue_status() is not None and not self.queue.busy
        running = lambda: up() and bool(self.queue_status().get("running"))
        confirming = lambda: time.time() < self.confirm_pause_until
        b["queue"] += [
            Button(pg.Rect(bx, 60, 210, 30), lambda: "Unhold solver" if self.queue_held() else "Hold solver",
                   self.toggle_hold, running, self.queue_held),
            Button(pg.Rect(bx, 100, 210, 30),
                   lambda: "Resume queue" if self.queue_paused() else
                   ("Click again to pause" if confirming() else "Pause queue"),
                   self.toggle_pause, up, self.queue_paused),
            Button(pg.Rect(bx, 140, 210, 30), "Mini view", lambda: self.set_view("mini")),
            Button(pg.Rect(bx, 180, 210, 30), "Open dashboard", self.open_dashboard,
                   lambda: self.queue_status() is not None),
        ]
        my = MINI_H - TB - 8 - 24
        b["mini"] += [
            Button(pg.Rect(MINI_W - 12 - 2 * 84 - 6, my, 84, 24), lambda: "Unhold" if self.queue_held() else "Hold",
                   self.toggle_hold, running, self.queue_held, small=True),
            Button(pg.Rect(MINI_W - 12 - 84, my, 84, 24),
                   lambda: "Resume" if self.queue_paused() else ("Click again" if confirming() else "Pause"),
                   self.toggle_pause, up, self.queue_paused, small=True),
        ]

    def set_tab(self, tab: str) -> None:
        self.tab = tab

    # ------------------------------------------------------------------
    # window: own title bar, pin, mini view
    # ------------------------------------------------------------------

    def _open_window(self, keep: tuple[int, int, int, int] | None = None) -> None:
        """(Re)open the borderless window for the current view. `keep` = the old window
        rect: its top-right corner stays put (moved inside the screen if needed)."""
        size = (W, TB + H) if self.view == "full" else (MINI_W, MINI_H)
        self.window_surface = pg.display.set_mode(size, pg.NOFRAME)
        self.screen = self.window_surface.subsurface(pg.Rect(0, TB, size[0], size[1] - TB))
        self.hwnd = wf.window_handle()
        wf.add_taskbar_styles(self.hwnd)
        wf.set_relaunch(self.hwnd, paths.APP_ID, paths.relaunch_command(), gw.OWN_TITLE, f"{paths.ICON_ICO},0")
        wf.set_topmost(self.hwnd, self.pinned)
        work = wf.work_area(self.hwnd)
        rect = wf.window_rect(self.hwnd)
        if keep is not None:
            wf.move_window(self.hwnd, *wf.keep_top_right(keep, size, work))
        elif rect is not None:
            wf.move_window(self.hwnd, *wf.clamp_into(rect[0], rect[1], size, work))

    def set_view(self, view: str) -> None:
        if view != self.view:
            old = wf.window_rect(self.hwnd)
            self.view = view
            self._open_window(old)

    def toggle_pin(self) -> None:
        self.pinned = not self.pinned
        wf.set_topmost(self.hwnd, self.pinned)
        try:
            settings.put("pinned", self.pinned)
        except OSError as exc:
            self.log(f"Could not save settings: {exc}", WARN)

    def window_mouse(self):
        """Mouse position in window pixels, None when the mouse is not over the window."""
        return pg.mouse.get_pos() if pg.mouse.get_focused() else None

    def content_mouse(self) -> tuple[int, int]:
        pos = self.window_mouse()
        return (pos[0], pos[1] - TB) if pos else (-1, -1)

    def title_press(self, pos) -> None:
        bar = self.title_bars[self.view]
        hit = bar.hit(pos, self.window_surface.get_width())
        if hit == "drag":
            wf.start_drag(self.hwnd)        # Windows moves the window until the button is released
        elif hit is not None and hit not in bar.disabled:
            bar.pressed = hit               # acts on release over the same button, like Windows

    def title_release(self, pos) -> None:
        bar = self.title_bars[self.view]
        pressed, bar.pressed = bar.pressed, None
        if pressed is None or bar.hit(pos, self.window_surface.get_width()) != pressed:
            return
        if pressed == "pin":
            self.toggle_pin()
        elif pressed == "min":
            pg.display.iconify()
        elif pressed == "expand":
            self.set_view("full")
        elif pressed == "close":
            pg.event.post(pg.event.Event(pg.QUIT))      # the normal quit path

    def draw_frame(self) -> None:
        """Title bar and a 1 px border (a borderless window has none)."""
        surf = self.window_surface
        self.title_bars[self.view].draw(surf, self.small, self.window_mouse(), self.pinned, pg.key.get_focused())
        pg.draw.rect(surf, tb.LINE, surf.get_rect(), 1)

    # ------------------------------------------------------------------
    # solver queue
    # ------------------------------------------------------------------

    def queue_status(self) -> dict | None:
        return self.queue.snapshot[0]

    def queue_held(self) -> bool:
        s = self.queue_status()
        return bool(s and s.get("running") and s["running"].get("held_since"))

    def queue_paused(self) -> bool:
        s = self.queue_status()
        return bool(s and s.get("paused"))

    def toggle_hold(self) -> None:
        self._queue_command("unhold" if self.queue_held() else "hold")

    def toggle_pause(self) -> None:
        if self.queue_paused():
            self._queue_command("resume")
        elif time.time() > self.confirm_pause_until:    # pausing stops the queue: ask first
            self.confirm_pause_until = time.time() + 3.0
        else:
            self.confirm_pause_until = 0.0
            self._queue_command("pause")

    def _queue_command(self, command: str) -> None:
        def done(result, error):
            def show():
                if error:
                    self.log(f"Queue: {command} failed: {error}", BAD)
                elif command == "pause":
                    self.log("Queue paused: the running job finishes, no new one starts.", GOOD)
                elif command == "resume":
                    self.log("Queue resumed.", GOOD)
                else:
                    state = "ON HOLD" if result.get("held_since") else "running again"
                    self.log(f"Queue: {qc.describe(result)}  {state}", GOOD)
            self.post(show)
        self.queue.send(command, done)

    def open_dashboard(self) -> None:
        os.startfile(self.queue.base_url + "/")

    # ------------------------------------------------------------------
    # levels / preview
    # ------------------------------------------------------------------

    def refresh_levels(self) -> None:
        """Reload the boards, build/levels_final and build/README.md's bounds (the solver queue
        adds solutions while the app runs). Loads in the background; levels whose solutions
        changed are ranked again."""
        if self.refreshing:
            return
        self.refreshing = True

        def work():
            try:
                new = lv.load_levels()
            except Exception as e:  # noqa: BLE001 - shown to the user
                self.post(lambda: (self.log(f"Refresh failed: {e}", BAD), setattr(self, "refreshing", False)))
                return
            self.post(lambda: self._apply_levels(new))

        threading.Thread(target=work, daemon=True).start()

    def _apply_levels(self, new: dict[str, lv.Level]) -> None:
        def sig(level: lv.Level | None):
            if level is None:
                return None
            f = level.solution_file
            stamp = (f.name, f.stat().st_size, int(f.stat().st_mtime)) if f is not None and f.exists() else None
            return level.solved, level.optimum, len(level.solutions), level.lower_bound, stamp

        old = self.levels
        new_all = {level.name: level for level in lv.all_levels(new)}
        changed = [name for name in new_all if sig(lv.find_level(old, name)) != sig(new_all[name])]
        newly = [name for name in changed
                 if new_all[name].solved and not ((o := lv.find_level(old, name)) and o.solved)]
        self.levels = new
        for name in changed:
            self.ranked.pop(name, None)
            self.rank_progress.pop(name, None)
        self.refreshing = False
        if lv.find_level(self.levels, self.selected) is None:
            self.selected = "1-4"
        if self.selected in changed or self.selected not in self.ranked:
            self.select_level(self.selected)
        if newly:
            self.log("Refresh: newly solved " + ", ".join(
                f"{n} ({new_all[n].optimum} moves, {len(new_all[n].solutions)} solutions)" for n in sorted(newly)), GOOD)
        others = [n for n in changed if n not in newly]
        if others:
            self.log(f"Refresh: updated {', '.join(sorted(others))}")
        if not changed:
            self.log("Refresh: no changes")

    def select_level(self, name: str) -> None:
        self.selected = name
        self.solution_index = 0
        self.preview_step = 0
        self.animating = False
        self.played_moves = 0
        level = self.level
        if not level.solved or name in self.ranked or name in self.ranking:
            return
        cached = routes.cached_ranking(level)
        if cached is not None:
            self.ranked[name] = cached
            return
        self.ranking.add(name)

        def work():
            def progress(i, n):
                self.post(lambda: self.rank_progress.__setitem__(name, f"{i}/{n}"))
            t0 = time.time()
            ranked = routes.rank_solutions(level, on_progress=progress)
            routes.store_ranking(level, ranked)

            def done():
                self.ranking.discard(name)
                if lv.find_level(self.levels, name) is not level:   # refreshed meanwhile: rank the new list
                    if self.selected == name:
                        self.select_level(name)
                    return
                self.ranked[name] = ranked
                if self.selected == name:
                    self.solution_index = 0
                    self.preview_step = 0
                if len(ranked) > 50:
                    self.log(f"{name}: ranked {len(ranked)} solutions by mouse travel in {time.time() - t0:.1f}s")
            self.post(done)

        threading.Thread(target=work, daemon=True).start()

    def preview_states(self) -> list[board_sim.BoardState]:
        sol = self.current_solution()
        start = board_sim.BoardState(list(self.level.start), self.level.start_fat)
        return board_sim.play(start, sol) if sol else [start]

    def set_step(self, step: int) -> None:
        moves = len(self.current_solution().split()) if self.current_solution() else 0
        self.preview_step = max(0, min(moves, step))
        self.anim_t = 0.0

    def set_solution(self, index: int) -> None:
        n = self.solution_count()
        if n:
            self.solution_index = index % n
            self.preview_step = 0
            self.played_moves = 0
            self.animating = False

    def toggle_animate(self) -> None:
        moves = len(self.current_solution().split())
        if self.preview_step >= moves:
            self.preview_step = 0
        self.animating = not self.animating
        self.anim_t = 0.0

    # ------------------------------------------------------------------
    # game window
    # ------------------------------------------------------------------

    def find_game(self, quiet: bool = False) -> None:
        windows = gw.find_game_windows()
        if not windows:
            self.window = None
            if not quiet:
                self.log("No Chuzzle window found. Start the game and press Find game.", WARN)
            return
        self._use_window(windows[0])

    def _use_window(self, window: gw.GameWindow) -> None:
        self.window = window
        self.geometry = gw.load_geometry(self.window)
        _, _, cw, ch = self.window.client_rect()
        self.log(f"Found '{self.window.title}' ({cw}x{ch}){' - calibrated' if self.geometry.calibrated else ''}",
                 GOOD)
        self._update_active_location()
        self._tas_use_window_timing()

    def _update_active_location(self) -> None:
        exe = self.window.exe if self.window else None
        title = self.window.title if self.window else ""
        self.active_loc = saves.active_location(self.locations, exe, title)

    def toggle_calibration(self) -> None:
        if self.calib_step:
            self.calib_step = 0
            self.log("Calibration cancelled.")
            return
        self.calib_step = 1
        self.calib_points = []
        self._f9_was_down = gw.key_down(gw.VK_F9)
        self.log("Calibrate: hover the CENTRE of the TOP-LEFT chuzzle in the game and press F9.", ACCENT)

    def _poll_calibration(self) -> None:
        down = gw.key_down(gw.VK_F9)
        pressed = down and not self._f9_was_down
        self._f9_was_down = down
        if not pressed or self.window is None:
            return
        self.calib_points.append(gw.client_fraction(self.window, gw.cursor_pos()))
        if self.calib_step == 1:
            self.calib_step = 2
            self.log("Now hover the CENTRE of the BOTTOM-RIGHT chuzzle and press F9.", ACCENT)
        else:
            self.calib_step = 0
            geometry = gw.Geometry.from_corners(self.calib_points[0], self.calib_points[1])
            if geometry.cell[0] <= 0 or geometry.cell[1] <= 0:
                self.log("Calibration points look swapped; try again.", BAD)
                return
            self.geometry = geometry
            gw.save_geometry(self.window, geometry)
            self.log("Calibration saved (python/chuzzle_suite/user_data/calibration.json).", GOOD)

    def reset_calibration(self) -> None:
        if self.window:
            gw.clear_geometry(self.window)
        self.geometry = gw.Geometry()
        self.log("Using the default 800x600 layout.")

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.2, min(5.0, speed))
        if self.player is not None:
            self.player.speed = self.speed

    def set_tracked_world(self, world: int) -> None:
        self.tracked_world = max(1, min(20, world))

    def _start_player(self, drags, after=None, label="") -> None:
        from .player import Player

        def progress(i):
            self.post(lambda: setattr(self, "played_moves", self._play_base + i))

        def done(msg):
            color = GOOD if msg == "done" else WARN
            self.post(lambda: (self.log(f"{label}: {msg}", color), setattr(self, "player_status", "")))

        def status(msg):
            self.post(lambda: setattr(self, "player_status", msg))

        overlay = None
        if self.background_input and self.show_sprite:
            if self.cursor_overlay is None:
                from .cursor_overlay import CursorOverlay
                self.cursor_overlay = CursorOverlay()
            overlay = self.cursor_overlay
        self._play_base = self.played_moves
        self.player = Player(self.window, self.geometry, drags, self.speed, background=self.background_input,
                             after=after, on_progress=progress, on_status=status, on_done=done, overlay=overlay)
        self.player.start()

    def _route(self) -> routes.Route:
        return routes.route_for(self.level, self.current_solution(), end_next_puzzle=self.click_next_after)

    def play_all(self) -> None:
        route = self._route()
        self.played_moves = 0
        after = [gw.NEXT_PUZZLE_PROMPT] if self.click_next_after else None
        self.log(f"Playing {self.selected} ({len(route.drags)} moves): {lv.to_display(route.move_string)}")
        self._start_player(route.drags, after, f"Play {self.selected}")

    def play_step(self) -> None:
        route = self._route()
        if self.played_moves >= len(route.drags):
            self.played_moves = 0
            self.log("Step: starting over from move 1 (reset the board in the game first).", WARN)
        drag = route.drags[self.played_moves]
        self._start_player([drag], None, f"Move {self.played_moves + 1} {lv.to_display(drag.move)}")
        self.preview_step = self.played_moves + 1

    def stop_player(self) -> None:
        if self.player is not None:
            self.player.stop()

    def _click_points(self, points, label) -> None:
        from .routes import Drag
        drags = [Drag("click", [p]) for p in points]
        self._start_player(drags, None, label)

    def click_level_button(self, i: int) -> None:
        self._click_points([gw.LEVEL_BUTTONS[i]], f"Level button {i + 1}")

    def click_world(self, delta: int) -> None:
        self._click_points([gw.NEXT_WORLD if delta > 0 else gw.PREV_WORLD], "World arrow")
        self.set_tracked_world(self.tracked_world + delta)

    def open_selected_in_game(self) -> None:
        delta = self.level.world - self.tracked_world
        arrow = gw.NEXT_WORLD if delta > 0 else gw.PREV_WORLD
        points = [arrow] * abs(delta) + [gw.LEVEL_BUTTONS[self.level.puzzle - 1]]
        self._click_points(points, f"Open {self.selected}")
        self.tracked_world = self.level.world
        self.played_moves = 0

    def reset_speed_save(self) -> None:
        loc = self.active_loc
        profile_name = loc.current_profile if loc else None
        profile = loc.profile(profile_name) if loc and profile_name else None
        if profile is None:
            self.log("No current profile found for the active save location.", WARN)
            return
        path = profile.path / "SAVEGAME-SPEEDCHUZZLE.DAT"
        moved = saves.stash_file(path, f"{loc.label}_{profile.name}")
        self._speed_cache = None
        if moved is None:
            self.log(f"No Speed Chuzzle save for {profile.name} (already reset).", WARN)
        else:
            self.log(f"Speed save of {profile.name} moved to {moved.parent.name}\\{moved.name}; "
                     "start Speed Chuzzle again in the game.", GOOD)

    # ------------------------------------------------------------------
    # saves
    # ------------------------------------------------------------------

    def rescan_saves(self) -> None:
        self.scanning = True

        def work():
            extra = [self.window.exe] if self.window and self.window.exe else []
            found = saves.discover(extra_exes=extra)

            def done():
                self.locations = found
                self.scanning = False
                self._update_active_location()
                if self.watcher is None:
                    self.watcher = saves.SaveWatcher(found)
                else:
                    self.watcher.set_locations(found)
                n = sum(len(l.profiles) for l in found)
                self.log(f"Save scan: {len(found)} locations, {n} profiles.", GOOD)
            self.post(done)

        threading.Thread(target=work, daemon=True).start()

    def _watch_loop(self) -> None:
        while True:
            time.sleep(2.0)
            watcher = self.watcher
            if watcher is None or self.scanning:
                continue
            try:
                events = watcher.poll()
            except Exception:
                continue
            if events:
                def show(events=events):
                    for e in events:
                        self.log(f"save {e.change}: {e.path}", ACCENT)
                    for loc in self.locations:
                        saves.scan_location(loc)
                self.post(show)

    def _selected_folder(self):
        if self.selected_save is None:
            return None, None
        loc, profile = self.selected_save
        return loc, (profile.path if profile else loc.root)

    def _backups_for_selected(self):
        loc, folder = self._selected_folder()
        return saves.list_backups(folder, loc.label) if folder else []

    def backup_selected(self) -> None:
        loc, folder = self._selected_folder()
        dest = saves.backup_folder(folder, loc.label)
        self.log(f"Backed up {folder.name} -> {dest}", GOOD)

    def restore_selected(self) -> None:
        if time.time() > self.confirm_restore_until:
            self.confirm_restore_until = time.time() + 3.0
            if self.window and self.window.alive():
                self.log("The game is running; it may overwrite restored saves when it exits.", WARN)
            return
        self.confirm_restore_until = 0.0
        loc, folder = self._selected_folder()
        latest = self._backups_for_selected()[0]
        safety = saves.restore_backup(latest, folder, loc.label)
        self.log(f"Restored {latest.name} into {folder.name} (previous files saved to {safety.name}).", GOOD)

    def open_selected_folder(self) -> None:
        _, folder = self._selected_folder()
        self._open(folder)

    def _open(self, folder) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        os.startfile(str(folder))

    # ------------------------------------------------------------------
    # drawing
    # ------------------------------------------------------------------

    def text(self, s, pos, font=None, color=TEXT, center=False) -> pg.Rect:
        surf = (font or self.font).render(str(s), True, color)
        rect = surf.get_rect(center=pos) if center else surf.get_rect(topleft=pos)
        self.screen.blit(surf, rect)
        return rect

    def wrap(self, s: str, x: int, y: int, width: int, font=None, color=TEXT, line_h=19) -> int:
        font = font or self.font
        line = ""
        for word in s.split():
            trial = f"{line} {word}".strip()
            if font.size(trial)[0] > width and line:
                self.text(line, (x, y), font, color)
                y += line_h
                line = word
            else:
                line = trial
        if line:
            self.text(line, (x, y), font, color)
            y += line_h
        return y

    def info_icon(self, center, text: str, mouse) -> pg.Rect:
        """A small (i); hovering it shows `text` as a tooltip (drawn last)."""
        rect = pg.Rect(0, 0, 18, 18)
        rect.center = center
        hover = rect.inflate(6, 6).collidepoint(mouse)
        pg.draw.circle(self.screen, ACCENT if hover else (70, 76, 94), rect.center, 9)
        self.text("i", (rect.centerx, rect.centery - 1), self.small, (255, 255, 255), center=True)
        if hover:
            self.tooltip = (text, rect)
        return rect

    def draw_tooltip(self) -> None:
        if not self.tooltip:
            return
        text, anchor = self.tooltip
        width = 300
        lines, line = [], ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if self.small.size(trial)[0] > width - 20 and line:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
        box = pg.Rect(0, 0, width, 12 + 17 * len(lines))
        box.topleft = (anchor.x - width + anchor.width, anchor.bottom + 6)
        box.clamp_ip(self.screen.get_rect().inflate(-8, -8))
        pg.draw.rect(self.screen, (20, 22, 28), box, border_radius=6)
        pg.draw.rect(self.screen, ACCENT, box, 1, border_radius=6)
        for i, ln in enumerate(lines):
            self.text(ln, (box.x + 10, box.y + 6 + 17 * i), self.small, TEXT)

    def draw_button(self, b: Button, mouse) -> None:
        enabled = b.enabled()
        hover = enabled and b.rect.collidepoint(mouse)
        fill = (70, 110, 170) if b.active() else ((66, 72, 90) if hover else (50, 54, 68))
        if not enabled:
            fill = (40, 42, 50)
        pg.draw.rect(self.screen, fill, b.rect, border_radius=6)
        pg.draw.rect(self.screen, PANEL_LINE, b.rect, 1, border_radius=6)
        self.text(b.text(), b.rect.center, self.small if b.small else self.font,
                  TEXT if enabled else (100, 104, 116), center=True)

    def progress_bar(self, rect: pg.Rect, percent: float | None, color=ACCENT) -> None:
        pg.draw.rect(self.screen, (16, 18, 24), rect, border_radius=4)
        if percent:
            fill = pg.Rect(rect.x, rect.y, max(rect.height, round(rect.width * min(100.0, percent) / 100)),
                           rect.height)
            pg.draw.rect(self.screen, color, fill, border_radius=4)
        pg.draw.rect(self.screen, PANEL_LINE, rect, 1, border_radius=4)

    def panel(self, rect: pg.Rect, title: str | None = None) -> None:
        pg.draw.rect(self.screen, PANEL, rect, border_radius=8)
        pg.draw.rect(self.screen, PANEL_LINE, rect, 1, border_radius=8)
        if title:
            self.text(title, (rect.x + 12, rect.y + 8), self.bold)

    def draw_board(self, state: board_sim.BoardState, origin, cell: int,
                   moving: tuple[str, list[int], float] | None = None) -> None:
        ox, oy = origin
        area = pg.Rect(ox, oy, cell * 6, cell * 6)
        pg.draw.rect(self.screen, (16, 18, 24), area.inflate(8, 8), border_radius=6)
        clip = self.screen.get_clip()
        self.screen.set_clip(area)
        fat = state.fat
        fat_cells = {(fat[0] + dx, fat[1] + dy) for dx in (0, 1) for dy in (0, 1)} if fat else set()

        def offset_for(x, y):
            if moving is None:
                return 0.0, 0.0
            axis, lines, off = moving
            if axis == "R" and y in lines:
                return off, 0.0
            if axis == "C" and x in lines:
                return 0.0, off
            return 0.0, 0.0

        r = cell // 2 - 3
        for y in range(6):
            for x in range(6):
                if (x, y) in fat_cells:
                    continue
                color = lv.COLOR_RGB[state.get(x, y) % len(lv.COLOR_RGB)]
                dx, dy = offset_for(x, y)
                # a sliding lane wraps: draw the copy entering from the other side too
                for wrap in ((-6, 0, 6) if (dx or dy) else (0,)):
                    cx = ox + (x + dx + (wrap if dx else 0)) * cell + cell / 2
                    cy = oy + (y + dy + (wrap if dy else 0)) * cell + cell / 2
                    if area.inflate(cell, cell).collidepoint(cx, cy):
                        pg.draw.circle(self.screen, color, (cx, cy), r)
                        pg.draw.circle(self.screen, (0, 0, 0), (cx, cy), r, 1)
        if fat:
            color = lv.COLOR_RGB[state.get(*fat) % len(lv.COLOR_RGB)]
            dx, dy = offset_for(fat[0], fat[1])
            cx = ox + (fat[0] + 1 + dx) * cell
            cy = oy + (fat[1] + 1 + dy) * cell
            pg.draw.circle(self.screen, color, (cx, cy), cell - 4)
            pg.draw.circle(self.screen, (0, 0, 0), (cx, cy), cell - 4, 2)
        self.screen.set_clip(clip)
        for i in range(7):
            pg.draw.line(self.screen, (40, 44, 54), (ox + i * cell, oy), (ox + i * cell, oy + 6 * cell))
            pg.draw.line(self.screen, (40, 44, 54), (ox, oy + i * cell), (ox + 6 * cell, oy + i * cell))

    def _preview_motion(self, states, step) -> tuple[str, list[int], float] | None:
        """During animation, slide the lane of move `step` (0-based) part way."""
        sol = self.current_solution()
        if not self.animating or not sol or step >= len(sol.split()):
            return None
        axis, lines, amount = board_sim.parse_token(sol.split()[step])
        disp = amount if amount <= 3 else amount - 6
        fat = states[step].fat
        if fat is not None and set(lines) == ({fat[1], fat[1] + 1} if axis == "R" else {fat[0], fat[0] + 1}):
            pos = fat[0] if axis == "R" else fat[1]
            disp = amount if 0 <= pos + amount <= 4 else amount - 6
        t = min(1.0, self.anim_t / 0.55)
        t = t * t * (3 - 2 * t)
        return axis, lines, disp * t

    def draw_grid(self, mouse) -> None:
        rect = pg.Rect(12, 52, 410, 716)
        self.panel(rect, "Mindbenders")
        self.info_icon((rect.x + 12 + self.bold.size("Mindbenders")[0] + 16, rect.y + 19), GRID_HELP, mouse)
        cw, ch = 78, 32
        self.grid_rects = {}
        for world in range(1, 21):
            for puzzle in range(1, 6):
                name = f"{world}-{puzzle}"
                level = self.levels.get(name)
                r = pg.Rect(rect.x + 10 + (puzzle - 1) * (cw + 2), rect.y + 36 + (world - 1) * (ch + 2), cw, ch)
                self.grid_rects[name] = r
                if level is None:
                    continue
                if level.solved:
                    shade = max(0, min(1, (level.optimum - 3) / 10))
                    fill = (int(40 + 30 * shade), int(120 - 40 * shade), int(70 - 10 * shade))
                else:
                    fill = (110, 44, 48)
                if r.collidepoint(mouse):
                    fill = tuple(min(255, c + 30) for c in fill)
                pg.draw.rect(self.screen, fill, r, border_radius=5)
                if name == self.selected.split("@")[0]:
                    pg.draw.rect(self.screen, (255, 255, 255), r, 2, border_radius=5)
                self.text(name, (r.x + 6, r.y + 7), self.font)
                tag = str(level.optimum) if level.solved else f">={level.lower_bound}"
                self.text(tag, (r.right - 6 - self.small.size(tag)[0], r.y + 9), self.small, (230, 230, 230))
                if level.has_fat:
                    pg.draw.rect(self.screen, (235, 235, 235), (r.right - 10, r.y + 3, 6, 6))
                if level.placements:
                    pg.draw.rect(self.screen, WARN, (r.right - 18, r.y + 3, 6, 6))

    def draw_preview(self, mouse) -> None:
        rect = pg.Rect(432, 52, 362, 716)
        level = self.level
        self.panel(rect, f"Level {level.official}, game-accepted goal" if level.is_placement else f"Level {level.name}")
        y = rect.y + 34
        if level.is_placement:
            if level.solved:
                info = f"Not the official goal: {level.optimum} moves  |  {len(level.solutions)} solutions"
            elif level.lower_bound is not None:
                info = f"Not the official goal: needs at least {level.lower_bound} moves"
            else:
                info = "Not the official goal: not searched yet"
            self.text(info, (rect.x + 12, y), self.font, WARN)
        else:
            if level.solved:
                info = f"Optimal: {level.optimum} moves  |  {len(level.solutions)} solutions"
            else:
                info = f"Unsolved: needs at least {level.lower_bound} moves"
            self.text(info + ("  |  2x2 piece" if level.has_fat else ""), (rect.x + 12, y), self.font, DIM)

        states = self.preview_states()
        step = min(self.preview_step, len(states) - 1)
        motion = self._preview_motion(states, step)
        self.draw_board(states[step], (rect.x + 25, rect.y + 70), 52, motion)

        sol = self.current_solution()
        y = 530
        if sol:
            moves = sol.split()
            step_text = f"step {step}/{len(moves)}"
            sx = rect.right - 14 - self.bold.size(step_text)[0]
            self.text(step_text, (sx, rect.y + 8), self.bold, ACCENT)
            self.info_icon((sx - 16, rect.y + 19), MOVES_HELP, mouse)
            ranked = self.ranked.get(level.name)
            label = f"{self.solution_index + 1}/{len(level.solutions)}"
            if ranked:
                label += f"  mouse {ranked[min(self.solution_index, len(ranked) - 1)][0]:.1f}"
            elif level.name in self.ranking:
                label += f"  ranking {self.rank_progress.get(level.name, '')}"
            self.text(label, (rect.centerx, 500), self.font, DIM, center=True)
            # move list, 1-indexed, current move highlighted
            x0, x = rect.x + 14, rect.x + 14
            for i, token in enumerate(lv.to_display(sol).split()):
                surf_w = self.mono.size(token + " ")[0]
                if x + surf_w > rect.right - 10:
                    x, y = x0, y + 22
                color = ACCENT if i == step else (TEXT if i < step else DIM)
                if i < self.played_moves:
                    color = GOOD
                self.text(token, (x, y), self.mono, color)
                x += surf_w
            y += 30
        else:
            y = 530     # below the step and solution button rows (452-514)
        self.text("Goal" + (" (game also accepts)" if level.is_placement else ""), (rect.x + 14, y), self.small,
                  WARN if level.is_placement else DIM)
        cell = max(14, min(24, (rect.bottom - 14 - (y + 20)) // 6))
        self.draw_board(board_sim.BoardState(list(level.goal), level.goal_fat), (rect.x + 14, y + 20), cell)
        self.draw_goal_switch(level)

    def draw_goal_switch(self, level: lv.Level) -> None:
        """Bottom right of the preview, above the < Goal / Goal > buttons: which goal is shown."""
        goals = self.goals()
        if len(goals) < 2:
            return
        i = next((k for k, g in enumerate(goals) if g.name == level.name), 0)
        x, y = 612, 636
        self.text(f"Goal {i + 1} of {len(goals)}", (x, y), self.bold, WARN if level.is_placement else TEXT)
        lines = (["game also accepts", "(not the official goal)"] if level.is_placement
                 else ["official goal", f"+{len(goals) - 1} the game also accepts"])
        lines.append("piece at " + level.piece_text.replace("columns", "cols"))
        if level.solved:
            lines.append(f"{level.optimum} moves, {len(level.solutions)} sol.")
        elif level.lower_bound is not None:
            lines.append(f"unsolved, >= {level.lower_bound} moves")
        else:
            lines.append("not searched yet")
        for k, s in enumerate(lines):
            self.text(self._elide(s, self.small, 172), (x, y + 22 + 17 * k), self.small, DIM)

    def draw_game_panel(self) -> None:
        rect = pg.Rect(804, 52, 384, 716)
        self.panel(rect, f"Auto-play {self.selected}")
        x = rect.x + 12
        hint = ("Background input: your mouse stays free; keep it off the game window. Hold Esc to stop."
                if self.background_input else
                "Real cursor: hands off the mouse. Esc or alt-tab stops it.")
        if self.player_status:
            hint = self.player_status
        self.wrap(hint, x, RY["hint"], rect.width - 24, self.small, WARN if self.player_status else DIM, 15)
        self.text(f"{self.speed:.2f}x", (x + 104, RY["speed"] + 14), self.font, center=True)
        self.text("Navigate in game", (x, RY["nav"]), self.bold)
        self.text(f"Game shows world {self.tracked_world}", (x, RY["world"]), self.font)
        # log
        ly = RY["log"]
        pg.draw.line(self.screen, PANEL_LINE, (rect.x + 8, ly - 6), (rect.right - 8, ly - 6))
        for when, msg, color in self.log_lines[-20:]:
            ly = self.wrap(f"{time.strftime('%H:%M:%S', time.localtime(when))}  {msg}", x, ly,
                           rect.width - 24, self.small, color, 15)
            if ly > rect.bottom - 10:
                break

    def speed_info(self) -> dict:
        """Current profile's Speed Chuzzle save, high scores and stashed resets
        (re-read at most once a second)."""
        now = time.time()
        if self._speed_cache and now - self._speed_cache[0] < 1.0:
            return self._speed_cache[1]
        info: dict = {}
        loc = self.active_loc
        if loc is not None:
            info["loc"] = loc
            name = loc.current_profile
            profile = loc.profile(name) if name else None
            info["profile"] = profile
            if profile is not None:
                info["save"] = profile.path / "SAVEGAME-SPEEDCHUZZLE.DAT"
                label = f"{loc.label}_{profile.name}"
                info["stash_dir"] = saves.stash_dir(label)
                info["stashed"] = saves.list_stashed(label)
            for folder in (loc.root, loc.root.parent):
                hs = folder / "HSSpeedChuzzle.cfg"
                if hs.is_file():
                    cfg = saves.read_config(hs)
                    info["scores"] = [(cfg.get(f"NAME{i}", ""), cfg.get(f"SCORE{i}", ""))
                                      for i in range(10) if f"NAME{i}" in cfg]
                    break
        self._speed_cache = (now, info)
        return info

    def draw_speed(self) -> None:
        rect = pg.Rect(12, 52, W - 24, 716)
        self.panel(rect, "Speed Chuzzle")
        x, y = rect.x + 24, rect.y + 44
        info = self.speed_info()
        loc, profile = info.get("loc"), info.get("profile")
        if loc is None:
            self.text("No Chuzzle save folder found yet (see the Saves tab).", (x, y), self.font, WARN)
            return
        self.text(f"Game folder: {loc.label}", (x, y), self.font)
        self.text(str(loc.root), (x, y + 22), self.small, DIM)
        if profile is None:
            self.text("No current profile in config.cfg.", (x, y + 50), self.font, WARN)
            return
        self.text(f"Profile: {profile.name}", (x, y + 50), self.bold, GOOD)
        save = info["save"]
        if save.is_file():
            st = save.stat()
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            self.text(f"Speed save: {save.name}  {st.st_size} bytes, modified {when}", (x, y + 76), self.font)
        else:
            self.text("Speed save: none (the next Speed game starts fresh)", (x, y + 76), self.font, GOOD)
        self.wrap("Reset moves this profile's SAVEGAME-SPEEDCHUZZLE.DAT into the reset stash "
                  "(newest 20 kept, nothing is deleted from the game), so the next Speed Chuzzle game "
                  "starts from the beginning. Start Speed Chuzzle again in the game afterwards.",
                  x, y + 108, 720, self.small, DIM, 17)

        # high scores
        sx = rect.x + 780
        self.text("High scores (HSSpeedChuzzle.cfg)", (sx, rect.y + 44), self.bold)
        for i, (name, score) in enumerate(info.get("scores", [])):
            yy = rect.y + 74 + i * 22
            self.text(f"{i + 1}.", (sx, yy), self.font, DIM)
            self.text(name, (sx + 28, yy), self.font)
            self.text(score, (sx + 340 - self.font.size(score)[0], yy), self.font)
        if not info.get("scores"):
            self.text("none found", (sx, rect.y + 74), self.small, DIM)

        # stashed resets
        y = rect.y + 320
        stashed = info.get("stashed", [])
        self.text(f"Reset stash ({len(stashed)})", (x, y), self.bold)
        for i, path in enumerate(stashed[:18]):
            st = path.stat()
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
            self.text(f"{path.name}", (x, y + 28 + i * 20), self.small, TEXT)
            self.text(f"{st.st_size} bytes   saved {when}", (x + 460, y + 28 + i * 20), self.small, DIM)
        if not stashed:
            self.text("No resets yet.", (x, y + 28), self.small, DIM)

    # ------------------------------------------------------------------
    # TAS tab: a tool-assisted Mind Bender run for its trophies (tas_plan.py, tas_runner.py)
    # ------------------------------------------------------------------

    TAS_TARGETS = [(tp.PUZZLER, "PUZZLER: level 10"), (tp.PSYCHIC, "PSYCHIC: all of 1-10"),
                   (tp.MENTALIST_MASTER, "MENTALIST MASTER: level 20"),
                   (tp.MENTALIST_SUPREMO, "SUPREMO: all 100")]
    TAS_HINT = ("Plays Mind Bender in the real game with your real mouse, as fast as the game takes "
                "the inputs. Start it on the Mind Bender screen of a profile that has solved nothing "
                "(level 1, puzzle 1). Hands off the mouse; hold Esc to stop.")

    def _build_tas_buttons(self) -> None:
        b = self.buttons["tas"]
        x0, idle = 24, lambda: not self.tas_running()
        for i, (trophy, label) in enumerate(self.TAS_TARGETS):
            b.append(Button(pg.Rect(x0 + (i % 2) * 229, 170 + (i // 2) * 36, 217, 30), label,
                            lambda t=trophy: self.tas_set(target=t, last_level=None), idle,
                            lambda t=trophy: self.tas_target == t))
        fixed5 = lambda: self.tas_target in (tp.PSYCHIC, tp.MENTALIST_SUPREMO)
        b += [
            Button(pg.Rect(x0 + 206, 246, 116, 28), "3 fastest", lambda: self.tas_set(per_level=3),
                   lambda: idle() and not fixed5(), lambda: self.tas_options().resolved().per_level == 3),
            Button(pg.Rect(x0 + 330, 246, 116, 28), "all 5", lambda: self.tas_set(per_level=5),
                   lambda: idle() and not fixed5(), lambda: self.tas_options().resolved().per_level == 5),
            Button(pg.Rect(x0 + 206, 282, 36, 28), "-",
                   lambda: self.tas_set(last_level=self.tas_options().resolved().last_level - 1), idle),
            Button(pg.Rect(x0 + 306, 282, 36, 28), "+",
                   lambda: self.tas_set(last_level=self.tas_options().resolved().last_level + 1), idle),
            Button(pg.Rect(x0 + 350, 282, 96, 28), "Default", lambda: self.tas_set(last_level=None),
                   lambda: idle() and self.tas_last_level is not None),
            Button(pg.Rect(x0 + 330, 320, 116, 24), "Read profile", self.tas_read_profile,
                   lambda: idle() and self._tas_profile() is not None, small=True),
        ]
        for i, (trophy, label) in enumerate(self.TAS_TARGETS):
            name = tp.TROPHY_NAMES[trophy]
            b.append(Button(pg.Rect(x0 + (i % 2) * 229, 350 + (i // 2) * 34, 217, 28),
                            lambda t=trophy, n=name: ("[x] " if t in self.tas_owned else "[ ] ") + n,
                            lambda t=trophy: self.tas_toggle_owned(t), idle,
                            lambda t=trophy: t in self.tas_owned, small=True))
        b += [
            Button(pg.Rect(x0, 418, 446, 28),
                   lambda: ("[x] " if self.tas_others else "[ ] ") + "every other trophy too (then TROPHY COLLECTOR pops)",
                   lambda: self.tas_set(others_owned=not self.tas_others), idle, lambda: self.tas_others, small=True),
            Button(pg.Rect(x0, 478, 446, 28),
                   lambda: ("[x] " if self.tas_livesplit else "[ ] ") + "LiveSplit key (numpad 1) at the start and the end",
                   lambda: self.tas_set(livesplit=not self.tas_livesplit), idle, lambda: self.tas_livesplit,
                   small=True),
            Button(pg.Rect(x0, 510, 446, 28),
                   lambda: ("[x] " if self.tas_check else "[ ] ") + "check the board on screen (stops on a miss)",
                   lambda: self.tas_set(check_solves=not self.tas_check), idle, lambda: self.tas_check, small=True),
            Button(pg.Rect(x0, 542, 446, 28),
                   lambda: ("[x] " if self.tas_keep_awake else "[ ] ") + "keep the game running when it loses the focus",
                   lambda: self.tas_set(keep_awake=not self.tas_keep_awake), lambda: True, lambda: self.tas_keep_awake,
                   small=True),
            Button(pg.Rect(x0 + 296, 576, 150, 24),
                   lambda: "Click again to stash" if self._tas_confirming("stash") else "Stash the save",
                   self.tas_stash_save, lambda: idle() and self._tas_mb_save_path() is not None, small=True),
            Button(pg.Rect(x0, 694, 217, 28),
                   lambda: "Click again to measure" if self._tas_confirming("measure") else "Measure speed",
                   self.tas_measure, lambda: idle() and self._tas_game_ok()),
            Button(pg.Rect(x0 + 229, 694, 217, 28), "Reset timing", self.tas_reset_timing,
                   lambda: idle() and self.tas_timing.measured),
            Button(pg.Rect(x0, 730, 217, 34),
                   lambda: "Click again to start" if self._tas_confirming("start") else "Start run",
                   self.tas_start, lambda: idle() and self._tas_game_ok() and self.tas_current_plan().ok),
            Button(pg.Rect(x0 + 229, 730, 217, 34), "Stop (or hold Esc)", self.tas_stop, self.tas_running),
        ]

        for i, (burst, label) in enumerate((("drag", "per drag"), ("puzzle", "per puzzle"),
                                            ("level", "per level"), ("run", "whole run"))):
            b.append(Button(pg.Rect(x0 + i * 114, 664, 104, 24), label,
                            lambda v=burst: self.tas_set(burst=v), idle, lambda v=burst: self.tas_burst == v,
                            small=True))

    def _tas_profile(self):
        """The current profile of the save location in use (the game's, after Find game)."""
        loc = self.active_loc
        name = loc.current_profile if loc else None
        return loc.profile(name) if loc and name else None

    def _tas_mb_save_path(self):
        profile = self._tas_profile()
        if profile is None:
            return None
        path = profile.path / saves.MINDBENDER_SAVE
        return path if path.is_file() else None

    def tas_read_profile(self) -> None:
        """Set the "already has" toggles from the profile's INFO.CFG."""
        profile = self._tas_profile()
        if profile is None:
            return
        owned = saves.owned_trophies(profile.path)
        self.tas_owned = owned & set(tp.MB_TROPHIES)
        others = set(range(1, 27)) - set(tp.MB_TROPHIES) - {tp.TROPHY_COLLECTOR}
        self.tas_set(others_owned=others <= owned)
        names = ", ".join(tp.TROPHY_NAMES[t] for t in sorted(self.tas_owned)) or "no Mind Bender trophy"
        self.log(f"TAS: profile {profile.name} has {names}"
                 f"{' and every other trophy' if self.tas_others else ''}.", GOOD)

    def tas_stash_save(self) -> None:
        """Move the profile's Mind Bender save into the stash (like the Speed Chuzzle reset), so
        Mind Bender starts again at 1-1. The game writes the save when it leaves Mind Bender, so
        this is for the main menu (or with the game closed)."""
        path = self._tas_mb_save_path()
        if path is None or not self._tas_confirm("stash"):
            return
        profile = self._tas_profile()
        moved = saves.stash_file(path, f"{self.active_loc.label}_{profile.name}")
        if moved is not None:
            self.log(f"TAS: Mind Bender save of {profile.name} moved to the stash ({moved.name}); "
                     "Mind Bender now starts at 1-1.", GOOD)

    def _tas_save_info(self) -> tuple[str, tuple]:
        """The profile's Mind Bender save, for the tab (re-read at most once a second)."""
        now = time.time()
        cached = getattr(self, "_tas_save_cache", None)
        if cached and now - cached[0] < 1.0:
            return cached[1]
        profile = self._tas_profile()
        if profile is None:
            info = ("Mind Bender save: no profile (Find game first)", DIM)
        else:
            path = profile.path / saves.MINDBENDER_SAVE
            s = saves.read_mindbender_save(path)
            if not path.is_file():
                info = (f"{profile.name}: no Mind Bender save, starts at 1-1", GOOD)
            elif s is None:
                info = (f"{profile.name}: Mind Bender save in another format", WARN)
            elif not s.solved and s.level == 1:
                info = (f"{profile.name}: Mind Bender at 1-1, nothing solved", GOOD)
            else:
                info = (f"{profile.name}: Mind Bender at level {s.level}, {len(s.solved)} solved", WARN)
        self._tas_save_cache = (now, info)
        return info

    def tas_options(self) -> tp.TasOptions:
        return tp.TasOptions(self.tas_target, self.tas_per_level, self.tas_last_level,
                             frozenset(self.tas_owned), self.tas_others)

    def tas_current_plan(self) -> tp.TasPlan:
        if self.tas_plan is None:
            self.tas_plan = tp.build_plan(self.levels, self.tas_options(), self.tas_livesplit,
                                          self.tas_timing.cost_model(self.tas_check))
        return self.tas_plan

    def tas_set(self, **changes) -> None:
        names = {"target": "tas_target", "per_level": "tas_per_level", "last_level": "tas_last_level",
                 "others_owned": "tas_others", "livesplit": "tas_livesplit", "check_solves": "tas_check",
                 "keep_awake": "tas_keep_awake", "burst": "tas_burst"}
        for key, value in changes.items():
            if key == "last_level" and value is not None:
                value = max(1, min(tp.LEVELS, int(value)))
            setattr(self, names[key], value)
        self.focus_keeper.enabled = self.tas_keep_awake
        self.tas_plan = None
        self.tas_scroll = 0
        try:
            settings.put("tas", {"target": self.tas_target, "per_level": self.tas_per_level,
                                 "last_level": self.tas_last_level, "owned": sorted(self.tas_owned),
                                 "others_owned": self.tas_others, "livesplit": self.tas_livesplit,
                                 "check_solves": self.tas_check, "keep_awake": self.tas_keep_awake,
                                 "burst": self.tas_burst})
        except OSError as exc:
            self.log(f"Could not save settings: {exc}", WARN)

    def tas_toggle_owned(self, trophy: int) -> None:
        self.tas_owned ^= {trophy}
        self.tas_set()

    def tas_running(self) -> bool:
        return self.tas_runner is not None and self.tas_runner.running

    def _tas_game_ok(self) -> bool:
        return self.window is not None and self.window.alive() and not self.playing()

    def _tas_confirming(self, what: str) -> bool:
        return self._tas_confirm_what == what and time.time() < self.tas_confirm_until

    def _tas_confirm(self, what: str) -> bool:
        """Moving the real mouse takes over the PC: the first click arms, a second within 3 s goes."""
        if self._tas_confirming(what):
            self.tas_confirm_until = 0.0
            return True
        self._tas_confirm_what = what
        self.tas_confirm_until = time.time() + 3.0
        return False

    def _tas_use_window_timing(self) -> None:
        self.tas_timing = tr.load_timing(self.window)
        self.tas_plan = None

    def tas_start(self) -> None:
        if not self._tas_confirm("start"):
            return
        self._tas_use_window_timing()
        plan = self.tas_current_plan()
        self.tas_step = -1
        self.tas_last_result = ""
        self.log(f"TAS: {tp.TROPHY_NAMES[plan.options.target]} run, {plan.puzzles} puzzles, {plan.moves} moves")
        self.tas_runner = tr.TasRunner(
            self.window, self.geometry, self.levels, plan, self.tas_timing, check_solves=self.tas_check,
            livesplit_vk=tr.VK_NUMPAD1 if self.tas_livesplit else 0, burst=self.tas_burst,
            on_step=lambda i: self.post(lambda: setattr(self, "tas_step", i)),
            on_status=lambda s: self.post(lambda: setattr(self, "tas_status", s)),
            on_done=lambda msg, secs: self.post(lambda: self._tas_done(msg, secs)))
        self.tas_runner.start()

    def _tas_done(self, message: str, seconds: float | None) -> None:
        self.tas_status = ""
        when = f" in {self._tas_clock(seconds)}" if seconds is not None and message == "done" else ""
        r = self.tas_runner
        if isinstance(r, tr.TasRunner):
            when += f" ({r.bursts_sent} input bursts)"
        self.tas_last_result = f"{message}{when}"
        self.log(f"TAS: {self.tas_last_result}", GOOD if message == "done" else WARN)

    def tas_stop(self) -> None:
        if self.tas_runner is not None:
            self.tas_runner.stop()

    def tas_measure(self) -> None:
        if not self._tas_confirm("measure"):
            return
        self._tas_use_window_timing()
        self.tas_last_result = ""
        self.tas_step = -1
        self.log("TAS: measuring the input timing (hands off the mouse; Esc stops)")
        self.tas_runner = tr.SpeedMeasure(
            self.window, self.geometry, self.levels, self.tas_timing,
            on_status=lambda s: self.post(lambda: setattr(self, "tas_status", s)),
            on_done=lambda msg, timing: self.post(lambda: self._tas_measured(msg, timing)))
        self.tas_runner.start()

    def _tas_measured(self, message: str, timing: tr.TasTiming | None) -> None:
        self.tas_status = ""
        if timing is not None:
            self.tas_timing = timing
            tr.save_timing(self.window, timing)
            self.tas_plan = None
        self.tas_last_result = message
        self.log(f"TAS: {message}", GOOD if timing is not None else WARN)

    def tas_reset_timing(self) -> None:
        tr.clear_timing(self.window)
        self._tas_use_window_timing()
        self.log("TAS: back to the default timing.")

    @staticmethod
    def _tas_clock(seconds: float) -> str:
        minutes, secs = divmod(seconds, 60)
        return f"{int(minutes)}:{secs:06.3f}" if minutes else f"{secs:.3f} s"

    def draw_tas(self, mouse) -> None:
        left = pg.Rect(12, 52, 470, 716)
        self.panel(left, "Mind Bender TAS")
        x = left.x + 12
        self.wrap(self.TAS_HINT, x, 86, left.width - 24, self.small, DIM, 16)
        self.text("Trophy to earn", (x, 144), self.bold)
        options = self.tas_options().resolved()
        self.text("Puzzles per level", (x, 250), self.font)
        self.text("Last level played", (x, 286), self.font)
        self.text(str(options.last_level), (x + 274, 296), self.bold, center=True)
        self.text("The profile already has (no pop-up)", (x, 320), self.bold)
        self.text("Run", (x, 454), self.bold)
        save = self._tas_save_info()
        self.text(self._elide_end(save[0], self.small, 282), (x, 580), self.small, save[1])
        self.text("Timing", (x, 610), self.bold)
        t = self.tas_timing
        source = "measured" if t.measured else "default, not measured"
        apart = "an F24 press" if t.key_break else f"a {t.gap * 1000:.1f} ms wait"
        self.text(f"drags: {apart} mid-drag, {t.pace * 1000:.0f} ms after ({source})",
                  (x, 632), self.small, GOOD if t.measured else WARN)
        self.text(f"trophy: pop-up {tr.POPUP_AFTER_SOLVE_S:.2f}-{tr.POPUP_AFTER_ARROW_S:.2f} s, room "
                  f"{tr.ROOM_IN_S:.2f} + {tr.ROOM_OUT_S:.2f} s (+{t.margin:.2f} s each)", (x, 648), self.small, DIM)
        if not self._tas_game_ok() and not self.tas_running():
            self.text("Find the game first (bottom bar).", (x + 200, 613), self.small, WARN)

        right = pg.Rect(494, 52, 694, 716)
        plan = self.tas_current_plan()
        self.panel(right, f"Plan: {tp.TROPHY_NAMES[options.target]} ({tp.TROPHY_GOALS[options.target]})")
        rx = right.x + 12
        pops = ", ".join(tp.TROPHY_NAMES[t] for t in plan.popups) or "none"
        earned = ", ".join(tp.TROPHY_NAMES[t] for t in plan.earned) or "none new"
        self.text(f"Levels 1-{options.last_level}, {options.per_level} per level: {plan.puzzles} puzzles, "
                  f"{plan.moves} moves, {plan.clicks} clicks", (rx, 84), self.font)
        self.text(f"Earns: {earned}.  Pop-ups to click away: {pops}", (rx, 106), self.small, DIM)
        if plan.missing:
            self.text(f"No solution known for {', '.join(plan.missing)}", (rx, 126), self.small, BAD)
        else:
            self.text(f"Estimated time: {self._tas_clock(plan.estimate_s())}", (rx, 126), self.small, DIM)
        running = self.tas_running()
        r = self.tas_runner
        if running or self.tas_last_result:
            status = self.tas_status if running else self.tas_last_result
            elapsed = ""
            if running and getattr(r, "started_at", None) is not None:
                elapsed = f"  {self._tas_clock(time.perf_counter() - r.started_at)}"
            color = ACCENT if running else (GOOD if self.tas_last_result.startswith(("done", "measured")) else WARN)
            self.text(self._elide_end(f"{'Running' if running else 'Last'}: {status}{elapsed}", self.font,
                                      right.width - 24), (rx, 148), self.font, color)
            if running and isinstance(r, tr.TasRunner):
                done = max(0, self.tas_step)
                self.progress_bar(pg.Rect(rx, 172, right.width - 24, 10), 100.0 * done / max(1, len(plan.steps)))
        # the steps
        inner = pg.Rect(rx, 192, right.width - 24, right.bottom - 202)
        pg.draw.line(self.screen, PANEL_LINE, (right.x + 8, inner.y - 6), (right.right - 8, inner.y - 6))
        clip = self.screen.get_clip()
        self.screen.set_clip(inner)
        row_h = 18
        playing_plan = running and isinstance(r, tr.TasRunner)
        if playing_plan and self.tas_step >= 0:     # keep the current step in view
            self.tas_scroll = max(0, self.tas_step * row_h - inner.height // 3)
        self.tas_max_scroll = max(0, len(plan.steps) * row_h - inner.height)
        self.tas_scroll = min(self.tas_scroll, self.tas_max_scroll)
        y = inner.y - self.tas_scroll
        for i, step in enumerate(plan.steps):
            if y > inner.bottom:
                break
            if y + row_h >= inner.y:
                color = {"split": ACCENT, "arrow": ACCENT, "trophy": WARN}.get(step.kind, TEXT)
                if playing_plan and i == self.tas_step:
                    pg.draw.rect(self.screen, (60, 66, 84), pg.Rect(inner.x - 4, y - 1, inner.width + 8, row_h))
                elif playing_plan and i < self.tas_step:
                    color = DIM
                label = self._elide_end(step.describe(), self.small, inner.width - 110)
                self.text(f"{i + 1:>3}", (inner.x, y), self.small, DIM)
                self.text(label + ("   <- target" if step.final else ""), (inner.x + 36, y), self.small, color)
            y += row_h
        self.screen.set_clip(clip)

    def _elide(self, s: str, font, width: int) -> str:
        if font.size(s)[0] <= width:
            return s
        while len(s) > 8 and font.size("..." + s)[0] > width:
            s = s[1:]
        return "..." + s

    def _elide_end(self, s: str, font, width: int) -> str:
        if font.size(s)[0] <= width:
            return s
        while len(s) > 4 and font.size(s + "...")[0] > width:
            s = s[:-1]
        return s.rstrip() + "..."

    def draw_game_bar(self) -> None:
        self.panel(BAR)
        x, y = BAR.x + 14, BAR.y + 8
        right = BAR.right - 10 - 3 * 116 - 14
        sep = "   |   "
        if self.window and self.window.alive():
            left, top, cw, ch = self.window.client_rect()
            cx = self.text(self.window.title, (x, y), self.bold, GOOD).right
            cx = self.text(f"{sep}{cw}x{ch} at ({left},{top})", (cx, y + 1), self.font, TEXT).right
            if self.calib_step:
                state, color = f"Calibrating {self.calib_step}/2: hover a chuzzle centre, press F9", ACCENT
            elif self.geometry.calibrated:
                state, color = "calibrated layout", GOOD
            else:
                state, color = "default 800x600 layout", DIM
            self.text(sep + state, (cx, y + 1), self.font, color)
            line2 = str(self.window.exe or "exe unknown")
        else:
            self.text("Game not found", (x, y), self.bold, WARN)
            line2 = "Start Chuzzle, open a Mindbender puzzle, then press Find game."
        if self.active_loc:
            prof = self.active_loc.current_profile or "?"
            line2 = f"saves: {self.active_loc.label}, profile {prof}{sep}{line2}"
        self.text(self._elide(line2, self.small, right - x), (x, y + 23), self.small, DIM)

    def draw_saves(self, mouse) -> None:
        rect = pg.Rect(12, 52, W - 260, 716)
        self.panel(rect, "Chuzzle save locations (all places the game can write)")
        self.save_rows = []
        clip = self.screen.get_clip()
        inner = rect.inflate(-16, -48).move(0, 16)
        self.screen.set_clip(inner)
        y = inner.y - self.save_scroll
        for loc in self.locations:
            active = loc is self.active_loc
            head = pg.Rect(inner.x, y, inner.width, 24)
            sel = self.selected_save is not None and self.selected_save[0] is loc and self.selected_save[1] is None
            if sel or head.collidepoint(mouse):
                pg.draw.rect(self.screen, (54, 60, 78) if sel else (44, 48, 60), head, border_radius=4)
            self.text(("> " if active else "") + loc.label, (head.x + 6, y + 2), self.bold, GOOD if active else TEXT)
            self.text(f"{loc.kind}  {loc.root}", (head.x + 330, y + 5), self.small, DIM)
            self.save_rows.append((head, loc, None))
            y += 26
            if loc.config:
                cfg = f"config.cfg: CurrentProfile={loc.config.get('CurrentProfile', '-')}  " \
                      f"Profiles={loc.config.get('Profiles', '-')}  " \
                      f"Screen={loc.config.get('DX::ScreenWidth', '?')}x{loc.config.get('DX::ScreenHeight', '?')}"
                self.text(cfg, (inner.x + 24, y), self.small, DIM)
                y += 18
            if loc.shared_files:
                names = ", ".join(f.path.name for f in loc.shared_files)
                self.text(f"shared: {names}", (inner.x + 24, y), self.small, DIM)
                y += 18
            for prof in loc.profiles:
                row = pg.Rect(inner.x + 18, y, inner.width - 18, 20)
                sel = self.selected_save is not None and self.selected_save[1] is prof
                if sel or row.collidepoint(mouse):
                    pg.draw.rect(self.screen, (54, 60, 78) if sel else (44, 48, 60), row, border_radius=4)
                current = loc.current_profile and prof.name.lower() == loc.current_profile.lower()
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(prof.last_modified))
                self.text(("* " if current else "  ") + prof.name, (row.x + 6, y + 1), self.font,
                          GOOD if current else TEXT)
                files = "  ".join(f"{f.path.name.replace('SAVEGAME-', '')} {f.size}B" for f in prof.files)
                self.text(f"{when}   {files}", (row.x + 190, y + 3), self.small, DIM)
                self.save_rows.append((row, loc, prof))
                y += 22
            y += 10
        self.max_scroll = max(0, y + self.save_scroll - inner.bottom)
        self.screen.set_clip(clip)
        if not self.locations:
            self.text("Scanning..." if self.scanning else "No save locations found.", (inner.x, inner.y), self.font, DIM)

        side = pg.Rect(W - 240, 52, 228, 716)
        self.panel(side)
        y = 270
        self.wrap("> = folder the running (or last played) game uses. * = CurrentProfile in "
                  "config.cfg. Click a profile or location, then back it up or restore it. "
                  "Backups go to chuzzle_suite/user_data. Restoring first backs up what is there.",
                  side.x + 10, y, side.width - 20, self.small, DIM, 16)
        self.text("Recent save changes", (side.x + 10, 420), self.bold)
        ly = 446
        changes = [e for e in (self.watcher.events if self.watcher else [])][-14:]
        for e in reversed(changes):
            when = time.strftime("%H:%M:%S", time.localtime(e.when))
            ly = self.wrap(f"{when} {e.change} {e.path.parent.name}\\{e.path.name}",
                           side.x + 10, ly, side.width - 20, self.small, ACCENT, 15)
            if ly > side.bottom - 20:
                break
        if not changes:
            self.text("(watching, none yet)", (side.x + 10, ly), self.small, DIM)

    def rich(self, parts, pos, font, bold, color=TEXT) -> int:
        """Text in runs, each (text, is_bold), one after the other; returns the x after the last."""
        x, y = pos
        for s, b in parts:
            x = self.text(s, (x, y), bold if b else font, color).right
        return x

    def draw_queue(self, mouse) -> None:
        """The Queue tab, laid out like the scheduler's dashboard (python/scheduler/dashboard.html):
        the running job as a title, its progress and phases, the five-number strip, the queued jobs,
        and the solver log filling the rest."""
        status, error, when = self.queue.snapshot
        self.draw_queue_side()
        rect = pg.Rect(12, 52, W - 260, 716)
        self.panel(rect)
        pad = 20
        x, y, width = rect.x + pad, rect.y + 16, rect.width - 2 * pad
        where = self.queue.base_url + (f"   {'checked' if error else 'updated'} "
                                       f"{time.strftime('%H:%M:%S', time.localtime(when))}" if when else "")
        self.text(where, (rect.right - pad - self.small.size(where)[0], rect.y + 10), self.small, DIM)
        if status is None:
            if error is None:
                self.text("Asking the scheduler...", (x, y + 8), self.q_font, DIM)
            else:
                self.text("Scheduler not running", (x, y + 8), self.q_title, WARN)
                self.text(f"({error})   Start it with: python python\\scheduler\\mbq.py daemon",
                          (x, y + 52), self.q_small, DIM)
            return

        job = status.get("running")
        fin = status.get("finalizing")
        alarm = (status.get("alarm") or {}).get("active")
        warns = []
        if alarm:
            warns.append((f"Alarm: {alarm.get('message', '')}", BAD))
        if status.get("paused"):
            warns.append(("Queue is paused: the running job finishes, no new one starts", WARN))
        if job and job.get("held_since"):
            held_for = time.time() - job["held_since"]
            warns.append((f"Solver on hold since {qc.fmt_clock(job['held_since'])} ({qc.fmt_dur(held_for)}); "
                           "it carries on from the same point when released", WARN))
        for n, p in status.get("external_solvers") or []:
            warns.append((f"Another solver is running (not managed): {n} pid {p}", WARN))
        for line, color in warns:
            self.text(self._elide_end(line, self.q_bold, width), (x, y), self.q_bold, color)
            y += 26
        if warns:
            y += 4

        if job:
            e = job.get("eta") or {}
            title = f"{job['puzzle']} at {job['moves']} moves"
            end = self.text(title, (x, y), self.q_title).right
            kind = qc.split(job) or job.get("engine", "")
            if job.get("gpu"):
                kind += f" gpu {job['gpu']}"
            sub = f"{job.get('engine', '')} {kind}".strip() + (" · 2x2 piece" if job.get("fat") else "") + \
                  f" · job #{job['id']}"
            self.text(self._elide_end(sub, self.q_font, rect.right - pad - end - 14), (end + 14, y + 11),
                      self.q_font, DIM)
            y += 44
            self.text(self._elide_end(f"{e.get('phase') or 'starting'} {e.get('detail') or ''}", self.q_font, width),
                      (x, y), self.q_font, DIM)
            y += 28
            held = bool(job.get("held_since"))
            self.progress_bar(pg.Rect(x, y, width, 16), e.get("percent"), WARN if held else ACCENT)
            y += 28
            pct = e.get("percent")
            pct_text = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "-"
            self.rich([(f"{pct_text} of this phase · elapsed {qc.fmt_dur(e.get('elapsed'))} · "
                        "this phase ends in ", False), (qc.fmt_dur(e.get("phase_remaining")), True)],
                      (x, y), self.q_font, self.q_bold)
            y += 28
            if e.get("total_remaining") is not None:
                self.rich([("Whole job: about ", False), (qc.fmt_dur(e["total_remaining"]), True),
                           (" left, done around ", False), (qc.fmt_clock(e.get("finish_at")), True)],
                          (x, y), self.q_font, self.q_bold)
            else:
                self.text("Whole job: estimate appears once later phases have history", (x, y), self.q_font, DIM)
            y += 36
            # the phases as a row of pills
            px = x
            for name, state, spent in qc.phase_chips(job):
                label = name + (f" {qc.fmt_dur(spent)}" if spent else "")
                tw, th = self.q_small.size(label)
                pill = pg.Rect(px, y, tw + 22, th + 10)
                if pill.right > x + width:
                    break
                fill = ACCENT if state == "now" else (16, 18, 24)
                pg.draw.rect(self.screen, fill, pill, border_radius=7)
                color = (255, 255, 255) if state == "now" else GOOD if state == "done" else DIM
                self.text(label, (pill.x + 11, pill.y + 4), self.q_small, color)
                px = pill.right + 8
            y += 40
            for note in ([job["message"]] if job.get("message") else []) + (job.get("notes") or [])[-2:]:
                self.text(self._elide_end(f"note: {note}", self.q_small, width), (x, y), self.q_small, DIM)
                y += 21
            y += 6
        elif fin:
            self.text(f"{fin['puzzle']} at {fin['moves']} moves", (x, y), self.q_title)
            self.text("Search finished, validating results...", (x, y + 44), self.q_font, DIM)
            y += 76
        else:
            pending = status.get("pending") or []
            self.text("Nothing running", (x, y), self.q_title)
            self.text(pending[0].get("wait_reason") or "Starting the next job..." if pending else "The queue is empty.",
                      (x, y + 44), self.q_font, DIM)
            y += 76

        # the strip: solved | queued jobs | queue after this job | free on the table drive | finished today
        y += 6
        est = status.get("queue_estimate") or {}
        after = "-"
        if est.get("known_seconds") is not None:
            after = qc.fmt_dur(est["known_seconds"]) + (f" + {est['unknown_jobs']}?" if est.get("unknown_jobs") else "")
        free = status.get("table_root_free_gb")
        stats = [(f"{status.get('solved', '?')}/100", "Solved"),
                 (str(len(status.get("pending") or [])), "Queued jobs"),
                 (after, "Queue after this job"),
                 (f"{free} GB" if free is not None else "-", "Free on table drive"),
                 (str(status.get("finished_today", 0)), "Finished today")]
        gap = 10
        box_w = (width - gap * (len(stats) - 1)) // len(stats)
        for i, (value, key) in enumerate(stats):
            box = pg.Rect(x + i * (box_w + gap), y, box_w, 64)
            pg.draw.rect(self.screen, (28, 30, 38), box, border_radius=8)
            pg.draw.rect(self.screen, PANEL_LINE, box, 1, border_radius=8)
            self.text(self._elide_end(value, self.q_stat, box_w - 24), (box.x + 12, box.y + 7), self.q_stat)
            self.text(key, (box.x + 12, box.y + 39), self.q_small, DIM)
        y += 64 + 18

        # the queued jobs, one line each
        pending = status.get("pending") or []
        self.text("Queued" if pending else "Nothing queued", (x, y), self.q_bold, TEXT if pending else DIM)
        y += 26
        for j in pending[:6]:
            line = f"#{j['id']}  {j['puzzle']}  {j['moves']} moves  {qc.split(j)}" + \
                   (f"  gpu {j['gpu']}" if j.get("gpu") else "") + (" · 2x2" if j.get("fat") else "") + \
                   (f"   {j['wait_reason']}" if j.get("wait_reason") else "")
            self.text(self._elide_end(line, self.q_mono, width), (x + 8, y), self.q_mono)
            y += 22
        if len(pending) > 6:
            self.text(f"... and {len(pending) - 6} more", (x + 8, y), self.q_small, DIM)
            y += 22

        # the solver log, filling the rest of the panel
        y += 10
        self.text("Solver log", (x, y), self.q_bold)
        y += 26
        box = pg.Rect(x, y, width, rect.bottom - pad + 4 - y)
        if box.height < 40:
            return
        pg.draw.rect(self.screen, (16, 18, 24), box, border_radius=8)
        pg.draw.rect(self.screen, PANEL_LINE, box, 1, border_radius=8)
        live = job or fin
        log = self.queue.log
        if not live:
            self.text("(no job running)", (box.x + 12, box.y + 10), self.q_log, DIM)
            return
        if log is None or log[0] != live.get("id"):
            self.text("(loading the log...)", (box.x + 12, box.y + 10), self.q_log, DIM)
            return
        lines = [(s, (200, 204, 214)) for s in log[1].rstrip("\n").splitlines()]
        now = qc.live_line(status)
        if now:
            lines.append((now, ACCENT))
        char_w = max(1, self.q_log.size("M")[0])
        per_line = max(20, (box.width - 24) // char_w)
        rows = []   # (text, colour, continues the line above)
        for s, color in lines:
            s = s.replace("\t", "    ")
            first = True
            while len(s) > per_line:
                cut = s.rfind(" ", per_line // 2, per_line + 1)   # break at a space when there is one
                cut = cut if cut > 0 else per_line
                rows.append((s[:cut], color, not first))
                s = "  " + s[cut:].lstrip(" ")
                first = False
            rows.append((s, color, not first))
        line_h = 18
        fit = max(1, (box.height - 20) // line_h)
        shown = rows[-fit:]
        while len(shown) > 1 and shown[0][2]:   # start at the beginning of a log line
            shown = shown[1:]
        ly = box.y + 10
        for s, color, _ in shown:
            self.text(s, (box.x + 12, ly), self.q_log, color)
            ly += line_h

    def draw_queue_side(self) -> None:
        side = pg.Rect(W - 240, 52, 228, 716)
        self.panel(side)
        y = self.wrap("Hold freezes the running solver where it is (all cores free); Unhold lets it carry on "
                      "from the same point. Pause lets the running job finish but starts no new one; it asks "
                      "for a second click. Mini view: a small window for the queue that can stay on top "
                      "(pin in the title bar).", side.x + 12, 232, side.width - 24, self.q_small, DIM, 20) + 14
        self.text("Log", (side.x + 12, y), self.q_bold)
        y += 26
        for when_, msg, color in self.log_lines[-12:]:
            y = self.wrap(f"{time.strftime('%H:%M:%S', time.localtime(when_))}  {msg}", side.x + 12, y,
                          side.width - 24, self.q_small, color, 19)
            if y > side.bottom - 20:
                break

    def draw_mini(self) -> None:
        """The mini view: running job, phase, progress, time left, queued count, hold / pause."""
        self.screen.fill(BG)
        status, error, _ = self.queue.snapshot
        x, width = 12, MINI_W - 24
        bottom = MINI_H - TB - 8 - 24 + 4         # text row next to the buttons
        if status is None:
            if error is None:
                self.text("Asking the scheduler...", (x, 8), self.bold, DIM)
            else:
                self.text("scheduler not running", (x, 8), self.bold, WARN)
                self.text(f"({error}) at {self.queue.base_url}", (x, 32), self.small, DIM)
                self.text("Start it with: python python\\scheduler\\mbq.py daemon", (x, 50), self.small, DIM)
            return
        job = status.get("running")
        if job:
            e = job.get("eta") or {}
            held = bool(job.get("held_since"))
            pct = f"{e['percent']}%" if e.get("percent") is not None else ""
            pw = self.bold.size(pct)[0] if pct else 0
            self.text(self._elide_end(qc.short_describe(job), self.bold, width - pw - 10), (x, 4), self.bold)
            if pct:
                self.text(pct, (x + width - pw, 4), self.bold, WARN if held else ACCENT)
            phase = f"phase: {e.get('phase') or '-'} {e.get('detail') or ''}".rstrip()
            self.text(self._elide_end(phase, self.small, width), (x, 27), self.small, DIM)
            self.progress_bar(pg.Rect(x, 47, width, 8), e.get("percent"), WARN if held else ACCENT)
        elif status.get("finalizing"):
            self.text(qc.short_describe(status["finalizing"]) + "  validating...", (x, 4), self.bold, ACCENT)
        else:
            self.text("nothing running", (x, 4), self.bold, DIM)
        queued = f"{len(status.get('pending') or [])} queued"
        qw = self.small.size(queued)[0]
        self.text(queued, (x + width - qw, 60), self.small, DIM)
        if job:
            self.text(self._elide_end(qc.left_line(job.get("eta") or {}), self.small, width - qw - 12), (x, 60),
                      self.small, TEXT)
        # hold / pause state, left of the buttons
        parts = []
        if job and job.get("held_since"):
            parts.append((f"ON HOLD since {qc.fmt_clock(job['held_since'])}", WARN))
        if status.get("paused"):
            parts.append(("PAUSED", WARN))
        if not parts:
            parts.append(("queue running", DIM))
        cx = x
        for i, (text, color) in enumerate(parts):
            cx = self.text(("   " if i else "") + text, (cx, bottom), self.small, color).right

    # ------------------------------------------------------------------
    # launcher
    # ------------------------------------------------------------------

    def refresh_games(self, full: bool = False) -> None:
        """Re-check the cached exes and the usual install folders; with `full` (or when no
        full scan was ever done) also walk every fixed drive, in the background."""
        if self.games_scanning:
            return
        self.games_scanning = True

        def status(text: str) -> None:
            self.post(lambda: setattr(self, "games_scan_status", text))

        def show(games, when) -> None:
            def apply():
                self.games = games
                self.games_scanned_at = when
            self.post(apply)

        def work():
            cached, when = launcher.load_cache()
            show(launcher.describe_all(launcher.quick_list(cached)), when)
            if full or when is None:
                status("scanning all drives...")
                exes = launcher.full_scan(lambda folder, n: status(f"{n:,} folders: {folder}"))
                when = time.time()
                launcher.save_cache(exes, when)
                show(launcher.describe_all(launcher.quick_list(exes)), when)

            def done():
                self.games_scanning = False
                self.games_scan_status = ""
                self.log(f"Launcher: {len(self.games)} Chuzzle exes.", GOOD)
            self.post(done)

        threading.Thread(target=work, daemon=True).start()

    def _running_games(self) -> dict[str, gw.GameWindow]:
        """Open game windows by normalized exe path (re-read once a second)."""
        now = time.time()
        if now - self._running_cache[0] > 1.0:
            running: dict[str, gw.GameWindow] = {}
            for w in gw.find_game_windows():
                if w.exe is not None:
                    running.setdefault(os.path.normcase(str(w.exe)), w)
            self._running_cache = (now, running)
        return self._running_cache[1]

    def launch_game(self, game: launcher.GameExe) -> None:
        try:
            launcher.launch(game.path)
        except OSError as exc:
            self.log(f"Could not start {game.label}: {exc}", BAD)
            return
        self.log(f"Started {game.label}.", GOOD)
        self._running_cache = (0.0, {})
        key = os.path.normcase(str(game.path))

        def attach():
            # use its window as "the game" (bottom bar, auto-play) once it opens
            for _ in range(30):
                time.sleep(1.0)
                for w in gw.find_game_windows():
                    if w.exe is not None and os.path.normcase(str(w.exe)) == key:
                        self.post(lambda w=w: self._use_window(w) if not self.playing() else None)
                        return
        threading.Thread(target=attach, daemon=True).start()

    def draw_launcher(self, mouse) -> None:
        rect = pg.Rect(12, 52, W - 260, 716)
        self.panel(rect, f"Chuzzle games on this PC ({len(self.games)})")
        running = self._running_games()
        inner = rect.inflate(-16, -48).move(0, 16)
        self.launcher_inner = inner
        self.launcher_buttons = []
        clip = self.screen.get_clip()
        self.screen.set_clip(inner)
        y = inner.y - self.launcher_scroll
        for game in self.games:
            row = pg.Rect(inner.x, y, inner.width, 58)
            win = running.get(os.path.normcase(str(game.path)))
            if row.collidepoint(mouse) and inner.collidepoint(mouse):
                pg.draw.rect(self.screen, (44, 48, 60), row, border_radius=6)
            self.text(game.label, (row.x + 8, y + 5), self.bold, GOOD if win else TEXT)
            detail = game.describe() + f", {game.size / 1e6:.1f} MB"
            if win:
                detail += f"   |   running: {win.title}"
            self.text(detail, (row.x + 8, y + 26), self.small, GOOD if win else DIM)
            self.text(self._elide(str(game.path), self.small, row.width - 270), (row.x + 8, y + 41), self.small, DIM)
            bx = row.right - 250
            if win:
                act = Button(pg.Rect(bx, y + 14, 130, 30), "Bring to front", lambda w=win: w.bring_to_front())
            else:
                act = Button(pg.Rect(bx, y + 14, 130, 30), "Launch", lambda g=game: self.launch_game(g),
                             lambda g=game: not g.hollow)
            self.launcher_buttons += [act, Button(pg.Rect(bx + 136, y + 14, 106, 30), "Open folder",
                                                  lambda g=game: self._open(g.path.parent))]
            y += 64
        self.launcher_max_scroll = max(0, y + self.launcher_scroll - inner.bottom)
        for b in self.launcher_buttons:
            if b.rect.colliderect(inner):
                self.draw_button(b, mouse)
        self.screen.set_clip(clip)
        if not self.games:
            self.text("Looking for Chuzzle games..." if self.games_scanning else "No Chuzzle exes found.",
                      (inner.x, inner.y), self.font, DIM)

        side = pg.Rect(W - 240, 52, 228, 716)
        self.panel(side)
        y = 146
        if self.games_scanning and self.games_scan_status:
            y = self.wrap(self.games_scan_status, side.x + 10, y, side.width - 20, self.small, ACCENT, 15) + 8
        when = (time.strftime("%Y-%m-%d %H:%M", time.localtime(self.games_scanned_at))
                if self.games_scanned_at else "never")
        drives = ", ".join(str(d)[:2] for d in launcher.fixed_drives())
        y = self.wrap(f"Last full scan of all drives ({drives}): {when}.", side.x + 10, y, side.width - 20,
                      self.small, TEXT, 16) + 8
        y = self.wrap("Games start from their own folder, like their shortcuts, so they use their usual "
                      "data and saves. A started game becomes the app's game (bottom bar) once its "
                      "window opens. \"PopCap DRM wrapper\" builds unpack the real game when started. "
                      "Re-check is instant; a full scan walks every drive (minutes) and is cached in "
                      "user_data/launcher_cache.json.", side.x + 10, y, side.width - 20, self.small, DIM, 16) + 12
        self.text("Log", (side.x + 10, y), self.bold)
        y += 24
        for when_, msg, color in self.log_lines[-10:]:
            y = self.wrap(f"{time.strftime('%H:%M:%S', time.localtime(when_))}  {msg}", side.x + 10, y,
                          side.width - 20, self.small, color, 15)
            if y > side.bottom - 20:
                break

    def active_buttons(self) -> list[Button]:
        bs = self.buttons["mini"] if self.view == "mini" else self.buttons["all"] + self.buttons[self.tab]
        return [b for b in bs if b.visible()]

    def draw(self) -> None:
        mouse = self.content_mouse()
        self.tooltip = None
        if self.view == "mini":
            self.draw_mini()
            for b in self.active_buttons():
                self.draw_button(b, mouse)
            self.draw_frame()
            pg.display.flip()
            return
        self.screen.fill(BG)
        self.text("Chuzzle Solver", (14, 10), self.big)
        solved = sum(1 for l in self.levels.values() if l.solved and l.world <= 20)
        self.text(f"{solved}/100 mindbenders solved optimally", (14 + self.big.size("Chuzzle Solver")[0] + 16, 18),
                  self.font, DIM)
        if self.tab == "solutions":
            self.draw_grid(mouse)
            self.draw_preview(mouse)
            self.draw_game_panel()
        elif self.tab == "speed":
            self.draw_speed()
        elif self.tab == "launcher":
            self.draw_launcher(mouse)
        elif self.tab == "queue":
            self.draw_queue(mouse)
        elif self.tab == "tas":
            self.draw_tas(mouse)
        else:
            self.draw_saves(mouse)
        self.draw_game_bar()
        for b in self.active_buttons():
            self.draw_button(b, mouse)
        self.draw_tooltip()
        self.draw_frame()
        pg.display.flip()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def handle_click(self, pos) -> None:
        """A click in the content area (pos relative to it, below the title bar)."""
        for b in self.active_buttons():
            if b.rect.collidepoint(pos) and b.enabled():
                b.action()
                return
        if self.view == "mini":
            return
        if self.tab == "solutions":
            for name, r in getattr(self, "grid_rects", {}).items():
                if r.collidepoint(pos) and name in self.levels:
                    if not self.playing():
                        self.select_level(name)
                    return
        elif self.tab == "launcher":
            if self.launcher_inner.collidepoint(pos):
                for b in self.launcher_buttons:
                    if b.rect.collidepoint(pos) and b.enabled():
                        b.action()
                        return
        elif self.tab == "saves":
            for r, loc, prof in self.save_rows:
                if r.collidepoint(pos):
                    self.selected_save = (loc, prof)
                    self.confirm_restore_until = 0.0
                    return

    def run(self) -> None:
        while True:
            dt = self.clock.tick(60) / 1000.0
            while not self.events.empty():
                self.events.get()()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    self.stop_player()
                    self.tas_stop()
                    self.queue.stop()
                    pg.quit()
                    return
                full = self.view == "full"
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if event.pos[1] < TB:
                        self.title_press(event.pos)
                    else:
                        self.handle_click((event.pos[0], event.pos[1] - TB))
                elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                    self.title_release(event.pos)
                elif event.type == pg.MOUSEWHEEL and full and self.tab == "saves":
                    self.save_scroll = max(0, min(getattr(self, "max_scroll", 0), self.save_scroll - event.y * 40))
                elif event.type == pg.MOUSEWHEEL and full and self.tab == "tas":
                    self.tas_scroll = max(0, min(self.tas_max_scroll, self.tas_scroll - event.y * 54))
                elif event.type == pg.MOUSEWHEEL and full and self.tab == "launcher":
                    self.launcher_scroll = max(0, min(self.launcher_max_scroll, self.launcher_scroll - event.y * 40))
                elif event.type == pg.KEYDOWN and full and self.tab == "solutions":
                    if event.key == pg.K_RIGHT:
                        self.set_step(self.preview_step + 1)
                    elif event.key == pg.K_LEFT:
                        self.set_step(self.preview_step - 1)
                    elif event.key == pg.K_SPACE:
                        self.toggle_animate()
                    elif event.key == pg.K_F5:
                        self.refresh_levels()
            if self.calib_step:
                self._poll_calibration()
            if self.animating:
                self.anim_t += dt
                if self.anim_t >= 0.8:
                    self.anim_t = 0.0
                    moves = len(self.current_solution().split())
                    self.preview_step += 1
                    if self.preview_step >= moves:
                        self.preview_step = moves
                        self.animating = False
            if self.window is not None and not self.window.alive() and not self.playing():
                self.log("Game window closed.", WARN)
                self.window = None
            # poll the scheduler only while the queue is on screen
            want = (self.view == "mini" or self.tab == "queue") and not wf.is_minimized(self.hwnd)
            if want and not self.queue.active:
                self.queue.active = True
                self.queue.poke()
            self.queue.active = want
            want_log = self.view == "full" and self.tab == "queue"
            if want_log and not self.queue.want_log:
                self.queue.poke()
            self.queue.want_log = want_log
            self.draw()


def main() -> None:
    App().run()
