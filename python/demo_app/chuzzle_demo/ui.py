# python/demo_app/chuzzle_demo/ui.py
"""pygame control panel: level grid, solution preview, real-game auto-play
and save-file tracking."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import pygame as pg

from . import board_sim, game_window as gw, levels as lv, routes, saves

W, H = 1200, 840
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
RY = {"hint": 84, "play": 120, "stop": 160, "speed": 198, "nav": 246, "world": 276, "arrows": 304,
      "levels": 340, "speedc": 388, "reset": 448, "log": 496}

MOVES_HELP = ("Moves are 1-indexed. R = row (1 = top) shifted right, C = column (1 = left) shifted "
              "down; the last digit is how many cells. Moves of the 2x2 piece name both of its "
              "lines, e.g. C453 = columns 4 and 5 down 3.")
GRID_HELP = ("Number = optimal move count. Green = solved (darker = longer), red = unsolved "
             "(no solution of 12 or fewer moves). White dot = level has a 2x2 piece.")


@dataclass
class Button:
    rect: pg.Rect
    label: str | Callable[[], str]
    action: Callable[[], None]
    enabled: Callable[[], bool] = lambda: True
    active: Callable[[], bool] = lambda: False

    def text(self) -> str:
        return self.label() if callable(self.label) else self.label


class App:
    def __init__(self):
        gw.set_dpi_aware()
        pg.init()
        pg.display.set_caption(gw.OWN_TITLE)
        self.screen = pg.display.set_mode((W, H))
        self.clock = pg.time.Clock()
        self.font = pg.font.SysFont("segoeui", 15)
        self.small = pg.font.SysFont("segoeui", 13)
        self.bold = pg.font.SysFont("segoeui", 17, bold=True)
        self.big = pg.font.SysFont("segoeui", 24, bold=True)
        self.mono = pg.font.SysFont("consolas", 15)

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

        # game
        self.window: gw.GameWindow | None = None
        self.geometry = gw.Geometry()
        self.player = None
        self.played_moves = 0
        self.speed = 1.0
        self.click_next_after = False
        self.background_input = True
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

        self.buttons: dict[str, list[Button]] = {"solutions": [], "saves": [], "all": []}
        self._build_buttons()
        self.select_level(self.selected)
        self.find_game(quiet=True)
        self.rescan_saves()
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
        return self.levels[self.selected]

    def current_solution(self) -> str | None:
        ranked = self.ranked.get(self.selected)
        if ranked:
            return ranked[min(self.solution_index, len(ranked) - 1)][1]
        sols = self.level.solutions
        return sols[min(self.solution_index, len(sols) - 1)] if sols else None

    def solution_count(self) -> int:
        return len(self.level.solutions)

    def playing(self) -> bool:
        return self.player is not None and self.player.running

    # ------------------------------------------------------------------
    # buttons
    # ------------------------------------------------------------------

    def _build_buttons(self) -> None:
        b = self.buttons
        b["all"] += [
            Button(pg.Rect(W - 250, 8, 115, 30), "Solutions", lambda: self.set_tab("solutions"),
                   active=lambda: self.tab == "solutions"),
            Button(pg.Rect(W - 128, 8, 115, 30), "Saves", lambda: self.set_tab("saves"),
                   active=lambda: self.tab == "saves"),
        ]
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
            Button(pg.Rect(x + 184, RY["world"] - 3, 36, 26), "-", lambda: self.set_tracked_world(self.tracked_world - 1)),
            Button(pg.Rect(x + 224, RY["world"] - 3, 36, 26), "+", lambda: self.set_tracked_world(self.tracked_world + 1)),
            Button(pg.Rect(x, RY["arrows"], 112, 28), "< World", lambda: self.click_world(-1), game),
            Button(pg.Rect(x + 118, RY["arrows"], 112, 28), "World >", lambda: self.click_world(+1), game),
            Button(pg.Rect(x + 236, RY["arrows"], 124, 28), "Open selected", self.open_selected_in_game, game),
        ]
        for i in range(5):
            b["solutions"].append(Button(pg.Rect(x + i * 73, RY["levels"], 68, 28), f"Level {i + 1}",
                                         lambda i=i: self.click_level_button(i), game))
        b["solutions"].append(Button(pg.Rect(x, RY["reset"], 360, 30), "Reset Speed Chuzzle save",
                                     self.reset_speed_save, lambda: self.active_loc is not None))
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

    def set_tab(self, tab: str) -> None:
        self.tab = tab

    # ------------------------------------------------------------------
    # levels / preview
    # ------------------------------------------------------------------

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
                self.ranked[name] = ranked
                self.ranking.discard(name)
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
        self.window = windows[0]
        self.geometry = gw.load_geometry(self.window)
        _, _, cw, ch = self.window.client_rect()
        self.log(f"Found '{self.window.title}' ({cw}x{ch}){' - calibrated' if self.geometry.calibrated else ''}",
                 GOOD)
        self._update_active_location()

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
            self.log("Calibration saved (python/demo_app/user_data/calibration.json).", GOOD)

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

        self._play_base = self.played_moves
        self.player = Player(self.window, self.geometry, drags, self.speed, background=self.background_input,
                             after=after, on_progress=progress, on_status=status, on_done=done)
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
        self.text(b.text(), b.rect.center, self.font, TEXT if enabled else (100, 104, 116), center=True)

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
                if name == self.selected:
                    pg.draw.rect(self.screen, (255, 255, 255), r, 2, border_radius=5)
                self.text(name, (r.x + 6, r.y + 7), self.font)
                tag = str(level.optimum) if level.solved else ">=13"
                self.text(tag, (r.right - 6 - self.small.size(tag)[0], r.y + 9), self.small, (230, 230, 230))
                if level.has_fat:
                    pg.draw.rect(self.screen, (235, 235, 235), (r.right - 10, r.y + 3, 6, 6))

    def draw_preview(self, mouse) -> None:
        rect = pg.Rect(432, 52, 362, 716)
        level = self.level
        self.panel(rect, f"Level {level.name}")
        y = rect.y + 34
        if level.solved:
            info = f"Optimal: {level.optimum} moves  |  {len(level.solutions)} solutions"
        else:
            info = "Unsolved: no solution with 12 or fewer moves"
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
            y = 470
        self.text("Goal", (rect.x + 14, y), self.small, DIM)
        cell = max(14, min(24, (rect.bottom - 14 - (y + 20)) // 6))
        self.draw_board(board_sim.BoardState(list(level.goal), level.goal_fat), (rect.x + 14, y + 20), cell)

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
        self.text("Speed Chuzzle", (x, RY["speedc"]), self.bold)
        self.wrap("Moves the current profile's SAVEGAME-SPEEDCHUZZLE.DAT into user_data/save_backups "
                  "(last 20 kept) so the next Speed game starts fresh.",
                  x, RY["speedc"] + 24, rect.width - 24, self.small, DIM, 16)
        # log
        ly = RY["log"]
        pg.draw.line(self.screen, PANEL_LINE, (rect.x + 8, ly - 6), (rect.right - 8, ly - 6))
        for when, msg, color in self.log_lines[-14:]:
            ly = self.wrap(f"{time.strftime('%H:%M:%S', time.localtime(when))}  {msg}", x, ly,
                           rect.width - 24, self.small, color, 15)
            if ly > rect.bottom - 10:
                break

    def _elide(self, s: str, font, width: int) -> str:
        if font.size(s)[0] <= width:
            return s
        while len(s) > 8 and font.size("..." + s)[0] > width:
            s = s[1:]
        return "..." + s

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
                  "Backups go to demo_app/user_data. Restoring first backs up what is there.",
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

    def draw(self) -> None:
        mouse = pg.mouse.get_pos()
        self.tooltip = None
        self.screen.fill(BG)
        self.text("Chuzzle Demo", (14, 10), self.big)
        solved = sum(1 for l in self.levels.values() if l.solved and l.world <= 20)
        self.text(f"{solved}/100 mindbenders solved optimally", (190, 18), self.font, DIM)
        if self.tab == "solutions":
            self.draw_grid(mouse)
            self.draw_preview(mouse)
            self.draw_game_panel()
        else:
            self.draw_saves(mouse)
        self.draw_game_bar()
        for b in self.buttons["all"] + self.buttons[self.tab]:
            self.draw_button(b, mouse)
        self.draw_tooltip()
        pg.display.flip()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------

    def handle_click(self, pos) -> None:
        for b in self.buttons["all"] + self.buttons[self.tab]:
            if b.rect.collidepoint(pos) and b.enabled():
                b.action()
                return
        if self.tab == "solutions":
            for name, r in getattr(self, "grid_rects", {}).items():
                if r.collidepoint(pos) and name in self.levels:
                    if not self.playing():
                        self.select_level(name)
                    return
        else:
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
                    pg.quit()
                    return
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pg.MOUSEWHEEL and self.tab == "saves":
                    self.save_scroll = max(0, min(getattr(self, "max_scroll", 0), self.save_scroll - event.y * 40))
                elif event.type == pg.KEYDOWN and self.tab == "solutions":
                    if event.key == pg.K_RIGHT:
                        self.set_step(self.preview_step + 1)
                    elif event.key == pg.K_LEFT:
                        self.set_step(self.preview_step - 1)
                    elif event.key == pg.K_SPACE:
                        self.toggle_animate()
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
            self.draw()


def main() -> None:
    App().run()
