from __future__ import annotations
# pyth/viewer/chuzzle_mouse_studio.py

import colorsys
import math
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk

from chuzzle_mouse_adapter import (
    DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    DEFAULT_LOCK_THRESHOLD,
    DEFAULT_NEXT_PUZZLE_TARGETS,
    DpMouseSolver,
    FileScoreResult,
    GRID_SIZE,
    ScoredSolution,
)

APP_TITLE = "Chuzzle DP Studio"
DEFAULT_FOLDER = "../../build/levels_final"
FILE_PATTERN = re.compile(r"^(\d+)-(\d+)_.*\.txt$")

BOARD_CELL_SIZE = 96
BOARD_GRID_X = 1.5
BOARD_GRID_Y = 0.8
BOARD_GRID_W = GRID_SIZE
BOARD_GRID_H = GRID_SIZE
CANVAS_W = 900
CANVAS_H = 600
EPSILON = 1e-9

START_PRESET_NONE = "None"
START_PRESET_LEFT_STAR = "Left Star (-1, 5)"
START_PRESET_CUSTOM = "Custom"

END_PRESET_NONE = "None"
END_PRESET_NEXT = "Next puzzle banner"
END_PRESET_LEFT_STAR = "Left Star (-1, 5)"
END_PRESET_CUSTOM = "Custom"


@dataclass(frozen=True)
class StepView:
    step_index: int
    token: str
    axis: str
    is_fat: bool
    group_id: int
    click_down: tuple[float, float]
    lock_point: tuple[float, float]
    release: tuple[float, float]
    path_points: tuple[tuple[float, float], ...]
    selected_line: int
    displacement: int
    free_drag: bool
    drag_distance: float
    move_distance: float
    cumulative_cost: float


def board_to_canvas(x: float, y: float) -> tuple[float, float]:
    return (
        (BOARD_GRID_X + x + 0.5) * BOARD_CELL_SIZE,
        (BOARD_GRID_Y + y + 0.5) * BOARD_CELL_SIZE,
    )


def hex_color_from_hsv(h: float, s: float, v: float) -> str:
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0.0, min(1.0, s)), max(0.0, min(1.0, v)))
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def color_for_step(step_index: int, total_steps: int) -> str:
    if total_steps <= 1:
        return hex_color_from_hsv(0.58, 0.75, 0.92)
    hue = 0.66 * (step_index / max(1, total_steps - 1))
    return hex_color_from_hsv(hue, 0.75, 0.96)


def color_for_group(group_id: int) -> str:
    golden = 0.61803398875
    hue = (0.12 + group_id * golden) % 1.0
    return hex_color_from_hsv(hue, 0.65, 0.95)


def color_for_axis(axis: str, is_fat: bool) -> str:
    if axis == "R":
        return "#db4437" if not is_fat else "#8e24aa"
    return "#1a73e8" if not is_fat else "#00897b"


def with_alpha_like(color: str, factor: float) -> str:
    color = color.lstrip("#")
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    r = int(255 - (255 - r) * factor)
    g = int(255 - (255 - g) * factor)
    b = int(255 - (255 - b) * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def derive_step_views(solution: ScoredSolution) -> list[StepView]:
    views: list[StepView] = []
    group_id = 1

    for i, move in enumerate(solution.move_data):
        if i > 0 and move.move_distance > EPSILON:
            group_id += 1

        axis = move.move[0]
        is_fat = len(move.move) == 4
        path_points = tuple((float(p[0]), float(p[1])) for p in getattr(move, "path_points", [move.click_down, move.release]))
        lock_point = tuple(getattr(move, "lock_point", move.release))
        view = StepView(
            step_index=i,
            token=move.move,
            axis=axis,
            is_fat=is_fat,
            group_id=group_id,
            click_down=(float(move.click_down[0]), float(move.click_down[1])),
            lock_point=(float(lock_point[0]), float(lock_point[1])),
            release=(float(move.release[0]), float(move.release[1])),
            path_points=path_points,
            selected_line=int(move.selected_line),
            displacement=int(move.displacement),
            free_drag=bool(getattr(move, "free_drag", False)),
            drag_distance=float(move.drag_distance),
            move_distance=float(move.move_distance),
            cumulative_cost=float(move.total_distance_to_here),
        )
        views.append(view)

    return views


def overlap_key(step: StepView) -> tuple[str, float, float, float]:
    sx, sy = step.click_down
    ex, ey = step.release
    dx = ex - sx
    dy = ey - sy
    if abs(dy) <= EPSILON and abs(dx) > EPSILON:
        return ("h", sy, min(sx, ex), max(sx, ex))
    if abs(dx) <= EPSILON and abs(dy) > EPSILON:
        return ("v", sx, min(sy, ey), max(sy, ey))
    slope = round(dy / dx, 3) if abs(dx) > EPSILON else 999999.0
    intercept = round(sy - slope * sx, 3) if slope != 999999.0 else sx
    return ("d", slope, intercept, round(math.hypot(dx, dy), 3))


def collapse_duplicate_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return points

    out = [points[0]]
    for x, y in points[1:]:
        px, py = out[-1]
        if abs(x - px) > EPSILON or abs(y - py) > EPSILON:
            out.append((x, y))
    return out

def offset_points(points: list[tuple[float, float]], slot: int, total_slots: int) -> list[tuple[float, float]]:
    if len(points) < 2 or total_slots <= 1:
        return points

    sx, sy = points[0]
    ex, ey = points[-1]
    dx = ex - sx
    dy = ey - sy
    dist = math.hypot(dx, dy)
    if dist <= EPSILON:
        return points

    ux = dx / dist
    uy = dy / dist
    px = -uy
    py = ux
    max_offset = 12.0
    center = (total_slots - 1) / 2.0
    amount = (slot - center) * (max_offset / max(1.0, center if center > 0 else 1.0))
    return [(x + px * amount, y + py * amount) for x, y in points]


def parse_point_text(text: str) -> tuple[float, float]:
    text = text.strip()
    if not text:
        raise ValueError("Expected a point like x,y")
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
    else:
        parts = text.split()
    if len(parts) != 2:
        raise ValueError(f"Invalid point: {text!r}. Use x,y")
    return float(parts[0]), float(parts[1])


def parse_point_list_text(text: str) -> list[tuple[float, float]]:
    text = text.strip()
    if not text:
        return []
    out: list[tuple[float, float]] = []
    chunks = text.replace("\n", ";").split(";")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        out.append(parse_point_text(chunk))
    return out


def format_point(point: tuple[float, float] | None) -> str:
    if point is None:
        return "none"
    return f"({point[0]:g}, {point[1]:g})"


class StudioApp:
    def __init__(self, folder: str = DEFAULT_FOLDER):
        self.folder = folder
        self.solver = DpMouseSolver()
        self.current_result: FileScoreResult | None = None
        self.current_solution_index = 0
        self.current_step_index: int | None = None
        self.filtered_files: list[str] = []
        self.step_views: list[StepView] = []
        self._suppress_solution_select = False
        self._suppress_step_select = False
        self._options_window: tk.Toplevel | None = None
        self._pending_load_after_id: str | None = None

        self.root = tk.Tk()
        self.root.title(APP_TITLE)
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        window_w = min(1700, max(1180, screen_w - 40))
        window_h = min(940, max(720, screen_h - 80))
        self.root.geometry(f"{window_w}x{window_h}+10+10")
        self.root.minsize(980, 620)

        self.dedupe_var = tk.BooleanVar(value=False)
        self.show_numbers_var = tk.BooleanVar(value=True)
        self.show_travel_var = tk.BooleanVar(value=True)
        self.color_mode_var = tk.StringVar(value="chain")
        self.file_filter_var = tk.StringVar(value="")
        self.summary_var = tk.StringVar(value="No puzzle loaded")
        self.status_var = tk.StringVar(value="Ready")

        self.start_preset_var = tk.StringVar(value=START_PRESET_NONE)
        self.start_custom_var = tk.StringVar(value="")
        self.end_preset_var = tk.StringVar(value=END_PRESET_NONE)
        self.end_custom_var = tk.StringVar(value="")
        self.lock_threshold_var = tk.StringVar(value=f"{DEFAULT_LOCK_THRESHOLD:g}")
        self.free_drag_min_disp_var = tk.StringVar(value=str(DEFAULT_FREE_DRAG_MIN_DISPLACEMENT))
        self.active_options_var = tk.StringVar(value="")

        self._build_ui()
        self._update_active_options_summary()
        self._load_file_list(initial=True)

    def _build_ui(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        root_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        root_pane.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(root_pane, padding=10)
        center = ttk.Frame(root_pane, padding=10)
        right = ttk.Frame(root_pane, padding=10)
        root_pane.add(left, weight=1)
        root_pane.add(center, weight=5)
        root_pane.add(right, weight=4)

        self._build_left_panel(left)
        self._build_center_panel(center)
        self._build_right_panel(right)

        status_bar = ttk.Label(
            self.root,
            textvariable=self.status_var,
            anchor="w",
            relief=tk.SUNKEN,
            padding=(8, 4),
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Puzzles", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(parent, text="Filter by file name").pack(anchor="w", pady=(10, 2))

        search_entry = ttk.Entry(parent, textvariable=self.file_filter_var)
        search_entry.pack(fill=tk.X)
        search_entry.bind("<KeyRelease>", lambda _e: self._load_file_list())

        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.file_listbox = tk.Listbox(list_frame, exportselection=False, activestyle="none")
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.config(yscrollcommand=file_scroll.set)
        self.file_listbox.bind("<<ListboxSelect>>", self._on_puzzle_select)
        self.file_listbox.bind("<Double-1>", lambda _e: self._load_selected_from_list())

        controls = ttk.LabelFrame(parent, text="Controls", padding=10)
        controls.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(controls, text="Load selected puzzle", command=self._load_selected_from_list).pack(anchor="w")
        ttk.Button(controls, text="Options...", command=self._open_options_window).pack(anchor="w", pady=(8, 0))

        options_box = ttk.LabelFrame(parent, text="Active options", padding=10)
        options_box.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            options_box,
            textvariable=self.active_options_var,
            justify=tk.LEFT,
            wraplength=260,
        ).pack(anchor="w")

    def _build_center_panel(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent)
        top_bar.pack(fill=tk.X)
        ttk.Label(top_bar, textvariable=self.summary_var, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, anchor="w")

        nav = ttk.Frame(parent)
        nav.pack(fill=tk.X, pady=(6, 4))
        ttk.Button(nav, text="Top", command=self._go_first_solution).pack(side=tk.LEFT)
        ttk.Button(nav, text="Prev", command=self._go_prev_solution).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Next", command=self._go_next_solution).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Bottom", command=self._go_last_solution).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(nav, text="Reload Puzzle", command=self._reload_current_puzzle).pack(side=tk.LEFT, padx=(12, 0))

        self.solution_text = tk.Text(
            parent,
            height=1,
            wrap=tk.WORD,
            font=("Consolas", 12),
            relief=tk.FLAT,
            bg="#f7f7f7",
        )
        self.solution_text.pack(fill=tk.X)
        self.solution_text.config(state=tk.DISABLED)

        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_W,
            height=CANVAS_H,
            bg="#f3f4f6",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Ranked solutions", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        solution_frame = ttk.Frame(parent)
        solution_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 10))
        self.solution_tree = ttk.Treeview(
            solution_frame,
            columns=("rank", "total", "move", "drag", "steps", "fat"),
            show="headings",
            height=10,
        )
        for col, text, width in [
            ("rank", "#", 44),
            ("total", "Total", 72),
            ("move", "Move", 72),
            ("drag", "Drag", 72),
            ("steps", "Steps", 56),
            ("fat", "Fat", 42),
        ]:
            self.solution_tree.heading(col, text=text)
            self.solution_tree.column(col, width=width, anchor=tk.CENTER, stretch=False)
        self.solution_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        solution_scroll = ttk.Scrollbar(solution_frame, orient=tk.VERTICAL, command=self.solution_tree.yview)
        solution_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.solution_tree.config(yscrollcommand=solution_scroll.set)
        self.solution_tree.bind("<<TreeviewSelect>>", self._on_solution_select)

        ttk.Label(parent, text="Steps", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(4, 0))
        steps_frame = ttk.Frame(parent)
        steps_frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.steps_tree = ttk.Treeview(
            steps_frame,
            columns=("step", "move", "grp", "click", "lock", "release", "gap", "drag"),
            show="headings",
            height=14,
        )
        for col, text, width in [
            ("step", "#", 42),
            ("move", "Move", 64),
            ("grp", "Grp", 42),
            ("click", "Click", 72),
            ("lock", "Lock", 72),
            ("release", "Release", 78),
            ("gap", "Gap", 62),
            ("drag", "Drag", 62),
        ]:
            self.steps_tree.heading(col, text=text)
            self.steps_tree.column(col, width=width, anchor=tk.CENTER, stretch=False)
        self.steps_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        steps_scroll = ttk.Scrollbar(steps_frame, orient=tk.VERTICAL, command=self.steps_tree.yview)
        steps_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.steps_tree.config(yscrollcommand=steps_scroll.set)
        self.steps_tree.bind("<<TreeviewSelect>>", self._on_step_select)

    def _resolve_start_mouse_position(self) -> tuple[float, float] | None:
        preset = self.start_preset_var.get()
        if preset == START_PRESET_NONE:
            return None
        if preset == START_PRESET_LEFT_STAR:
            return -1.0, 5.0
        if preset == START_PRESET_CUSTOM:
            return parse_point_text(self.start_custom_var.get())
        return None

    def _resolve_end_config(self) -> tuple[list[tuple[float, float]] | None, bool]:
        preset = self.end_preset_var.get()
        if preset == END_PRESET_NEXT:
            return None, True
        if preset == END_PRESET_NONE:
            return None, False
        if preset == END_PRESET_LEFT_STAR:
            return [(-1.0, 5.0)], False
        if preset == END_PRESET_CUSTOM:
            points = parse_point_list_text(self.end_custom_var.get())
            if not points:
                raise ValueError("Custom end preset needs at least one point")
            return points, False
        return None, False

    def _resolve_lock_threshold(self) -> float:
        text = self.lock_threshold_var.get().strip()
        value = float(text)
        if value < 0.0:
            raise ValueError("Lock threshold must be >= 0")
        return value

    def _resolve_free_drag_min_disp(self) -> int:
        text = self.free_drag_min_disp_var.get().strip()
        value = int(text)
        if value < 0:
            raise ValueError("Free-drag minimum displacement must be >= 0")
        return value

    def _describe_start_option(self) -> str:
        preset = self.start_preset_var.get()
        if preset == START_PRESET_NONE:
            return "none"
        if preset == START_PRESET_LEFT_STAR:
            return "left star (-1,5)"
        if preset == START_PRESET_CUSTOM:
            try:
                return format_point(parse_point_text(self.start_custom_var.get()))
            except Exception:
                return "custom ?"
        return "none"

    def _describe_end_option(self) -> str:
        preset = self.end_preset_var.get()
        if preset == END_PRESET_NONE:
            return "none"
        if preset == END_PRESET_NEXT:
            return "next puzzle banner"
        if preset == END_PRESET_LEFT_STAR:
            return "left star (-1,5)"
        if preset == END_PRESET_CUSTOM:
            try:
                points = parse_point_list_text(self.end_custom_var.get())
                if not points:
                    return "custom ?"
                return "; ".join(format_point(p) for p in points)
            except Exception:
                return "custom ?"
        return "none"

    def _update_active_options_summary(self) -> None:
        text = (
            f"Start: {self._describe_start_option()}\n"
            f"End: {self._describe_end_option()}\n"
            f"Lock threshold: {self.lock_threshold_var.get()}\n"
            f"Free-drag at |d| >= {self.free_drag_min_disp_var.get()}\n"
            f"Dedupe: {'on' if self.dedupe_var.get() else 'off'}\n"
            f"Move numbers: {'on' if self.show_numbers_var.get() else 'off'}\n"
            f"Cursor gaps: {'on' if self.show_travel_var.get() else 'off'}\n"
            f"Color mode: {self.color_mode_var.get()}\n"
            f"Solver: {self.solver.backend_name()}"
        )
        self.active_options_var.set(text)

    def _open_options_window(self) -> None:
        if self._options_window is not None and self._options_window.winfo_exists():
            self._options_window.lift()
            self._options_window.focus_force()
            return

        win = tk.Toplevel(self.root)
        win.title("Studio 2 Options")
        win.transient(self.root)
        win.grab_set()
        win.resizable(False, False)
        self._options_window = win

        outer = ttk.Frame(win, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        solver_box = ttk.LabelFrame(outer, text="Solver options", padding=10)
        solver_box.pack(fill=tk.X)

        ttk.Checkbutton(solver_box, text="Dedupe identical solution lines", variable=self.dedupe_var).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(solver_box, text="Start preset").grid(row=1, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            solver_box,
            textvariable=self.start_preset_var,
            state="readonly",
            values=[START_PRESET_NONE, START_PRESET_LEFT_STAR, START_PRESET_CUSTOM],
            width=24,
        ).grid(row=2, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="Start custom point").grid(row=1, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.start_custom_var, width=24).grid(row=2, column=1, sticky="we")

        ttk.Label(solver_box, text="End preset").grid(row=3, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            solver_box,
            textvariable=self.end_preset_var,
            state="readonly",
            values=[END_PRESET_NONE, END_PRESET_NEXT, END_PRESET_LEFT_STAR, END_PRESET_CUSTOM],
            width=24,
        ).grid(row=4, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="End custom points").grid(row=3, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.end_custom_var, width=24).grid(row=4, column=1, sticky="we")

        ttk.Label(solver_box, text="Lock threshold").grid(row=5, column=0, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.lock_threshold_var, width=24).grid(row=6, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="Free-drag minimum |displacement|").grid(row=5, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.free_drag_min_disp_var, width=24).grid(row=6, column=1, sticky="we")

        ttk.Label(
            solver_box,
            text=(
                "Custom point format: x,y\n"
                "Multiple end points: x,y; x,y; x,y\n"
                "Adapter routes to the active solver backend."
            ),
            justify=tk.LEFT,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(8, 0))

        view_box = ttk.LabelFrame(outer, text="View options", padding=10)
        view_box.pack(fill=tk.X, pady=(12, 0))
        ttk.Checkbutton(view_box, text="Show move numbers on board", variable=self.show_numbers_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(view_box, text="Show cursor travel gaps", variable=self.show_travel_var).grid(row=1, column=0, sticky="w")

        ttk.Label(view_box, text="Color strategy").grid(row=2, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            view_box,
            textvariable=self.color_mode_var,
            state="readonly",
            values=["chain", "sequence", "axis"],
            width=18,
        ).grid(row=3, column=0, sticky="w")

        legend_box = ttk.LabelFrame(outer, text="Legend", padding=10)
        legend_box.pack(fill=tk.X, pady=(12, 0))
        ttk.Label(
            legend_box,
            text=(
                "chain: same color until the cursor must reposition\n"
                "sequence: color ramps by step order\n"
                "axis: rows vs columns, fat moves get distinct hues\n\n"
                "Solid path uses click -> lock -> release.\n"
                "Dashed gray lines show pure mouse travel."
            ),
            justify=tk.LEFT,
        ).pack(anchor="w")

        buttons = ttk.Frame(outer)
        buttons.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(buttons, text="Apply", command=self._apply_options_from_window).pack(side=tk.RIGHT)
        ttk.Button(buttons, text="Cancel", command=self._close_options_window).pack(side=tk.RIGHT, padx=(0, 8))

        win.protocol("WM_DELETE_WINDOW", self._close_options_window)

    def _close_options_window(self) -> None:
        if self._options_window is not None and self._options_window.winfo_exists():
            self._options_window.destroy()
        self._options_window = None

    def _apply_options_from_window(self) -> None:
        try:
            self._resolve_start_mouse_position()
            self._resolve_end_config()
            self._resolve_lock_threshold()
            self._resolve_free_drag_min_disp()
        except Exception as exc:
            self.status_var.set(f"Options error: {exc}")
            self.root.bell()
            return

        self._update_active_options_summary()
        self._close_options_window()

        if self.current_result is not None:
            self._reload_current_puzzle()
        else:
            self._refresh_solution_view(reset_status=False)
            self.status_var.set("Options applied")

    def _load_file_list(self, initial: bool = False) -> None:
        query = self.file_filter_var.get().strip().lower()
        path = Path(self.folder)
        if not path.is_dir():
            self.filtered_files = []
            self.file_listbox.delete(0, tk.END)
            self.status_var.set(f"Folder not found: {path}")
            return

        files = [
            entry.name
            for entry in path.iterdir()
            if entry.is_file() and entry.name.endswith(".txt") and FILE_PATTERN.match(entry.name)
        ]
        files.sort(key=lambda name: (int(name.split("_")[0].split("-")[0]), int(name.split("_")[0].split("-")[1]), name))
        if query:
            files = [name for name in files if query in name.lower()]

        self.filtered_files = files
        self.file_listbox.delete(0, tk.END)
        for name in files:
            self.file_listbox.insert(tk.END, name)

        if initial and files:
            default_name = "9-4_c8_5896.txt" if "9-4_c8_5896.txt" in files else files[0]
            index = files.index(default_name)
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index)
            self.file_listbox.activate(index)
            self.file_listbox.see(index)
            self.summary_var.set("Select a puzzle to load")
            self.status_var.set(f"Ready. {len(files)} puzzle files found. No puzzle loaded yet.")
            self._clear_solution_views(reset_status=False)
        elif not files:
            self.status_var.set("No matching puzzle files")
            self.summary_var.set("No puzzle files found")
            self._clear_solution_views(reset_status=False)

    def _select_file_by_name(self, filename: str) -> None:
        if filename not in self.filtered_files:
            return
        index = self.filtered_files.index(filename)
        self.file_listbox.selection_clear(0, tk.END)
        self.file_listbox.selection_set(index)
        self.file_listbox.activate(index)
        self.file_listbox.see(index)

    def _load_selected_from_list(self) -> None:
        selected = self.file_listbox.curselection()
        if not selected:
            self.status_var.set("Select a puzzle file first")
            return
        filename = self.filtered_files[selected[0]]
        self._schedule_load_puzzle_file(filename)

    def _on_puzzle_select(self, _event=None) -> None:
        selected = self.file_listbox.curselection()
        if not selected:
            return
        filename = self.filtered_files[selected[0]]
        self.summary_var.set(f"Selected: {filename}")
        self.status_var.set("Puzzle selected. Double-click it or use 'Load selected puzzle'.")

    def _schedule_load_puzzle_file(self, filename: str) -> None:
        if self._pending_load_after_id is not None:
            try:
                self.root.after_cancel(self._pending_load_after_id)
            except tk.TclError:
                pass
            self._pending_load_after_id = None
        self.status_var.set(f"Loading {filename}...")
        self.root.update_idletasks()
        self._pending_load_after_id = self.root.after(10, lambda: self._load_puzzle_file(filename))

    def _load_puzzle_file(self, filename: str) -> None:
        self._pending_load_after_id = None
        self._select_file_by_name(filename)
        full_path = Path(self.folder) / filename
        try:
            start_mouse_position = self._resolve_start_mouse_position()
            end_positions, end_next_puzzle = self._resolve_end_config()
            lock_threshold = self._resolve_lock_threshold()
            free_drag_min_disp = self._resolve_free_drag_min_disp()

            self.current_result = self.solver.score_file(
                full_path,
                dedupe=self.dedupe_var.get(),
                start_mouse_position=start_mouse_position,
                end_positions=end_positions,
                end_next_puzzle=end_next_puzzle,
                lock_threshold=lock_threshold,
                free_drag_min_displacement=free_drag_min_disp,
            )
        except Exception as exc:
            self.current_result = None
            self.status_var.set(f"Failed to load {filename}: {exc}")
            self.summary_var.set("Load failed")
            self._clear_solution_views(reset_status=False)
            return

        self.current_solution_index = 0
        self.current_step_index = None
        self._populate_solution_tree()
        self._set_solution_index(0)
        count = len(self.current_result.solutions) if self.current_result else 0
        self.status_var.set(f"Loaded {filename} with {count} ranked solution{'s' if count != 1 else ''}")

    def _reload_current_puzzle(self) -> None:
        selected = self.file_listbox.curselection()
        if not selected:
            return
        filename = self.filtered_files[selected[0]]
        current_move_string = None
        if self.current_result and self.current_result.solutions:
            current_move_string = self.current_result.solutions[self.current_solution_index].move_string
        self._load_puzzle_file(filename)
        if current_move_string and self.current_result:
            for idx, solution in enumerate(self.current_result.solutions):
                if solution.move_string == current_move_string:
                    self._set_solution_index(idx)
                    break

    def _clear_solution_views(self, reset_status: bool = True) -> None:
        self.step_views = []
        self.current_step_index = None
        for item in self.solution_tree.get_children():
            self.solution_tree.delete(item)
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)
        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete("1.0", tk.END)
        self.solution_text.config(state=tk.DISABLED)
        self.summary_var.set("No puzzle loaded")
        self.canvas.delete("all")
        self._draw_board_background()
        if reset_status:
            self.status_var.set("Ready")

    def _populate_solution_tree(self) -> None:
        for item in self.solution_tree.get_children():
            self.solution_tree.delete(item)
        if not self.current_result:
            return
        for idx, solution in enumerate(self.current_result.solutions):
            fat_count = sum(1 for move in solution.move_data if len(move.move) == 4)
            self.solution_tree.insert(
                "",
                tk.END,
                iid=f"sol_{idx}",
                values=(
                    idx + 1,
                    f"{solution.total_cost:.3f}",
                    f"{solution.total_move:.3f}",
                    f"{solution.total_drag:.3f}",
                    len(solution.move_data),
                    fat_count,
                ),
            )

    def _populate_steps_tree(self) -> None:
        for item in self.steps_tree.get_children():
            self.steps_tree.delete(item)
        for step in self.step_views:
            click = f"({step.click_down[0]:.2f},{step.click_down[1]:.2f})"
            lock = f"({step.lock_point[0]:.2f},{step.lock_point[1]:.2f})"
            release = f"({step.release[0]:.2f},{step.release[1]:.2f})"
            self.steps_tree.insert(
                "",
                tk.END,
                iid=f"step_{step.step_index}",
                values=(
                    step.step_index + 1,
                    step.token,
                    step.group_id,
                    click,
                    lock,
                    release,
                    f"{step.move_distance:.3f}",
                    f"{step.drag_distance:.3f}",
                ),
            )

    def _on_solution_select(self, _event=None) -> None:
        if self._suppress_solution_select:
            return
        selection = self.solution_tree.selection()
        if not selection:
            return
        iid = selection[0]
        index = int(iid.split("_")[1])
        if index == self.current_solution_index:
            return
        self._set_solution_index(index)

    def _on_step_select(self, _event=None) -> None:
        if self._suppress_step_select:
            return
        selection = self.steps_tree.selection()
        if not selection:
            self.current_step_index = None
            self._refresh_solution_view()
            return
        iid = selection[0]
        index = int(iid.split("_")[1])
        if index == self.current_step_index:
            return
        self.current_step_index = index
        self._refresh_solution_view()

    def _set_solution_index(self, index: int) -> None:
        if not self.current_result or not self.current_result.solutions:
            return

        index = max(0, min(index, len(self.current_result.solutions) - 1))
        self.current_solution_index = index
        self.current_step_index = None

        solution = self.current_result.solutions[index]
        self.step_views = derive_step_views(solution)

        self._suppress_step_select = True
        try:
            self._populate_steps_tree()
            if self.step_views:
                first_step_iid = f"step_{self.step_views[0].step_index}"
                if self.steps_tree.exists(first_step_iid):
                    self.steps_tree.selection_set(first_step_iid)
                    self.steps_tree.focus(first_step_iid)
                    self.steps_tree.see(first_step_iid)
                    self.current_step_index = self.step_views[0].step_index
            else:
                self.current_step_index = None
        finally:
            self._suppress_step_select = False

        self._refresh_solution_view()

        sol_iid = f"sol_{index}"
        self._suppress_solution_select = True
        try:
            if self.solution_tree.exists(sol_iid):
                self.solution_tree.selection_set(sol_iid)
                self.solution_tree.focus(sol_iid)
                self.solution_tree.see(sol_iid)
        finally:
            self._suppress_solution_select = False

    def _go_first_solution(self) -> None:
        self._set_solution_index(0)

    def _go_prev_solution(self) -> None:
        self._set_solution_index(self.current_solution_index - 1)

    def _go_next_solution(self) -> None:
        self._set_solution_index(self.current_solution_index + 1)

    def _go_last_solution(self) -> None:
        if self.current_result and self.current_result.solutions:
            self._set_solution_index(len(self.current_result.solutions) - 1)

    def _refresh_solution_view(self, reset_status: bool = True) -> None:
        self.canvas.delete("all")
        self._draw_board_background()
        if reset_status:
            self.status_var.set("Ready")

        if not self.current_result or not self.current_result.solutions:
            self.summary_var.set("No puzzle loaded")
            self.solution_text.config(state=tk.NORMAL)
            self.solution_text.delete("1.0", tk.END)
            self.solution_text.config(state=tk.DISABLED)
            return

        solution = self.current_result.solutions[self.current_solution_index]
        total_solutions = len(self.current_result.solutions)
        fat_count = sum(1 for step in self.step_views if step.is_fat)
        group_count = max((step.group_id for step in self.step_views), default=0)
        self.summary_var.set(
            f"{self.current_result.path.name}    "
            f"Solution {self.current_solution_index + 1}/{total_solutions}    "
            f"Total {solution.total_cost:.3f}    "
            f"Groups {group_count}    Fat {fat_count}    "
            f"Start {self._describe_start_option()}    "
            f"End {self._describe_end_option()}"
        )

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete("1.0", tk.END)
        self.solution_text.insert("1.0", solution.move_string)
        self.solution_text.config(state=tk.DISABLED)

        self._draw_mouse_travel()
        self._draw_drag_paths()
        self._draw_start_and_end_markers(solution)
        self._draw_header_metrics(solution)

        self.status_var.set(
            f"Showing {self.current_result.path.name} | solution {self.current_solution_index + 1}/{total_solutions} | "
            f"steps {len(solution.move_data)}"
        )

    def _draw_board_background(self) -> None:
        self.canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill="#eef1f5", outline="")

        left = BOARD_GRID_X * BOARD_CELL_SIZE
        top = BOARD_GRID_Y * BOARD_CELL_SIZE
        right = (BOARD_GRID_X + BOARD_GRID_W) * BOARD_CELL_SIZE
        bottom = (BOARD_GRID_Y + BOARD_GRID_H) * BOARD_CELL_SIZE

        self.canvas.create_rectangle(left - 20, top - 20, right + 210, bottom + 100, fill="#f8fafc", outline="")
        self.canvas.create_rectangle(left, top, right, bottom, fill="#ffffff", outline="#d0d7de", width=2)

        for y in range(BOARD_GRID_H):
            for x in range(BOARD_GRID_W):
                x0 = (BOARD_GRID_X + x) * BOARD_CELL_SIZE
                y0 = (BOARD_GRID_Y + y) * BOARD_CELL_SIZE
                x1 = x0 + BOARD_CELL_SIZE
                y1 = y0 + BOARD_CELL_SIZE
                fill = "#f3f4f6" if (x + y) % 2 else "#ffffff"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d6dbe3")

        for x in range(BOARD_GRID_W + 1):
            x_pos = (BOARD_GRID_X + x) * BOARD_CELL_SIZE
            self.canvas.create_line(x_pos, top, x_pos, bottom, fill="#252b34", width=3)
        for y in range(BOARD_GRID_H + 1):
            y_pos = (BOARD_GRID_Y + y) * BOARD_CELL_SIZE
            self.canvas.create_line(left, y_pos, right, y_pos, fill="#252b34", width=3)

        for x in range(BOARD_GRID_W):
            cx, cy = board_to_canvas(x, -0.78)
            self.canvas.create_text(cx, cy, text=str(x), font=("Segoe UI", 10, "bold"), fill="#57606a")
        for y in range(BOARD_GRID_H):
            cx, cy = board_to_canvas(-0.78, y)
            self.canvas.create_text(cx, cy, text=str(y), font=("Segoe UI", 10, "bold"), fill="#57606a")

        self.canvas.create_text(
            left,
            bottom + 22,
            anchor="w",
            text="Solid colored polylines = click -> lock -> release",
            font=("Segoe UI", 9),
            fill="#4b5563",
            )
        self.canvas.create_text(
            left,
            bottom + 38,
            anchor="w",
            text="Dashed gray lines = cursor reposition between drags",
            font=("Segoe UI", 9),
            fill="#4b5563",
            )

    def _draw_header_metrics(self, solution: ScoredSolution) -> None:
        left = (BOARD_GRID_X + BOARD_GRID_W) * BOARD_CELL_SIZE + 36
        top = BOARD_GRID_Y * BOARD_CELL_SIZE + 10
        box_w = 150
        box_h = 52

        initial_move = getattr(solution, "initial_move_distance", 0.0)
        inter_move = getattr(solution, "inter_move_distance", max(0.0, solution.total_move - getattr(solution, "final_move_distance", 0.0) - initial_move))
        final_move = getattr(solution, "final_move_distance", 0.0)

        metrics = [
            ("Total", f"{solution.total_cost:.3f}", "#111827"),
            ("Start", f"{initial_move:.3f}", "#6b7280"),
            ("Between", f"{inter_move:.3f}", "#6b7280"),
            ("Final", f"{final_move:.3f}", "#6b7280"),
            ("Drag", f"{solution.total_drag:.3f}", "#6b7280"),
        ]

        for i, (label, value, value_color) in enumerate(metrics):
            y = top + i * (box_h + 8)
            self.canvas.create_rectangle(left, y, left + box_w, y + box_h, fill="#ffffff", outline="#d0d7de", width=2)
            self.canvas.create_text(left + 12, y + 14, anchor="w", text=label, font=("Segoe UI", 9, "bold"), fill="#6b7280")
            self.canvas.create_text(left + 12, y + 34, anchor="w", text=value, font=("Segoe UI", 13, "bold"), fill=value_color)

    def _step_color(self, step: StepView, total_steps: int) -> str:
        mode = self.color_mode_var.get()
        if mode == "sequence":
            return color_for_step(step.step_index, total_steps)
        if mode == "axis":
            return color_for_axis(step.axis, step.is_fat)
        return color_for_group(step.group_id)

    def _draw_start_and_end_markers(self, solution: ScoredSolution) -> None:
        initial_mouse_position = getattr(solution, "initial_mouse_position", None)
        if self.step_views and initial_mouse_position is not None:
            start_x, start_y = board_to_canvas(*initial_mouse_position)
            first = self.step_views[0]
            first_x, first_y = board_to_canvas(*first.click_down)

            self.canvas.create_oval(
                start_x - 8, start_y - 8, start_x + 8, start_y + 8,
                fill="#fb923c", outline="#7c2d12", width=2
            )
            self.canvas.create_text(start_x, start_y - 15, text="START", font=("Segoe UI", 8, "bold"), fill="#7c2d12")

            if getattr(solution, "initial_move_distance", 0.0) > EPSILON:
                self.canvas.create_line(start_x, start_y, first_x, first_y, fill="#fb923c", width=2, dash=(6, 6))

        if self.end_preset_var.get() == END_PRESET_NEXT:
            for target in DEFAULT_NEXT_PUZZLE_TARGETS:
                tx, ty = board_to_canvas(*target)
                self.canvas.create_oval(tx - 5, ty - 5, tx + 5, ty + 5, outline="#16a34a", width=2)

        final_target = getattr(solution, "final_mouse_target", None)
        final_move_distance = getattr(solution, "final_move_distance", 0.0)
        if self.step_views and final_target is not None:
            last = self.step_views[-1]
            last_x, last_y = board_to_canvas(*last.release)
            end_x, end_y = board_to_canvas(*final_target)

            if final_move_distance > EPSILON:
                self.canvas.create_line(last_x, last_y, end_x, end_y, fill="#16a34a", width=2, dash=(6, 6))

            self.canvas.create_oval(end_x - 8, end_y - 8, end_x + 8, end_y + 8, fill="#22c55e", outline="#14532d", width=2)
            self.canvas.create_text(end_x, end_y - 15, text="END", font=("Segoe UI", 8, "bold"), fill="#14532d")

    def _draw_mouse_travel(self) -> None:
        if not self.show_travel_var.get():
            return
        for i in range(1, len(self.step_views)):
            prev_step = self.step_views[i - 1]
            curr_step = self.step_views[i]
            sx, sy = board_to_canvas(*prev_step.release)
            ex, ey = board_to_canvas(*curr_step.click_down)
            width = 3 if self.current_step_index in {i - 1, i} else 2
            fill = "#5b6472" if self.current_step_index in {i - 1, i} else "#a0a7b4"
            self.canvas.create_line(sx, sy, ex, ey, fill=fill, width=width, dash=(7, 7))
            if curr_step.move_distance > EPSILON:
                mx = (sx + ex) / 2
                my = (sy + ey) / 2
                self.canvas.create_text(mx, my - 10, text=f"{curr_step.move_distance:.2f}", font=("Segoe UI", 9, "bold"), fill="#6b7280")

    def _draw_drag_paths(self) -> None:
        grouped: dict[tuple[str, float, float, float], list[StepView]] = {}
        for step in self.step_views:
            grouped.setdefault(overlap_key(step), []).append(step)

        slot_lookup: dict[int, tuple[int, int]] = {}
        for group_steps in grouped.values():
            for slot, step in enumerate(sorted(group_steps, key=lambda s: s.step_index)):
                slot_lookup[step.step_index] = (slot, len(group_steps))

        total_steps = len(self.step_views)
        for step in self.step_views:
            points = [board_to_canvas(*pt) for pt in step.path_points]
            points = collapse_duplicate_points(points)
            points = offset_points(points, *slot_lookup.get(step.step_index, (0, 1)))

            color = self._step_color(step, total_steps)
            highlight = step.step_index == self.current_step_index
            width = 8 if highlight else 5
            outline = "#111827" if highlight else with_alpha_like(color, 0.45)

            flat_points = []
            for x, y in points:
                flat_points.extend([x, y])

            self.canvas.create_line(*flat_points, fill=outline, width=width + 3, arrow=tk.LAST, arrowshape=(16, 20, 7), smooth=False, joinstyle=tk.ROUND)
            self.canvas.create_line(*flat_points, fill=color, width=width, arrow=tk.LAST, arrowshape=(16, 20, 7), smooth=False, joinstyle=tk.ROUND)

            sx, sy = points[0]
            radius = 10 if highlight else 7
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill=color, outline="#111827", width=2)

            if step.free_drag and len(points) >= 2:
                lx, ly = points[1]
                self.canvas.create_oval(lx - 5, ly - 5, lx + 5, ly + 5, fill="#ffffff", outline=color, width=2)

            if self.show_numbers_var.get():
                tx = sx + 16
                ty = sy - 16
                text_fill = "#111827" if highlight else "#1f2937"
                self.canvas.create_oval(tx - 11, ty - 11, tx + 11, ty + 11, fill="#ffffff", outline=color, width=2)
                self.canvas.create_text(tx, ty, text=str(step.step_index + 1), font=("Segoe UI", 8, "bold"), fill=text_fill)

        if self.current_step_index is not None and 0 <= self.current_step_index < len(self.step_views):
            step = self.step_views[self.current_step_index]
            box_x = (BOARD_GRID_X + BOARD_GRID_W) * BOARD_CELL_SIZE + 36
            box_y = BOARD_GRID_Y * BOARD_CELL_SIZE + 320
            self.canvas.create_rectangle(box_x, box_y, box_x + 190, box_y + 145, fill="#ffffff", outline="#d0d7de", width=2)
            details = [
                f"Step: {step.step_index + 1}",
                f"Token: {step.token}",
                f"Group: {step.group_id}",
                f"Click: ({step.click_down[0]:.2f}, {step.click_down[1]:.2f})",
                f"Lock: ({step.lock_point[0]:.2f}, {step.lock_point[1]:.2f})",
                f"Release: ({step.release[0]:.2f}, {step.release[1]:.2f})",
                f"Free drag: {'yes' if step.free_drag else 'no'}",
                f"Gap: {step.move_distance:.3f}",
                f"Drag: {step.drag_distance:.3f}",
            ]
            for i, line in enumerate(details):
                self.canvas.create_text(box_x + 10, box_y + 12 + i * 15, anchor="w", text=line, font=("Segoe UI", 8), fill="#374151")


def main() -> None:
    app = StudioApp(DEFAULT_FOLDER)
    app.root.mainloop()


if __name__ == "__main__":
    main()
