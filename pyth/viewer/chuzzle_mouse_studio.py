from __future__ import annotations
# pyth/viewer/chuzzle_mouse_studio.py

import colorsys
import math
import re
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import ttk
from enum import Enum

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

LEFT_X_MARKERS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("1", (-3.18, 3.0475)),
    ("2", (-2.67, 3.4205)),
    ("3", (-2.11, 3.5125)),
    ("4", (-1.52, 3.4205)),
    ("5", (-0.98, 3.1125)),
)
NEXT_LEVEL_MARKER_POSITION: tuple[float, float] = (-1.22, 4.73)

# X-N marker diameter is 4/7 of one board cell.
LEFT_X_MARKER_RADIUS = BOARD_CELL_SIZE * (2.0 / 7.0)
# Star is approximately one board-cell tall.
NEXT_LEVEL_STAR_OUTER_RADIUS = BOARD_CELL_SIZE * 0.5
NEXT_LEVEL_STAR_INNER_RADIUS = NEXT_LEVEL_STAR_OUTER_RADIUS * 0.47
# Opposite direction and 2x magnitude from the original +7 clockwise.
NEXT_LEVEL_STAR_ROTATION_CW_DEG = -14.0

# Top-left fat start positions by level key, sourced from `levels.hpp`.
FAT_START_BY_LEVEL: dict[str, tuple[float, float]] = {
    "4-2": (4.0, 4.0),
    "4-4": (2.0, 2.0),
    "5-1": (0.0, 2.0),
    "6-1": (4.0, 4.0),
    "6-2": (1.0, 3.0),
    "6-3": (0.0, 1.0),
    "6-4": (3.0, 3.0),
    "6-5": (3.0, 0.0),
    "8-2": (4.0, 4.0),
    "8-4": (3.0, 4.0),
    "9-1": (1.0, 3.0),
    "12-2": (1.0, 3.0),
    "13-4": (1.0, 4.0),
    "13-5": (2.0, 4.0),
    "15-2": (3.0, 4.0),
    "15-3": (1.0, 0.0),
    "15-4": (1.0, 4.0),
    "16-1": (3.0, 4.0),
    "16-5": (2.0, 0.0),
    "17-2": (4.0, 4.0),
    "17-4": (2.0, 1.0),
    "18-1": (1.0, 4.0),
    "18-2": (0.0, 0.0),
    "18-4": (1.0, 2.0),
    "18-5": (4.0, 3.0),
    "19-2": (0.0, 3.0),
    "19-4": (1.0, 4.0),
    "20-3": (2.0, 2.0),
}


def star_polygon_points(
    center_x: float,
    center_y: float,
    outer_radius: float,
    inner_radius: float,
    rotation_cw_deg: float,
    points: int = 5,
) -> list[float]:
    coords: list[float] = []
    step = math.pi / points
    start = math.radians(-90.0 - rotation_cw_deg)
    for i in range(points * 2):
        radius = outer_radius if i % 2 == 0 else inner_radius
        angle = start + i * step
        coords.extend((center_x + radius * math.cos(angle), center_y + radius * math.sin(angle)))
    return coords

@dataclass(frozen=True)
class MousePreset:
    title: str
    positions: tuple[tuple[float, float], ...] | None = None
    next_puzzle_banner: bool = False
    allow_custom_positions: bool = False
    description: str = "none"
    
class SharedMousePresets(Enum):
    NONE = MousePreset(title="None", positions=None, description="none")
    NEXT_WORLD_STAR = MousePreset(title="Next World Star", positions=((-1.22, 4.73),), description="next world star")
    LEVEL_1 = MousePreset(title="Level 1", positions=((-3.18, 3.0475),), description="level 1")
    LEVEL_2 = MousePreset(title="Level 2", positions=((-2.67, 3.4205),), description="level 2")
    LEVEL_3 = MousePreset(title="Level 3", positions=((-2.11, 3.5125),), description="level 3")
    LEVEL_4 = MousePreset(title="Level 4", positions=((-1.52, 3.4205),), description="level 4")
    LEVEL_5 = MousePreset(title="Level 5", positions=((-0.98, 3.1125),), description="level 5")
    CUSTOM = MousePreset(title="Custom", allow_custom_positions=True, description="custom")

# Add/edit presets here. The UI and resolver logic consume these lists directly.
START_PRESETS: tuple[MousePreset, ...] = (
    SharedMousePresets.NONE.value,
    SharedMousePresets.LEVEL_1.value,
    SharedMousePresets.LEVEL_2.value,
    SharedMousePresets.LEVEL_3.value,
    SharedMousePresets.LEVEL_4.value,
    SharedMousePresets.LEVEL_5.value,
    SharedMousePresets.NEXT_WORLD_STAR.value,
    SharedMousePresets.CUSTOM.value
)

END_PRESETS: tuple[MousePreset, ...] = (
    SharedMousePresets.NONE.value,
    MousePreset(title="Next Puzzle Banner", positions=None, next_puzzle_banner=True, description="next puzzle banner"),
    SharedMousePresets.LEVEL_1.value,
    SharedMousePresets.LEVEL_2.value,
    SharedMousePresets.LEVEL_3.value,
    SharedMousePresets.LEVEL_4.value,
    SharedMousePresets.LEVEL_5.value,
    SharedMousePresets.NEXT_WORLD_STAR.value,
    SharedMousePresets.CUSTOM.value
)

START_PRESETS_BY_TITLE = {preset.title: preset for preset in START_PRESETS}
END_PRESETS_BY_TITLE = {preset.title: preset for preset in END_PRESETS}
START_PRESET_TITLES = [preset.title for preset in START_PRESETS]
END_PRESET_TITLES = [preset.title for preset in END_PRESETS]
DEFAULT_START_PRESET_TITLE = START_PRESETS[0].title
DEFAULT_END_PRESET_TITLE = END_PRESETS[0].title

POST_END_CLICK_NONE = "None"
POST_END_CLICK_TARGETS = [
    POST_END_CLICK_NONE,
    SharedMousePresets.LEVEL_1.value.title,
    SharedMousePresets.LEVEL_2.value.title,
    SharedMousePresets.LEVEL_3.value.title,
    SharedMousePresets.LEVEL_4.value.title,
    SharedMousePresets.LEVEL_5.value.title,
]
POST_END_CLICK_BY_TITLE: dict[str, tuple[float, float]] = {
    SharedMousePresets.LEVEL_1.value.title: SharedMousePresets.LEVEL_1.value.positions[0],
    SharedMousePresets.LEVEL_2.value.title: SharedMousePresets.LEVEL_2.value.positions[0],
    SharedMousePresets.LEVEL_3.value.title: SharedMousePresets.LEVEL_3.value.positions[0],
    SharedMousePresets.LEVEL_4.value.title: SharedMousePresets.LEVEL_4.value.positions[0],
    SharedMousePresets.LEVEL_5.value.title: SharedMousePresets.LEVEL_5.value.positions[0],
}


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

def offset_points(
    points: list[tuple[float, float]],
    slot: int,
    total_slots: int,
    max_offset: float = 12.0,
) -> list[tuple[float, float]]:
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
        self.metrics_var = tk.StringVar(value="")
        self.step_details_var = tk.StringVar(value="")
        self.grid_cell_size_var = tk.StringVar(value=f"{BOARD_CELL_SIZE:g}")
        self._canvas_zoom_scale = 1.0
        self._initial_canvas_view_set = False
        self.post_end_click_enabled_var = tk.BooleanVar(value=False)
        self.post_end_click_target_var = tk.StringVar(value=POST_END_CLICK_NONE)

        self.start_preset_var = tk.StringVar(value=DEFAULT_START_PRESET_TITLE)
        self.start_custom_var = tk.StringVar(value="")
        self.end_preset_var = tk.StringVar(value=DEFAULT_END_PRESET_TITLE)
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
        self.canvas.configure(scrollregion=(-500, -300, CANVAS_W + 500, CANVAS_H + 300))
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_canvas_pan_drag)
        self.canvas.bind("<MouseWheel>", self._on_canvas_mouse_wheel)

        self.metrics_row = tk.Label(
            parent,
            textvariable=self.metrics_var,
            anchor="w",
            font=("Consolas", 10),
            relief=tk.FLAT,
            bg="#f7f7f7",
            padx=4,
            pady=2,
        )
        self.metrics_row.pack(fill=tk.X, pady=(2, 0))

        self.step_details_row = tk.Label(
            parent,
            textvariable=self.step_details_var,
            anchor="w",
            font=("Consolas", 9),
            relief=tk.FLAT,
            bg="#f7f7f7",
            padx=4,
            pady=2,
        )
        self.step_details_row.pack(fill=tk.X, pady=(1, 0))

    def _on_canvas_pan_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def _on_canvas_pan_drag(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_canvas_mouse_wheel(self, event: tk.Event) -> None:
        base_size = self._resolve_grid_cell_size(allow_empty_fallback=True)
        factor = 1.08 if event.delta > 0 else (1.0 / 1.08)
        new_size = max(36.0, min(220.0, base_size * factor))
        self.grid_cell_size_var.set(f"{new_size:.3g}")
        self._canvas_zoom_scale = new_size / BOARD_CELL_SIZE
        self._refresh_solution_view(reset_status=False)

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

    def _selected_start_preset(self) -> MousePreset:
        return START_PRESETS_BY_TITLE.get(self.start_preset_var.get(), START_PRESETS[0])

    def _selected_end_preset(self) -> MousePreset:
        return END_PRESETS_BY_TITLE.get(self.end_preset_var.get(), END_PRESETS[0])

    def _resolve_start_mouse_position(self) -> tuple[float, float] | None:
        preset = self._selected_start_preset()
        if preset.allow_custom_positions:
            return parse_point_text(self.start_custom_var.get())
        if not preset.positions:
            return None
        return preset.positions[0]

    def _resolve_custom_end_positions(self) -> list[tuple[float, float]]:
        points = parse_point_list_text(self.end_custom_var.get())
        if not points:
            raise ValueError("Custom end preset needs at least one point")
        return points

    def _resolve_end_config(self) -> tuple[list[tuple[float, float]] | None, bool]:
        preset = self._selected_end_preset()
        if preset.allow_custom_positions:
            return self._resolve_custom_end_positions(), False
        if preset.next_puzzle_banner:
            return None, True
        if not preset.positions:
            return None, False
        return list(preset.positions), False

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

    def _resolve_grid_cell_size(self, allow_empty_fallback: bool = False) -> float:
        text = self.grid_cell_size_var.get().strip()
        if not text and allow_empty_fallback:
            return BOARD_CELL_SIZE * self._canvas_zoom_scale
        value = float(text)
        if value <= 0.0:
            raise ValueError("Grid-cell size must be > 0")
        return value

    def _reset_grid_cell_size(self) -> None:
        self.grid_cell_size_var.set(f"{BOARD_CELL_SIZE:g}")

    def _resolve_level_start_fat(self, filename: str) -> tuple[bool, tuple[float, float] | None]:
        match = FILE_PATTERN.match(filename)
        if not match:
            return False, None
        level_key = f"{match.group(1)}-{match.group(2)}"
        fat_pos = FAT_START_BY_LEVEL.get(level_key)
        if fat_pos is None:
            return False, None
        return True, fat_pos

    def _resolve_post_end_click_target(self) -> tuple[float, float] | None:
        if not self.post_end_click_enabled_var.get():
            return None
        return POST_END_CLICK_BY_TITLE.get(self.post_end_click_target_var.get())

    def _describe_post_end_click_option(self) -> str:
        if not self.post_end_click_enabled_var.get():
            return "off"
        target = self.post_end_click_target_var.get()
        if target == POST_END_CLICK_NONE:
            return "on, no target"
        return f"on -> {target}"

    def _describe_start_option(self) -> str:
        preset = self._selected_start_preset()
        if preset.allow_custom_positions:
            try:
                return format_point(parse_point_text(self.start_custom_var.get()))
            except Exception:
                return "custom ?"
        return preset.description

    def _describe_end_option(self) -> str:
        preset = self._selected_end_preset()
        if preset.allow_custom_positions:
            try:
                points = self._resolve_custom_end_positions()
                return "; ".join(format_point(p) for p in points)
            except Exception:
                return "custom ?"
        return preset.description

    def _update_active_options_summary(self) -> None:
        text = (
            f"Start: {self._describe_start_option()}\n"
            f"End: {self._describe_end_option()}\n"
            f"Post-end click: {self._describe_post_end_click_option()}\n"
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
            values=START_PRESET_TITLES,
            width=24,
        ).grid(row=2, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="Start custom point").grid(row=1, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.start_custom_var, width=24).grid(row=2, column=1, sticky="we")

        ttk.Label(solver_box, text="End preset").grid(row=3, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            solver_box,
            textvariable=self.end_preset_var,
            state="readonly",
            values=END_PRESET_TITLES,
            width=24,
        ).grid(row=4, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="End custom points").grid(row=3, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.end_custom_var, width=24).grid(row=4, column=1, sticky="we")

        ttk.Label(solver_box, text="Lock threshold").grid(row=5, column=0, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.lock_threshold_var, width=24).grid(row=6, column=0, sticky="we", padx=(0, 8))

        ttk.Label(solver_box, text="Free-drag minimum |displacement|").grid(row=5, column=1, sticky="w", pady=(10, 2))
        ttk.Entry(solver_box, textvariable=self.free_drag_min_disp_var, width=24).grid(row=6, column=1, sticky="we")

        ttk.Label(solver_box, text="Grid-cell size").grid(row=7, column=0, sticky="w", pady=(10, 2))
        size_row = ttk.Frame(solver_box)
        size_row.grid(row=8, column=0, sticky="w")
        ttk.Entry(size_row, textvariable=self.grid_cell_size_var, width=10).pack(side=tk.LEFT)
        ttk.Button(size_row, text="Reset", command=self._reset_grid_cell_size).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            solver_box,
            text=(
                "Custom point format: x,y\n"
                "Multiple end points: x,y; x,y; x,y\n"
                "Adapter routes to the active solver backend."
            ),
            justify=tk.LEFT,
        ).grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))

        view_box = ttk.LabelFrame(outer, text="View options", padding=10)
        view_box.pack(fill=tk.X, pady=(12, 0))
        ttk.Checkbutton(view_box, text="Show move numbers on board", variable=self.show_numbers_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(view_box, text="Show cursor travel gaps", variable=self.show_travel_var).grid(row=1, column=0, sticky="w")

        ttk.Checkbutton(
            view_box,
            text="Show post-end click arrow",
            variable=self.post_end_click_enabled_var,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        ttk.Label(view_box, text="Post-end target").grid(row=3, column=0, sticky="w", pady=(8, 2))
        ttk.Combobox(
            view_box,
            textvariable=self.post_end_click_target_var,
            state="readonly",
            values=POST_END_CLICK_TARGETS,
            width=18,
        ).grid(row=4, column=0, sticky="w")

        ttk.Label(view_box, text="Color strategy").grid(row=5, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            view_box,
            textvariable=self.color_mode_var,
            state="readonly",
            values=["chain", "sequence", "axis"],
            width=18,
        ).grid(row=6, column=0, sticky="w")

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
            grid_cell_size = self._resolve_grid_cell_size()
        except Exception as exc:
            self.status_var.set(f"Options error: {exc}")
            self.root.bell()
            return

        self._canvas_zoom_scale = grid_cell_size / BOARD_CELL_SIZE

        self._update_active_options_summary()

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
            has_fat, initial_fat_position = self._resolve_level_start_fat(filename)

            self.current_result = self.solver.score_file(
                full_path,
                dedupe=self.dedupe_var.get(),
                start_mouse_position=start_mouse_position,
                has_fat=has_fat,
                initial_fat_position=initial_fat_position,
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
        self.metrics_var.set("")
        self.step_details_var.set("")
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
            self.metrics_var.set("")
            self.step_details_var.set("")
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
            f"End {self._describe_end_option()}    "
            f"Post-end {self._describe_post_end_click_option()}"
        )

        self.solution_text.config(state=tk.NORMAL)
        self.solution_text.delete("1.0", tk.END)
        self.solution_text.insert("1.0", solution.move_string)
        self.solution_text.config(state=tk.DISABLED)

        initial_move = getattr(solution, "initial_move_distance", 0.0)
        inter_move = getattr(
            solution,
            "inter_move_distance",
            max(0.0, solution.total_move - getattr(solution, "final_move_distance", 0.0) - initial_move),
        )
        final_move = getattr(solution, "final_move_distance", 0.0)
        self.metrics_var.set(
            " | ".join(
                [
                    f"Total {solution.total_cost:.3f}",
                    f"Start {initial_move:.3f}",
                    f"Between {inter_move:.3f}",
                    f"Final {final_move:.3f}",
                    f"Drag {solution.total_drag:.3f}",
                ]
            )
        )

        self._update_step_details_line()

        self._draw_mouse_travel()
        self._draw_drag_paths()
        self._draw_start_and_end_markers(solution)

        self.status_var.set(
            f"Showing {self.current_result.path.name} | solution {self.current_solution_index + 1}/{total_solutions} | "
            f"steps {len(solution.move_data)}"
        )

    def _update_step_details_line(self) -> None:
        if self.current_step_index is None:
            self.step_details_var.set("")
            return
        if not (0 <= self.current_step_index < len(self.step_views)):
            self.step_details_var.set("")
            return

        step = self.step_views[self.current_step_index]
        self.step_details_var.set(
            " | ".join(
                [
                    f"Step {step.step_index + 1}",
                    f"Token {step.token}",
                    f"Group {step.group_id}",
                    f"Click ({step.click_down[0]:.2f}, {step.click_down[1]:.2f})",
                    f"Lock ({step.lock_point[0]:.2f}, {step.lock_point[1]:.2f})",
                    f"Release ({step.release[0]:.2f}, {step.release[1]:.2f})",
                    f"Free Drag {'yes' if step.free_drag else 'no'}",
                    f"Gap {step.move_distance:.3f}",
                    f"Drag {step.drag_distance:.3f}",
                ]
            )
        )

    def _cell_size(self) -> float:
        return BOARD_CELL_SIZE * self._canvas_zoom_scale

    def _board_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        cell = self._cell_size()
        return (
            (BOARD_GRID_X + x + 0.5) * cell,
            (BOARD_GRID_Y + y + 0.5) * cell,
        )

    def _draw_board_background(self) -> None:
        self.canvas.create_rectangle(0, 0, CANVAS_W, CANVAS_H, fill="#eef1f5", outline="")

        cell = self._cell_size()
        left = BOARD_GRID_X * cell
        top = BOARD_GRID_Y * cell
        right = (BOARD_GRID_X + BOARD_GRID_W) * cell
        bottom = (BOARD_GRID_Y + BOARD_GRID_H) * cell

        self.canvas.create_rectangle(left - 20, top - 20, right + 210, bottom + 100, fill="#f8fafc", outline="")
        self.canvas.create_rectangle(left, top, right, bottom, fill="#ffffff", outline="#d0d7de", width=2)

        for y in range(BOARD_GRID_H):
            for x in range(BOARD_GRID_W):
                x0 = (BOARD_GRID_X + x) * cell
                y0 = (BOARD_GRID_Y + y) * cell
                x1 = x0 + cell
                y1 = y0 + cell
                fill = "#f3f4f6" if (x + y) % 2 else "#ffffff"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#d6dbe3")

        for x in range(BOARD_GRID_W + 1):
            x_pos = (BOARD_GRID_X + x) * cell
            self.canvas.create_line(x_pos, top, x_pos, bottom, fill="#252b34", width=3)
        for y in range(BOARD_GRID_H + 1):
            y_pos = (BOARD_GRID_Y + y) * cell
            self.canvas.create_line(left, y_pos, right, y_pos, fill="#252b34", width=3)

        for x in range(BOARD_GRID_W):
            cx, cy = self._board_to_canvas(x, -0.78)
            self.canvas.create_text(cx, cy, text=str(x), font=("Segoe UI", 10, "bold"), fill="#57606a")
        for y in range(BOARD_GRID_H):
            cx, cy = self._board_to_canvas(-0.78, y)
            self.canvas.create_text(cx, cy, text=str(y), font=("Segoe UI", 10, "bold"), fill="#57606a")

        self._draw_left_reference_markers()

        bbox = self.canvas.bbox("all")
        if bbox is not None:
            pad = 80
            self.canvas.configure(scrollregion=(bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad))
            self._ensure_initial_canvas_view()


    def _ensure_initial_canvas_view(self) -> None:
        if self._initial_canvas_view_set:
            return
        width = self.canvas.winfo_width()
        if width <= 1:
            self.root.after(20, self._ensure_initial_canvas_view)
            return

        bbox = self.canvas.bbox("all")
        if bbox is None:
            return

        leftmost_cx = min(self._board_to_canvas(*marker_pos)[0] for _marker_label, marker_pos in LEFT_X_MARKERS)
        x1, _y1, x2, _y2 = bbox
        span = max(1.0, x2 - x1)
        target_left = leftmost_cx - width / 2.0
        frac = (target_left - x1) / span
        frac = max(0.0, min(1.0, frac))
        self.canvas.xview_moveto(frac)
        self._initial_canvas_view_set = True

    def _current_world_plus_one_label(self) -> str | None:
        if self.current_result is None:
            return None
        name = self.current_result.path.name
        match = FILE_PATTERN.match(name)
        if not match:
            return None
        return str(int(match.group(1)) + 1)

    def _draw_left_reference_markers(self) -> None:
        zoom = self._canvas_zoom_scale
        for marker_label, marker_pos in LEFT_X_MARKERS:
            cx, cy = self._board_to_canvas(*marker_pos)
            r = LEFT_X_MARKER_RADIUS * zoom
            self.canvas.create_oval(
                cx - r,
                cy - r,
                cx + r,
                cy + r,
                fill="#ffffff",
                outline="#2563eb",
                width=2,
            )
            self.canvas.create_text(
                cx,
                cy,
                text=marker_label,
                font=("Segoe UI", 10, "bold"),
                fill="#1d4ed8",
            )

        sx, sy = self._board_to_canvas(*NEXT_LEVEL_MARKER_POSITION)
        star_points = star_polygon_points(
            sx,
            sy,
            outer_radius=NEXT_LEVEL_STAR_OUTER_RADIUS * zoom,
            inner_radius=NEXT_LEVEL_STAR_INNER_RADIUS * zoom,
            rotation_cw_deg=NEXT_LEVEL_STAR_ROTATION_CW_DEG,
        )
        outline = "#a16207"
        self.canvas.create_polygon(star_points, fill="#facc15", outline=outline, width=2)

        star_label = self._current_world_plus_one_label()
        if star_label is not None:
            self.canvas.create_text(
                sx,
                sy,
                text=star_label,
                font=("Segoe UI", 11, "bold"),
                fill=outline,
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
        zoom = self._canvas_zoom_scale
        initial_mouse_position = getattr(solution, "initial_mouse_position", None)
        if self.step_views and initial_mouse_position is not None:
            start_x, start_y = self._board_to_canvas(*initial_mouse_position)
            first = self.step_views[0]
            first_x, first_y = self._board_to_canvas(*first.click_down)

            self.canvas.create_oval(
                start_x - 8 * zoom, start_y - 8 * zoom, start_x + 8 * zoom, start_y + 8 * zoom,
                fill="#fb923c", outline="#7c2d12", width=2
            )
            self.canvas.create_text(start_x, start_y - 15, text="START", font=("Segoe UI", 8, "bold"), fill="#7c2d12")

            if getattr(solution, "initial_move_distance", 0.0) > EPSILON:
                self.canvas.create_line(start_x, start_y, first_x, first_y, fill="#fb923c", width=2, dash=(6, 6))

        if self._selected_end_preset().next_puzzle_banner:
            for target in DEFAULT_NEXT_PUZZLE_TARGETS:
                tx, ty = self._board_to_canvas(*target)
                self.canvas.create_oval(tx - 5 * zoom, ty - 5 * zoom, tx + 5 * zoom, ty + 5 * zoom, outline="#16a34a", width=2)

        final_target = getattr(solution, "final_mouse_target", None)
        final_move_distance = getattr(solution, "final_move_distance", 0.0)
        if self.step_views and final_target is not None:
            last = self.step_views[-1]
            last_x, last_y = self._board_to_canvas(*last.release)
            end_x, end_y = self._board_to_canvas(*final_target)

            if final_move_distance > EPSILON:
                self.canvas.create_line(last_x, last_y, end_x, end_y, fill="#16a34a", width=2, dash=(6, 6))

            self.canvas.create_oval(
                end_x - 8 * zoom,
                end_y - 8 * zoom,
                end_x + 8 * zoom,
                end_y + 8 * zoom,
                fill="#22c55e",
                outline="#14532d",
                width=2,
            )
            self.canvas.create_text(end_x, end_y - 15, text="END", font=("Segoe UI", 8, "bold"), fill="#14532d")

            post_end_target = self._resolve_post_end_click_target()
            if post_end_target is not None:
                post_x, post_y = self._board_to_canvas(*post_end_target)
                arrowshape = (14 * zoom, 18 * zoom, 6 * zoom)
                self.canvas.create_line(
                    end_x,
                    end_y,
                    post_x,
                    post_y,
                    fill="#7c3aed",
                    width=2 * zoom,
                    dash=(6, 6),
                    arrow=tk.LAST,
                    arrowshape=arrowshape,
                )

    def _draw_mouse_travel(self) -> None:
        if not self.show_travel_var.get():
            return
        zoom = self._canvas_zoom_scale
        for i in range(1, len(self.step_views)):
            prev_step = self.step_views[i - 1]
            curr_step = self.step_views[i]
            sx, sy = self._board_to_canvas(*prev_step.release)
            ex, ey = self._board_to_canvas(*curr_step.click_down)
            width = (3 if self.current_step_index in {i - 1, i} else 2) * zoom
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
        zoom = self._canvas_zoom_scale
        used_number_positions: list[tuple[float, float]] = []
        for step in self.step_views:
            points = [self._board_to_canvas(*pt) for pt in step.path_points]
            points = collapse_duplicate_points(points)
            points = offset_points(points, *slot_lookup.get(step.step_index, (0, 1)), max_offset=12.0 * zoom)

            color = self._step_color(step, total_steps)
            highlight = step.step_index == self.current_step_index
            width = (8 if highlight else 5) * zoom
            outline = "#111827" if highlight else with_alpha_like(color, 0.45)
            arrowshape = (16 * zoom, 20 * zoom, 7 * zoom)

            flat_points = []
            for x, y in points:
                flat_points.extend([x, y])

            self.canvas.create_line(*flat_points, fill=outline, width=width + 3 * zoom, arrow=tk.LAST, arrowshape=arrowshape, smooth=False, joinstyle=tk.ROUND)
            self.canvas.create_line(*flat_points, fill=color, width=width, arrow=tk.LAST, arrowshape=arrowshape, smooth=False, joinstyle=tk.ROUND)

            sx, sy = points[0]
            radius = (10 if highlight else 7) * zoom
            self.canvas.create_oval(sx - radius, sy - radius, sx + radius, sy + radius, fill=color, outline="#111827", width=2)

            if step.free_drag and len(points) >= 2:
                lx, ly = points[1]
                r2 = 5 * zoom
                self.canvas.create_oval(lx - r2, ly - r2, lx + r2, ly + r2, fill="#ffffff", outline=color, width=2)

            if self.show_numbers_var.get():
                tx, ty = self._choose_move_number_position(sx, sy, used_number_positions)
                used_number_positions.append((tx, ty))
                text_fill = "#111827" if highlight else "#1f2937"
                nr = 11 * zoom
                self.canvas.create_oval(tx - nr, ty - nr, tx + nr, ty + nr, fill="#ffffff", outline=color, width=2)
                self.canvas.create_text(tx, ty, text=str(step.step_index + 1), font=("Segoe UI", 8, "bold"), fill=text_fill)


    def _choose_move_number_position(
        self,
        sx: float,
        sy: float,
        used_positions: list[tuple[float, float]],
    ) -> tuple[float, float]:
        zoom = self._canvas_zoom_scale
        candidates = [
            (16.0, -16.0),
            (16.0, 16.0),
            (-16.0, -16.0),
            (-16.0, 16.0),
            (0.0, -24.0),
            (24.0, 0.0),
            (0.0, 24.0),
            (-24.0, 0.0),
            (28.0, -20.0),
            (28.0, 20.0),
            (-28.0, -20.0),
            (-28.0, 20.0),
        ]

        min_gap_sq = (17.0 * zoom) * (17.0 * zoom)
        for dx, dy in candidates:
            tx = sx + dx * zoom
            ty = sy + dy * zoom
            if all((tx - ux) * (tx - ux) + (ty - uy) * (ty - uy) >= min_gap_sq for ux, uy in used_positions):
                return tx, ty

        fallback_index = len(used_positions)
        angle = fallback_index * 0.9
        radius = (30.0 + 5.0 * min(8, fallback_index)) * zoom
        return sx + radius * math.cos(angle), sy + radius * math.sin(angle)


def main() -> None:
    app = StudioApp(DEFAULT_FOLDER)
    app.root.mainloop()


if __name__ == "__main__":
    main()
