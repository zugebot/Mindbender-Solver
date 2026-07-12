from __future__ import annotations
# pyth/viewer/chuzzle_mouse_presets.py

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

GRID_SIZE = 6
BOARD_CELL_SIZE = 96
BOARD_GRID_X = 1.5
BOARD_GRID_Y = 0.8
BOARD_GRID_W = GRID_SIZE
BOARD_GRID_H = GRID_SIZE
CANVAS_W = 900
CANVAS_H = 600
EPSILON = 1e-9

DEFAULT_FOLDER = "../../build/levels_final"
APP_TITLE = "Chuzzle DP Studio"

LEFT_X_MARKERS: tuple[tuple[str, tuple[float, float]], ...] = (
    ("1", (-3.18, 3.0475)),
    ("2", (-2.67, 3.4205)),
    ("3", (-2.11, 3.5125)),
    ("4", (-1.52, 3.4205)),
    ("5", (-0.98, 3.1125)),
)
NEXT_LEVEL_MARKER_POSITION: tuple[float, float] = (-1.22, 4.73)

LEFT_X_MARKER_RADIUS = BOARD_CELL_SIZE * (2.0 / 7.0)
NEXT_LEVEL_STAR_OUTER_RADIUS = BOARD_CELL_SIZE * 0.5
NEXT_LEVEL_STAR_INNER_RADIUS = NEXT_LEVEL_STAR_OUTER_RADIUS * 0.47
NEXT_LEVEL_STAR_ROTATION_CW_DEG = -14.0

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

DEFAULT_NEXT_PUZZLE_TARGETS: tuple[tuple[float, float], ...] = (
    (1.0, 3.0),
    (2.0, 3.0),
    (3.0, 3.0),
    (4.0, 3.0),
)


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


START_PRESETS: tuple[MousePreset, ...] = (
    SharedMousePresets.NONE.value,
    SharedMousePresets.LEVEL_1.value,
    SharedMousePresets.LEVEL_2.value,
    SharedMousePresets.LEVEL_3.value,
    SharedMousePresets.LEVEL_4.value,
    SharedMousePresets.LEVEL_5.value,
    SharedMousePresets.NEXT_WORLD_STAR.value,
    SharedMousePresets.CUSTOM.value,
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
    SharedMousePresets.CUSTOM.value,
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
    SharedMousePresets.NEXT_WORLD_STAR.value.title,
]
POST_END_CLICK_BY_TITLE: dict[str, tuple[float, float]] = {
    SharedMousePresets.LEVEL_1.value.title: SharedMousePresets.LEVEL_1.value.positions[0],
    SharedMousePresets.LEVEL_2.value.title: SharedMousePresets.LEVEL_2.value.positions[0],
    SharedMousePresets.LEVEL_3.value.title: SharedMousePresets.LEVEL_3.value.positions[0],
    SharedMousePresets.LEVEL_4.value.title: SharedMousePresets.LEVEL_4.value.positions[0],
    SharedMousePresets.LEVEL_5.value.title: SharedMousePresets.LEVEL_5.value.positions[0],
    SharedMousePresets.NEXT_WORLD_STAR.value.title: SharedMousePresets.NEXT_WORLD_STAR.value.positions[0],
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