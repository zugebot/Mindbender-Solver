from __future__ import annotations
# pyth/viewer/chuzzle_mouse_domain.py

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ScoredMove:
    move: str
    click_down: list[float]
    lock_point: list[float]
    release: list[float]
    path_points: list[list[float]]
    selected_line: int
    displacement: int
    free_drag: bool
    drag_distance: float
    move_distance: float
    total_distance_to_here: float


@dataclass(slots=True)
class ScoredSolution:
    move_string: str
    total_drag: float
    total_move: float
    total_cost: float
    move_data: list[ScoredMove]
    initial_mouse_position: tuple[float, float] | None = None
    initial_move_distance: float = 0.0
    inter_move_distance: float = 0.0
    final_mouse_target: tuple[float, float] | None = None
    final_move_distance: float = 0.0


@dataclass(slots=True)
class FileScoreResult:
    path: Path
    solutions: list[ScoredSolution]


@dataclass(slots=True)
class RouteOverlayMove:
    move: str
    click_down: tuple[float, float]
    lock_point: tuple[float, float]
    release: tuple[float, float]
    path_points: tuple[tuple[float, float], ...]
    selected_line: int
    displacement: int
    free_drag: bool
    drag_distance: float
    move_distance: float
    total_distance_to_here: float


@dataclass(slots=True)
class RouteStep:
    step_id: str
    world_number: int
    puzzle_number: int
    puzzle_id: str
    source_file: str
    move_string: str
    total_cost: float
    total_move: float
    total_drag: float

    start_mouse_position: tuple[float, float] | None = None
    end_positions: tuple[tuple[float, float], ...] = ()
    end_next_puzzle: bool = False
    post_end_click_target: tuple[float, float] | None = None

    has_fat: bool = False
    initial_fat_position: tuple[float, float] | None = None
    notes: str = ""

    initial_move_distance: float = 0.0
    inter_move_distance: float = 0.0
    final_mouse_target: tuple[float, float] | None = None
    final_move_distance: float = 0.0

    overlay_moves: list[RouteOverlayMove] = field(default_factory=list)


@dataclass(slots=True)
class RouteProfile:
    name: str
    description: str = ""
    category: str = "custom"
    steps: list[RouteStep] = field(default_factory=list)


@dataclass(slots=True)
class CachedScoreFile:
    source_file: str
    source_hash: str
    options_hash: str
    result: FileScoreResult


def scored_move_to_dict(move: ScoredMove) -> dict[str, Any]:
    return asdict(move)


def scored_move_from_dict(data: dict[str, Any]) -> ScoredMove:
    return ScoredMove(
        move=str(data["move"]),
        click_down=[float(v) for v in data["click_down"]],
        lock_point=[float(v) for v in data["lock_point"]],
        release=[float(v) for v in data["release"]],
        path_points=[[float(p[0]), float(p[1])] for p in data["path_points"]],
        selected_line=int(data["selected_line"]),
        displacement=int(data["displacement"]),
        free_drag=bool(data["free_drag"]),
        drag_distance=float(data["drag_distance"]),
        move_distance=float(data["move_distance"]),
        total_distance_to_here=float(data["total_distance_to_here"]),
    )


def scored_solution_to_dict(solution: ScoredSolution) -> dict[str, Any]:
    return {
        "move_string": solution.move_string,
        "total_drag": solution.total_drag,
        "total_move": solution.total_move,
        "total_cost": solution.total_cost,
        "move_data": [scored_move_to_dict(move) for move in solution.move_data],
        "initial_mouse_position": list(solution.initial_mouse_position) if solution.initial_mouse_position is not None else None,
        "initial_move_distance": solution.initial_move_distance,
        "inter_move_distance": solution.inter_move_distance,
        "final_mouse_target": list(solution.final_mouse_target) if solution.final_mouse_target is not None else None,
        "final_move_distance": solution.final_move_distance,
    }


def scored_solution_from_dict(data: dict[str, Any]) -> ScoredSolution:
    initial_mouse_position = None
    if data.get("initial_mouse_position") is not None:
        initial_mouse_position = (
            float(data["initial_mouse_position"][0]),
            float(data["initial_mouse_position"][1]),
        )

    final_mouse_target = None
    if data.get("final_mouse_target") is not None:
        final_mouse_target = (
            float(data["final_mouse_target"][0]),
            float(data["final_mouse_target"][1]),
        )

    return ScoredSolution(
        move_string=str(data["move_string"]),
        total_drag=float(data["total_drag"]),
        total_move=float(data["total_move"]),
        total_cost=float(data["total_cost"]),
        move_data=[scored_move_from_dict(item) for item in data["move_data"]],
        initial_mouse_position=initial_mouse_position,
        initial_move_distance=float(data.get("initial_move_distance", 0.0)),
        inter_move_distance=float(data.get("inter_move_distance", 0.0)),
        final_mouse_target=final_mouse_target,
        final_move_distance=float(data.get("final_move_distance", 0.0)),
    )


def file_score_result_to_dict(result: FileScoreResult) -> dict[str, Any]:
    return {
        "path": str(result.path),
        "solutions": [scored_solution_to_dict(solution) for solution in result.solutions],
    }


def file_score_result_from_dict(data: dict[str, Any]) -> FileScoreResult:
    return FileScoreResult(
        path=Path(data["path"]),
        solutions=[scored_solution_from_dict(item) for item in data["solutions"]],
    )


def route_overlay_move_to_dict(move: RouteOverlayMove) -> dict[str, Any]:
    return {
        "move": move.move,
        "click_down": [move.click_down[0], move.click_down[1]],
        "lock_point": [move.lock_point[0], move.lock_point[1]],
        "release": [move.release[0], move.release[1]],
        "path_points": [[p[0], p[1]] for p in move.path_points],
        "selected_line": move.selected_line,
        "displacement": move.displacement,
        "free_drag": move.free_drag,
        "drag_distance": move.drag_distance,
        "move_distance": move.move_distance,
        "total_distance_to_here": move.total_distance_to_here,
    }


def route_overlay_move_from_dict(data: dict[str, Any]) -> RouteOverlayMove:
    return RouteOverlayMove(
        move=str(data["move"]),
        click_down=(float(data["click_down"][0]), float(data["click_down"][1])),
        lock_point=(float(data["lock_point"][0]), float(data["lock_point"][1])),
        release=(float(data["release"][0]), float(data["release"][1])),
        path_points=tuple((float(p[0]), float(p[1])) for p in data.get("path_points", [])),
        selected_line=int(data["selected_line"]),
        displacement=int(data["displacement"]),
        free_drag=bool(data["free_drag"]),
        drag_distance=float(data["drag_distance"]),
        move_distance=float(data["move_distance"]),
        total_distance_to_here=float(data["total_distance_to_here"]),
    )


def route_step_to_dict(step: RouteStep) -> dict[str, Any]:
    return {
        "step_id": step.step_id,
        "world_number": step.world_number,
        "puzzle_number": step.puzzle_number,
        "puzzle_id": step.puzzle_id,
        "source_file": step.source_file,
        "move_string": step.move_string,
        "total_cost": step.total_cost,
        "total_move": step.total_move,
        "total_drag": step.total_drag,
        "start_mouse_position": list(step.start_mouse_position) if step.start_mouse_position is not None else None,
        "end_positions": [list(point) for point in step.end_positions],
        "end_next_puzzle": step.end_next_puzzle,
        "post_end_click_target": list(step.post_end_click_target) if step.post_end_click_target is not None else None,
        "has_fat": step.has_fat,
        "initial_fat_position": list(step.initial_fat_position) if step.initial_fat_position is not None else None,
        "notes": step.notes,
        "initial_move_distance": step.initial_move_distance,
        "inter_move_distance": step.inter_move_distance,
        "final_mouse_target": list(step.final_mouse_target) if step.final_mouse_target is not None else None,
        "final_move_distance": step.final_move_distance,
        "overlay_moves": [route_overlay_move_to_dict(move) for move in step.overlay_moves],
    }


def route_step_from_dict(data: dict[str, Any]) -> RouteStep:
    start_mouse_position = None
    if data.get("start_mouse_position") is not None:
        start_mouse_position = (
            float(data["start_mouse_position"][0]),
            float(data["start_mouse_position"][1]),
        )

    post_end_click_target = None
    if data.get("post_end_click_target") is not None:
        post_end_click_target = (
            float(data["post_end_click_target"][0]),
            float(data["post_end_click_target"][1]),
        )

    initial_fat_position = None
    if data.get("initial_fat_position") is not None:
        initial_fat_position = (
            float(data["initial_fat_position"][0]),
            float(data["initial_fat_position"][1]),
        )

    final_mouse_target = None
    if data.get("final_mouse_target") is not None:
        final_mouse_target = (
            float(data["final_mouse_target"][0]),
            float(data["final_mouse_target"][1]),
        )

    return RouteStep(
        step_id=str(data["step_id"]),
        world_number=int(data["world_number"]),
        puzzle_number=int(data["puzzle_number"]),
        puzzle_id=str(data["puzzle_id"]),
        source_file=str(data["source_file"]),
        move_string=str(data["move_string"]),
        total_cost=float(data["total_cost"]),
        total_move=float(data["total_move"]),
        total_drag=float(data["total_drag"]),
        start_mouse_position=start_mouse_position,
        end_positions=tuple((float(p[0]), float(p[1])) for p in data.get("end_positions", [])),
        end_next_puzzle=bool(data.get("end_next_puzzle", False)),
        post_end_click_target=post_end_click_target,
        has_fat=bool(data.get("has_fat", False)),
        initial_fat_position=initial_fat_position,
        notes=str(data.get("notes", "")),
        initial_move_distance=float(data.get("initial_move_distance", 0.0)),
        inter_move_distance=float(data.get("inter_move_distance", 0.0)),
        final_mouse_target=final_mouse_target,
        final_move_distance=float(data.get("final_move_distance", 0.0)),
        overlay_moves=[route_overlay_move_from_dict(item) for item in data.get("overlay_moves", [])],
    )


def route_profile_to_dict(profile: RouteProfile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "description": profile.description,
        "category": profile.category,
        "steps": [route_step_to_dict(step) for step in profile.steps],
    }


def route_profile_from_dict(data: dict[str, Any]) -> RouteProfile:
    return RouteProfile(
        name=str(data["name"]),
        description=str(data.get("description", "")),
        category=str(data.get("category", "custom")),
        steps=[route_step_from_dict(step) for step in data.get("steps", [])],
    )


def cached_score_file_to_dict(cached: CachedScoreFile) -> dict[str, Any]:
    return {
        "source_file": cached.source_file,
        "source_hash": cached.source_hash,
        "options_hash": cached.options_hash,
        "result": file_score_result_to_dict(cached.result),
    }


def cached_score_file_from_dict(data: dict[str, Any]) -> CachedScoreFile:
    return CachedScoreFile(
        source_file=str(data["source_file"]),
        source_hash=str(data["source_hash"]),
        options_hash=str(data["options_hash"]),
        result=file_score_result_from_dict(data["result"]),
    )