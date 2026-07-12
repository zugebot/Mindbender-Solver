from __future__ import annotations
# pyth/viewer/chuzzle_mouse_routes.py

import re
import uuid
from pathlib import Path

from chuzzle_mouse_domain import (
    RouteOverlayMove,
    RouteProfile,
    RouteStep,
    ScoredSolution,
    route_profile_from_dict,
    route_profile_to_dict,
)

FILE_PATTERN = re.compile(r"^(\d+)-(\d+)_.*\.txt$")


class RouteProfileManager:
    def __init__(self, profiles_dir: str | Path):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)

    def profile_path(self, name: str) -> Path:
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
        if not safe_name:
            safe_name = "unnamed_profile"
        return self.profiles_dir / f"{safe_name}.route.json"

    def list_profiles(self) -> list[str]:
        files = sorted(self.profiles_dir.glob("*.route.json"))
        out: list[str] = []
        for file in files:
            name = file.name
            if name.endswith(".route.json"):
                out.append(name[:-11])
            else:
                out.append(file.stem)
        return out

    def load_profile(self, path_or_name: str | Path) -> RouteProfile:
        path = Path(path_or_name)
        if not path.exists():
            path = self.profile_path(str(path_or_name))
        data = path.read_text(encoding="utf-8")
        import json
        return route_profile_from_dict(json.loads(data))

    def save_profile(self, profile: RouteProfile) -> Path:
        path = self.profile_path(profile.name)
        import json
        path.write_text(json.dumps(route_profile_to_dict(profile), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def delete_profile(self, name: str) -> None:
        path = self.profile_path(name)
        if path.exists():
            path.unlink()

    @staticmethod
    def empty_profile(name: str, description: str = "", category: str = "custom") -> RouteProfile:
        return RouteProfile(name=name, description=description, category=category, steps=[])

    @staticmethod
    def route_total_cost(profile: RouteProfile) -> float:
        return sum(step.total_cost for step in profile.steps)

    @staticmethod
    def route_total_move(profile: RouteProfile) -> float:
        return sum(step.total_move for step in profile.steps)

    @staticmethod
    def route_total_drag(profile: RouteProfile) -> float:
        return sum(step.total_drag for step in profile.steps)

    @staticmethod
    def world_summary(profile: RouteProfile) -> dict[int, float]:
        out: dict[int, float] = {}
        for step in profile.steps:
            out.setdefault(step.world_number, 0.0)
            out[step.world_number] += step.total_cost
        return out

    @staticmethod
    def world_step_counts(profile: RouteProfile) -> dict[int, int]:
        out: dict[int, int] = {}
        for step in profile.steps:
            out.setdefault(step.world_number, 0)
            out[step.world_number] += 1
        return out

    @staticmethod
    def parse_world_and_puzzle_from_filename(filename: str) -> tuple[int, int]:
        match = FILE_PATTERN.match(filename)
        if not match:
            raise ValueError(f"Could not parse world/puzzle from filename: {filename}")
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def build_overlay_moves(solution: ScoredSolution) -> list[RouteOverlayMove]:
        out: list[RouteOverlayMove] = []
        for move in solution.move_data:
            out.append(
                RouteOverlayMove(
                    move=str(move.move),
                    click_down=(float(move.click_down[0]), float(move.click_down[1])),
                    lock_point=(float(move.lock_point[0]), float(move.lock_point[1])),
                    release=(float(move.release[0]), float(move.release[1])),
                    path_points=tuple((float(p[0]), float(p[1])) for p in move.path_points),
                    selected_line=int(move.selected_line),
                    displacement=int(move.displacement),
                    free_drag=bool(move.free_drag),
                    drag_distance=float(move.drag_distance),
                    move_distance=float(move.move_distance),
                    total_distance_to_here=float(move.total_distance_to_here),
                )
            )
        return out

    @staticmethod
    def build_route_step_from_solution(
            *,
            source_file: str,
            solution: ScoredSolution,
            start_mouse_position: tuple[float, float] | None = None,
            end_positions: tuple[tuple[float, float], ...] | list[tuple[float, float]] = (),
            end_next_puzzle: bool = False,
            post_end_click_target: tuple[float, float] | None = None,
            has_fat: bool = False,
            initial_fat_position: tuple[float, float] | None = None,
            notes: str = "",
    ) -> RouteStep:
        world_number, puzzle_number = RouteProfileManager.parse_world_and_puzzle_from_filename(source_file)
        puzzle_id = f"{world_number}-{puzzle_number}"

        return RouteStep(
            step_id=uuid.uuid4().hex[:12],
            world_number=world_number,
            puzzle_number=puzzle_number,
            puzzle_id=puzzle_id,
            source_file=source_file,
            move_string=solution.move_string,
            total_cost=float(solution.total_cost),
            total_move=float(solution.total_move),
            total_drag=float(solution.total_drag),
            start_mouse_position=start_mouse_position,
            end_positions=tuple(end_positions or ()),
            end_next_puzzle=bool(end_next_puzzle),
            post_end_click_target=post_end_click_target,
            has_fat=bool(has_fat),
            initial_fat_position=initial_fat_position,
            notes=notes,
            initial_move_distance=float(solution.initial_move_distance),
            inter_move_distance=float(solution.inter_move_distance),
            final_mouse_target=solution.final_mouse_target,
            final_move_distance=float(solution.final_move_distance),
            overlay_moves=RouteProfileManager.build_overlay_moves(solution),
        )

    @staticmethod
    def replace_step(profile: RouteProfile, index: int, new_step: RouteStep) -> None:
        if 0 <= index < len(profile.steps):
            profile.steps[index] = new_step

    @staticmethod
    def move_step(profile: RouteProfile, from_index: int, to_index: int) -> None:
        if not (0 <= from_index < len(profile.steps)):
            return
        to_index = max(0, min(len(profile.steps) - 1, to_index))
        if from_index == to_index:
            return
        step = profile.steps.pop(from_index)
        profile.steps.insert(to_index, step)