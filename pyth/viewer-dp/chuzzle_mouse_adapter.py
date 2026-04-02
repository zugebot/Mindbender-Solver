from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from chuzzle_mouse_dp3 import (
    DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    DEFAULT_LOCK_THRESHOLD,
    DEFAULT_NEXT_PUZZLE_TARGETS,
    GRID_SIZE,
    ScoredMove,
    ScoredSolution,
    solve_sequence as solve_sequence_dp3,
)

_BACKEND_NAME = "dp3_python"


@dataclass(slots=True)
class FileScoreResult:
    path: Path
    solutions: list[ScoredSolution]


def read_solution_lines(
    path: str | Path,
    *,
    dedupe: bool = False,
) -> list[str]:
    file_path = Path(path)
    raw_lines = file_path.read_text(encoding="utf-8").splitlines()

    lines: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if line:
            lines.append(line)

    if dedupe:
        lines = list(dict.fromkeys(lines))

    return lines


def score_sequences(
    sequences: Iterable[str | Sequence[str]],
    *,
    start_mouse_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> list[ScoredSolution]:
    scored: list[ScoredSolution] = []

    for sequence in sequences:
        scored.append(
            solve_sequence_dp3(
                sequence,
                start_mouse_position=start_mouse_position,
                end_positions=end_positions,
                end_next_puzzle=end_next_puzzle,
                lock_threshold=lock_threshold,
                free_drag_min_displacement=free_drag_min_displacement,
            )
        )

    scored.sort(key=lambda s: (s.total_cost, s.total_move, s.total_drag, s.move_string))
    return scored


def score_file(
    path: str | Path,
    dedupe: bool = False,
    *,
    start_mouse_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> FileScoreResult:
    file_path = Path(path)
    lines = read_solution_lines(file_path, dedupe=dedupe)
    solutions = score_sequences(
        lines,
        start_mouse_position=start_mouse_position,
        end_positions=end_positions,
        end_next_puzzle=end_next_puzzle,
        lock_threshold=lock_threshold,
        free_drag_min_displacement=free_drag_min_displacement,
    )
    return FileScoreResult(path=file_path, solutions=solutions)


class DpMouseSolver:
    """
    Compatibility wrapper for studio code that still expects the old class shape.
    Backed by chuzzle_mouse_dp3.
    """

    @staticmethod
    def solve_sequence(
        move_string: str | Sequence[str],
        *,
        start_mouse_position: Sequence[float] | None = None,
        end_positions: Iterable[Sequence[float]] | None = None,
        end_next_puzzle: bool = False,
        lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
        free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    ) -> ScoredSolution:
        return solve_sequence_dp3(
            move_string,
            start_mouse_position=start_mouse_position,
            end_positions=end_positions,
            end_next_puzzle=end_next_puzzle,
            lock_threshold=lock_threshold,
            free_drag_min_displacement=free_drag_min_displacement,
        )

    @staticmethod
    def score_file(
        path: str | Path,
        dedupe: bool = False,
        *,
        start_mouse_position: Sequence[float] | None = None,
        end_positions: Iterable[Sequence[float]] | None = None,
        end_next_puzzle: bool = False,
        lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
        free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    ) -> FileScoreResult:
        return score_file(
            path,
            dedupe=dedupe,
            start_mouse_position=start_mouse_position,
            end_positions=end_positions,
            end_next_puzzle=end_next_puzzle,
            lock_threshold=lock_threshold,
            free_drag_min_displacement=free_drag_min_displacement,
        )

    @staticmethod
    def backend_name() -> str:
        return _BACKEND_NAME
