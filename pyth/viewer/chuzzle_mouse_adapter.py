from __future__ import annotations
# pyth/viewer/chuzzle_mouse_adapter.py

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

GRID_SIZE = 6
DEFAULT_LOCK_THRESHOLD = 1.0
DEFAULT_FREE_DRAG_MIN_DISPLACEMENT = 2
DEFAULT_NEXT_PUZZLE_TARGETS: tuple[tuple[float, float], ...] = (
    (1.0, 3.0),
    (2.0, 3.0),
    (3.0, 3.0),
    (4.0, 3.0),
)

CHUZZLE_OK = 0


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


class ChuzzleMouseBackendUnavailableError(RuntimeError):
    pass


class ChuzzlePoint(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
    ]


class ChuzzleSolveConfig(ctypes.Structure):
    _fields_ = [
        ("has_start_mouse_position", ctypes.c_int),
        ("start_mouse_position", ChuzzlePoint),
        ("has_initial_fat_position", ctypes.c_int),
        ("initial_fat_position", ChuzzlePoint),
        ("end_next_puzzle", ctypes.c_int),
        ("end_position_count", ctypes.c_size_t),
        ("end_positions", ctypes.POINTER(ChuzzlePoint)),
        ("lock_threshold", ctypes.c_double),
        ("free_drag_min_displacement", ctypes.c_int),
    ]


class ChuzzleStep(ctypes.Structure):
    _fields_ = [
        ("move", ctypes.c_char * 5),
        ("selected_line", ctypes.c_int),
        ("displacement", ctypes.c_int),
        ("free_drag", ctypes.c_int),
        ("click_down", ChuzzlePoint),
        ("lock_point", ChuzzlePoint),
        ("release", ChuzzlePoint),
        ("drag_distance", ctypes.c_double),
        ("move_distance", ctypes.c_double),
        ("total_distance_to_here", ctypes.c_double),
    ]


class ChuzzleSolution(ctypes.Structure):
    _fields_ = [
        ("move_string", ctypes.c_void_p),
        ("total_drag", ctypes.c_double),
        ("total_move", ctypes.c_double),
        ("total_cost", ctypes.c_double),
        ("has_initial_mouse_position", ctypes.c_int),
        ("initial_mouse_position", ChuzzlePoint),
        ("initial_move_distance", ctypes.c_double),
        ("inter_move_distance", ctypes.c_double),
        ("has_final_mouse_target", ctypes.c_int),
        ("final_mouse_target", ChuzzlePoint),
        ("final_move_distance", ctypes.c_double),
        ("step_count", ctypes.c_size_t),
        ("steps", ctypes.POINTER(ChuzzleStep)),
    ]


def normalize_move_sequence(move_string: str | Sequence[str]) -> list[str]:
    tokens: list[str] = []

    if isinstance(move_string, str):
        raw_tokens = move_string.strip().split()
        for token in raw_tokens:
            token = token.strip().upper()
            if token:
                tokens.append(token)
        return tokens

    for raw_token in move_string:
        token = str(raw_token).strip().upper()
        if token:
            tokens.append(token)

    return tokens


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _candidate_library_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("CHUZZLE_MOUSE_LIB")
    if env_path:
        candidates.append(Path(env_path))

    base = _module_dir()
    names = [
        "libchuzzle_mouse.dll",
        "libchuzzle_mouse.so",
        "libchuzzle_mouse.dylib",
    ]
    for name in names:
        candidates.append(base / name)

    return candidates


def _try_load_library() -> ctypes.CDLL | None:
    for path in _candidate_library_paths():
        if not path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(path))
            lib.chuzzle_solve_sequence_utf8.argtypes = [
                ctypes.c_char_p,
                ctypes.POINTER(ChuzzleSolveConfig),
                ctypes.POINTER(ChuzzleSolution),
            ]
            lib.chuzzle_solve_sequence_utf8.restype = ctypes.c_int

            lib.chuzzle_free_solution.argtypes = [ctypes.POINTER(ChuzzleSolution)]
            lib.chuzzle_free_solution.restype = None
            return lib
        except OSError:
            continue
        except AttributeError:
            continue
    return None


_CHUZZLE_MOUSE_LIB = _try_load_library()


def backend_available() -> bool:
    return _CHUZZLE_MOUSE_LIB is not None


def backend_name() -> str:
    if backend_available():
        return "ctypes"
    return "ctypes_unavailable"


_BACKEND_NAME = backend_name()


def _as_point(point: Sequence[float]) -> ChuzzlePoint:
    return ChuzzlePoint(float(point[0]), float(point[1]))


def _decode_c_string(ptr_value: int | None) -> str:
    if not ptr_value:
        return ""
    raw = ctypes.cast(ptr_value, ctypes.c_char_p).value
    if raw is None:
        return ""
    return raw.decode("utf-8")


def _convert_solution(native: ChuzzleSolution) -> ScoredSolution:
    move_data: list[ScoredMove] = []

    for i in range(native.step_count):
        step = native.steps[i]
        move = bytes(step.move).split(b"\\0", 1)[0].decode("utf-8")

        click_down = [step.click_down.x, step.click_down.y]
        lock_point = [step.lock_point.x, step.lock_point.y]
        release = [step.release.x, step.release.y]

        move_data.append(
            ScoredMove(
                move=move,
                click_down=click_down,
                lock_point=lock_point,
                release=release,
                path_points=[click_down[:], lock_point[:], release[:]],
                selected_line=int(step.selected_line),
                displacement=int(step.displacement),
                free_drag=bool(step.free_drag),
                drag_distance=float(step.drag_distance),
                move_distance=float(step.move_distance),
                total_distance_to_here=float(step.total_distance_to_here),
            )
        )

    initial_mouse_position = None
    if native.has_initial_mouse_position:
        initial_mouse_position = (
            float(native.initial_mouse_position.x),
            float(native.initial_mouse_position.y),
        )

    final_mouse_target = None
    if native.has_final_mouse_target:
        final_mouse_target = (
            float(native.final_mouse_target.x),
            float(native.final_mouse_target.y),
        )

    return ScoredSolution(
        move_string=_decode_c_string(native.move_string),
        total_drag=float(native.total_drag),
        total_move=float(native.total_move),
        total_cost=float(native.total_cost),
        move_data=move_data,
        initial_mouse_position=initial_mouse_position,
        initial_move_distance=float(native.initial_move_distance),
        inter_move_distance=float(native.inter_move_distance),
        final_mouse_target=final_mouse_target,
        final_move_distance=float(native.final_move_distance),
    )


def solve_sequence(
    move_string: str | Sequence[str],
    *,
    start_mouse_position: Sequence[float] | None = None,
    has_fat: bool = False,
    initial_fat_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> ScoredSolution:
    if _CHUZZLE_MOUSE_LIB is None:
        raise ChuzzleMouseBackendUnavailableError("Chuzzle Mouse backend library is not available")

    tokens = normalize_move_sequence(move_string)
    if not tokens:
        raise ValueError("Cannot solve an empty move sequence")

    normalized_move_string = " ".join(tokens)

    if end_next_puzzle:
        final_targets = DEFAULT_NEXT_PUZZLE_TARGETS
    else:
        final_targets = tuple(
            (float(point[0]), float(point[1]))
            for point in (end_positions or ())
        )

    end_points_buffer = None
    end_points_ptr = ctypes.POINTER(ChuzzlePoint)()
    if final_targets and not end_next_puzzle:
        point_array_type = ChuzzlePoint * len(final_targets)
        end_points_buffer = point_array_type(*(_as_point(point) for point in final_targets))
        end_points_ptr = ctypes.cast(end_points_buffer, ctypes.POINTER(ChuzzlePoint))

    config = ChuzzleSolveConfig()
    if start_mouse_position is None:
        config.has_start_mouse_position = 0
        config.start_mouse_position = ChuzzlePoint(0.0, 0.0)
    else:
        config.has_start_mouse_position = 1
        config.start_mouse_position = _as_point(start_mouse_position)

    if has_fat:
        if initial_fat_position is None:
            raise ValueError("has_fat=True requires initial_fat_position")
        config.has_initial_fat_position = 1
        config.initial_fat_position = _as_point(initial_fat_position)
    else:
        config.has_initial_fat_position = 0
        config.initial_fat_position = ChuzzlePoint(0.0, 0.0)

    config.end_next_puzzle = 1 if end_next_puzzle else 0
    config.end_position_count = 0 if end_next_puzzle else len(final_targets)
    config.end_positions = end_points_ptr
    config.lock_threshold = float(lock_threshold)
    config.free_drag_min_displacement = int(free_drag_min_displacement)

    native_solution = ChuzzleSolution()

    rc = _CHUZZLE_MOUSE_LIB.chuzzle_solve_sequence_utf8(
        normalized_move_string.encode("utf-8"),
        ctypes.byref(config),
        ctypes.byref(native_solution),
    )
    if rc != CHUZZLE_OK:
        _CHUZZLE_MOUSE_LIB.chuzzle_free_solution(ctypes.byref(native_solution))
        raise RuntimeError(f"Chuzzle Mouse solver returned error code {rc}")

    try:
        return _convert_solution(native_solution)
    finally:
        _CHUZZLE_MOUSE_LIB.chuzzle_free_solution(ctypes.byref(native_solution))


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
    has_fat: bool = False,
    initial_fat_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> list[ScoredSolution]:
    scored: list[ScoredSolution] = []

    for sequence in sequences:
        scored.append(
            solve_sequence(
                sequence,
                start_mouse_position=start_mouse_position,
                has_fat=has_fat,
                initial_fat_position=initial_fat_position,
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
    has_fat: bool = False,
    initial_fat_position: Sequence[float] | None = None,
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
        has_fat=has_fat,
        initial_fat_position=initial_fat_position,
        end_positions=end_positions,
        end_next_puzzle=end_next_puzzle,
        lock_threshold=lock_threshold,
        free_drag_min_displacement=free_drag_min_displacement,
    )
    return FileScoreResult(path=file_path, solutions=solutions)


class DpMouseSolver:
    @staticmethod
    def solve_sequence(
        move_string: str | Sequence[str],
        *,
        start_mouse_position: Sequence[float] | None = None,
        has_fat: bool = False,
        initial_fat_position: Sequence[float] | None = None,
        end_positions: Iterable[Sequence[float]] | None = None,
        end_next_puzzle: bool = False,
        lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
        free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    ) -> ScoredSolution:
        return solve_sequence(
            move_string,
            start_mouse_position=start_mouse_position,
            has_fat=has_fat,
            initial_fat_position=initial_fat_position,
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
        has_fat: bool = False,
        initial_fat_position: Sequence[float] | None = None,
        end_positions: Iterable[Sequence[float]] | None = None,
        end_next_puzzle: bool = False,
        lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
        free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    ) -> FileScoreResult:
        return score_file(
            path,
            dedupe=dedupe,
            start_mouse_position=start_mouse_position,
            has_fat=has_fat,
            initial_fat_position=initial_fat_position,
            end_positions=end_positions,
            end_next_puzzle=end_next_puzzle,
            lock_threshold=lock_threshold,
            free_drag_min_displacement=free_drag_min_displacement,
        )

    @staticmethod
    def backend_name() -> str:
        return _BACKEND_NAME


