from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Sequence

from chuzzle_mouse_dp3 import (
    DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
    DEFAULT_LOCK_THRESHOLD,
    DEFAULT_NEXT_PUZZLE_TARGETS,
    ScoredMove,
    ScoredSolution,
    normalize_move_sequence,
    solve_sequence as solve_sequence_dp3,
)


class NativeBackendUnavailableError(RuntimeError):
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
        ("end_next_puzzle", ctypes.c_int),
        ("end_position_count", ctypes.c_size_t),
        ("end_positions", ctypes.POINTER(ChuzzlePoint)),
        ("lock_threshold", ctypes.c_double),
        ("free_drag_min_displacement", ctypes.c_int),
    ]


class ChuzzleNativeStep(ctypes.Structure):
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


class ChuzzleNativeSolution(ctypes.Structure):
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
        ("steps", ctypes.POINTER(ChuzzleNativeStep)),
    ]


CHUZZLE_OK = 0


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _candidate_library_paths() -> list[Path]:
    candidates: list[Path] = []

    env_path = os.environ.get("CHUZZLE_MOUSE_NATIVE_LIB")
    if env_path:
        candidates.append(Path(env_path))

    base = _module_dir()
    names = [
        "chuzzle_mouse_native.dll",
        "libchuzzle_mouse_native.so",
        "libchuzzle_mouse_native.dylib",
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
                ctypes.POINTER(ChuzzleNativeSolution),
            ]
            lib.chuzzle_solve_sequence_utf8.restype = ctypes.c_int

            lib.chuzzle_free_solution.argtypes = [ctypes.POINTER(ChuzzleNativeSolution)]
            lib.chuzzle_free_solution.restype = None
            return lib
        except OSError:
            continue
        except AttributeError:
            continue
    return None


_NATIVE_LIB = _try_load_library()


def native_backend_available() -> bool:
    return _NATIVE_LIB is not None


def backend_name() -> str:
    if native_backend_available():
        return "native_ctypes"
    return "dp3_python_fallback"


def _as_native_point(point: Sequence[float]) -> ChuzzlePoint:
    return ChuzzlePoint(float(point[0]), float(point[1]))


def _decode_c_string(ptr_value: int | None) -> str:
    if not ptr_value:
        return ""
    return ctypes.cast(ptr_value, ctypes.c_char_p).value.decode("utf-8")


def _convert_native_solution(native: ChuzzleNativeSolution) -> ScoredSolution:
    move_data: list[ScoredMove] = []

    for i in range(native.step_count):
        step = native.steps[i]
        move = bytes(step.move).split(b"\0", 1)[0].decode("utf-8")

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


def _solve_sequence_native(
    move_string: str | Sequence[str],
    *,
    start_mouse_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> ScoredSolution:
    if _NATIVE_LIB is None:
        raise NativeBackendUnavailableError("Native backend library is not available")

    tokens = normalize_move_sequence(move_string)
    if not tokens:
        raise ValueError("Cannot solve an empty move sequence")

    normalized_move_string = " ".join(tokens)

    if end_next_puzzle:
        final_targets = DEFAULT_NEXT_PUZZLE_TARGETS
    else:
        final_targets = tuple((float(point[0]), float(point[1])) for point in (end_positions or ()))

    end_points_buffer = None
    end_points_ptr = ctypes.POINTER(ChuzzlePoint)()
    if final_targets and not end_next_puzzle:
        point_array_type = ChuzzlePoint * len(final_targets)
        end_points_buffer = point_array_type(*(_as_native_point(point) for point in final_targets))
        end_points_ptr = ctypes.cast(end_points_buffer, ctypes.POINTER(ChuzzlePoint))

    config = ChuzzleSolveConfig()
    if start_mouse_position is None:
        config.has_start_mouse_position = 0
        config.start_mouse_position = ChuzzlePoint(0.0, 0.0)
    else:
        config.has_start_mouse_position = 1
        config.start_mouse_position = _as_native_point(start_mouse_position)

    config.end_next_puzzle = 1 if end_next_puzzle else 0
    config.end_position_count = 0 if end_next_puzzle else len(final_targets)
    config.end_positions = end_points_ptr
    config.lock_threshold = float(lock_threshold)
    config.free_drag_min_displacement = int(free_drag_min_displacement)

    native_solution = ChuzzleNativeSolution()

    rc = _NATIVE_LIB.chuzzle_solve_sequence_utf8(
        normalized_move_string.encode("utf-8"),
        ctypes.byref(config),
        ctypes.byref(native_solution),
    )
    if rc != CHUZZLE_OK:
        _NATIVE_LIB.chuzzle_free_solution(ctypes.byref(native_solution))
        raise RuntimeError(f"Native solver returned error code {rc}")

    try:
        return _convert_native_solution(native_solution)
    finally:
        _NATIVE_LIB.chuzzle_free_solution(ctypes.byref(native_solution))


def solve_sequence(
    move_string: str | Sequence[str],
    *,
    start_mouse_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> ScoredSolution:
    if native_backend_available():
        return _solve_sequence_native(
            move_string,
            start_mouse_position=start_mouse_position,
            end_positions=end_positions,
            end_next_puzzle=end_next_puzzle,
            lock_threshold=lock_threshold,
            free_drag_min_displacement=free_drag_min_displacement,
        )

    return solve_sequence_dp3(
        move_string,
        start_mouse_position=start_mouse_position,
        end_positions=end_positions,
        end_next_puzzle=end_next_puzzle,
        lock_threshold=lock_threshold,
        free_drag_min_displacement=free_drag_min_displacement,
    )
