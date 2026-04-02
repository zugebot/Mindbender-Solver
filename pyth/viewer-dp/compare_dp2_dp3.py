from __future__ import annotations

from pathlib import Path

from chuzzle_mouse_dp2 import solve_sequence as solve_sequence_dp2
from chuzzle_mouse_dp3 import solve_sequence as solve_sequence_dp3

ABS_TOL = 1e-9

SEQUENCES: list[str] = [
    "R01",
    "R12 C34 R451",
    "R12 C34 R45 C120",
    "R451 C231 R12 C45",
    "C45 R22 C53 R04 C32 C25 R13 C12 R54 R05 C32",
    "C45 R22 C53 R04 C25 C32 R13 C12 R54 R05 C32",
    "C25 C53 R13 R24 C12 C32 R05 C45 R54 C34 R22",
    "C24 R05 R52 C01 R44 R13 C21 R24 C13 R31 C45",
    "R22 C25 C32 C53 R13 C12 R05 C45 R54 C32 R22",
    "C25 C53 R13 R24 C32 C12 R05 C45 R54 C34 R22",
    "R25 C12 C44 R02 R12 R54 C13 R51 C25 R32",
    "R25 C12 C44 R02 R12 R54 C13 R51 R32 C25",
    "R25 C12 C44 R02 R54 R12 C13 R51 C25 R32",
    "R25 C12 C44 R02 R54 R12 C13 R51 R32 C25",
    "R25 C12 C44 R04 R12 R32 C33 R55 R04 C25",
    "R25 C12 C44 R04 R32 R12 C33 R55 R04 C25",
    "R25 C12 C44 R12 R32 R04 C33 R55 R04 C25",
    "R25 C12 C44 R12 R54 R02 C13 R51 C25 R32",
    "R25 C12 C44 R12 R54 R02 C13 R51 R32 C25",
    "R25 C15 C44 R02 C13 R55 R12 C25 R32 R01",
    "R25 C15 C44 R02 C13 R55 R12 R01 C25 R32",
    "R25 C15 C44 R02 C13 R55 R32 R12 C25 R01",
    "R25 C15 C44 R02 C13 R55 R32 R12 R01 C25",
    "R25 C15 R02 C13 R02 C44 R55 R12 C25 R32",
    "R25 C15 R02 C13 R02 C44 R55 R32 R12 C25",
    "R25 C44 R02 C12 R12 R32 C33 R55 C25 R01",
    "R25 C44 R02 C12 R12 R32 C33 R55 R01 C25",
    "R25 C44 R02 C12 R32 R12 C33 R55 C25 R01",
    "R25 C44 R02 C12 R32 R12 C33 R55 R01 C25",

]

LEVEL_FILE: str | None = None
LEVEL_FILE_LIMIT: int | None = None

START_MOUSE_POSITION: tuple[float, float] | None = (-1.0, 5.0)
END_POSITIONS: tuple[tuple[float, float], ...] | None = None
END_NEXT_PUZZLE = True
LOCK_THRESHOLD = 1.0
FREE_DRAG_MIN_DISPLACEMENT = 2


def read_sequences_from_file(path: str, limit: int | None = None) -> list[str]:
    raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if limit is not None:
        lines = lines[:limit]
    return lines


def assert_close(label: str, a: float, b: float, tol: float = ABS_TOL) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{label}: {a!r} != {b!r} within tolerance {tol}")


def assert_point_close(label: str, a, b, tol: float = ABS_TOL) -> None:
    if a is None or b is None:
        if a != b:
            raise AssertionError(f"{label}: {a!r} != {b!r}")
        return

    if len(a) != len(b):
        raise AssertionError(f"{label}: length mismatch {len(a)} != {len(b)}")

    for i, (av, bv) in enumerate(zip(a, b)):
        assert_close(f"{label}[{i}]", float(av), float(bv), tol)


def compare_solutions(sequence: str) -> None:
    kwargs = dict(
        start_mouse_position=START_MOUSE_POSITION,
        end_positions=END_POSITIONS,
        end_next_puzzle=END_NEXT_PUZZLE,
        lock_threshold=LOCK_THRESHOLD,
        free_drag_min_displacement=FREE_DRAG_MIN_DISPLACEMENT,
    )

    result2 = solve_sequence_dp2(sequence, **kwargs)
    result3 = solve_sequence_dp3(sequence, **kwargs)

    if result2.move_string != result3.move_string:
        raise AssertionError(f"move_string mismatch: {result2.move_string!r} != {result3.move_string!r}")

    assert_close("total_drag", result2.total_drag, result3.total_drag)
    assert_close("total_move", result2.total_move, result3.total_move)
    assert_close("total_cost", result2.total_cost, result3.total_cost)
    assert_close("initial_move_distance", result2.initial_move_distance, result3.initial_move_distance)
    assert_close("inter_move_distance", result2.inter_move_distance, result3.inter_move_distance)
    assert_close("final_move_distance", result2.final_move_distance, result3.final_move_distance)

    assert_point_close("initial_mouse_position", result2.initial_mouse_position, result3.initial_mouse_position)
    assert_point_close("final_mouse_target", result2.final_mouse_target, result3.final_mouse_target)

    if len(result2.move_data) != len(result3.move_data):
        raise AssertionError(f"move_data length mismatch: {len(result2.move_data)} != {len(result3.move_data)}")

    for index, (step2, step3) in enumerate(zip(result2.move_data, result3.move_data)):
        prefix = f"step[{index}]"

        if step2.move != step3.move:
            raise AssertionError(f"{prefix}.move mismatch: {step2.move!r} != {step3.move!r}")

        if step2.selected_line != step3.selected_line:
            raise AssertionError(
                f"{prefix}.selected_line mismatch: {step2.selected_line!r} != {step3.selected_line!r}"
            )

        if step2.displacement != step3.displacement:
            raise AssertionError(
                f"{prefix}.displacement mismatch: {step2.displacement!r} != {step3.displacement!r}"
            )

        if step2.free_drag != step3.free_drag:
            raise AssertionError(f"{prefix}.free_drag mismatch: {step2.free_drag!r} != {step3.free_drag!r}")

        assert_close(f"{prefix}.drag_distance", step2.drag_distance, step3.drag_distance)
        assert_close(f"{prefix}.move_distance", step2.move_distance, step3.move_distance)
        assert_close(f"{prefix}.total_distance_to_here", step2.total_distance_to_here, step3.total_distance_to_here)

        assert_point_close(f"{prefix}.click_down", step2.click_down, step3.click_down)
        assert_point_close(f"{prefix}.lock_point", step2.lock_point, step3.lock_point)
        assert_point_close(f"{prefix}.release", step2.release, step3.release)

        if len(step2.path_points) != len(step3.path_points):
            raise AssertionError(
                f"{prefix}.path_points length mismatch: {len(step2.path_points)} != {len(step3.path_points)}"
            )

        for point_index, (point2, point3) in enumerate(zip(step2.path_points, step3.path_points)):
            assert_point_close(f"{prefix}.path_points[{point_index}]", point2, point3)


def main() -> None:
    sequences = list(SEQUENCES)
    if LEVEL_FILE:
        sequences.extend(read_sequences_from_file(LEVEL_FILE, LEVEL_FILE_LIMIT))

    if not sequences:
        raise SystemExit("No sequences configured")

    print(f"Comparing {len(sequences)} sequence(s)")
    for index, sequence in enumerate(sequences, start=1):
        compare_solutions(sequence)
        print(f"[OK {index:03d}] {sequence}")

    print("All compared solutions matched.")


if __name__ == "__main__":
    main()
