from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence


GRID_SIZE = 6

DEFAULT_NEXT_PUZZLE_TARGETS: tuple[tuple[float, float], ...] = (
    (1.0, 3.0),
    (2.0, 3.0),
    (3.0, 3.0),
    (4.0, 3.0),
)


def normalize_point(point: Sequence[float] | None) -> tuple[float, float] | None:
    if point is None:
        return None
    if len(point) != 2:
        raise ValueError(f"Point must have exactly 2 coordinates, got: {point!r}")
    return float(point[0]), float(point[1])


def normalize_points(points: Iterable[Sequence[float]] | None) -> tuple[tuple[float, float], ...]:
    if points is None:
        return ()
    out: list[tuple[float, float]] = []
    for point in points:
        if len(point) != 2:
            raise ValueError(f"Point must have exactly 2 coordinates, got: {point!r}")
        out.append((float(point[0]), float(point[1])))
    return tuple(out)


def nearest_target(
        point: Sequence[float],
        targets: Sequence[Sequence[float]],
) -> tuple[tuple[float, float] | None, float]:
    if not targets:
        return None, 0.0

    best_target: tuple[float, float] | None = None
    best_distance = math.inf
    for target in targets:
        target_xy = float(target[0]), float(target[1])
        distance = euclidean_distance(point, target_xy)
        if distance < best_distance:
            best_distance = distance
            best_target = target_xy
    return best_target, best_distance


@dataclass(frozen=True)
class ParsedMove:
    token: str
    axis: str
    lines: tuple[int, ...]
    amount: int

    @property
    def is_fat(self) -> bool:
        return len(self.lines) == 2

    @property
    def candidate_count_hint(self) -> int:
        return len(self.lines) * GRID_SIZE * len(get_displacements(self.amount))


@dataclass(frozen=True)
class MoveCandidate:
    token: str
    axis: str
    lines: tuple[int, ...]
    amount: int
    click_down: tuple[int, int]
    release: tuple[int, int]
    selected_line: int
    displacement: int
    drag_distance: float


@dataclass
class ScoredMove:
    move: str
    click_down: list[int]
    release: list[int]
    selected_line: int
    displacement: int
    drag_distance: float
    move_distance: float
    total_distance_to_here: float


@dataclass
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


@dataclass
class FileScoreResult:
    path: Path
    solutions: list[ScoredSolution]


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


@lru_cache(maxsize=None)
def parse_move(token: str) -> ParsedMove:
    token = token.strip().upper()
    if not token:
        raise ValueError("Empty move token")
    if token[0] not in {"R", "C"}:
        raise ValueError(f"Move must start with R or C: {token!r}")
    payload = token[1:]
    if not payload.isdigit():
        raise ValueError(f"Move payload must be numeric: {token!r}")
    if len(payload) == 2:
        lines = (int(payload[0]),)
        amount = int(payload[1])
    elif len(payload) == 3:
        lines = (int(payload[0]), int(payload[1]))
        amount = int(payload[2])
    else:
        raise ValueError(
            f"Unsupported move format {token!r}. Expected Rxy / Cxy or Rxyz / Cxyz."
        )
    if any(line < 0 or line >= GRID_SIZE for line in lines):
        raise ValueError(f"Move line out of range 0..{GRID_SIZE - 1}: {token!r}")
    return ParsedMove(token=token, axis=token[0], lines=lines, amount=amount % GRID_SIZE)


@lru_cache(maxsize=None)
def get_displacements(amount: int) -> tuple[int, ...]:
    amount %= GRID_SIZE
    if amount == 0:
        return (0,)
    forward = amount
    backward = -(GRID_SIZE - amount)
    if forward == backward:
        return (forward,)
    return forward, backward


@lru_cache(maxsize=None)
def enumerate_candidates(token: str) -> tuple[MoveCandidate, ...]:
    move = parse_move(token)
    candidates: dict[tuple[tuple[int, int], tuple[int, int], int], MoveCandidate] = {}
    for line in move.lines:
        for anchor in range(GRID_SIZE):
            for displacement in get_displacements(move.amount):
                if move.axis == "R":
                    click_down = (anchor, line)
                    release = (anchor + displacement, line)
                else:
                    click_down = (line, anchor)
                    release = (line, anchor + displacement)
                candidate = MoveCandidate(
                    token=move.token,
                    axis=move.axis,
                    lines=move.lines,
                    amount=move.amount,
                    click_down=click_down,
                    release=release,
                    selected_line=line,
                    displacement=displacement,
                    drag_distance=float(abs(displacement)),
                )
                key = (candidate.click_down, candidate.release, candidate.selected_line)
                candidates[key] = candidate
    ordered = sorted(
        candidates.values(),
        key=lambda c: (
            c.selected_line,
            c.click_down[1],
            c.click_down[0],
            c.release[1],
            c.release[0],
            c.displacement,
        ),
    )
    return tuple(ordered)


@lru_cache(maxsize=None)
def transition_matrix(prev_token: str, next_token: str) -> tuple[tuple[float, ...], ...]:
    prev_candidates = enumerate_candidates(prev_token)
    next_candidates = enumerate_candidates(next_token)
    rows: list[tuple[float, ...]] = []
    for prev_candidate in prev_candidates:
        row = tuple(
            euclidean_distance(prev_candidate.release, next_candidate.click_down)
            for next_candidate in next_candidates
        )
        rows.append(row)
    return tuple(rows)


class DpMouseSolver:
    """
    Exact dynamic-programming solver for the mouse-path optimization problem.

    Assumptions baked into this model:
    1. First click has zero move-to cost unless start_mouse_position is provided.
    2. Click-down must occur on a valid row/column location inside the 6x6 board.
    3. Release positions are allowed outside the board. This models dragging past an edge.
    4. 4-character moves such as C014 / R452 are treated as fat moves. The drag can start on
       either affected line, which matches your note about grabbing either side of the 2x2 piece.
    5. The mouse-cost objective is:
          optional initial cursor travel
        + sum(drag lengths)
        + sum(cursor travel between consecutive gestures)
        + optional final cursor travel to a target
    """

    @staticmethod
    def solve_sequence(
            move_string: str | Sequence[str],
            *,
            start_mouse_position: Sequence[float] | None = None,
            end_positions: Iterable[Sequence[float]] | None = None,
            end_next_puzzle: bool = False,
    ) -> ScoredSolution:
        tokens = normalize_move_sequence(move_string)
        if not tokens:
            raise ValueError("Cannot solve an empty move sequence")

        start_pos = normalize_point(start_mouse_position)

        if end_next_puzzle:
            final_targets = DEFAULT_NEXT_PUZZLE_TARGETS
        else:
            final_targets = normalize_points(end_positions)

        optimize_end = len(final_targets) > 0

        candidate_layers = [enumerate_candidates(token) for token in tokens]

        costs: list[float] = []
        for candidate in candidate_layers[0]:
            initial_move = 0.0 if start_pos is None else euclidean_distance(start_pos, candidate.click_down)
            costs.append(initial_move + candidate.drag_distance)

        parents: list[list[int]] = [[-1] * len(candidate_layers[0])]

        for move_index in range(1, len(tokens)):
            prev_token = tokens[move_index - 1]
            current_token = tokens[move_index]
            prev_candidates = candidate_layers[move_index - 1]
            current_candidates = candidate_layers[move_index]
            trans = transition_matrix(prev_token, current_token)

            next_costs = [math.inf] * len(current_candidates)
            next_parents = [-1] * len(current_candidates)
            current_drag = [candidate.drag_distance for candidate in current_candidates]

            for curr_idx in range(len(current_candidates)):
                best_cost = math.inf
                best_parent = -1
                for prev_idx in range(len(prev_candidates)):
                    cost = costs[prev_idx] + trans[prev_idx][curr_idx] + current_drag[curr_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_parent = prev_idx
                next_costs[curr_idx] = best_cost
                next_parents[curr_idx] = best_parent

            costs = next_costs
            parents.append(next_parents)

        if optimize_end:
            adjusted_costs: list[float] = []
            chosen_targets: list[tuple[float, float] | None] = []
            chosen_target_costs: list[float] = []

            for candidate, base_cost in zip(candidate_layers[-1], costs):
                best_target, extra_cost = nearest_target(candidate.release, final_targets)
                adjusted_costs.append(base_cost + extra_cost)
                chosen_targets.append(best_target)
                chosen_target_costs.append(extra_cost)

            best_final_index = min(range(len(adjusted_costs)), key=adjusted_costs.__getitem__)
            final_mouse_target = chosen_targets[best_final_index]
            final_move_distance = chosen_target_costs[best_final_index]
        else:
            best_final_index = min(range(len(costs)), key=costs.__getitem__)
            final_mouse_target = None
            final_move_distance = 0.0

        chosen_indices = [0] * len(tokens)
        chosen_indices[-1] = best_final_index
        for move_index in range(len(tokens) - 1, 0, -1):
            chosen_indices[move_index - 1] = parents[move_index][chosen_indices[move_index]]

        move_data: list[ScoredMove] = []
        total_drag = 0.0
        initial_move_distance = 0.0
        inter_move_distance = 0.0
        previous_release: tuple[int, int] | None = None

        for move_index, candidate_index in enumerate(chosen_indices):
            candidate = candidate_layers[move_index][candidate_index]

            if previous_release is None:
                move_distance = 0.0 if start_pos is None else euclidean_distance(start_pos, candidate.click_down)
                initial_move_distance = move_distance
            else:
                move_distance = euclidean_distance(previous_release, candidate.click_down)
                inter_move_distance += move_distance

            total_drag += candidate.drag_distance
            total_cost_to_here = total_drag + initial_move_distance + inter_move_distance

            move_data.append(
                ScoredMove(
                    move=candidate.token,
                    click_down=[candidate.click_down[0], candidate.click_down[1]],
                    release=[candidate.release[0], candidate.release[1]],
                    selected_line=candidate.selected_line,
                    displacement=candidate.displacement,
                    drag_distance=candidate.drag_distance,
                    move_distance=move_distance,
                    total_distance_to_here=total_cost_to_here,
                )
            )
            previous_release = candidate.release

        total_move = initial_move_distance + inter_move_distance + final_move_distance

        return ScoredSolution(
            move_string=" ".join(tokens),
            total_drag=total_drag,
            total_move=total_move,
            total_cost=total_drag + total_move,
            move_data=move_data,
            initial_mouse_position=start_pos,
            initial_move_distance=initial_move_distance,
            inter_move_distance=inter_move_distance,
            final_mouse_target=final_mouse_target,
            final_move_distance=final_move_distance,
        )

    @staticmethod
    def score_file(
            path: str | Path,
            dedupe: bool = False,
            *,
            start_mouse_position: Sequence[float] | None = None,
            end_positions: Iterable[Sequence[float]] | None = None,
            end_next_puzzle: bool = False,
    ) -> FileScoreResult:
        path = Path(path)
        raw_lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if dedupe:
            raw_lines = list(dict.fromkeys(raw_lines))

        scored = [
            DpMouseSolver.solve_sequence(
                line,
                start_mouse_position=start_mouse_position,
                end_positions=end_positions,
                end_next_puzzle=end_next_puzzle,
            )
            for line in raw_lines
        ]
        scored.sort(key=lambda s: (s.total_cost, s.total_move, s.total_drag, s.move_string))
        return FileScoreResult(path=path, solutions=scored)


def normalize_move_sequence(move_string: str | Sequence[str]) -> list[str]:
    if isinstance(move_string, str):
        return [token for token in move_string.strip().split() if token]
    return [str(token).strip().upper() for token in move_string if str(token).strip()]


def format_solution_summary(solution: ScoredSolution) -> str:
    return (
        f"{solution.move_string}\n"
        f"  total={solution.total_cost:.3f}  drag={solution.total_drag:.3f}  move={solution.total_move:.3f}"
    )


def preview_ranked_solutions(
        path: str | Path,
        limit: int = 10,
        dedupe: bool = False,
        *,
        start_mouse_position: Sequence[float] | None = None,
        end_positions: Iterable[Sequence[float]] | None = None,
        end_next_puzzle: bool = False
) -> list[str]:
    result = DpMouseSolver.score_file(
        path,
        dedupe=dedupe,
        start_mouse_position=start_mouse_position,
        end_positions=end_positions,
        end_next_puzzle=end_next_puzzle,
    )
    return [format_solution_summary(solution) for solution in result.solutions[:limit]]


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank Chuzzle solution strings by exact DP mouse cost."
    )
    parser.add_argument(
        "path",
        help="Path to a text file containing one solution sequence per line.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many ranked solutions to print.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop duplicate lines before scoring.",
    )

    parser.add_argument(
        "--start",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Optional initial mouse position, for example: --start -1 5",
    )
    parser.add_argument(
        "--end",
        nargs=2,
        type=float,
        action="append",
        metavar=("X", "Y"),
        help=(
            "Allowed final mouse position. Can be provided multiple times. "
            "Example: --end 1 3 --end 2 3 --end 3 3 --end 4 3"
        ),
    )
    parser.add_argument(
        "--end-next-puzzle",
        "--use-next-puzzle-targets",
        dest="end_next_puzzle",
        action="store_true",
        help=(
            "Use the default next-puzzle banner targets "
            "(1,3), (2,3), (3,3), (4,3). This overrides any --end values."
        ),
    )

    args = parser.parse_args()

    for line in preview_ranked_solutions(
            args.path,
            limit=args.top,
            dedupe=args.dedupe,
            start_mouse_position=args.start,
            end_positions=args.end,
            end_next_puzzle=args.end_next_puzzle,
    ):
        print(line)


if __name__ == "__main__":
    _main()
