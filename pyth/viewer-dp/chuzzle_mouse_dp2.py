from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

GRID_SIZE = 6
DEFAULT_LOCK_THRESHOLD = 1.0 / 1.0
DEFAULT_FREE_DRAG_MIN_DISPLACEMENT = 2
DEFAULT_NEXT_PUZZLE_TARGETS: tuple[tuple[float, float], ...] = (
    (1.0, 3.0),
    (2.0, 3.0),
    (3.0, 3.0),
    (4.0, 3.0),
)


def normalize_move_sequence(move_string: str | Sequence[str]) -> list[str]:
    if isinstance(move_string, str):
        return [token for token in move_string.strip().split() if token]
    return [str(token).strip().upper() for token in move_string if str(token).strip()]


def euclidean_distance(p1: Sequence[float], p2: Sequence[float]) -> float:
    dx: float = p1[0] - p2[0]
    dy: float = p1[1] - p2[1]
    return math.hypot(dx, dy)


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
    click_down: tuple[float, float]
    exact_release: tuple[float, float]
    lock_point: tuple[float, float]
    selected_line: int
    displacement: int
    free_drag: bool
    min_drag_distance: float


@dataclass(frozen=True)
class TransitionChoice:
    total_cost: float
    drag_distance: float
    move_to_next_distance: float
    release: tuple[float, float]


@dataclass(frozen=True)
class TerminalChoice:
    total_cost: float
    drag_distance: float
    final_move_distance: float
    release: tuple[float, float]
    target: tuple[float, float] | None


@dataclass
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


def _is_free_drag(displacement: int, lock_threshold: float, free_drag_min_displacement: int) -> bool:
    return abs(displacement) >= free_drag_min_displacement and abs(displacement) > lock_threshold


@lru_cache(maxsize=None)
def enumerate_candidates(
    token: str,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> tuple[MoveCandidate, ...]:
    move = parse_move(token)
    candidates: dict[tuple[tuple[float, float], int, int], MoveCandidate] = {}

    for line in move.lines:
        for anchor in range(GRID_SIZE):
            for displacement in get_displacements(move.amount):
                free_drag = _is_free_drag(displacement, lock_threshold, free_drag_min_displacement)
                sign = 0 if displacement == 0 else (1 if displacement > 0 else -1)

                if move.axis == "R":
                    click_down = (float(anchor), float(line))
                    exact_release = (float(anchor + displacement), float(line))
                    if free_drag:
                        lock_point = (float(anchor) + sign * lock_threshold, float(line))
                    else:
                        lock_point = exact_release
                else:
                    click_down = (float(line), float(anchor))
                    exact_release = (float(line), float(anchor + displacement))
                    if free_drag:
                        lock_point = (float(line), float(anchor) + sign * lock_threshold)
                    else:
                        lock_point = exact_release

                candidate = MoveCandidate(
                    token=move.token,
                    axis=move.axis,
                    lines=move.lines,
                    amount=move.amount,
                    click_down=click_down,
                    exact_release=exact_release,
                    lock_point=lock_point,
                    selected_line=line,
                    displacement=displacement,
                    free_drag=free_drag,
                    min_drag_distance=float(abs(displacement)),
                )
                key = (candidate.click_down, candidate.selected_line, candidate.displacement)
                candidates[key] = candidate

    ordered = sorted(
        candidates.values(),
        key=lambda c: (
            c.selected_line,
            c.click_down[1],
            c.click_down[0],
            c.displacement,
        ),
    )
    return tuple(ordered)


def _best_point_on_vertical_line(
    x_value: float,
    lock_point: tuple[float, float],
    target_point: tuple[float, float],
) -> tuple[float, float]:
    _, py = lock_point
    _, qy = target_point
    lo = min(py, qy)
    hi = max(py, qy)
    if abs(hi - lo) <= 1e-12:
        return x_value, lo

    def objective(y_value: float) -> float:
        release = (x_value, y_value)
        return euclidean_distance(lock_point, release) + euclidean_distance(release, target_point)

    for _ in range(70):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) <= objective(m2):
            hi = m2
        else:
            lo = m1

    y_best = (lo + hi) / 2.0
    return x_value, y_best


def _best_point_on_horizontal_line(
    y_value: float,
    lock_point: tuple[float, float],
    target_point: tuple[float, float],
) -> tuple[float, float]:
    px, _ = lock_point
    qx, _ = target_point
    lo = min(px, qx)
    hi = max(px, qx)
    if abs(hi - lo) <= 1e-12:
        return lo, y_value

    def objective(x_value: float) -> float:
        release = (x_value, y_value)
        return euclidean_distance(lock_point, release) + euclidean_distance(release, target_point)

    for _ in range(70):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if objective(m1) <= objective(m2):
            hi = m2
        else:
            lo = m1

    x_best = (lo + hi) / 2.0
    return x_best, y_value


def optimize_release_to_point(candidate: MoveCandidate, point: Sequence[float]) -> TransitionChoice:
    target = (float(point[0]), float(point[1]))

    if not candidate.free_drag:
        release = candidate.exact_release
        drag_distance = candidate.min_drag_distance
        move_distance = euclidean_distance(release, target)
        return TransitionChoice(
            total_cost=drag_distance + move_distance,
            drag_distance=drag_distance,
            move_to_next_distance=move_distance,
            release=release,
        )

    if candidate.axis == "R":
        release = _best_point_on_vertical_line(candidate.exact_release[0], candidate.lock_point, target)
    else:
        release = _best_point_on_horizontal_line(candidate.exact_release[1], candidate.lock_point, target)

    drag_distance = euclidean_distance(candidate.click_down, candidate.lock_point) + euclidean_distance(candidate.lock_point, release)
    move_distance = euclidean_distance(release, target)
    return TransitionChoice(
        total_cost=drag_distance + move_distance,
        drag_distance=drag_distance,
        move_to_next_distance=move_distance,
        release=release,
    )


@lru_cache(maxsize=None)
def transition_plan(
    prev_token: str,
    next_token: str,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> tuple[tuple[TransitionChoice, ...], ...]:
    prev_candidates = enumerate_candidates(prev_token, lock_threshold, free_drag_min_displacement)
    next_candidates = enumerate_candidates(next_token, lock_threshold, free_drag_min_displacement)

    rows: list[tuple[TransitionChoice, ...]] = []
    for prev_candidate in prev_candidates:
        row: list[TransitionChoice] = []
        for next_candidate in next_candidates:
            row.append(optimize_release_to_point(prev_candidate, next_candidate.click_down))
        rows.append(tuple(row))
    return tuple(rows)


def best_terminal_choice(
    candidate: MoveCandidate,
    end_targets: Sequence[Sequence[float]],
) -> TerminalChoice:
    if not end_targets:
        release = candidate.exact_release
        drag_distance = candidate.min_drag_distance
        return TerminalChoice(
            total_cost=drag_distance,
            drag_distance=drag_distance,
            final_move_distance=0.0,
            release=release,
            target=None,
        )

    best_choice: TerminalChoice | None = None
    for target in end_targets:
        transition = optimize_release_to_point(candidate, target)
        choice = TerminalChoice(
            total_cost=transition.total_cost,
            drag_distance=transition.drag_distance,
            final_move_distance=transition.move_to_next_distance,
            release=transition.release,
            target=(float(target[0]), float(target[1])),
        )
        if best_choice is None or choice.total_cost < best_choice.total_cost:
            best_choice = choice

    assert best_choice is not None
    return best_choice



"""
Dynamic-programming solver with a two-phase drag model.

Model:
1. Click-down begins on a legal cell in a row/column.
2. If abs(displacement) < free_drag_min_displacement, the move behaves like the old solver.
3. If abs(displacement) >= free_drag_min_displacement, the drag "locks" after lock_threshold
   along the primary axis.
4. After lock, the mouse may end anywhere on the final target line:
     - row drag    -> any y on x = target_x
     - column drag -> any x on y = target_y
5. The optimal release point depends on the next click target, so edge costs include:
     drag for previous move + reposition to next move's click-down.
"""

def solve_sequence(
    move_string: str | Sequence[str],
    *,
    start_mouse_position: Sequence[float] | None = None,
    end_positions: Iterable[Sequence[float]] | None = None,
    end_next_puzzle: bool = False,
    lock_threshold: float = DEFAULT_LOCK_THRESHOLD,
    free_drag_min_displacement: int = DEFAULT_FREE_DRAG_MIN_DISPLACEMENT,
) -> ScoredSolution:
    tokens = normalize_move_sequence(move_string)
    if not tokens:
        raise ValueError("Cannot solve an empty move sequence")

    start_pos = start_mouse_position
    if end_next_puzzle:
        final_targets = DEFAULT_NEXT_PUZZLE_TARGETS
    else:
        final_targets = end_positions

    candidate_layers = [
        enumerate_candidates(token, lock_threshold, free_drag_min_displacement)
        for token in tokens
    ]

    if len(tokens) == 1:
        terminal_choices = [best_terminal_choice(candidate, final_targets) for candidate in candidate_layers[0]]
        best_index = min(
            range(len(candidate_layers[0])),
            key=lambda i: (0.0 if start_pos is None else euclidean_distance(start_pos, candidate_layers[0][i].click_down))
            + terminal_choices[i].total_cost,
        )

        candidate = candidate_layers[0][best_index]
        terminal = terminal_choices[best_index]
        initial_move_distance = 0.0 if start_pos is None else euclidean_distance(start_pos, candidate.click_down)
        total_drag = terminal.drag_distance
        inter_move_distance = 0.0
        final_move_distance = terminal.final_move_distance
        total_move = initial_move_distance + inter_move_distance + final_move_distance

        move_data = [
            ScoredMove(
                move=candidate.token,
                click_down=[candidate.click_down[0], candidate.click_down[1]],
                lock_point=[candidate.lock_point[0], candidate.lock_point[1]],
                release=[terminal.release[0], terminal.release[1]],
                path_points=[
                    [candidate.click_down[0], candidate.click_down[1]],
                    [candidate.lock_point[0], candidate.lock_point[1]],
                    [terminal.release[0], terminal.release[1]],
                ],
                selected_line=candidate.selected_line,
                displacement=candidate.displacement,
                free_drag=candidate.free_drag,
                drag_distance=terminal.drag_distance,
                move_distance=initial_move_distance,
                total_distance_to_here=initial_move_distance + total_drag,
            )
        ]

        return ScoredSolution(
            move_string=" ".join(tokens),
            total_drag=total_drag,
            total_move=total_move,
            total_cost=total_drag + total_move,
            move_data=move_data,
            initial_mouse_position=start_pos,
            initial_move_distance=initial_move_distance,
            inter_move_distance=inter_move_distance,
            final_mouse_target=terminal.target,
            final_move_distance=final_move_distance,
        )

    dp_costs: list[list[float]] = [[] for _ in tokens]
    parents: list[list[int]] = [[-1] * len(candidate_layers[0])] + [[] for _ in tokens[1:]]

    dp_costs[0] = [
        0.0 if start_pos is None else euclidean_distance(start_pos, candidate.click_down)
        for candidate in candidate_layers[0]
    ]

    for move_index in range(1, len(tokens)):
        prev_token = tokens[move_index - 1]
        current_token = tokens[move_index]
        prev_candidates = candidate_layers[move_index - 1]
        current_candidates = candidate_layers[move_index]
        trans = transition_plan(prev_token, current_token, lock_threshold, free_drag_min_displacement)

        next_costs = [math.inf] * len(current_candidates)
        next_parents = [-1] * len(current_candidates)

        for curr_idx in range(len(current_candidates)):
            best_cost = math.inf
            best_parent = -1
            for prev_idx in range(len(prev_candidates)):
                edge = trans[prev_idx][curr_idx]
                cost = dp_costs[move_index - 1][prev_idx] + edge.total_cost
                if cost < best_cost:
                    best_cost = cost
                    best_parent = prev_idx
            next_costs[curr_idx] = best_cost
            next_parents[curr_idx] = best_parent

        dp_costs[move_index] = next_costs
        parents[move_index] = next_parents

    final_layer = candidate_layers[-1]
    terminal_choices = [best_terminal_choice(candidate, final_targets) for candidate in final_layer]

    best_final_index = min(
        range(len(final_layer)),
        key=lambda idx: dp_costs[-1][idx] + terminal_choices[idx].total_cost,
    )

    chosen_indices = [0] * len(tokens)
    chosen_indices[-1] = best_final_index
    for move_index in range(len(tokens) - 1, 0, -1):
        chosen_indices[move_index - 1] = parents[move_index][chosen_indices[move_index]]

    chosen_candidates = [candidate_layers[i][chosen_indices[i]] for i in range(len(tokens))]

    chosen_releases: list[tuple[float, float]] = []
    chosen_drags: list[float] = []
    chosen_moves_between: list[float] = []

    for move_index in range(len(tokens) - 1):
        prev_candidate = chosen_candidates[move_index]
        next_candidate = chosen_candidates[move_index + 1]
        edge = optimize_release_to_point(prev_candidate, next_candidate.click_down)
        chosen_releases.append(edge.release)
        chosen_drags.append(edge.drag_distance)
        chosen_moves_between.append(edge.move_to_next_distance)

    terminal = best_terminal_choice(chosen_candidates[-1], final_targets)
    chosen_releases.append(terminal.release)
    chosen_drags.append(terminal.drag_distance)

    move_data: list[ScoredMove] = []
    total_drag = 0.0
    initial_move_distance = 0.0 if start_pos is None else euclidean_distance(start_pos, chosen_candidates[0].click_down)
    inter_move_distance = 0.0

    for move_index, candidate in enumerate(chosen_candidates):
        if move_index == 0:
            move_distance = initial_move_distance
        else:
            move_distance = chosen_moves_between[move_index - 1]
            inter_move_distance += move_distance

        total_drag += chosen_drags[move_index]
        total_cost_to_here = initial_move_distance + inter_move_distance + total_drag
        release = chosen_releases[move_index]

        move_data.append(
            ScoredMove(
                move=candidate.token,
                click_down=[candidate.click_down[0], candidate.click_down[1]],
                lock_point=[candidate.lock_point[0], candidate.lock_point[1]],
                release=[release[0], release[1]],
                path_points=[
                    [candidate.click_down[0], candidate.click_down[1]],
                    [candidate.lock_point[0], candidate.lock_point[1]],
                    [release[0], release[1]],
                ],
                selected_line=candidate.selected_line,
                displacement=candidate.displacement,
                free_drag=candidate.free_drag,
                drag_distance=chosen_drags[move_index],
                move_distance=move_distance,
                total_distance_to_here=total_cost_to_here,
            )
        )

    final_move_distance = terminal.final_move_distance
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
        final_mouse_target=terminal.target,
        final_move_distance=final_move_distance,
    )


