from __future__ import annotations

import math
from dataclasses import dataclass
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

EPSILON = 1e-12
TERNARY_SEARCH_STEPS = 70

Point = tuple[float, float]


@dataclass(slots=True, frozen=True)
class ParsedMove:
    token: str
    axis: str
    line_count: int
    lines: tuple[int, int]
    amount: int


@dataclass(slots=True, frozen=True)
class MoveCandidate:
    token: str
    axis: str
    amount: int
    selected_line: int
    displacement: int
    free_drag: bool
    click_down: Point
    lock_point: Point
    exact_release: Point
    min_drag_distance: float


@dataclass(slots=True, frozen=True)
class TransitionChoice:
    total_cost: float
    drag_distance: float
    move_to_next_distance: float
    release: Point


@dataclass(slots=True, frozen=True)
class TerminalChoice:
    total_cost: float
    drag_distance: float
    final_move_distance: float
    release: Point
    target: Point | None


@dataclass(slots=True, frozen=True)
class CoreStep:
    token: str
    selected_line: int
    displacement: int
    free_drag: bool
    click_down: Point
    lock_point: Point
    release: Point
    drag_distance: float
    move_distance: float
    total_distance_to_here: float


@dataclass(slots=True, frozen=True)
class CoreSolution:
    move_string: str
    total_drag: float
    total_move: float
    total_cost: float
    steps: tuple[CoreStep, ...]
    initial_mouse_position: Point | None
    initial_move_distance: float
    inter_move_distance: float
    final_mouse_target: Point | None
    final_move_distance: float


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


def normalize_point(point: Sequence[float] | None) -> Point | None:
    if point is None:
        return None
    if len(point) != 2:
        raise ValueError(f"Point must have exactly 2 coordinates, got: {point!r}")
    return float(point[0]), float(point[1])


def normalize_points(points: Iterable[Sequence[float]] | None) -> tuple[Point, ...]:
    if points is None:
        return ()

    out: list[Point] = []
    for point in points:
        if len(point) != 2:
            raise ValueError(f"Point must have exactly 2 coordinates, got: {point!r}")
        out.append((float(point[0]), float(point[1])))
    return tuple(out)


def euclidean_distance(p1: Point, p2: Point) -> float:
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.hypot(dx, dy)


def get_displacements(amount: int) -> tuple[int, ...]:
    wrapped_amount = amount % GRID_SIZE
    if wrapped_amount == 0:
        return (0,)
    return wrapped_amount, -(GRID_SIZE - wrapped_amount)


def parse_move(token: str) -> ParsedMove:
    token = token.strip().upper()
    if not token:
        raise ValueError("Empty move token")

    axis = token[0]
    if axis != "R" and axis != "C":
        raise ValueError(f"Move must start with R or C: {token!r}")

    payload = token[1:]
    if not payload.isdigit():
        raise ValueError(f"Move payload must be numeric: {token!r}")

    if len(payload) == 2:
        line0 = int(payload[0])
        line1 = -1
        line_count = 1
        amount = int(payload[1])
    elif len(payload) == 3:
        line0 = int(payload[0])
        line1 = int(payload[1])
        line_count = 2
        amount = int(payload[2])
    else:
        raise ValueError(
            f"Unsupported move format {token!r}. Expected Rxy / Cxy or Rxyz / Cxyz."
        )

    if line0 < 0 or line0 >= GRID_SIZE:
        raise ValueError(f"Move line out of range 0..{GRID_SIZE - 1}: {token!r}")
    if line_count == 2 and (line1 < 0 or line1 >= GRID_SIZE):
        raise ValueError(f"Move line out of range 0..{GRID_SIZE - 1}: {token!r}")

    return ParsedMove(
        token=token,
        axis=axis,
        line_count=line_count,
        lines=(line0, line1),
        amount=amount % GRID_SIZE,
    )


def is_free_drag(displacement: int, lock_threshold: float, free_drag_min_displacement: int) -> bool:
    displacement_abs = abs(displacement)
    return displacement_abs >= free_drag_min_displacement and displacement_abs > lock_threshold


def build_candidates(
        move: ParsedMove,
        lock_threshold: float,
        free_drag_min_displacement: int,
) -> list[MoveCandidate]:
    candidates: list[MoveCandidate] = []
    displacements = get_displacements(move.amount)

    line_enabled = [False] * GRID_SIZE
    line_enabled[move.lines[0]] = True
    if move.line_count == 2 and move.lines[1] >= 0:
        line_enabled[move.lines[1]] = True

    for selected_line in range(GRID_SIZE):
        if not line_enabled[selected_line]:
            continue

        for anchor in range(GRID_SIZE):
            for displacement in displacements:
                free_drag = is_free_drag(displacement, lock_threshold, free_drag_min_displacement)

                if displacement > 0:
                    sign = 1.0
                elif displacement < 0:
                    sign = -1.0
                else:
                    sign = 0.0

                if move.axis == "R":
                    click_down = (float(anchor), float(selected_line))
                    exact_release = (float(anchor + displacement), float(selected_line))
                    if free_drag:
                        lock_point = (float(anchor) + sign * lock_threshold, float(selected_line))
                    else:
                        lock_point = exact_release
                else:
                    click_down = (float(selected_line), float(anchor))
                    exact_release = (float(selected_line), float(anchor + displacement))
                    if free_drag:
                        lock_point = (float(selected_line), float(anchor) + sign * lock_threshold)
                    else:
                        lock_point = exact_release

                candidates.append(
                    MoveCandidate(
                        token=move.token,
                        axis=move.axis,
                        amount=move.amount,
                        selected_line=selected_line,
                        displacement=displacement,
                        free_drag=free_drag,
                        click_down=click_down,
                        lock_point=lock_point,
                        exact_release=exact_release,
                        min_drag_distance=float(abs(displacement)),
                    )
                )

    return candidates


def optimize_scalar_on_segment(lo: float, hi: float, objective) -> float:
    if abs(hi - lo) <= EPSILON:
        return lo

    left = lo
    right = hi

    for _ in range(TERNARY_SEARCH_STEPS):
        m1 = left + (right - left) / 3.0
        m2 = right - (right - left) / 3.0
        if objective(m1) <= objective(m2):
            right = m2
        else:
            left = m1

    return (left + right) * 0.5


def best_point_on_vertical_line(x_value: float, lock_point: Point, target_point: Point) -> Point:
    lo = min(lock_point[1], target_point[1])
    hi = max(lock_point[1], target_point[1])

    def objective(y_value: float) -> float:
        release = (x_value, y_value)
        return euclidean_distance(lock_point, release) + euclidean_distance(release, target_point)

    y_best = optimize_scalar_on_segment(lo, hi, objective)
    return x_value, y_best


def best_point_on_horizontal_line(y_value: float, lock_point: Point, target_point: Point) -> Point:
    lo = min(lock_point[0], target_point[0])
    hi = max(lock_point[0], target_point[0])

    def objective(x_value: float) -> float:
        release = (x_value, y_value)
        return euclidean_distance(lock_point, release) + euclidean_distance(release, target_point)

    x_best = optimize_scalar_on_segment(lo, hi, objective)
    return x_best, y_value


def optimize_release_to_point(candidate: MoveCandidate, point: Point) -> TransitionChoice:
    target = point

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
        release = best_point_on_vertical_line(candidate.exact_release[0], candidate.lock_point, target)
    else:
        release = best_point_on_horizontal_line(candidate.exact_release[1], candidate.lock_point, target)

    drag_distance = (
            euclidean_distance(candidate.click_down, candidate.lock_point)
            + euclidean_distance(candidate.lock_point, release)
    )
    move_distance = euclidean_distance(release, target)

    return TransitionChoice(
        total_cost=drag_distance + move_distance,
        drag_distance=drag_distance,
        move_to_next_distance=move_distance,
        release=release,
    )


def build_transition_matrix(
        prev_candidates: list[MoveCandidate],
        next_candidates: list[MoveCandidate],
) -> list[list[TransitionChoice]]:
    matrix: list[list[TransitionChoice]] = []

    for prev_candidate in prev_candidates:
        row: list[TransitionChoice] = []
        for next_candidate in next_candidates:
            row.append(optimize_release_to_point(prev_candidate, next_candidate.click_down))
        matrix.append(row)

    return matrix


def best_terminal_choice(
        candidate: MoveCandidate,
        end_targets: Sequence[Point],
) -> TerminalChoice:
    if not end_targets:
        return TerminalChoice(
            total_cost=candidate.min_drag_distance,
            drag_distance=candidate.min_drag_distance,
            final_move_distance=0.0,
            release=candidate.exact_release,
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
            target=target,
        )

        if best_choice is None or choice.total_cost < best_choice.total_cost:
            best_choice = choice

    if best_choice is None:
        raise RuntimeError("Internal error: failed to choose a terminal target")

    return best_choice


def build_core_step(
        candidate: MoveCandidate,
        release: Point,
        drag_distance: float,
        move_distance: float,
        total_distance_to_here: float,
) -> CoreStep:
    return CoreStep(
        token=candidate.token,
        selected_line=candidate.selected_line,
        displacement=candidate.displacement,
        free_drag=candidate.free_drag,
        click_down=candidate.click_down,
        lock_point=candidate.lock_point,
        release=release,
        drag_distance=drag_distance,
        move_distance=move_distance,
        total_distance_to_here=total_distance_to_here,
    )


def convert_core_step_to_scored_move(step: CoreStep) -> ScoredMove:
    return ScoredMove(
        move=step.token,
        click_down=[step.click_down[0], step.click_down[1]],
        lock_point=[step.lock_point[0], step.lock_point[1]],
        release=[step.release[0], step.release[1]],
        path_points=[
            [step.click_down[0], step.click_down[1]],
            [step.lock_point[0], step.lock_point[1]],
            [step.release[0], step.release[1]],
        ],
        selected_line=step.selected_line,
        displacement=step.displacement,
        free_drag=step.free_drag,
        drag_distance=step.drag_distance,
        move_distance=step.move_distance,
        total_distance_to_here=step.total_distance_to_here,
    )


def convert_core_solution_to_scored_solution(core: CoreSolution) -> ScoredSolution:
    move_data: list[ScoredMove] = []
    for step in core.steps:
        move_data.append(convert_core_step_to_scored_move(step))

    return ScoredSolution(
        move_string=core.move_string,
        total_drag=core.total_drag,
        total_move=core.total_move,
        total_cost=core.total_cost,
        move_data=move_data,
        initial_mouse_position=core.initial_mouse_position,
        initial_move_distance=core.initial_move_distance,
        inter_move_distance=core.inter_move_distance,
        final_mouse_target=core.final_mouse_target,
        final_move_distance=core.final_move_distance,
    )


def solve_sequence_core(
        tokens: Sequence[str],
        *,
        start_pos: Point | None,
        final_targets: Sequence[Point],
        lock_threshold: float,
        free_drag_min_displacement: int,
) -> CoreSolution:
    if not tokens:
        raise ValueError("Cannot solve an empty move sequence")

    candidate_layers: list[list[MoveCandidate]] = []
    for token in tokens:
        parsed = parse_move(token)
        candidate_layers.append(
            build_candidates(parsed, lock_threshold, free_drag_min_displacement)
        )

    sequence_len = len(tokens)

    if sequence_len == 1:
        only_candidates = candidate_layers[0]
        best_index = -1
        best_total_cost = math.inf
        best_terminal: TerminalChoice | None = None
        best_initial_move = 0.0

        for candidate_index in range(len(only_candidates)):
            candidate = only_candidates[candidate_index]
            terminal = best_terminal_choice(candidate, final_targets)

            if start_pos is None:
                initial_move_distance = 0.0
            else:
                initial_move_distance = euclidean_distance(start_pos, candidate.click_down)

            total_cost = initial_move_distance + terminal.total_cost
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_index = candidate_index
                best_terminal = terminal
                best_initial_move = initial_move_distance

        if best_index < 0 or best_terminal is None:
            raise RuntimeError("Internal error: failed to choose a solution")

        best_candidate = only_candidates[best_index]
        total_drag = best_terminal.drag_distance
        inter_move_distance = 0.0
        final_move_distance = best_terminal.final_move_distance
        total_move = best_initial_move + final_move_distance

        steps = (
            build_core_step(
                candidate=best_candidate,
                release=best_terminal.release,
                drag_distance=best_terminal.drag_distance,
                move_distance=best_initial_move,
                total_distance_to_here=best_initial_move + total_drag,
            ),
        )

        return CoreSolution(
            move_string=" ".join(tokens),
            total_drag=total_drag,
            total_move=total_move,
            total_cost=total_drag + total_move,
            steps=steps,
            initial_mouse_position=start_pos,
            initial_move_distance=best_initial_move,
            inter_move_distance=inter_move_distance,
            final_mouse_target=best_terminal.target,
            final_move_distance=final_move_distance,
        )

    transition_matrices: list[list[list[TransitionChoice]]] = []
    for move_index in range(sequence_len - 1):
        transition_matrices.append(
            build_transition_matrix(
                candidate_layers[move_index],
                candidate_layers[move_index + 1],
            )
        )

    dp_costs: list[list[float]] = []
    parents: list[list[int]] = []

    first_layer = candidate_layers[0]
    first_costs = [0.0] * len(first_layer)
    for candidate_index in range(len(first_layer)):
        if start_pos is None:
            first_costs[candidate_index] = 0.0
        else:
            first_costs[candidate_index] = euclidean_distance(start_pos, first_layer[candidate_index].click_down)

    dp_costs.append(first_costs)
    parents.append([-1] * len(first_layer))

    for move_index in range(1, sequence_len):
        prev_candidates = candidate_layers[move_index - 1]
        curr_candidates = candidate_layers[move_index]
        transitions = transition_matrices[move_index - 1]

        curr_costs = [math.inf] * len(curr_candidates)
        curr_parents = [-1] * len(curr_candidates)

        for curr_index in range(len(curr_candidates)):
            best_cost = math.inf
            best_parent = -1

            for prev_index in range(len(prev_candidates)):
                edge = transitions[prev_index][curr_index]
                candidate_cost = dp_costs[move_index - 1][prev_index] + edge.total_cost
                if candidate_cost < best_cost:
                    best_cost = candidate_cost
                    best_parent = prev_index

            curr_costs[curr_index] = best_cost
            curr_parents[curr_index] = best_parent

        dp_costs.append(curr_costs)
        parents.append(curr_parents)

    final_layer = candidate_layers[-1]
    terminal_choices: list[TerminalChoice] = []
    for candidate in final_layer:
        terminal_choices.append(best_terminal_choice(candidate, final_targets))

    best_final_index = -1
    best_final_total = math.inf

    for final_index in range(len(final_layer)):
        total_cost = dp_costs[-1][final_index] + terminal_choices[final_index].total_cost
        if total_cost < best_final_total:
            best_final_total = total_cost
            best_final_index = final_index

    if best_final_index < 0:
        raise RuntimeError("Internal error: failed to choose a final solution")

    chosen_indices = [0] * sequence_len
    chosen_indices[-1] = best_final_index

    for move_index in range(sequence_len - 1, 0, -1):
        chosen_indices[move_index - 1] = parents[move_index][chosen_indices[move_index]]

    chosen_candidates: list[MoveCandidate] = []
    for move_index in range(sequence_len):
        chosen_candidates.append(candidate_layers[move_index][chosen_indices[move_index]])

    chosen_releases: list[Point] = []
    chosen_drags: list[float] = []
    chosen_moves_between: list[float] = []

    for move_index in range(sequence_len - 1):
        transition = transition_matrices[move_index][chosen_indices[move_index]][chosen_indices[move_index + 1]]
        chosen_releases.append(transition.release)
        chosen_drags.append(transition.drag_distance)
        chosen_moves_between.append(transition.move_to_next_distance)

    terminal = terminal_choices[best_final_index]
    chosen_releases.append(terminal.release)
    chosen_drags.append(terminal.drag_distance)

    if start_pos is None:
        initial_move_distance = 0.0
    else:
        initial_move_distance = euclidean_distance(start_pos, chosen_candidates[0].click_down)

    steps_list: list[CoreStep] = []
    total_drag = 0.0
    inter_move_distance = 0.0

    for move_index in range(sequence_len):
        if move_index == 0:
            move_distance = initial_move_distance
        else:
            move_distance = chosen_moves_between[move_index - 1]
            inter_move_distance += move_distance

        total_drag += chosen_drags[move_index]
        total_distance_to_here = initial_move_distance + inter_move_distance + total_drag

        steps_list.append(
            build_core_step(
                candidate=chosen_candidates[move_index],
                release=chosen_releases[move_index],
                drag_distance=chosen_drags[move_index],
                move_distance=move_distance,
                total_distance_to_here=total_distance_to_here,
            )
        )

    final_move_distance = terminal.final_move_distance
    total_move = initial_move_distance + inter_move_distance + final_move_distance

    return CoreSolution(
        move_string=" ".join(tokens),
        total_drag=total_drag,
        total_move=total_move,
        total_cost=total_drag + total_move,
        steps=tuple(steps_list),
        initial_mouse_position=start_pos,
        initial_move_distance=initial_move_distance,
        inter_move_distance=inter_move_distance,
        final_mouse_target=terminal.target,
        final_move_distance=final_move_distance,
    )


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

    start_pos = normalize_point(start_mouse_position)
    if end_next_puzzle:
        final_targets = DEFAULT_NEXT_PUZZLE_TARGETS
    else:
        final_targets = normalize_points(end_positions)

    core = solve_sequence_core(
        tokens=tokens,
        start_pos=start_pos,
        final_targets=final_targets,
        lock_threshold=lock_threshold,
        free_drag_min_displacement=free_drag_min_displacement,
    )
    return convert_core_solution_to_scored_solution(core)