#include "../include/chuzzle_mouse_api.h"

#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHUZZLE_GRID_SIZE 6
#define CHUZZLE_EPSILON 1e-12
#define CHUZZLE_TERNARY_SEARCH_STEPS 70
#define CHUZZLE_MOVE_TOKEN_CAPACITY 5

typedef struct ParsedMove {
    char token[CHUZZLE_MOVE_TOKEN_CAPACITY];
    char axis;
    int line_count;
    int lines[2];
    int amount;
} ParsedMove;

typedef struct MoveCandidate {
    char token[CHUZZLE_MOVE_TOKEN_CAPACITY];
    char axis;
    int amount;
    int selected_line;
    int displacement;
    int free_drag;
    ChuzzlePoint click_down;
    ChuzzlePoint lock_point;
    ChuzzlePoint exact_release;
    double min_drag_distance;
} MoveCandidate;

typedef struct TransitionChoice {
    double total_cost;
    double drag_distance;
    double move_to_next_distance;
    ChuzzlePoint release;
} TransitionChoice;

typedef struct TerminalChoice {
    double total_cost;
    double drag_distance;
    double final_move_distance;
    ChuzzlePoint release;
    int has_target;
    ChuzzlePoint target;
} TerminalChoice;

typedef struct CandidateLayer {
    MoveCandidate *items;
    size_t count;
} CandidateLayer;

static const ChuzzlePoint CHUZZLE_DEFAULT_NEXT_PUZZLE_TARGETS[4] = {
        {1.0, 3.0},
        {2.0, 3.0},
        {3.0, 3.0},
        {4.0, 3.0},
};

static void chuzzle_clear_solution(ChuzzleSolution *solution) {
    if (solution == NULL) {
        return;
    }
    memset(solution, 0, sizeof(*solution));
}

static double chuzzle_distance(ChuzzlePoint a, ChuzzlePoint b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return hypot(dx, dy);
}

static char *chuzzle_strdup_local(const char *text) {
    size_t len;
    char *copy;

    if (text == NULL) {
        return NULL;
    }

    len = strlen(text);
    copy = (char *) malloc(len + 1);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, len + 1);
    return copy;
}

static int chuzzle_tokenize_move_string(
        const char *move_string,
        char (**out_tokens)[CHUZZLE_MOVE_TOKEN_CAPACITY],
        size_t *out_count,
        char **out_normalized_string) {
    size_t capacity = 0;
    size_t count = 0;
    char(*tokens)[CHUZZLE_MOVE_TOKEN_CAPACITY] = NULL;
    const unsigned char *cursor;

    char *normalized = NULL;
    size_t normalized_length = 0;
    size_t i;

    if (move_string == NULL || out_tokens == NULL || out_count == NULL || out_normalized_string == NULL) {
        return CHUZZLE_ERR_NULL_ARGUMENT;
    }

    cursor = (const unsigned char *) move_string;
    while (*cursor != '\0') {
        size_t token_length = 0;
        char token[CHUZZLE_MOVE_TOKEN_CAPACITY];

        while (*cursor != '\0' && isspace(*cursor)) {
            cursor++;
        }
        if (*cursor == '\0') {
            break;
        }

        while (*cursor != '\0' && !isspace(*cursor)) {
            if (token_length + 1 >= CHUZZLE_MOVE_TOKEN_CAPACITY) {
                free(tokens);
                return CHUZZLE_ERR_PARSE_MOVE;
            }
            token[token_length++] = (char) toupper(*cursor);
            cursor++;
        }
        token[token_length] = '\0';

        if (capacity == count) {
            size_t new_capacity = (capacity == 0) ? 8 : (capacity * 2);
            char(*new_tokens)[CHUZZLE_MOVE_TOKEN_CAPACITY] =
                    (char(*)[CHUZZLE_MOVE_TOKEN_CAPACITY]) realloc(tokens, new_capacity * sizeof(*tokens));
            if (new_tokens == NULL) {
                free(tokens);
                return CHUZZLE_ERR_INVALID_ARGUMENT;
            }
            tokens = new_tokens;
            capacity = new_capacity;
        }

        memcpy(tokens[count], token, CHUZZLE_MOVE_TOKEN_CAPACITY);
        count++;
    }

    if (count == 0) {
        free(tokens);
        return CHUZZLE_ERR_INVALID_ARGUMENT;
    }

    for (i = 0; i < count; ++i) {
        normalized_length += strlen(tokens[i]);
        if (i + 1 < count) {
            normalized_length += 1;
        }
    }

    normalized = (char *) malloc(normalized_length + 1);
    if (normalized == NULL) {
        free(tokens);
        return CHUZZLE_ERR_INVALID_ARGUMENT;
    }

    normalized[0] = '\0';
    for (i = 0; i < count; ++i) {
        strcat(normalized, tokens[i]);
        if (i + 1 < count) {
            strcat(normalized, " ");
        }
    }

    *out_tokens = tokens;
    *out_count = count;
    *out_normalized_string = normalized;
    return CHUZZLE_OK;
}

static int chuzzle_parse_move_token(const char token[CHUZZLE_MOVE_TOKEN_CAPACITY], ParsedMove *out_move) {
    size_t payload_length;
    ParsedMove move;
    int i;

    if (token == NULL || out_move == NULL) {
        return CHUZZLE_ERR_NULL_ARGUMENT;
    }

    memset(&move, 0, sizeof(move));
    strncpy(move.token, token, CHUZZLE_MOVE_TOKEN_CAPACITY - 1);
    move.token[CHUZZLE_MOVE_TOKEN_CAPACITY - 1] = '\0';

    if (move.token[0] == '\0') {
        return CHUZZLE_ERR_PARSE_MOVE;
    }

    move.axis = move.token[0];
    if (move.axis != 'R' && move.axis != 'C') {
        return CHUZZLE_ERR_PARSE_MOVE;
    }

    payload_length = strlen(move.token + 1);
    for (i = 1; move.token[i] != '\0'; ++i) {
        if (!isdigit((unsigned char) move.token[i])) {
            return CHUZZLE_ERR_PARSE_MOVE;
        }
    }

    if (payload_length == 2) {
        move.lines[0] = move.token[1] - '0';
        move.lines[1] = -1;
        move.line_count = 1;
        move.amount = move.token[2] - '0';
    } else if (payload_length == 3) {
        move.lines[0] = move.token[1] - '0';
        move.lines[1] = move.token[2] - '0';
        move.line_count = 2;
        move.amount = move.token[3] - '0';
    } else {
        return CHUZZLE_ERR_PARSE_MOVE;
    }

    if (move.lines[0] < 0 || move.lines[0] >= CHUZZLE_GRID_SIZE) {
        return CHUZZLE_ERR_PARSE_MOVE;
    }
    if (move.line_count == 2 && (move.lines[1] < 0 || move.lines[1] >= CHUZZLE_GRID_SIZE)) {
        return CHUZZLE_ERR_PARSE_MOVE;
    }

    move.amount %= CHUZZLE_GRID_SIZE;
    *out_move = move;
    return CHUZZLE_OK;
}

static int chuzzle_get_displacements(int amount, int out_displacements[2], int *out_count) {
    int wrapped_amount;

    if (out_displacements == NULL || out_count == NULL) {
        return CHUZZLE_ERR_NULL_ARGUMENT;
    }

    wrapped_amount = amount % CHUZZLE_GRID_SIZE;
    if (wrapped_amount < 0) {
        wrapped_amount += CHUZZLE_GRID_SIZE;
    }

    if (wrapped_amount == 0) {
        out_displacements[0] = 0;
        *out_count = 1;
        return CHUZZLE_OK;
    }

    out_displacements[0] = wrapped_amount;
    out_displacements[1] = -(CHUZZLE_GRID_SIZE - wrapped_amount);
    *out_count = 2;
    return CHUZZLE_OK;
}

static int chuzzle_is_free_drag(int displacement, double lock_threshold, int free_drag_min_displacement) {
    int displacement_abs = displacement < 0 ? -displacement : displacement;
    return displacement_abs >= free_drag_min_displacement && (double) displacement_abs > lock_threshold;
}

static int chuzzle_build_candidates(
        const ParsedMove *move,
        double lock_threshold,
        int free_drag_min_displacement,
        CandidateLayer *out_layer) {
    int displacements[2];
    int displacement_count = 0;
    int line_enabled[CHUZZLE_GRID_SIZE];
    size_t candidate_count = 0;
    MoveCandidate *items = NULL;
    size_t index = 0;
    int selected_line;
    int anchor;
    int displacement_index;

    if (move == NULL || out_layer == NULL) {
        return CHUZZLE_ERR_NULL_ARGUMENT;
    }

    memset(line_enabled, 0, sizeof(line_enabled));
    line_enabled[move->lines[0]] = 1;
    if (move->line_count == 2 && move->lines[1] >= 0) {
        line_enabled[move->lines[1]] = 1;
    }

    if (chuzzle_get_displacements(move->amount, displacements, &displacement_count) != CHUZZLE_OK) {
        return CHUZZLE_ERR_INVALID_ARGUMENT;
    }

    candidate_count = (size_t) move->line_count * (size_t) CHUZZLE_GRID_SIZE * (size_t) displacement_count;
    items = (MoveCandidate *) malloc(candidate_count * sizeof(*items));
    if (items == NULL) {
        return CHUZZLE_ERR_INVALID_ARGUMENT;
    }

    for (selected_line = 0; selected_line < CHUZZLE_GRID_SIZE; ++selected_line) {
        if (!line_enabled[selected_line]) {
            continue;
        }

        for (anchor = 0; anchor < CHUZZLE_GRID_SIZE; ++anchor) {
            for (displacement_index = 0; displacement_index < displacement_count; ++displacement_index) {
                int displacement = displacements[displacement_index];
                int free_drag = chuzzle_is_free_drag(displacement, lock_threshold, free_drag_min_displacement);
                double sign = 0.0;
                MoveCandidate candidate;

                if (displacement > 0) {
                    sign = 1.0;
                } else if (displacement < 0) {
                    sign = -1.0;
                }

                memset(&candidate, 0, sizeof(candidate));
                strncpy(candidate.token, move->token, CHUZZLE_MOVE_TOKEN_CAPACITY - 1);
                candidate.axis = move->axis;
                candidate.amount = move->amount;
                candidate.selected_line = selected_line;
                candidate.displacement = displacement;
                candidate.free_drag = free_drag;
                candidate.min_drag_distance = (double) (displacement < 0 ? -displacement : displacement);

                if (move->axis == 'R') {
                    candidate.click_down.x = (double) anchor;
                    candidate.click_down.y = (double) selected_line;
                    candidate.exact_release.x = (double) (anchor + displacement);
                    candidate.exact_release.y = (double) selected_line;

                    if (free_drag) {
                        candidate.lock_point.x = (double) anchor + sign * lock_threshold;
                        candidate.lock_point.y = (double) selected_line;
                    } else {
                        candidate.lock_point = candidate.exact_release;
                    }
                } else {
                    candidate.click_down.x = (double) selected_line;
                    candidate.click_down.y = (double) anchor;
                    candidate.exact_release.x = (double) selected_line;
                    candidate.exact_release.y = (double) (anchor + displacement);

                    if (free_drag) {
                        candidate.lock_point.x = (double) selected_line;
                        candidate.lock_point.y = (double) anchor + sign * lock_threshold;
                    } else {
                        candidate.lock_point = candidate.exact_release;
                    }
                }

                items[index++] = candidate;
            }
        }
    }

    out_layer->items = items;
    out_layer->count = index;
    return CHUZZLE_OK;
}

static void chuzzle_free_candidate_layers(CandidateLayer *layers, size_t layer_count) {
    size_t i;
    if (layers == NULL) {
        return;
    }

    for (i = 0; i < layer_count; ++i) {
        free(layers[i].items);
        layers[i].items = NULL;
        layers[i].count = 0;
    }
    free(layers);
}

static double chuzzle_optimize_scalar_on_segment_vertical(
        double x_value,
        ChuzzlePoint lock_point,
        ChuzzlePoint target_point) {
    double left = lock_point.y < target_point.y ? lock_point.y : target_point.y;
    double right = lock_point.y > target_point.y ? lock_point.y : target_point.y;
    int iteration;

    if (fabs(right - left) <= CHUZZLE_EPSILON) {
        return left;
    }

    for (iteration = 0; iteration < CHUZZLE_TERNARY_SEARCH_STEPS; ++iteration) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        ChuzzlePoint p1 = {x_value, m1};
        ChuzzlePoint p2 = {x_value, m2};
        double f1 = chuzzle_distance(lock_point, p1) + chuzzle_distance(p1, target_point);
        double f2 = chuzzle_distance(lock_point, p2) + chuzzle_distance(p2, target_point);

        if (f1 <= f2) {
            right = m2;
        } else {
            left = m1;
        }
    }

    return (left + right) * 0.5;
}

static double chuzzle_optimize_scalar_on_segment_horizontal(
        double y_value,
        ChuzzlePoint lock_point,
        ChuzzlePoint target_point) {
    double left = lock_point.x < target_point.x ? lock_point.x : target_point.x;
    double right = lock_point.x > target_point.x ? lock_point.x : target_point.x;
    int iteration;

    if (fabs(right - left) <= CHUZZLE_EPSILON) {
        return left;
    }

    for (iteration = 0; iteration < CHUZZLE_TERNARY_SEARCH_STEPS; ++iteration) {
        double m1 = left + (right - left) / 3.0;
        double m2 = right - (right - left) / 3.0;
        ChuzzlePoint p1 = {m1, y_value};
        ChuzzlePoint p2 = {m2, y_value};
        double f1 = chuzzle_distance(lock_point, p1) + chuzzle_distance(p1, target_point);
        double f2 = chuzzle_distance(lock_point, p2) + chuzzle_distance(p2, target_point);

        if (f1 <= f2) {
            right = m2;
        } else {
            left = m1;
        }
    }

    return (left + right) * 0.5;
}

static ChuzzlePoint chuzzle_best_point_on_vertical_line(
        double x_value,
        ChuzzlePoint lock_point,
        ChuzzlePoint target_point) {
    ChuzzlePoint result;
    result.x = x_value;
    result.y = chuzzle_optimize_scalar_on_segment_vertical(x_value, lock_point, target_point);
    return result;
}

static ChuzzlePoint chuzzle_best_point_on_horizontal_line(
        double y_value,
        ChuzzlePoint lock_point,
        ChuzzlePoint target_point) {
    ChuzzlePoint result;
    result.x = chuzzle_optimize_scalar_on_segment_horizontal(y_value, lock_point, target_point);
    result.y = y_value;
    return result;
}

static TransitionChoice chuzzle_optimize_release_to_point(const MoveCandidate *candidate, ChuzzlePoint target) {
    TransitionChoice transition;

    memset(&transition, 0, sizeof(transition));

    if (!candidate->free_drag) {
        transition.release = candidate->exact_release;
        transition.drag_distance = candidate->min_drag_distance;
        transition.move_to_next_distance = chuzzle_distance(transition.release, target);
        transition.total_cost = transition.drag_distance + transition.move_to_next_distance;
        return transition;
    }

    if (candidate->axis == 'R') {
        transition.release = chuzzle_best_point_on_vertical_line(candidate->exact_release.x, candidate->lock_point, target);
    } else {
        transition.release = chuzzle_best_point_on_horizontal_line(candidate->exact_release.y, candidate->lock_point, target);
    }

    transition.drag_distance =
            chuzzle_distance(candidate->click_down, candidate->lock_point) + chuzzle_distance(candidate->lock_point, transition.release);
    transition.move_to_next_distance = chuzzle_distance(transition.release, target);
    transition.total_cost = transition.drag_distance + transition.move_to_next_distance;
    return transition;
}

static TerminalChoice chuzzle_best_terminal_choice(
        const MoveCandidate *candidate,
        const ChuzzlePoint *end_targets,
        size_t end_target_count) {
    TerminalChoice best_choice;
    size_t i;
    int has_choice = 0;

    memset(&best_choice, 0, sizeof(best_choice));

    if (end_target_count == 0 || end_targets == NULL) {
        best_choice.total_cost = candidate->min_drag_distance;
        best_choice.drag_distance = candidate->min_drag_distance;
        best_choice.final_move_distance = 0.0;
        best_choice.release = candidate->exact_release;
        best_choice.has_target = 0;
        return best_choice;
    }

    for (i = 0; i < end_target_count; ++i) {
        TransitionChoice transition = chuzzle_optimize_release_to_point(candidate, end_targets[i]);
        TerminalChoice choice;

        memset(&choice, 0, sizeof(choice));
        choice.total_cost = transition.total_cost;
        choice.drag_distance = transition.drag_distance;
        choice.final_move_distance = transition.move_to_next_distance;
        choice.release = transition.release;
        choice.has_target = 1;
        choice.target = end_targets[i];

        if (!has_choice || choice.total_cost < best_choice.total_cost) {
            best_choice = choice;
            has_choice = 1;
        }
    }

    return best_choice;
}

static void chuzzle_free_token_storage(char (*tokens)[CHUZZLE_MOVE_TOKEN_CAPACITY], char *normalized_moves) {
    free(tokens);
    free(normalized_moves);
}

int chuzzle_solve_sequence_utf8(
        const char *move_string,
        const ChuzzleSolveConfig *config,
        ChuzzleSolution *out_solution) {
    char(*tokens)[CHUZZLE_MOVE_TOKEN_CAPACITY] = NULL;
    size_t token_count = 0;
    char *normalized_moves = NULL;
    int result = CHUZZLE_OK;

    CandidateLayer *candidate_layers = NULL;
    size_t sequence_len = 0;
    ChuzzlePoint start_pos = {0.0, 0.0};
    int has_start_pos = 0;
    const ChuzzlePoint *final_targets = NULL;
    size_t final_target_count = 0;
    double lock_threshold = 1.0;
    int free_drag_min_displacement = 2;

    double *prev_costs = NULL;
    int **parents = NULL;
    size_t *candidate_counts = NULL;

    TerminalChoice *terminal_choices = NULL;
    int *chosen_indices = NULL;
    ChuzzleStep *steps = NULL;

    size_t move_index;
    size_t final_index;
    int best_final_index = -1;
    double best_final_total = DBL_MAX;

    if (move_string == NULL || config == NULL || out_solution == NULL) {
        return CHUZZLE_ERR_NULL_ARGUMENT;
    }

    chuzzle_clear_solution(out_solution);

    result = chuzzle_tokenize_move_string(move_string, &tokens, &token_count, &normalized_moves);
    if (result != CHUZZLE_OK) {
        chuzzle_free_token_storage(tokens, normalized_moves);
        return result;
    }

    sequence_len = token_count;
    has_start_pos = config->has_start_mouse_position ? 1 : 0;
    start_pos = config->start_mouse_position;
    lock_threshold = config->lock_threshold;
    free_drag_min_displacement = config->free_drag_min_displacement;

    if (config->end_next_puzzle) {
        final_targets = CHUZZLE_DEFAULT_NEXT_PUZZLE_TARGETS;
        final_target_count = 4;
    } else {
        final_targets = config->end_positions;
        final_target_count = config->end_position_count;
    }

    candidate_layers = (CandidateLayer *) calloc(sequence_len, sizeof(*candidate_layers));
    candidate_counts = (size_t *) calloc(sequence_len, sizeof(*candidate_counts));
    parents = (int **) calloc(sequence_len, sizeof(*parents));
    if (candidate_layers == NULL || candidate_counts == NULL || parents == NULL) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    for (move_index = 0; move_index < sequence_len; ++move_index) {
        ParsedMove parsed_move;
        result = chuzzle_parse_move_token(tokens[move_index], &parsed_move);
        if (result != CHUZZLE_OK) {
            goto cleanup;
        }

        result = chuzzle_build_candidates(
                &parsed_move,
                lock_threshold,
                free_drag_min_displacement,
                &candidate_layers[move_index]);
        if (result != CHUZZLE_OK) {
            goto cleanup;
        }

        candidate_counts[move_index] = candidate_layers[move_index].count;
    }

    if (sequence_len == 1) {
        CandidateLayer *only_layer = &candidate_layers[0];
        int best_index = -1;
        double best_total_cost = DBL_MAX;
        TerminalChoice best_terminal;
        double best_initial_move = 0.0;
        int has_terminal = 0;

        memset(&best_terminal, 0, sizeof(best_terminal));

        for (final_index = 0; final_index < only_layer->count; ++final_index) {
            MoveCandidate *candidate = &only_layer->items[final_index];
            TerminalChoice terminal = chuzzle_best_terminal_choice(candidate, final_targets, final_target_count);
            double initial_move_distance = 0.0;
            double total_cost;

            if (has_start_pos) {
                initial_move_distance = chuzzle_distance(start_pos, candidate->click_down);
            }

            total_cost = initial_move_distance + terminal.total_cost;
            if (total_cost < best_total_cost) {
                best_total_cost = total_cost;
                best_index = (int) final_index;
                best_terminal = terminal;
                best_initial_move = initial_move_distance;
                has_terminal = 1;
            }
        }

        if (!has_terminal || best_index < 0) {
            result = CHUZZLE_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        out_solution->move_string = normalized_moves;
        normalized_moves = NULL;

        out_solution->total_drag = best_terminal.drag_distance;
        out_solution->total_move = best_initial_move + best_terminal.final_move_distance;
        out_solution->total_cost = out_solution->total_drag + out_solution->total_move;
        out_solution->has_initial_mouse_position = has_start_pos;
        if (has_start_pos) {
            out_solution->initial_mouse_position = start_pos;
        }
        out_solution->initial_move_distance = best_initial_move;
        out_solution->inter_move_distance = 0.0;
        out_solution->has_final_mouse_target = best_terminal.has_target;
        if (best_terminal.has_target) {
            out_solution->final_mouse_target = best_terminal.target;
        }
        out_solution->final_move_distance = best_terminal.final_move_distance;
        out_solution->step_count = 1;
        out_solution->steps = (ChuzzleStep *) calloc(1, sizeof(*out_solution->steps));
        if (out_solution->steps == NULL) {
            result = CHUZZLE_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        {
            MoveCandidate *candidate = &only_layer->items[best_index];
            ChuzzleStep *step = &out_solution->steps[0];
            strncpy(step->move, candidate->token, CHUZZLE_MOVE_TOKEN_CAPACITY - 1);
            step->selected_line = candidate->selected_line;
            step->displacement = candidate->displacement;
            step->free_drag = candidate->free_drag;
            step->click_down = candidate->click_down;
            step->lock_point = candidate->lock_point;
            step->release = best_terminal.release;
            step->drag_distance = best_terminal.drag_distance;
            step->move_distance = best_initial_move;
            step->total_distance_to_here = best_initial_move + best_terminal.drag_distance;
        }

        result = CHUZZLE_OK;
        goto cleanup;
    }

    prev_costs = (double *) malloc(candidate_counts[0] * sizeof(*prev_costs));
    parents[0] = (int *) malloc(candidate_counts[0] * sizeof(*parents[0]));
    if (prev_costs == NULL || parents[0] == NULL) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    for (final_index = 0; final_index < candidate_counts[0]; ++final_index) {
        if (has_start_pos) {
            prev_costs[final_index] = chuzzle_distance(start_pos, candidate_layers[0].items[final_index].click_down);
        } else {
            prev_costs[final_index] = 0.0;
        }
        parents[0][final_index] = -1;
    }

    for (move_index = 1; move_index < sequence_len; ++move_index) {
        CandidateLayer *prev_layer = &candidate_layers[move_index - 1];
        CandidateLayer *curr_layer = &candidate_layers[move_index];
        double *curr_costs = (double *) malloc(curr_layer->count * sizeof(*curr_costs));
        int *curr_parents = (int *) malloc(curr_layer->count * sizeof(*curr_parents));
        size_t curr_index;
        size_t prev_index;

        if (curr_costs == NULL || curr_parents == NULL) {
            free(curr_costs);
            free(curr_parents);
            result = CHUZZLE_ERR_INVALID_ARGUMENT;
            goto cleanup;
        }

        for (curr_index = 0; curr_index < curr_layer->count; ++curr_index) {
            MoveCandidate *curr_candidate = &curr_layer->items[curr_index];
            double best_cost = DBL_MAX;
            int best_parent = -1;

            for (prev_index = 0; prev_index < prev_layer->count; ++prev_index) {
                MoveCandidate *prev_candidate = &prev_layer->items[prev_index];
                TransitionChoice edge = chuzzle_optimize_release_to_point(prev_candidate, curr_candidate->click_down);
                double candidate_cost = prev_costs[prev_index] + edge.total_cost;

                if (candidate_cost < best_cost) {
                    best_cost = candidate_cost;
                    best_parent = (int) prev_index;
                }
            }

            curr_costs[curr_index] = best_cost;
            curr_parents[curr_index] = best_parent;
        }

        parents[move_index] = curr_parents;
        free(prev_costs);
        prev_costs = curr_costs;
    }

    terminal_choices = (TerminalChoice *) calloc(candidate_counts[sequence_len - 1], sizeof(*terminal_choices));
    if (terminal_choices == NULL) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    for (final_index = 0; final_index < candidate_counts[sequence_len - 1]; ++final_index) {
        terminal_choices[final_index] = chuzzle_best_terminal_choice(
                &candidate_layers[sequence_len - 1].items[final_index],
                final_targets,
                final_target_count);

        {
            double total_cost = prev_costs[final_index] + terminal_choices[final_index].total_cost;
            if (total_cost < best_final_total) {
                best_final_total = total_cost;
                best_final_index = (int) final_index;
            }
        }
    }

    if (best_final_index < 0) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    chosen_indices = (int *) malloc(sequence_len * sizeof(*chosen_indices));
    if (chosen_indices == NULL) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    chosen_indices[sequence_len - 1] = best_final_index;
    for (move_index = sequence_len - 1; move_index > 0; --move_index) {
        chosen_indices[move_index - 1] = parents[move_index][chosen_indices[move_index]];
    }

    steps = (ChuzzleStep *) calloc(sequence_len, sizeof(*steps));
    if (steps == NULL) {
        result = CHUZZLE_ERR_INVALID_ARGUMENT;
        goto cleanup;
    }

    {
        double total_drag = 0.0;
        double inter_move_distance = 0.0;
        double initial_move_distance = 0.0;
        TerminalChoice terminal = terminal_choices[best_final_index];

        if (has_start_pos) {
            initial_move_distance = chuzzle_distance(start_pos, candidate_layers[0].items[chosen_indices[0]].click_down);
        }

        for (move_index = 0; move_index < sequence_len; ++move_index) {
            MoveCandidate *candidate = &candidate_layers[move_index].items[chosen_indices[move_index]];
            ChuzzleStep *step = &steps[move_index];
            TransitionChoice transition;
            double move_distance;
            ChuzzlePoint release;

            if (move_index + 1 < sequence_len) {
                MoveCandidate *next_candidate = &candidate_layers[move_index + 1].items[chosen_indices[move_index + 1]];
                transition = chuzzle_optimize_release_to_point(candidate, next_candidate->click_down);
                release = transition.release;
            } else {
                transition.total_cost = terminal.total_cost;
                transition.drag_distance = terminal.drag_distance;
                transition.move_to_next_distance = terminal.final_move_distance;
                transition.release = terminal.release;
                release = terminal.release;
            }

            if (move_index == 0) {
                move_distance = initial_move_distance;
            } else {
                move_distance = chuzzle_optimize_release_to_point(
                                        &candidate_layers[move_index - 1].items[chosen_indices[move_index - 1]],
                                        candidate->click_down)
                                        .move_to_next_distance;
                inter_move_distance += move_distance;
            }

            total_drag += transition.drag_distance;

            memset(step, 0, sizeof(*step));
            strncpy(step->move, candidate->token, CHUZZLE_MOVE_TOKEN_CAPACITY - 1);
            step->selected_line = candidate->selected_line;
            step->displacement = candidate->displacement;
            step->free_drag = candidate->free_drag;
            step->click_down = candidate->click_down;
            step->lock_point = candidate->lock_point;
            step->release = release;
            step->drag_distance = transition.drag_distance;
            step->move_distance = move_distance;
            step->total_distance_to_here = initial_move_distance + inter_move_distance + total_drag;
        }

        out_solution->move_string = normalized_moves;
        normalized_moves = NULL;
        out_solution->total_drag = total_drag;
        out_solution->total_move = initial_move_distance + inter_move_distance + terminal.final_move_distance;
        out_solution->total_cost = out_solution->total_drag + out_solution->total_move;
        out_solution->has_initial_mouse_position = has_start_pos;
        if (has_start_pos) {
            out_solution->initial_mouse_position = start_pos;
        }
        out_solution->initial_move_distance = initial_move_distance;
        out_solution->inter_move_distance = inter_move_distance;
        out_solution->has_final_mouse_target = terminal.has_target;
        if (terminal.has_target) {
            out_solution->final_mouse_target = terminal.target;
        }
        out_solution->final_move_distance = terminal.final_move_distance;
        out_solution->step_count = sequence_len;
        out_solution->steps = steps;
        steps = NULL;
    }

    result = CHUZZLE_OK;

cleanup:
    if (steps != NULL) {
        free(steps);
    }
    free(chosen_indices);
    free(terminal_choices);
    free(prev_costs);

    if (parents != NULL) {
        for (move_index = 0; move_index < sequence_len; ++move_index) {
            free(parents[move_index]);
        }
    }
    free(parents);
    free(candidate_counts);
    chuzzle_free_candidate_layers(candidate_layers, sequence_len);
    chuzzle_free_token_storage(tokens, normalized_moves);

    if (result != CHUZZLE_OK) {
        chuzzle_free_solution(out_solution);
    }

    return result;
}

void chuzzle_free_solution(
        ChuzzleSolution *solution) {
    if (solution == NULL) {
        return;
    }

    free(solution->move_string);
    free(solution->steps);
    chuzzle_clear_solution(solution);
}
