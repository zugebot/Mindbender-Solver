#ifndef CHUZZLE_MOUSE_NATIVE_API_H
#define CHUZZLE_MOUSE_NATIVE_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#if defined(_WIN32)
#define CHUZZLE_API __declspec(dllexport)
#else
#define CHUZZLE_API
#endif

enum {
    CHUZZLE_OK = 0,
    CHUZZLE_ERR_NULL_ARGUMENT = 1,
    CHUZZLE_ERR_INVALID_ARGUMENT = 2,
    CHUZZLE_ERR_PARSE_MOVE = 3,
    CHUZZLE_ERR_NOT_IMPLEMENTED = 100
};

typedef struct ChuzzlePoint {
    double x;
    double y;
} ChuzzlePoint;

typedef struct ChuzzleSolveConfig {
    int has_start_mouse_position;
    ChuzzlePoint start_mouse_position;

    int end_next_puzzle;
    size_t end_position_count;
    const ChuzzlePoint* end_positions;

    double lock_threshold;
    int free_drag_min_displacement;
} ChuzzleSolveConfig;

typedef struct ChuzzleNativeStep {
    char move[5];
    int selected_line;
    int displacement;
    int free_drag;

    ChuzzlePoint click_down;
    ChuzzlePoint lock_point;
    ChuzzlePoint release;

    double drag_distance;
    double move_distance;
    double total_distance_to_here;
} ChuzzleNativeStep;

typedef struct ChuzzleNativeSolution {
    char* move_string;

    double total_drag;
    double total_move;
    double total_cost;

    int has_initial_mouse_position;
    ChuzzlePoint initial_mouse_position;
    double initial_move_distance;
    double inter_move_distance;

    int has_final_mouse_target;
    ChuzzlePoint final_mouse_target;
    double final_move_distance;

    size_t step_count;
    ChuzzleNativeStep* steps;
} ChuzzleNativeSolution;

CHUZZLE_API int chuzzle_solve_sequence_utf8(
    const char* move_string,
    const ChuzzleSolveConfig* config,
    ChuzzleNativeSolution* out_solution
);

CHUZZLE_API void chuzzle_free_solution(
    ChuzzleNativeSolution* solution
);

#ifdef __cplusplus
}
#endif

#endif
