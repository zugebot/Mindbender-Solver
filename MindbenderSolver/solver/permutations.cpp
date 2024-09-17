#include "permutations.hpp"
#include "rotations.hpp"


static constexpr uint64_t BOARD_PRE_ALLOC_SIZES[5] = {
        60,
        2550,
        104000,
        4245000,
        173325000,
        // 7076687500,
        // 288933750000,
        // 11796869531250,
        // 481654101562500,
        // 19665443613281250,
        // 802919920312500000,
};



std::vector<Board> make_permutation_list_depth_1(Board &board) {
    static constexpr int DEPTH = 1 - 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (int a = 0; a < 60; ++a) {
        Board *currentBoard = &boards[count];
        *currentBoard = board;
        actions[a](*currentBoard);
        currentBoard->precompute_hash();
        uint64_t move = a;
        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }

    return boards;
}


std::vector<Board> make_permutation_list_depth_2(Board &board) {
    static constexpr int DEPTH = 2 - 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (int a = 0; a < 12; ++a) {
        int a_dir = a / 6;
        int a_sect = a % 6;
        for (int b_dir = 0; b_dir <= 1; ++b_dir) {
            int b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b = b_dir * 6 + b_sect_start; b < b_dir * 6 + 6; ++b) {
                for (int a_cur = a * 5; a_cur < a * 5 + 5; ++a_cur) {
                    for (int b_cur = b * 5; b_cur < b * 5 + 5; ++b_cur) {
                        Board *currentBoard = &boards[count];
                        *currentBoard = board;
                        actions[a](*currentBoard);
                        actions[b](*currentBoard);
                        currentBoard->precompute_hash();
                        uint64_t move = a_cur | (b_cur << 6);
                        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                        count++;
                    }
                }
            }
        }
    }

    return boards;
}


std::vector<Board> make_permutation_list_depth_3(Board &board) {
    static constexpr int DEPTH = 3 - 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    int count = 0;
    for (int a = 0; a < 12; ++a) {
        int a_dir = a / 6;
        int a_sect = a % 6;
        for (int b_dir = 0; b_dir <= 1; ++b_dir) {
            int b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                int b = b_dir * 6 + b_sect;
                for (int c_dir = 0; c_dir <= 1; ++c_dir) {
                    int c_sect_start = (c_dir == b_dir) ? b_sect + 1 : 0;
                    for (int c_sect = c_sect_start; c_sect < 6; ++c_sect) {
                        int c = c_dir * 6 + c_sect;
                        for (int a_cur = a * 5; a_cur < a * 5 + 5; a_cur++) {
                            for (int b_cur = b * 5; b_cur < b * 5 + 5; b_cur++) {
                                for (int c_cur = c * 5; c_cur < c * 5 + 5; c_cur++) {
                                    Board *currentBoard = &boards[count];
                                    *currentBoard = board;
                                    actions[a](*currentBoard);
                                    actions[b](*currentBoard);
                                    actions[c](*currentBoard);
                                    currentBoard->precompute_hash();
                                    uint64_t move = a_cur | (b_cur << 6) | (c_cur << 12);
                                    (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                                    count++;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return boards;
}



template<int N, int DEPTH>
__forceinline void unrolled_loop_d_depth_4(Board& board,
                                           const int a_cur, const int b_cur, const int c_cur, const int d_base,
                                           uint64_t move, int& count, std::vector<Board>& boards) {

    const int d_cur = d_base + N;

    boards[count] = board;
    actions[a_cur](boards[count]);
    actions[b_cur](boards[count]);
    actions[c_cur](boards[count]);
    actions[d_cur](boards[count]);
    boards[count].precompute_hash();

    move |= d_cur << 18;
    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
    count++;
}


std::vector<Board> make_permutation_list_depth_4(Board &board) {
    static constexpr int DEPTH = 4 - 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    int count = 0;
    for (int a = 0; a < 12; a++) {
        const int a_base = a * 5;
        const int a_dir = a / 6;
        const int a_sect = a % 6;
        for (int b = 0; b < 12; b++) {
            const int b_base = b * 5;
            const int b_dir = b / 6;
            const int b_sect = b % 6;
            if (a_dir == b_dir && a_sect >= b_sect) { continue; }
            for (int c = 0; c < 12; c++) {
                const int c_base = c * 5;
                const int c_dir = c / 6;
                const int c_sect = c % 6;
                if (b_dir == c_dir && b_sect >= c_sect) { continue; }
                for (int d = 0; d < 12; d++) {
                    const int d_base = d * 5;
                    const int d_dir = d / 6;
                    const int d_sect = d % 6;
                    if (c_dir == d_dir && c_sect >= d_sect) { continue; }
                    for (int a_cur = a_base; a_cur < a_base + 5; a_cur++) {
                        for (int b_cur = b_base; b_cur < b_base + 5; b_cur++) {
                            const uint64_t final_move_1 = a_cur | (b_cur << 6);
                            for (int c_cur = c_base; c_cur < c_base + 5; c_cur++) {
                                const uint64_t final_move_2 = final_move_1 | (c_cur << 12);
                                unrolled_loop_d_depth_4<0, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards);
                                unrolled_loop_d_depth_4<1, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards);
                                unrolled_loop_d_depth_4<2, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards);
                                unrolled_loop_d_depth_4<3, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards);
                                unrolled_loop_d_depth_4<4, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards);
                            }
                        }
                    }
                }
            }
        }
    }
    return boards;
}


std::vector<Board> make_permutation_list_depth_5(Board &board) {
    static constexpr int DEPTH = 5 - 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    int count = 0;
    for (int a_dir = 0; a_dir <= 1; ++a_dir) {
        for (int a_sect = 0; a_sect < 6; ++a_sect) {
            int a = a_dir * 6 + a_sect;
            for (int b_dir = 0; b_dir <= 1; ++b_dir) {
                int b_sect_start = (a_dir == b_dir) ? a_sect + 1 : 0;
                for (int b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                    int b = b_dir * 6 + b_sect;
                    for (int c_dir = 0; c_dir <= 1; ++c_dir) {
                        int c_sect_start = (b_dir == c_dir) ? b_sect + 1 : 0;
                        for (int c_sect = c_sect_start; c_sect < 6; ++c_sect) {
                            int c = c_dir * 6 + c_sect;
                            for (int d_dir = 0; d_dir <= 1; ++d_dir) {
                                int d_sect_start = (c_dir == d_dir) ? c_sect + 1 : 0;
                                for (int d_sect = d_sect_start; d_sect < 6; ++d_sect) {
                                    int d = d_dir * 6 + d_sect;
                                    for (int e_dir = 0; e_dir <= 1; ++e_dir) {
                                        int e_sect_start = (d_dir == e_dir) ? d_sect + 1 : 0;
                                        for (int e_sect = e_sect_start; e_sect < 6; ++e_sect) {
                                            int e = e_dir * 6 + e_sect;
                                            for (int a_amount = 1; a_amount < 6; ++a_amount) {
                                                int a_cur = a * 5 + a_amount - 1;
                                                for (int b_amount = 1; b_amount < 6; ++b_amount) {
                                                    int b_cur = b * 5 + b_amount - 1;
                                                    for (int c_amount = 1; c_amount < 6; ++c_amount) {
                                                        int c_cur = c * 5 + c_amount - 1;
                                                        for (int d_amount = 1; d_amount < 6; ++d_amount) {
                                                            int d_cur = d * 5 + d_amount - 1;
                                                            for (int e_amount = 1; e_amount < 6; ++e_amount) {
                                                                int e_cur = e * 5 + e_amount - 1;
                                                                Board *currentBoard = &boards[count];
                                                                *currentBoard = board;
                                                                actions[a_cur](*currentBoard);
                                                                actions[b_cur](*currentBoard);
                                                                actions[c_cur](*currentBoard);
                                                                actions[d_cur](*currentBoard);
                                                                actions[e_cur](*currentBoard);
                                                                currentBoard->precompute_hash();
                                                                uint64_t move = (uint64_t(a_cur) <<  0) |
                                                                                (uint64_t(b_cur) <<  6) |
                                                                                (uint64_t(c_cur) << 12) |
                                                                                (uint64_t(d_cur) << 18) |
                                                                                (uint64_t(e_cur) << 24);
                                                                (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                                                                count++;
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return boards;
}






make_permutation_list_func makePermutationListFuncs[] = {
        nullptr,
        &make_permutation_list_depth_1,
        &make_permutation_list_depth_2,
        &make_permutation_list_depth_3,
        &make_permutation_list_depth_4,
        &make_permutation_list_depth_5
};
