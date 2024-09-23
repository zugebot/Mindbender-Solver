#include "perms.hpp"
#include "rotations.hpp"


static constexpr uint64_t BOARD_PRE_ALLOC_SIZES[6] = {
        1,
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


static constexpr uint64_t BOARD_FAT_PRE_ALLOC_SIZES[6] = {
        1,
        48,
        2304,
        110592,
        5308416,
        254803968,
};



std::vector<Board> make_permutation_list_depth_0(Board &board, c_u32 colorCount) {
    std::vector<Board> boards = {board};
    boards[0].precomputeHash(colorCount);
    return boards;
}


std::vector<Board> make_permutation_list_depth_1(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (int a = 0; a < 60; ++a) {
        Board *currentBoard = &boards[count];
        *currentBoard = board;
        actions[a](*currentBoard);
        currentBoard->precomputeHash(colorCount);
        uint64_t move = a;
        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }

    return boards;
}


std::vector<Board> make_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 2;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a;
    uint64_t move_a;

    int count = 0;
    for (int a = 0; a < 12; ++a) {
        int a_dir = a / 6;
        int a_sect = a % 6;
        for (int b_dir = 0; b_dir <= 1; ++b_dir) {
            int b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b = b_dir * 6 + b_sect_start; b < b_dir * 6 + 6; ++b) {

                for (int a_cur = a * 5; a_cur < a * 5 + 5; ++a_cur) {
                    board_a = board;
                    actions[a_cur](board_a);

                    for (int b_cur = b * 5; b_cur < b * 5 + 5; ++b_cur) {

                        boards[count] = board_a;
                        actions[b_cur](boards[count]);
                        boards[count].precomputeHash(colorCount);
                        uint64_t move = a_cur | (b_cur << 6);
                        (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                        count++;
                    }
                }
            }
        }
    }

    return boards;
}


std::vector<Board> make_permutation_list_depth_3(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 3;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    uint64_t move_a, move_b;

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
                            board_a = board;
                            actions[a_cur](board_a);
                            move_a = a_cur;
                            for (int b_cur = b * 5; b_cur < b * 5 + 5; b_cur++) {
                                board_b = board_a;
                                actions[b_cur](board_b);
                                move_b = move_a | (b_cur << 6);
                                for (int c_cur = c * 5; c_cur < c * 5 + 5; c_cur++) {
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);
                                    boards[count].precomputeHash(colorCount);
                                    uint64_t move = move_b | (c_cur << 12);
                                    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
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
                                           uint64_t move, int& count, std::vector<Board>& boards, c_u32 colorCount) {

    const int d_cur = d_base + N;

    boards[count] = board;
    actions[a_cur](boards[count]);
    actions[b_cur](boards[count]);
    actions[c_cur](boards[count]);
    actions[d_cur](boards[count]);
    boards[count].precomputeHash(colorCount);

    move |= d_cur << 18;
    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
    count++;
}


std::vector<Board> make_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 4;
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
                                unrolled_loop_d_depth_4<0, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards, colorCount);
                                unrolled_loop_d_depth_4<1, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards, colorCount);
                                unrolled_loop_d_depth_4<2, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards, colorCount);
                                unrolled_loop_d_depth_4<3, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards, colorCount);
                                unrolled_loop_d_depth_4<4, DEPTH>(board, a_cur, b_cur, c_cur, d_base, final_move_2, count, boards, colorCount);
                            }
                        }
                    }
                }
            }
        }
    }
    return boards;
}


std::vector<Board> make_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 5;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    uint64_t move_a, move_b, move_c, move_d;

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
                    for (int e = 0; e < 12; e++) {
                        const int e_base = e * 5;
                        const int e_dir = e / 6;
                        const int e_sect = e % 6;
                        if (d_dir == e_dir && d_sect >= e_sect) { continue; }
                        for (uint64_t a_cur = a_base; a_cur < a_base + 5; a_cur++) {
                            board_a = board;
                            actions[a_cur](board_a);
                            move_a = a_cur;
                            for (uint64_t b_cur = b_base; b_cur < b_base + 5; b_cur++) {
                                board_b = board_a;
                                actions[b_cur](board_b);
                                move_b = move_a | (uint64_t(b_cur) << 6);
                                for (uint64_t c_cur = c_base; c_cur < c_base + 5; c_cur++) {
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);
                                    std::fill_n(&boards[count], 25, boards[count]);
                                    move_c = move_b | (uint64_t(c_cur) << 12);
                                    for (uint64_t d_cur = d_base; d_cur < d_base + 5; d_cur++) {
                                        actions[d_cur](boards[count]);
                                        std::fill_n(&boards[count], 5, boards[count]);
                                        move_d = move_c | (uint64_t(d_cur) << 18);
                                        for (uint64_t e_cur = e_base; e_cur < e_base + 5; e_cur++) {
                                            actions[e_cur](boards[count]);
                                            boards[count].precomputeHash(colorCount);
                                            uint64_t move = move_d | (uint64_t(e_cur) << 24);
                                            (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
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
    return boards;
}


MakePermFuncArray makePermutationListFuncs[] = {
        &make_permutation_list_depth_0,
        &make_permutation_list_depth_1,
        &make_permutation_list_depth_2,
        &make_permutation_list_depth_3,
        &make_permutation_list_depth_4,
        &make_permutation_list_depth_5};





















std::vector<Board> make_fat_permutation_list_depth_0(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 0;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    boards[0] = board;
    boards[0].precomputeHash(colorCount);
    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_1(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 1;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (uint64_t a = 0; a < 48; ++a) {
        Board *currentBoard = &boards[count];
        *currentBoard = board;
        fatActions[currentBoard->getFatXY()][a](*currentBoard);
        currentBoard->precomputeHash(colorCount);
        uint64_t move = a;
        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }

    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 2;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (uint64_t a = 0; a < 48; a++) {
        for (uint64_t b = 0; b < 48; b++) {
            Board *currentBoard = &boards[count];
            *currentBoard = board;
            fatActions[currentBoard->getFatXY()][a](*currentBoard);
            fatActions[currentBoard->getFatXY()][b](*currentBoard);
            currentBoard->precomputeHash(colorCount);
            uint64_t move = a | b << 6;
            (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
            count++;
        }
    }


    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_3(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 3;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (uint64_t a = 0; a < 48; a++) {
        for (uint64_t b = 0; b < 48; b++) {
            for (uint64_t c = 0; c < 48; c++) {
                Board *currentBoard = &boards[count];
                *currentBoard = board;
                fatActions[currentBoard->getFatXY()][a](*currentBoard);
                fatActions[currentBoard->getFatXY()][b](*currentBoard);
                fatActions[currentBoard->getFatXY()][c](*currentBoard);
                currentBoard->precomputeHash(colorCount);
                uint64_t move = a | (b << 6) | (c << 12);
                (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                count++;
            }
        }
    }

    return boards;
}





std::vector<Board> make_fat_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 4;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (uint64_t a = 0; a < 48; a++) {
        for (uint64_t b = 0; b < 48; b++) {
            for (uint64_t c = 0; c < 48; c++) {
                for (uint64_t d = 0; d < 48; d++) {
                    Board *currentBoard = &boards[count];
                    volatile uint64_t move = a | (b << 6) | (c << 12) | (d << 18);
                    *currentBoard = board;
                    fatActions[currentBoard->getFatXY()][a](*currentBoard);
                    fatActions[currentBoard->getFatXY()][b](*currentBoard);
                    fatActions[currentBoard->getFatXY()][c](*currentBoard);
                    fatActions[currentBoard->getFatXY()][d](*currentBoard);
                    currentBoard->precomputeHash(colorCount);
                    (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                    count++;
                }
            }
        }
    }

    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr int DEPTH = 5;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    int count = 0;
    for (uint64_t a = 0; a < 48; a++) {
        for (uint64_t b = 0; b < 48; b++) {
            for (uint64_t c = 0; c < 48; c++) {
                for (uint64_t d = 0; d < 48; d++) {
                    for (uint64_t e = 0; e < 48; e++) {
                        Board *currentBoard = &boards[count];
                        *currentBoard = board;
                        fatActions[currentBoard->getFatXY()][a](*currentBoard);
                        fatActions[currentBoard->getFatXY()][b](*currentBoard);
                        fatActions[currentBoard->getFatXY()][c](*currentBoard);
                        fatActions[currentBoard->getFatXY()][d](*currentBoard);
                        fatActions[currentBoard->getFatXY()][e](*currentBoard);
                        currentBoard->precomputeHash(colorCount);
                        uint64_t move = a | (b << 6) | (c << 12) | (d << 18) | (e << 24);
                        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
                        count++;
                    }
                }
            }
        }
    }

    return boards;
}


MakePermFuncArray makeFatPermutationListFuncs[] = {
        &make_fat_permutation_list_depth_0,
        &make_fat_permutation_list_depth_1,
        &make_fat_permutation_list_depth_2,
        &make_fat_permutation_list_depth_3,
        &make_fat_permutation_list_depth_4,
        &make_fat_permutation_list_depth_5};

