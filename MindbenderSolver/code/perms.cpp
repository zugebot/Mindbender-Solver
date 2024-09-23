#include "perms.hpp"
#include "rotations.hpp"


static constexpr u64 BOARD_PRE_ALLOC_SIZES[6] = {
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


static constexpr u64 BOARD_FAT_PRE_ALLOC_SIZES[6] = {
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
    static constexpr i32 DEPTH = 1;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);

    i32 count = 0;
    for (i32 a = 0; a < 60; ++a) {
        Board *currentBoard = &boards[count];
        *currentBoard = board;
        actions[a](*currentBoard);
        currentBoard->precomputeHash(colorCount);
        u64 move = a;
        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }

    return boards;
}


std::vector<Board> make_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr i32 DEPTH = 2;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a;

    i32 count = 0;
    // find valid row/column part
    for (i32 a = 0; a < 12; ++a) {
        i32 a_dir = a / 6;
        i32 a_sect = a % 6;
        for (i32 b_dir = 0; b_dir <= 1; ++b_dir) {
            i32 b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (i32 b = b_dir * 6 + b_sect_start; b < b_dir * 6 + 6; ++b) {
                // actual creation part
                for (i32 a_cur = a * 5; a_cur < a * 5 + 5; ++a_cur) {
                    board_a = board;
                    actions[a_cur](board_a);
                    for (i32 b_cur = b * 5; b_cur < b * 5 + 5; ++b_cur) {
                        boards[count] = board_a;
                        actions[b_cur](boards[count]);
                        boards[count].precomputeHash(colorCount);
                        u64 move = a_cur | (b_cur << 6);
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
    static constexpr i32 DEPTH = 3;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    u64 move_a, move_b;

    i32 count = 0;
    // find valid row/column part
    for (i32 a = 0; a < 12; ++a) {
        i32 a_dir = a / 6;
        i32 a_sect = a % 6;
        for (i32 b_dir = 0; b_dir <= 1; ++b_dir) {
            i32 b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (i32 b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                i32 b = b_dir * 6 + b_sect;
                for (i32 c_dir = 0; c_dir <= 1; ++c_dir) {
                    i32 c_sect_start = (c_dir == b_dir) ? b_sect + 1 : 0;
                    for (i32 c_sect = c_sect_start; c_sect < 6; ++c_sect) {
                        i32 c = c_dir * 6 + c_sect;
                        // actual creation part
                        for (i32 a_cur = a * 5; a_cur < a * 5 + 5; a_cur++) {
                            board_a = board;
                            actions[a_cur](board_a);
                            move_a = a_cur;
                            for (i32 b_cur = b * 5; b_cur < b * 5 + 5; b_cur++) {
                                board_b = board_a;
                                actions[b_cur](board_b);
                                move_b = move_a | (b_cur << 6);
                                for (i32 c_cur = c * 5; c_cur < c * 5 + 5; c_cur++) {
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);
                                    boards[count].precomputeHash(colorCount);
                                    u64 move = move_b | (c_cur << 12);
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


std::vector<Board> make_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a;
    u64 move_b, move_c, move_d;

    u32 count = 0;
    // find valid row/column part
    for (u32 a = 0; a < 12; a++) {
        c_u32 a_base = a * 5;
        c_u32 a_dir = a / 6;
        c_u32 a_sect = a % 6;
        for (u32 b = 0; b < 12; b++) {
            c_u32 b_base = b * 5;
            c_u32 b_dir = b / 6;
            c_u32 b_sect = b % 6;
            if (a_dir == b_dir && a_sect >= b_sect) { continue; }
            for (u32 c = 0; c < 12; c++) {
                c_u32 c_base = c * 5;
                c_u32 c_dir = c / 6;
                c_u32 c_sect = c % 6;
                if (b_dir == c_dir && b_sect >= c_sect) { continue; }
                for (u32 d = 0; d < 12; d++) {
                    c_u32 d_base = d * 5;
                    c_u32 d_dir = d / 6;
                    c_u32 d_sect = d % 6;
                    if (c_dir == d_dir && c_sect >= d_sect) { continue; }
                    // actual creation part
                    for (u32 a_cur = a_base; a_cur < a_base + 5; a_cur++) {

                        board_a = board;
                        actions[a_cur](board_a);

                        std::fill_n(&boards[count], 125, board_a);
                        for (u32 b_cur = b_base; b_cur < b_base + 5; b_cur++) {
                            move_b = a_cur | (b_cur << 6);
                            actions[b_cur](boards[count]);

                            std::fill_n(&boards[count], 25, boards[count]);
                            for (u32 c_cur = c_base; c_cur < c_base + 5; c_cur++) {
                                move_c = move_b | (c_cur << 12);
                                actions[c_cur](boards[count]);

                                std::fill_n(&boards[count], 5, boards[count]);
                                for (u32 d_cur = d_base; d_cur < d_base + 5; d_cur++) {
                                    move_d = move_c | d_cur << 18;
                                    actions[d_cur](boards[count]);

                                    boards[count].precomputeHash(colorCount);
                                    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_d);
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


std::vector<Board> make_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
    std::vector<Board> boards(BOARD_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    u64 move_b, move_c, move_d, move_e;

    u32 count = 0;
    // find valid row/column part
    for (u32 a = 0; a < 12; a++) {
        c_u32 a_base = a * 5;
        c_u32 a_dir = a / 6;
        c_u32 a_sect = a % 6;
        for (u32 b = 0; b < 12; b++) {
            c_u32 b_base = b * 5;
            c_u32 b_dir = b / 6;
            c_u32 b_sect = b % 6;
            if (a_dir == b_dir && a_sect >= b_sect) { continue; }
            for (u32 c = 0; c < 12; c++) {
                c_u32 c_base = c * 5;
                c_u32 c_dir = c / 6;
                c_u32 c_sect = c % 6;
                if (b_dir == c_dir && b_sect >= c_sect) { continue; }
                for (u32 d = 0; d < 12; d++) {
                    c_u32 d_base = d * 5;
                    c_u32 d_dir = d / 6;
                    c_u32 d_sect = d % 6;
                    if (c_dir == d_dir && c_sect >= d_sect) { continue; }
                    for (u32 e = 0; e < 12; e++) {
                        c_u32 e_base = e * 5;
                        c_u32 e_dir = e / 6;
                        c_u32 e_sect = e % 6;
                        if (d_dir == e_dir && d_sect >= e_sect) { continue; }
                        // actual creation part
                        for (u64 a_cur = a_base; a_cur < a_base + 5; a_cur++) {
                            board_a = board;
                            actions[a_cur](board_a);
                            for (u64 b_cur = b_base; b_cur < b_base + 5; b_cur++) {
                                move_b = a_cur | (u64(b_cur) << 6);
                                board_b = board_a;
                                actions[b_cur](board_b);

                                for (u64 c_cur = c_base; c_cur < c_base + 5; c_cur++) {
                                    move_c = move_b | (u64(c_cur) << 12);
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);

                                    std::fill_n(&boards[count], 25, boards[count]);
                                    for (u64 d_cur = d_base; d_cur < d_base + 5; d_cur++) {
                                        move_d = move_c | (u64(d_cur) << 18);
                                        actions[d_cur](boards[count]);

                                        std::fill_n(&boards[count], 5, boards[count]);
                                        for (u64 e_cur = e_base; e_cur < e_base + 5; e_cur++) {
                                            move_e = move_d | (u64(e_cur) << 24);

                                            actions[e_cur](boards[count]);
                                            boards[count].precomputeHash(colorCount);
                                            (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_e);
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


















static constexpr u64 FAT_PERM_COUNT = 48;


std::vector<Board> make_fat_permutation_list_depth_0(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 0;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    boards[0] = board;
    boards[0].precomputeHash(colorCount);
    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_1(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 1;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        boards[count] = board;
        fatActions[boards[count].getFatXY()][a](boards[count]);
        
        boards[count].precomputeHash(colorCount);
        u64 move = a;
        (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }

    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 2;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; a++) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        for (u64 b = 0; b < FAT_PERM_COUNT; b++) {
            c_u64 move = a | b << 6;
            boards[count] = board_a;
            fatActions[boards[count].getFatXY()][b](boards[count]);

            boards[count].precomputeHash(colorCount);
            (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
            count++;
        }
    }


    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_3(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 3;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    u64 move_b;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; a++) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        for (u64 b = 0; b < FAT_PERM_COUNT; b++) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            for (u64 c = 0; c < FAT_PERM_COUNT; c++) {
                c_u64 move = move_b | (c << 12);
                boards[count] = board_b;
                fatActions[boards[count].getFatXY()][c](boards[count]);

                boards[count].precomputeHash(colorCount);
                (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                count++;
            }
        }
    }

    return boards;
}





std::vector<Board> make_fat_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b, board_c;
    u64 move_b, move_c;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; a++) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        for (u64 b = 0; b < FAT_PERM_COUNT; b++) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            for (u64 c = 0; c < FAT_PERM_COUNT; c++) {
                move_c = move_b | (c << 12);
                board_c = board_b;
                fatActions[board_c.getFatXY()][c](board_c);
                for (u64 d = 0; d < FAT_PERM_COUNT; d++) {
                    c_u64 move_d = move_c | (d << 18);
                    boards[count] = board_c;
                    fatActions[boards[count].getFatXY()][d](boards[count]);

                    boards[count].precomputeHash(colorCount);
                    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_d);
                    count++;
                }
            }
        }
    }

    return boards;
}


std::vector<Board> make_fat_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b, board_c, board_d;
    u64 move_b, move_c, move_d;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; a++) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        for (u64 b = 0; b < FAT_PERM_COUNT; b++) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            for (u64 c = 0; c < FAT_PERM_COUNT; c++) {
                move_c = move_b | (c << 12);
                board_c = board_b;
                fatActions[board_c.getFatXY()][c](board_c);
                for (u64 d = 0; d < FAT_PERM_COUNT; d++) {
                    move_d = move_c | (d << 18);
                    board_d = board_c;
                    fatActions[board_d.getFatXY()][d](board_d);
                    for (u64 e = 0; e < FAT_PERM_COUNT; e++) {
                        c_u64 move_e = move_d | (e << 24);
                        boards[count] = board_d;
                        fatActions[boards[count].getFatXY()][e](boards[count]);

                        boards[count].precomputeHash(colorCount);
                        (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_e);
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

