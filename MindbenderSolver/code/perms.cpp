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


#define CONTINUE_IF_EQUIV(board1, board2) \
    if constexpr (CHECK_SIMILAR) { \
        if (board1.b1 == board2.b1 && board1.b2 == board2.b2) { continue; } \
    }


#define INTERSECT_MAKE_CACHE(intersects, board, sect1, sect2, amount1) \
    if constexpr (CHECK_INTERSECTION) { \
        intersects = board.doActISColMatchBatched(sect1, sect2, amount1); \
    }

#define INTERSECT_CONTINUE_IF_CACHE(rc_check, intersects, offset) \
    if constexpr (CHECK_INTERSECTION) { \
        if (rc_check && intersects & (1 << (offset))) { continue; } \
    }



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
                        for (i32 a_cur = a * 5; a_cur < a * 5 + 5; ++a_cur) {
                            board_a = board;
                            actions[a_cur](board_a);
                            move_a = a_cur;
                            for (i32 b_cur = b * 5; b_cur < b * 5 + 5; ++b_cur) {
                                board_b = board_a;
                                actions[b_cur](board_b);
                                move_b = move_a | (b_cur << 6);
                                for (i32 c_cur = c * 5; c_cur < c * 5 + 5; ++c_cur) {
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);
                                    boards[count].precomputeHash(colorCount);
                                    u64 move = move_b | (c_cur << 12);
                                    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                                    ++count;
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
    for (u32 a = 0; a < 12; ++a) {
        c_u32 a_base = a * 5;
        c_u32 a_dir = a / 6;
        c_u32 a_sect = a % 6;
        for (u32 b = 0; b < 12; ++b) {
            c_u32 b_base = b * 5;
            c_u32 b_dir = b / 6;
            c_u32 b_sect = b % 6;
            if (a_dir == b_dir && a_sect >= b_sect) { continue; }
            for (u32 c = 0; c < 12; ++c) {
                c_u32 c_base = c * 5;
                c_u32 c_dir = c / 6;
                c_u32 c_sect = c % 6;
                if (b_dir == c_dir && b_sect >= c_sect) { continue; }
                for (u32 d = 0; d < 12; ++d) {
                    c_u32 d_base = d * 5;
                    c_u32 d_dir = d / 6;
                    c_u32 d_sect = d % 6;
                    if (c_dir == d_dir && c_sect >= d_sect) { continue; }
                    // actual creation part
                    for (u32 a_cur = a_base; a_cur < a_base + 5; ++a_cur) {

                        board_a = board;
                        actions[a_cur](board_a);

                        std::fill_n(&boards[count], 125, board_a);
                        for (u32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                            move_b = a_cur | (b_cur << 6);
                            actions[b_cur](boards[count]);

                            std::fill_n(&boards[count], 25, boards[count]);
                            for (u32 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                move_c = move_b | (c_cur << 12);
                                actions[c_cur](boards[count]);

                                std::fill_n(&boards[count], 5, boards[count]);
                                for (u32 d_cur = d_base; d_cur < d_base + 5; ++d_cur) {
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
    for (u32 a = 0; a < 12; ++a) {
        c_u32 a_base = a * 5;
        c_u32 a_dir = a / 6;
        c_u32 a_sect = a % 6;
        for (u32 b = 0; b < 12; ++b) {
            c_u32 b_base = b * 5;
            c_u32 b_dir = b / 6;
            c_u32 b_sect = b % 6;
            if (a_dir == b_dir && a_sect >= b_sect) { continue; }
            for (u32 c = 0; c < 12; ++c) {
                c_u32 c_base = c * 5;
                c_u32 c_dir = c / 6;
                c_u32 c_sect = c % 6;
                if (b_dir == c_dir && b_sect >= c_sect) { continue; }
                for (u32 d = 0; d < 12; ++d) {
                    c_u32 d_base = d * 5;
                    c_u32 d_dir = d / 6;
                    c_u32 d_sect = d % 6;
                    if (c_dir == d_dir && c_sect >= d_sect) { continue; }
                    for (u32 e = 0; e < 12; ++e) {
                        c_u32 e_base = e * 5;
                        c_u32 e_dir = e / 6;
                        c_u32 e_sect = e % 6;
                        if (d_dir == e_dir && d_sect >= e_sect) { continue; }
                        // actual creation part
                        for (u64 a_cur = a_base; a_cur < a_base + 5; ++a_cur) {
                            board_a = board;
                            actions[a_cur](board_a);
                            for (u64 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                                move_b = a_cur | (u64(b_cur) << 6);
                                board_b = board_a;
                                actions[b_cur](board_b);

                                for (u64 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                    move_c = move_b | (u64(c_cur) << 12);
                                    boards[count] = board_b;
                                    actions[c_cur](boards[count]);

                                    std::fill_n(&boards[count], 25, boards[count]);
                                    for (u64 d_cur = d_base; d_cur < d_base + 5; ++d_cur) {
                                        move_d = move_c | (u64(d_cur) << 18);
                                        actions[d_cur](boards[count]);

                                        std::fill_n(&boards[count], 5, boards[count]);
                                        for (u64 e_cur = e_base; e_cur < e_base + 5; ++e_cur) {
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


template<bool CHECK_SIMILAR = true>
std::vector<Board> make_fat_permutation_list_depth_1(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 1;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        boards[count] = board;
        fatActions[boards[count].getFatXY()][a](boards[count]);
        CONTINUE_IF_EQUIV(board, boards[count])
        
        boards[count].precomputeHash(colorCount);
        (boards[count].mem.*setNextMoveFuncs[DEPTH])(a);
        ++count;
    }

    return boards;
}


template<bool CHECK_SIMILAR = true>
std::vector<Board> make_fat_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 2;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board, board_a)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            c_u64 move = a | b << 6;
            boards[count] = board_a;
            fatActions[boards[count].getFatXY()][b](boards[count]);
            CONTINUE_IF_EQUIV(board_a, boards[count])

            boards[count].precomputeHash(colorCount);
            (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
            ++count;
        }
    }


    return boards;
}


template<bool CHECK_SIMILAR = true>
std::vector<Board> make_fat_permutation_list_depth_3(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 3;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b;
    u64 move_b;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board_a, board)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            CONTINUE_IF_EQUIV(board_a, board_b)


            for (u64 c = 0; c < FAT_PERM_COUNT; ++c) {
                c_u64 move = move_b | (c << 12);
                boards[count] = board_b;
                fatActions[boards[count].getFatXY()][c](boards[count]);
                CONTINUE_IF_EQUIV(board_b, boards[count])

                boards[count].precomputeHash(colorCount);
                (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                ++count;
            }
        }
    }

    return boards;
}




template<bool CHECK_SIMILAR = true>
std::vector<Board> make_fat_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b, board_c;
    u64 move_b, move_c;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board, board_a)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            CONTINUE_IF_EQUIV(board_b, board_a)

            for (u64 c = 0; c < FAT_PERM_COUNT; ++c) {
                move_c = move_b | (c << 12);
                board_c = board_b;
                fatActions[board_c.getFatXY()][c](board_c);
                CONTINUE_IF_EQUIV(board_b, board_c)

                for (u64 d = 0; d < FAT_PERM_COUNT; ++d) {
                    c_u64 move_d = move_c | (d << 18);
                    boards[count] = board_c;
                    fatActions[boards[count].getFatXY()][d](boards[count]);
                    CONTINUE_IF_EQUIV(board_c, boards[count])

                    boards[count].precomputeHash(colorCount);
                    (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_d);
                    ++count;
                }
            }
        }
    }

    return boards;
}


template<bool CHECK_SIMILAR = true>
std::vector<Board> make_fat_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
    std::vector<Board> boards(BOARD_FAT_PRE_ALLOC_SIZES[DEPTH]);
    Board board_a, board_b, board_c, board_d;
    u64 move_b, move_c, move_d;

    u32 count = 0;
    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board, board_a)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            CONTINUE_IF_EQUIV(board_a, board_b)

            for (u64 c = 0; c < FAT_PERM_COUNT; ++c) {
                move_c = move_b | (c << 12);
                board_c = board_b;
                fatActions[board_c.getFatXY()][c](board_c);
                CONTINUE_IF_EQUIV(board_b, board_c)

                for (u64 d = 0; d < FAT_PERM_COUNT; ++d) {
                    move_d = move_c | (d << 18);
                    board_d = board_c;
                    fatActions[board_d.getFatXY()][d](board_d);
                    CONTINUE_IF_EQUIV(board_c, board_d)

                    for (u64 e = 0; e < FAT_PERM_COUNT; ++e) {
                        c_u64 move_e = move_d | (e << 24);
                        boards[count] = board_d;
                        fatActions[boards[count].getFatXY()][e](boards[count]);
                        CONTINUE_IF_EQUIV(board_d, boards[count])

                        boards[count].precomputeHash(colorCount);
                        (boards[count].mem.*setNextMoveFuncs[DEPTH])(move_e);
                        ++count;
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




template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
std::vector<Board> make2_permutation_list_depth_2(Board &board, c_u32 colorCount) {
    static constexpr i32 DEPTH = 2;
    std::vector<Board> boards(2550);
    Board board_a;
    u8 intersects = 0;
    i32 count = 0;

    for (i32 a_dir = 0; a_dir < 2; ++a_dir) {
        for (i32 a_sect = 0; a_sect < 6; ++a_sect) {
            i32 a_base = a_dir * 30 + a_sect * 5;

            for (i32 b_dir = 0; b_dir < 2; ++b_dir) {
                bool do_RC_check = a_dir != b_dir && a_dir != 0;
                i32 b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
                for (i32 b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                    i32 b_base = b_dir * 30 + b_sect * 5;

                    for (i32 a_cur = a_base; a_cur < a_base + 5; ++a_cur) {
                        board_a = board;
                        actions[a_cur](board_a);
                        CONTINUE_IF_EQUIV(board, board_a)

                        INTERSECT_MAKE_CACHE(intersects, board, a_sect, b_sect, a_cur - a_base + 1)
                        for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check, intersects, b_cur - b_base)
                            boards[count] = board_a;
                            actions[b_cur](boards[count]);
                            CONTINUE_IF_EQUIV(board_a, boards[count])

                            u64 move = a_cur | (b_cur << 6);
                            boards[count].precomputeHash(colorCount);
                            (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                            ++count;
                        }
                    }
                }
            }
        }
    }

    boards.resize(count);
    return boards;
}





template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
std::vector<Board> make2_permutation_list_depth_3(Board &board, c_u32 colorCount) {
    static constexpr i32 DEPTH = 3;
    std::vector<Board> boards(104000);
    Board board_a, board_b;

    bool do_RC_check[2] = {false};
    u8 intersects[2] = {0};

    i32 count = 0;

    for (i32 a_dir = 0; a_dir < 2; ++a_dir) {
        for (i32 a_sect = 0; a_sect < 6; ++a_sect) {
            i32 a_base = a_dir * 30 + a_sect * 5;

            for (i32 b_dir = 0; b_dir < 2; ++b_dir) {
                do_RC_check[0] = a_dir != b_dir && a_dir != 0;
                i32 b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
                for (i32 b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                    i32 b_base = b_dir * 30 + b_sect * 5;

                    for (i32 c_dir = 0; c_dir < 2; ++c_dir) {
                        do_RC_check[1] = b_dir != c_dir && b_dir != 0;
                        i32 c_sect_start = (c_dir == b_dir) ? b_sect + 1 : 0;
                        for (i32 c_sect = c_sect_start; c_sect < 6; ++c_sect) {
                            i32 c_base = c_dir * 30 + c_sect * 5;

                            for (i32 a_cur = a_base; a_cur < a_base + 5; ++a_cur) {
                                board_a = board;
                                actions[a_cur](board_a);
                                CONTINUE_IF_EQUIV(board, board_a)

                                INTERSECT_MAKE_CACHE(intersects[0], board, a_sect, b_sect, a_cur - a_base + 1)
                                for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], b_cur - b_base)
                                    board_b = board_a;
                                    actions[b_cur](board_b);
                                    CONTINUE_IF_EQUIV(board_a, board_b)

                                    INTERSECT_MAKE_CACHE(intersects[1], board, b_sect, c_sect, b_cur - b_base + 1)
                                    for (i32 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                        INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], c_cur - c_base)
                                        boards[count] = board_b;
                                        actions[c_cur](boards[count]);
                                        CONTINUE_IF_EQUIV(board_b, boards[count])

                                        u64 move = a_cur | (b_cur << 6) | (c_cur << 12);
                                        boards[count].precomputeHash(colorCount);
                                        (boards[count].mem.*setNextMoveFuncs[DEPTH])(move);
                                        ++count;

                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    boards.resize(count);
    return boards;
}



template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
std::vector<Board> make2_permutation_list_depth_4(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
    std::vector<Board> boards(4245000);
    Board board_a, board_b, board_c;

    bool do_RC_check[3] = {false};
    u8 intersects[3] = {0};

    u32 count = 0;

    for (i32 a_dir = 0; a_dir < 2; ++a_dir) {
        for (i32 a_sect = 0; a_sect < 6; ++a_sect) {
            i32 a_base = a_dir * 30 + a_sect * 5;

            for (i32 b_dir = 0; b_dir < 2; ++b_dir) {
                do_RC_check[0] = a_dir != b_dir && a_dir != 0;
                i32 b_sect_start = (b_dir == a_dir) ? a_sect + 1 : 0;
                for (i32 b_sect = b_sect_start; b_sect < 6; ++b_sect) {
                    i32 b_base = b_dir * 30 + b_sect * 5;

                    for (i32 c_dir = 0; c_dir < 2; ++c_dir) {
                        do_RC_check[1] = b_dir != c_dir && b_dir != 0;
                        i32 c_sect_start = (c_dir == b_dir) ? b_sect + 1 : 0;
                        for (i32 c_sect = c_sect_start; c_sect < 6; ++c_sect) {
                            i32 c_base = c_dir * 30 + c_sect * 5;

                            for (i32 d_dir = 0; d_dir < 2; ++d_dir) {
                                do_RC_check[2] = c_dir != d_dir && c_dir != 0;
                                i32 d_sect_start = (d_dir == c_dir) ? c_sect + 1 : 0;
                                for (i32 d_sect = d_sect_start; d_sect < 6; ++d_sect) {
                                    i32 d_base = d_dir * 30 + d_sect * 5;


                                    for (i32 a_cur = a_base; a_cur < a_base + 5; ++a_cur) {
                                        board_a = board;
                                        actions[a_cur](board_a);
                                        CONTINUE_IF_EQUIV(board, board_a)

                                        INTERSECT_MAKE_CACHE(intersects[0], board, a_sect, b_sect, a_cur - a_base + 1)
                                        for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], b_cur - b_base)

                                            board_b = board_a;
                                            actions[b_cur](board_b);
                                            CONTINUE_IF_EQUIV(board_a, board_b)

                                            INTERSECT_MAKE_CACHE(intersects[1], board, b_sect, c_sect, b_cur - b_base + 1)
                                            for (i32 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                                INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], c_cur - c_base)
                                                board_c = board_b;
                                                actions[c_cur](board_c);
                                                CONTINUE_IF_EQUIV(board_b, board_c)

                                                INTERSECT_MAKE_CACHE(intersects[2], board, c_sect, d_sect, c_cur - c_base + 1)
                                                for (i32 d_cur = d_base; d_cur < d_base + 5; ++d_cur) {
                                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[2], intersects[2], d_cur - d_base)

                                                    boards[count] = board_c;
                                                    actions[d_cur](boards[count]);
                                                    CONTINUE_IF_EQUIV(board_c, boards[count])

                                                    u64 move = a_cur | (b_cur << 6) | (c_cur << 12) | (d_cur << 18);
                                                    boards[count].precomputeHash(colorCount);
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
        }
    }
    boards.resize(count);
    return boards;
}


template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
std::vector<Board> make2_permutation_list_depth_5(Board &board, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
    std::vector<Board> boards(173325000);
    Board board_a, board_b, board_c, board_d;



    bool do_RC_check[4] = {false};
    u8 intersects[4] = {0};
    i32 sect_start[4] = {0};
    i32 dir[5] = {0};
    i32 sect[5] = {0};
    i32 base[5] = {0};
    i32 curr[5] = {0};

    u32 count = 0;

    for (dir[0] = 0; dir[0] < 2; ++dir[0]) {
        for (sect[0] = 0; sect[0] < 6; ++sect[0]) {
            base[0] = dir[0] * 30 + sect[0] * 5;
            for (dir[1] = 0; dir[1] < 2; ++dir[1]) {
                do_RC_check[0] = dir[0] != dir[1] && dir[0] != 0;
                sect_start[0] = (dir[1] == dir[0]) ? sect[0] + 1 : 0;
                for (sect[1] = sect_start[0]; sect[1] < 6; ++sect[1]) {
                    base[1] = dir[1] * 30 + sect[1] * 5;
                    for (dir[2] = 0; dir[2] < 2; ++dir[2]) {
                        do_RC_check[1] = dir[1] != dir[2] && dir[1] != 0;
                        sect_start[1] = (dir[2] == dir[1]) ? sect[1] + 1 : 0;
                        for (sect[2] = sect_start[1]; sect[2] < 6; ++sect[2]) {
                            base[2] = dir[2] * 30 + sect[2] * 5;
                            for (dir[3] = 0; dir[3] < 2; ++dir[3]) {
                                do_RC_check[2] = dir[2] != dir[3] && dir[2] != 0;
                                sect_start[2] = (dir[3] == dir[2]) ? sect[2] + 1 : 0;
                                for (sect[3] = sect_start[2]; sect[3] < 6; ++sect[3]) {
                                    base[3] = dir[3] * 30 + sect[3] * 5;
                                    for (dir[4] = 0; dir[4] < 2; ++dir[4]) {
                                        do_RC_check[3] = dir[3] != dir[4] && dir[3] != 0;
                                        sect_start[3] = (dir[4] == dir[3]) ? sect[3] + 1 : 0;
                                        for (sect[4] = sect_start[3]; sect[4] < 6; ++sect[4]) {
                                            base[4] = dir[4] * 30 + sect[4] * 5;

                                            for (curr[0] = base[0]; curr[0] < base[0] + 5; ++curr[0]) {
                                                board_a = board;
                                                actions[curr[0]](board_a);
                                                CONTINUE_IF_EQUIV(board, board_a)

                                                INTERSECT_MAKE_CACHE(intersects[0], board, sect[0], sect[1], curr[0] - base[0] + 1)
                                                for (curr[1] = base[1]; curr[1] < base[1] + 5; ++curr[1]) {
                                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], curr[1] - base[1])
                                                    board_b = board_a;
                                                    actions[curr[1]](board_b);
                                                    CONTINUE_IF_EQUIV(board_a, board_b)

                                                    INTERSECT_MAKE_CACHE(intersects[1], board, sect[1], sect[2], curr[1] - base[1] + 1)
                                                    for (curr[2] = base[2]; curr[2] < base[2] + 5; ++curr[2]) {
                                                        INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], curr[2] - base[2])
                                                        board_c = board_b;
                                                        actions[curr[2]](board_c);
                                                        CONTINUE_IF_EQUIV(board_b, board_c)

                                                        INTERSECT_MAKE_CACHE(intersects[2], board, sect[2], sect[3], curr[2] - base[2] + 1)
                                                        for (curr[3] = base[3]; curr[3] < base[3] + 5; ++curr[3]) {
                                                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check[2], intersects[2], curr[3] - base[3])
                                                            board_d = board_c;
                                                            actions[curr[3]](board_d);
                                                            CONTINUE_IF_EQUIV(board_c, board_d)

                                                            INTERSECT_MAKE_CACHE(intersects[3], board, sect[3], sect[4], curr[3] - base[3] + 1)
                                                            for (curr[4] = base[4]; curr[4] < base[4] + 5; ++curr[4]) {
                                                                INTERSECT_CONTINUE_IF_CACHE(do_RC_check[3], intersects[3], curr[4] - base[4])
                                                                boards[count] = board_d;
                                                                actions[curr[4]](boards[count]);
                                                                CONTINUE_IF_EQUIV(board_d, boards[count])

                                                                u64 move = curr[0] | (curr[1] << 6) | (curr[2] << 12)
                                                                           | (curr[3] << 18) | (curr[4] << 24);
                                                                boards[count].precomputeHash(colorCount);
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
                    }
                }
            }
        }
    }
    boards.resize(count);
    return boards;
}



MakePermFuncArray make2PermutationListFuncs[] = {
        &make_permutation_list_depth_0,
        &make_permutation_list_depth_1,
        &make2_permutation_list_depth_2,
        &make2_permutation_list_depth_3,
        &make2_permutation_list_depth_4,
        &make2_permutation_list_depth_5};
