#include "perms.hpp"
#include "rotations.hpp"


#define CONTINUE_IF_EQUIV(board1, board2) \
    if constexpr (CHECK_SIMILAR) { \
        if (board1.b1 == board2.b1 && board1.b2 == board2.b2) { continue; } \
    }


#define INTERSECT_MAKE_CACHE(do_RC_check, intersects, board, sect1, sect2, amount1) \
    if constexpr (CHECK_INTERSECTION) { \
        if (do_RC_check) { \
            intersects = board.doActISColMatchBatched(sect1, sect2, amount1); \
        } \
    }

#define INTERSECT_CONTINUE_IF_CACHE(rc_check, intersects, offset) \
    if constexpr (CHECK_INTERSECTION) { \
        if (rc_check && intersects & (1 << (offset))) { continue; } \
    }






static constexpr u64 FAT_PERM_COUNT = 48;
static constexpr bool CHECK_FAT_SIMILAR = true;


void make_fat_permutation_list_depth_0(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 0;
    boards_out[0] = board_in;
    boards_out[0].precomputeHash(colorCount);
}


template<bool CHECK_SIMILAR = CHECK_FAT_SIMILAR>
void make_fat_permutation_list_depth_1(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 1;
    u32 count = 0;

    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        boards_out[count] = board_in;
        fatActions[boards_out[count].getFatXY()][a](boards_out[count]);
        CONTINUE_IF_EQUIV(board_in, boards_out[count])

        boards_out[count].precomputeHash(colorCount);
        (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(a);
        ++count;
    }

    boards_out.resize(count);
}


template<bool CHECK_SIMILAR = CHECK_FAT_SIMILAR>
void make_fat_permutation_list_depth_2(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 2;
    Board board_a;
    u32 count = 0;

    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board_in;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board_in, board_a)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            c_u64 move = a | b << 6;
            boards_out[count] = board_a;
            fatActions[boards_out[count].getFatXY()][b](boards_out[count]);
            CONTINUE_IF_EQUIV(board_a, boards_out[count])

            boards_out[count].precomputeHash(colorCount);
            (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
            ++count;
        }
    }

    boards_out.resize(count);
}


template<bool CHECK_SIMILAR = CHECK_FAT_SIMILAR>
void make_fat_permutation_list_depth_3(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 3;
    Board board_a, board_b;
    u64 move_b;
    u32 count = 0;

    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board_in;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board_in, board_a)

        for (u64 b = 0; b < FAT_PERM_COUNT; ++b) {
            move_b = a | (b << 6);
            board_b = board_a;
            fatActions[board_b.getFatXY()][b](board_b);
            CONTINUE_IF_EQUIV(board_a, board_b)


            for (u64 c = 0; c < FAT_PERM_COUNT; ++c) {
                c_u64 move = move_b | (c << 12);
                boards_out[count] = board_b;
                fatActions[boards_out[count].getFatXY()][c](boards_out[count]);
                CONTINUE_IF_EQUIV(board_b, boards_out[count])

                boards_out[count].precomputeHash(colorCount);
                (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
                ++count;
            }
        }
    }

    boards_out.resize(count);
}


template<bool CHECK_SIMILAR = CHECK_FAT_SIMILAR>
void make_fat_permutation_list_depth_4(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
    Board board_a, board_b, board_c;
    u64 move_b, move_c;
    u32 count = 0;

    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board_in;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board_in, board_a)

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
                    c_u64 move_d = move_c | (d << 18);
                    boards_out[count] = board_c;
                    fatActions[boards_out[count].getFatXY()][d](boards_out[count]);
                    CONTINUE_IF_EQUIV(board_c, boards_out[count])

                    boards_out[count].precomputeHash(colorCount);
                    (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move_d);
                    ++count;
                }
            }
        }
    }

    boards_out.resize(count);
}


template<bool CHECK_SIMILAR = CHECK_FAT_SIMILAR>
void make_fat_permutation_list_depth_5(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
    Board board_a, board_b, board_c, board_d;
    u64 move_b, move_c, move_d;
    u32 count = 0;

    for (u64 a = 0; a < FAT_PERM_COUNT; ++a) {
        board_a = board_in;
        fatActions[board_a.getFatXY()][a](board_a);
        CONTINUE_IF_EQUIV(board_in, board_a)

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
                        boards_out[count] = board_d;
                        fatActions[boards_out[count].getFatXY()][e](boards_out[count]);
                        CONTINUE_IF_EQUIV(board_d, boards_out[count])

                        boards_out[count].precomputeHash(colorCount);
                        (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move_e);
                        ++count;
                    }
                }
            }
        }
    }

    boards_out.resize(count);
}










void make_permutation_list_depth_0(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    boards_out[0].precomputeHash(colorCount);
}


void make_permutation_list_depth_1(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr i32 DEPTH = 1;
    i32 count = 0;

    for (i32 a = 0; a < 60; ++a) {
        Board *currentBoard = &boards_out[count];
        *currentBoard = board_in;
        actions[a](*currentBoard);
        currentBoard->precomputeHash(colorCount);
        u64 move = a;
        (currentBoard->mem.*setNextMoveFuncs[DEPTH])(move);
        count++;
    }
}


template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
void make_permutation_list_depth_2(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr i32 DEPTH = 2;
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
                        board_a = board_in;
                        actions[a_cur](board_a);
                        CONTINUE_IF_EQUIV(board_in, board_a)

                        INTERSECT_MAKE_CACHE(do_RC_check, intersects, board_in, a_sect, b_sect, a_cur - a_base + 1)
                        for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check, intersects, b_cur - b_base)
                            boards_out[count] = board_a;
                            actions[b_cur](boards_out[count]);
                            CONTINUE_IF_EQUIV(board_a, boards_out[count])

                            u64 move = a_cur | (b_cur << 6);
                            boards_out[count].precomputeHash(colorCount);
                            (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
                            ++count;
                        }
                    }
                }
            }
        }
    }

    boards_out.resize(count);
}


template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
void make_permutation_list_depth_3(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr i32 DEPTH = 3;
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
                                board_a = board_in;
                                actions[a_cur](board_a);
                                CONTINUE_IF_EQUIV(board_in, board_a)

                                INTERSECT_MAKE_CACHE(do_RC_check[0], intersects[0], board_in, a_sect, b_sect, a_cur - a_base + 1)
                                for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], b_cur - b_base)
                                    board_b = board_a;
                                    actions[b_cur](board_b);
                                    CONTINUE_IF_EQUIV(board_a, board_b)

                                    INTERSECT_MAKE_CACHE(do_RC_check[1], intersects[1], board_in, b_sect, c_sect, b_cur - b_base + 1)
                                    for (i32 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                        INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], c_cur - c_base)
                                        boards_out[count] = board_b;
                                        actions[c_cur](boards_out[count]);
                                        CONTINUE_IF_EQUIV(board_b, boards_out[count])

                                        u64 move = a_cur | (b_cur << 6) | (c_cur << 12);
                                        boards_out[count].precomputeHash(colorCount);
                                        (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
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

    boards_out.resize(count);
}


template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
void make_permutation_list_depth_4(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 4;
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
                                        board_a = board_in;
                                        actions[a_cur](board_a);
                                        CONTINUE_IF_EQUIV(board_in, board_a)

                                        INTERSECT_MAKE_CACHE(do_RC_check[0], intersects[0], board_in, a_sect, b_sect, a_cur - a_base + 1)
                                        for (i32 b_cur = b_base; b_cur < b_base + 5; ++b_cur) {
                                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], b_cur - b_base)

                                            board_b = board_a;
                                            actions[b_cur](board_b);
                                            CONTINUE_IF_EQUIV(board_a, board_b)

                                            INTERSECT_MAKE_CACHE(do_RC_check[1], intersects[1], board_in, b_sect, c_sect, b_cur - b_base + 1)
                                            for (i32 c_cur = c_base; c_cur < c_base + 5; ++c_cur) {
                                                INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], c_cur - c_base)
                                                board_c = board_b;
                                                actions[c_cur](board_c);
                                                CONTINUE_IF_EQUIV(board_b, board_c)

                                                INTERSECT_MAKE_CACHE(do_RC_check[2], intersects[2], board_in, c_sect, d_sect, c_cur - c_base + 1)
                                                // #pragma unroll 5
                                                for (i32 d_cur = d_base; d_cur < d_base + 5; ++d_cur) {
                                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[2], intersects[2], d_cur - d_base)

                                                    boards_out[count] = board_c;
                                                    actions[d_cur](boards_out[count]);
                                                    CONTINUE_IF_EQUIV(board_c, boards_out[count])

                                                    u64 move = a_cur | (b_cur << 6) | (c_cur << 12) | (d_cur << 18);
                                                    boards_out[count].precomputeHash(colorCount);
                                                    (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
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

    boards_out.resize(count);
}


template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
void make_permutation_list_depth_5(vecBoard_t& boards_out, const Board &board_in, c_u32 colorCount) {
    static constexpr u32 DEPTH = 5;
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
                                                board_a = board_in;
                                                actions[curr[0]](board_a);
                                                CONTINUE_IF_EQUIV(board_in, board_a)

                                                INTERSECT_MAKE_CACHE(do_RC_check[0], intersects[0], board_in, sect[0], sect[1], curr[0] - base[0] + 1)
                                                for (curr[1] = base[1]; curr[1] < base[1] + 5; ++curr[1]) {
                                                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check[0], intersects[0], curr[1] - base[1])
                                                    board_b = board_a;
                                                    actions[curr[1]](board_b);
                                                    CONTINUE_IF_EQUIV(board_a, board_b)

                                                    INTERSECT_MAKE_CACHE(do_RC_check[1], intersects[1], board_in, sect[1], sect[2], curr[1] - base[1] + 1)
                                                    for (curr[2] = base[2]; curr[2] < base[2] + 5; ++curr[2]) {
                                                        INTERSECT_CONTINUE_IF_CACHE(do_RC_check[1], intersects[1], curr[2] - base[2])
                                                        board_c = board_b;
                                                        actions[curr[2]](board_c);
                                                        CONTINUE_IF_EQUIV(board_b, board_c)

                                                        INTERSECT_MAKE_CACHE(do_RC_check[2], intersects[2], board_in, sect[2], sect[3], curr[2] - base[2] + 1)
                                                        for (curr[3] = base[3]; curr[3] < base[3] + 5; ++curr[3]) {
                                                            INTERSECT_CONTINUE_IF_CACHE(do_RC_check[2], intersects[2], curr[3] - base[3])
                                                            board_d = board_c;
                                                            actions[curr[3]](board_d);
                                                            CONTINUE_IF_EQUIV(board_c, board_d)

                                                            INTERSECT_MAKE_CACHE(do_RC_check[3], intersects[3], board_in, sect[3], sect[4], curr[3] - base[3] + 1)
                                                            for (curr[4] = base[4]; curr[4] < base[4] + 5; ++curr[4]) {
                                                                INTERSECT_CONTINUE_IF_CACHE(do_RC_check[3], intersects[3], curr[4] - base[4])
                                                                boards_out[count] = board_d;
                                                                actions[curr[4]](boards_out[count]);
                                                                CONTINUE_IF_EQUIV(board_d, boards_out[count])

                                                                u64 move = curr[0] | (curr[1] << 6) | (curr[2] << 12)
                                                                           | (curr[3] << 18) | (curr[4] << 24);
                                                                boards_out[count].precomputeHash(colorCount);
                                                                (boards_out[count].mem.*setNextMoveFuncs[DEPTH])(move);
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

    boards_out.resize(count);
}








template<bool CHECK_INTERSECTION = true, bool CHECK_SIMILAR = true>
void make_permutation_list_depth_plus_one(const vecBoard_t & boards_in, vecBoard_t & boards_out, c_u32 colorCount) {
    int count = 0;
    u8 intersects ;

    for (int board_index = 0; board_index < boards_in.size(); board_index++) {
        u8 a = boards_in[board_index].mem.getLastMove();
        u8 a_dir = a / 30;
        u8 a_sect = (a % 30) / 5;
        u8 a_amount = a % 5 + 1;


        for (int b_dir = 0; b_dir < 2; b_dir++) {
            bool do_RC_check = a_dir != b_dir && a_dir != 0;

            int b_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_start; b_sect < 6; b_sect++) {
                int b_base = b_dir * 30 + b_sect * 5;

                INTERSECT_MAKE_CACHE(do_RC_check, intersects, boards_in[board_index],
                                     a_sect, b_sect, a_amount)

                for (int b_amount = 0; b_amount < 5; b_amount++) {
                    INTERSECT_CONTINUE_IF_CACHE(do_RC_check, intersects, b_amount)

                    c_int b_cur = b_base + b_amount;

                    boards_out[count] = boards_in[board_index];
                    actions[b_cur](boards_out[count]);
                    CONTINUE_IF_EQUIV(boards_out[count], boards_in[board_index])

                    boards_out[count].precomputeHash(colorCount);
                    boards_out[count].mem.setNext1Move(b_cur);
                    count++;
                }
            }
        }

    }

    boards_out.resize(count);
}



Permutations::toDepthFuncPtr_t Permutations::toDepthFatFuncPtrs[] = {
        &make_fat_permutation_list_depth_0,
        &make_fat_permutation_list_depth_1,
        &make_fat_permutation_list_depth_2,
        &make_fat_permutation_list_depth_3,
        &make_fat_permutation_list_depth_4,
        &make_fat_permutation_list_depth_5
};


Permutations::toDepthFuncPtr_t Permutations::toDepthFuncPtrs[] = {
        &make_permutation_list_depth_0,
        &make_permutation_list_depth_1,
        &make_permutation_list_depth_2,
        &make_permutation_list_depth_3,
        &make_permutation_list_depth_4,
        &make_permutation_list_depth_5};


Permutations::toDepthPlusOneFuncPtr_t Permutations::toDepthPlusOneFuncPtr
        = make_permutation_list_depth_plus_one;


/*
static Permutations::depthMap_t makeDepthMap(c_u32 maxDepthTotal, c_u32 maxDepthOneSide) {
    Permutations::depthMap_t depthMap;

    for (int totalDepth = 1; totalDepth < maxDepthTotal; )

}*/


const Permutations::depthMap_t Permutations::depthMap = {
        {1, {{1, 0}, {0, 1}}},
        {2, {{1, 1}, {0, 2}, {2, 0}}},
        {3, {{1, 2}, {2, 1}, {0, 3}, {3, 0}}},
        {4, {{2, 2}, {3, 1}, {1, 3}, {4, 0}, {0, 4}}},
        {5, {{3, 2}, {3, 2}, {4, 1}, {1, 4}, {5, 0}, {0, 5}}},
        {6, {{3, 3}, {4, 2}, {2, 4}, {5, 1}, {1, 5}}},
        {7, {{4, 3}, {3, 4}, {5, 2}, {2, 5}}},
        {8, {{4, 4}, {5, 3}, {3, 5}}},
        {9, {{4, 5},{5, 4}}},
        {10, {{5, 5}}},
};


MU void Permutations::reserveForDepth(const Board& board_in, vecBoard_t& boards_out, c_u32 depth, c_bool isFat) {
    double fraction = board_in.getDuplicateEstimateAtDepth(depth);
    u64 allocSize = isFat ? BOARD_FAT_PRE_ALLOC_SIZES[depth] : BOARD_PRE_ALLOC_SIZES[depth];

    allocSize = u64((double(allocSize)) * fraction);
    boards_out.reserve(allocSize);
}


void Permutations::getDepthFunc(const Board &board_in, vecBoard_t &boards_out, u32 depth, c_bool shouldResize) {
    if (depth >= PTR_LIST_SIZE) {
        return;
    }

    const bool isFat = board_in.hasFat();

    u8 colorCount = board_in.getColorCount();

    if (shouldResize) {
        reserveForDepth(board_in, boards_out, depth, isFat);
    }
    boards_out.resize(boards_out.capacity());


    if (!isFat) {
        toDepthFuncPtrs[depth](boards_out, board_in, colorCount);
    } else {
        toDepthFatFuncPtrs[depth](boards_out, board_in, colorCount);
    }

}


void Permutations::getDepthPlus1Func(const vecBoard_t& boards_in, vecBoard_t& boards_out, bool shouldResize) {
    u8 colorCount = boards_in[0].getColorCount();

    if (shouldResize) {
        boards_out.resize(boards_in.size() * 60);
    }
    boards_out.resize(boards_out.capacity());

    toDepthPlusOneFuncPtr(boards_in, boards_out, colorCount);
}
