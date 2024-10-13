#pragma once

#include "reference.hpp"


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
void make_perm_list_inner(c_PERMOBJ_t &board_in,
                          vec_PERMOBJ_t &boards_out,
                          Ref<MAX_DEPTH> &ref,
                          c_u64 move_prev,
                          int& count) {

    if constexpr (MAX_DEPTH == 0) {
        boards_out.push_back(board_in);
        return;

    } else if constexpr (CUR_DEPTH == MAX_DEPTH) {
        // Base case: process and store the final board
        boards_out[count] = board_in;
        (boards_out[count].*ref.hasher)();
        boards_out[count].getMemory().setNextNMove<MAX_DEPTH>(move_prev);
        ++count;
        return;

    } else {

        // Handle intersection checks if needed
        if constexpr (CHECK_CROSS && CUR_DEPTH > 0) {
            if (ref.checkRC_seq[CUR_DEPTH]) {
                ref.intersect_seq[CUR_DEPTH] = board_in.doActISColMatchBatched(
                        ref.sect_seq[CUR_DEPTH - 1],
                        ref.sect_seq[CUR_DEPTH],
                        ref.cur_seq[CUR_DEPTH - 1] - ref.base_seq[CUR_DEPTH - 1] + 1);
            }
        }

        // Loop over cur indices
        int base = ref.base_seq[CUR_DEPTH];
        u64& cur = ref.cur_seq[CUR_DEPTH];

        for (cur = base; cur < base + 5; ++cur) {

            PERMOBJ_t board_next = board_in;
            allActionsList[cur](board_next);

            if constexpr (CHECK_SIM) {
                if (board_in.b1 == board_next.b1 && board_in.b2 == board_next.b2) { continue; }
            }

            // check intersection here
            if constexpr (CHECK_CROSS) {
                if constexpr (CUR_DEPTH > 0) {
                    if (ref.checkRC_seq[CUR_DEPTH] && ref.intersect_seq[CUR_DEPTH] & (1 << (cur - base))) { continue; }
                }
            }

            // Update move
            u64 move_next = move_prev | (cur << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));

            // Recursive call to the next depth
            make_perm_list_inner<CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
                    board_next, boards_out, ref, move_next, count);
        }
    }
}


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
void make_perm_list_outer(c_PERMOBJ_t &board_in,
                          vec_PERMOBJ_t &boards_out,
                          Ref<MAX_DEPTH> &ref,
                          int& count) {
    if constexpr (CUR_DEPTH == MAX_DEPTH) {
        // All depths have been processed; start the inner loops
        make_perm_list_inner<0, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(board_in, boards_out, ref, 0, count);
        return;
    } else {
        for (int dir = 0; dir < 2; ++dir) {
            ref.dir_seq[CUR_DEPTH] = dir;

            i32 sect_start;
            if constexpr (CUR_DEPTH != 0) {
                sect_start = (ref.dir_seq[CUR_DEPTH] == ref.dir_seq[CUR_DEPTH - 1])
                                     ? ref.sect_seq[CUR_DEPTH - 1] + 1 : 0;
            } else {
                sect_start = 0;
            }

            for (int sect = sect_start; sect < 6; ++sect) {
                ref.sect_seq[CUR_DEPTH] = sect;
                ref.base_seq[CUR_DEPTH] = dir * 30 + sect * 5;

                // Determine do_RC_check
                if (CUR_DEPTH > 0) {
                    int prev_dir = ref.dir_seq[CUR_DEPTH - 1];
                    ref.checkRC_seq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                }

                // Recursive call to the next depth
                make_perm_list_outer<CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
                        board_in, boards_out, ref, count);
            }
        }
    }
}


template<int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
void make_perm_list(c_PERMOBJ_t &board_in,
                    vec_PERMOBJ_t &boards_out,
                    c_PERMOBJ_t::HasherPtr hasher) {
    Ref<MAX_DEPTH> ref;
    ref.hasher = hasher;
    int count = 0;

    make_perm_list_outer<0, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
            board_in, boards_out, ref, count);
    boards_out.resize(count);
}


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_SIM>
void make_fat_perm_list_recursive_helper(
        c_PERMOBJ_t &board,
        vec_PERMOBJ_t &boards_out,
        c_PERMOBJ_t::HasherPtr hasher,
        u64 move,
        u32& count) {
    static constexpr u64 FAT_PERM_COUNT = 48;

    // Get the function indexes for the current board state
    u8 *funcIndexes = fatActionsIndexes[board.getFatXY()];

    for (u64 actn_i = 0; actn_i < FAT_PERM_COUNT; ++actn_i) {
        PERMOBJ_t board_next = board;
        allActionsList[funcIndexes[actn_i]](board_next);

        if constexpr (CHECK_SIM) {
            if (board.b1 == board_next.b1 && board.b2 == board_next.b2) { continue; }
        }

        // Update move
        u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

        if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
            // Base case: process and store the final board
            boards_out[count] = board_next;
            (boards_out[count].*hasher)();
            boards_out[count].getMemory().setNextNMove<MAX_DEPTH>(move_next);
            count++;
        } else {
            // Recursive call to the next depth
            make_fat_perm_list_recursive_helper<CUR_DEPTH + 1, MAX_DEPTH, CHECK_SIM>(
                    board_next, boards_out, hasher, move_next, count);
        }
    }
}


template<int DEPTH, bool CHECK_SIM>
void make_fat_perm_list(c_PERMOBJ_t &board_in,
                        vec_PERMOBJ_t &boards_out,
                        c_PERMOBJ_t::HasherPtr hasher) {
    if constexpr (DEPTH == 0) {
        // Special case for depth 0
        boards_out[0] = board_in;
        (boards_out[0].*hasher)();
        boards_out.resize(1);
    } else {
        u32 count = 0;
        make_fat_perm_list_recursive_helper<0, DEPTH, CHECK_SIM>(
                board_in, boards_out, hasher, 0, count);
        boards_out.resize(count);
    }
}
