#pragma once

#include "utils/processor.hpp"
#include "utils/hasGetHash.hpp"
#include "utils/jvec.hpp"

#include "board.hpp"
#include "rotations.hpp"
#include "allowed_type.hpp"
#include "reference.hpp"


template<typename T,
        int CUR_DEPTH, int MAX_DEPTH,
        bool MOVES_ASCENDING, bool DIRECTION
        >
static void make_fat_perm_list_helper(
        C Board &board,
        JVec<T> &boards_out,
        u32 &count,
        C typename T::HasherPtr hasher,
        C u64 move,
        C ActStruct& lastActStruct,
        C u8 startIndex,
        C u8 endIndex) {
    static_assert(AllowedPermsType<T>, "T must be Memory or Board");
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    // MAKE_FAT_PERM_LIST_HELPER_CALLS++;

    MU bool lastActIsRow;
    MU bool lastActIsCol;
    if constexpr (DIRECTION) {
        lastActIsRow = lastActStruct.isColNotFat & 2;
    } else {
        lastActIsCol = lastActStruct.isColNotFat & 1;
    }


    C u8 *funcIndexes = fatActionsIndexes[board.getFatXY()];

    for (u64 actn_i = startIndex; actn_i < endIndex; ++actn_i) {

        C ActStruct& actStruct = allActStructList[funcIndexes[actn_i]];
        Board board_next = board;
        actStruct.action(board_next);


        if constexpr (CUR_DEPTH != 0) {

            if constexpr (DIRECTION) {
                if (lastActIsRow && actStruct.isColNotFat & 2) {
                    if constexpr (MOVES_ASCENDING) {
                        if (lastActStruct.index <= actStruct.index) {
                            continue;
                        }
                    } else {
                        if (lastActStruct.index >= actStruct.index) {
                            continue;
                        }
                    }
                }
            }

            if constexpr (!DIRECTION) {
                if (lastActIsCol && actStruct.isColNotFat & 1) {
                    if constexpr (MOVES_ASCENDING) {
                        if (lastActStruct.index <= actStruct.index) {
                            // MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS++;
                            continue;
                        }
                    } else {
                        if (lastActStruct.index >= actStruct.index) {
                            // MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS++;
                            continue;
                        }
                    }
                }
            }
        }

        if (board == board_next) {
            // MAKE_FAT_PERM_LIST_HELPER_FOUND_SIMILAR++;
            continue;
        }

        if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
            // Base case: process and store the final board

            if constexpr (std::is_same_v<T, Memory>) {
                boards_out[count] = board_next.memory;
                (boards_out[count].*hasher)(board_next.b1, board_next.b2);
                u64 move_next = move | actn_i << 6 * CUR_DEPTH;
                boards_out[count].template setNextNMove<MAX_DEPTH>(move_next);
            } else if constexpr (std::is_same_v<T, Board>) {
                boards_out[count] = board_next;
                (boards_out[count].*hasher)();
                u64 move_next = move | actn_i << 6 * CUR_DEPTH;
                boards_out[count].memory.template setNextNMove<MAX_DEPTH>(move_next);
            }

            count++;
        } else if constexpr (DIRECTION) {
            u8 nextStart = 0;
            u8 nextEnd = 24;

            if constexpr (MOVES_ASCENDING) {
                nextStart = actn_i + actStruct.tillNext;
            } else {
                nextEnd = actn_i - actStruct.tillLast;
                if (nextEnd == 255) { nextEnd = 0; }
            }
            u64 move_next = move | actn_i << 6 * CUR_DEPTH;
            make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, MOVES_ASCENDING, true>(
                    board_next, boards_out, count, hasher, move_next, actStruct, nextStart, nextEnd);
            make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, MOVES_ASCENDING, false>(
                    board_next, boards_out, count, hasher, move_next, actStruct, 24, 48);

        } else { // if constexpr (!DIRECTION)
            u8 nextStart = 24;
            u8 nextEnd = 48;
            if constexpr (MOVES_ASCENDING) {
                nextStart = actn_i + actStruct.tillNext;
            } else {
                nextEnd = actn_i - actStruct.tillLast;
                nextEnd += (nextEnd == 255);
            }
            u64 move_next = move | actn_i << 6 * CUR_DEPTH;
            make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, MOVES_ASCENDING, true>(
                    board_next, boards_out, count, hasher, move_next, actStruct, 0, 24);

            make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, MOVES_ASCENDING, false>(
                    board_next, boards_out, count, hasher, move_next, actStruct, nextStart, nextEnd);
        }
    }
}


template<typename T,
         int DEPTH, bool MOVES_ASCENDING
        >
void make_fat_perm_list(C Board &board_in,
                        JVec<T> &boards_out,
                        C typename T::HasherPtr hasher) {
    static_assert(AllowedPermsType<T>, "T must be Memory or Board");
    static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

    MU u32 count = 0;
    if constexpr (DEPTH == 0) {

        if constexpr (std::is_same_v<T, Memory>) {
            boards_out[count] = board_in.memory;
            (boards_out[count].*hasher)(board_in.b1, board_in.b2);
        } else if constexpr (std::is_same_v<T, Board>) {
            boards_out[0] = board_in;
            (boards_out[0].*hasher)();
        }

        boards_out.resize(1);
    } else {
        make_fat_perm_list_helper<T, 0, DEPTH, MOVES_ASCENDING, true>(
                board_in, boards_out, count, hasher, 0,
                {nullptr, 0, 0, 0, 0, "\0\0\0\0"}, 0, 24);
        make_fat_perm_list_helper<T, 0, DEPTH, MOVES_ASCENDING, false>(
                board_in, boards_out, count, hasher, 0,
                {nullptr, 0, 0, 0, 0, "\0\0\0\0"}, 24, 48);
        boards_out.resize(count);
    }
}
