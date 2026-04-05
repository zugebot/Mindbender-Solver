#pragma once
// code/perms_fat.tpp

#include "utils/hasGetHash.hpp"

namespace perms_detail {

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR, bool DIRECTION>
    static void make_fat_perm_list_helper(
            C Board& board,
            JVec<T>& boards_out,
            u32& count,
            C typename T::HasherPtr hasher,
            C u64 move,
            C ActStruct& lastActStruct,
            C u8 startIndex,
            C u8 endIndex) {
        static_assert(AllowedPermsType<T>, "T must be Memory, Board, or B1B2");
        static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

        MU bool lastActIsRow = false;
        MU bool lastActIsCol = false;

        if constexpr (DIRECTION) {
            lastActIsRow = (lastActStruct.isColNotFat & 2) != 0;
        } else {
            lastActIsCol = (lastActStruct.isColNotFat & 1) != 0;
        }

        C u8* funcIndexes = fatActionsIndexes[board.getFatXY()];

        for (u64 actn_i = startIndex; actn_i < endIndex; ++actn_i) {
            C ActStruct& actStruct = allActStructList[funcIndexes[actn_i]];
            Board board_next = board;
            actStruct.action(board_next);

            if constexpr (CUR_DEPTH != 0) {
                if constexpr (SECT_DIR != eSequenceDir::NONE) {
                    if constexpr (DIRECTION) {
                        if (lastActIsRow && ((actStruct.isColNotFat & 2) != 0)) {
                            if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                                if (lastActStruct.index <= actStruct.index) {
                                    continue;
                                }
                            } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                                if (lastActStruct.index >= actStruct.index) {
                                    continue;
                                }
                            }
                        }
                    } else {
                        if (lastActIsCol && ((actStruct.isColNotFat & 1) != 0)) {
                            if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                                if (lastActStruct.index <= actStruct.index) {
                                    continue;
                                }
                            } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                                if (lastActStruct.index >= actStruct.index) {
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            if (board == board_next) {
                continue;
            }

            if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                C u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                if constexpr (std::is_same_v<T, Memory>) {
                    boards_out[count] = board_next.memory;
                    (boards_out[count].*hasher)(board_next.b1, board_next.b2);
                    boards_out[count].template setNextNMove<MAX_DEPTH>(move_next);

                } else if constexpr (std::is_same_v<T, Board>) {
                    boards_out[count] = board_next;
                    (boards_out[count].*hasher)();
                    boards_out[count].memory.template setNextNMove<MAX_DEPTH>(move_next);

                } else if constexpr (std::is_same_v<T, B1B2>) {
                    boards_out[count] = board_next.asB1B2();
                }

                ++count;

            } else if constexpr (DIRECTION) {
                u8 nextStart = 0;
                u8 nextEnd = 24;

                if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                    nextStart = actn_i + actStruct.tillNext;
                } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                    nextEnd = actn_i - actStruct.tillLast;
                    if (nextEnd == 255) { nextEnd = 0; }
                } else {
                    nextStart = 0;
                    nextEnd = 24;
                }

                C u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, true>(
                        board_next, boards_out, count, hasher, move_next, actStruct, nextStart, nextEnd);

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, false>(
                        board_next, boards_out, count, hasher, move_next, actStruct, 24, 48);

            } else {
                u8 nextStart = 24;
                u8 nextEnd = 48;

                if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                    nextStart = actn_i + actStruct.tillNext;
                } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                    nextEnd = actn_i - actStruct.tillLast;
                    if (nextEnd == 255) { nextEnd = 24; }
                } else {
                    nextStart = 24;
                    nextEnd = 48;
                }

                C u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, true>(
                        board_next, boards_out, count, hasher, move_next, actStruct, 0, 24);

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, false>(
                        board_next, boards_out, count, hasher, move_next, actStruct, nextStart, nextEnd);
            }
        }
    }

    template<typename T,
             i32 DEPTH, eSequenceDir SECT_DIR>
    void make_fat_perm_list(
            C Board& board_in,
            JVec<T>& boards_out,
            C typename T::HasherPtr hasher) {
        static_assert(AllowedPermsType<T>, "T must be Memory, Board, or B1B2");
        static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");

        MU u32 count = 0;

        if constexpr (DEPTH == 0) {
            if constexpr (std::is_same_v<T, Memory>) {
                boards_out[0] = board_in.memory;
                (boards_out[0].*hasher)(board_in.b1, board_in.b2);

            } else if constexpr (std::is_same_v<T, Board>) {
                boards_out[0] = board_in;
                (boards_out[0].*hasher)();

            } else if constexpr (std::is_same_v<T, B1B2>) {
                boards_out[0] = board_in.asB1B2();
            }

            boards_out.resize(1);

        } else {
            make_fat_perm_list_helper<T, 0, DEPTH, SECT_DIR, true>(
                    board_in, boards_out, count, hasher, 0,
                    {nullptr, 0, 0, 0, 0, "\0\0\0\0"}, 0, 24);

            make_fat_perm_list_helper<T, 0, DEPTH, SECT_DIR, false>(
                    board_in, boards_out, count, hasher, 0,
                    {nullptr, 0, 0, 0, 0, "\0\0\0\0"}, 24, 48);

            boards_out.resize(count);
        }
    }

}

/**
 * these are an upper-limit for each depth,
 * it's dependent on the fat location
 *
 * so a good upper limit for guessing more is
 * 1: 48
 * 2: 27.5
 * 3: 27.577272727
 * 4: 27.503104225
 * 5: 27.4814706423
 * SIZE = 48 * 27.6 ^ (depth - 1)
 */