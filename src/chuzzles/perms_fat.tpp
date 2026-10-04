#pragma once
// src/perms_fat.tpp

namespace perms_detail {

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR, bool DIRECTION>
    static void make_fat_perm_list_helper(
            const Board& board,
            JVec<T>& boards_out,
            JVec<u64>& hashes_out,
            u32& count,
            const u64 move,
            const u8 previousGlobalAction,
            const u8 startIndex,
            const u8 endIndex) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        // A normal C followed by a normal R that commutes on this board is
        // skipped; the same state is kept through the R -> C order.
        const u8* funcIndexes = fatActionsIndexes[board.getFatXY()];
        const u32 crossMask = DIRECTION
                ? board.getFatNormalColThenRowCommutingMask(previousGlobalAction)
                : 0;

        for (u64 actn_i = startIndex; actn_i < endIndex; ++actn_i) {
            if (crossMask & (u32{1} << actn_i)) {
                continue;
            }
            const ActStruct& actStruct = allActStructList[funcIndexes[actn_i]];
            Board board_next = board;
            actStruct.action(board_next);

            // The board-relative [startIndex, endIndex) restriction below
            // already chooses one ordering for consecutive moves in the same
            // row/column lane. Do not also compare ActStruct::index here:
            // fatActionsIndexes changes with the fat position, so combining
            // both orderings can reject every representative of a reachable
            // state (notably the R04/R54 segment in puzzle 6-2).

            if (board == board_next) {
                continue;
            }

            if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                const u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                if constexpr (std::is_same_v<T, Board>) {
                    boards_out[count] = board_next;
                    boards_out[count].memory.template setNextNMove<MAX_DEPTH>(move_next);
                    hashes_out[count] = StateHash::computeHash(boards_out[count]);

                } else if constexpr (std::is_same_v<T, B1B2>) {
                    boards_out[count] = board_next.asB1B2();
                    hashes_out[count] = StateHash::computeHash(boards_out[count]);
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

                const u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, true>(
                        board_next, boards_out, hashes_out, count, move_next, funcIndexes[actn_i], nextStart, nextEnd);

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, false>(
                        board_next, boards_out, hashes_out, count, move_next, funcIndexes[actn_i], 24, 48);

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

                const u64 move_next = move | (actn_i << (6 * CUR_DEPTH));

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, true>(
                        board_next, boards_out, hashes_out, count, move_next, funcIndexes[actn_i], 0, 24);

                make_fat_perm_list_helper<T, CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR, false>(
                        board_next, boards_out, hashes_out, count, move_next, funcIndexes[actn_i], nextStart, nextEnd);
            }
        }
    }

    template<typename T,
             i32 DEPTH, eSequenceDir SECT_DIR>
    void make_fat_perm_list(
            const Board& board_in,
            JVec<T>& boards_out,
            JVec<u64>& hashes_out) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        MU u32 count = 0;

        if constexpr (DEPTH == 0) {
            if constexpr (std::is_same_v<T, Board>) {
                boards_out[0] = board_in;
                hashes_out[0] = StateHash::computeHash(boards_out[0]);

            } else if constexpr (std::is_same_v<T, B1B2>) {
                boards_out[0] = board_in.asB1B2();
                hashes_out[0] = StateHash::computeHash(boards_out[0]);
            }

            boards_out.resize(1);
            hashes_out.resize(1);

        } else {
            make_fat_perm_list_helper<T, 0, DEPTH, SECT_DIR, true>(
                    board_in, boards_out, hashes_out, count, 0, 255, 0, 24);

            make_fat_perm_list_helper<T, 0, DEPTH, SECT_DIR, false>(
                    board_in, boards_out, hashes_out, count, 0, 255, 24, 48);

            boards_out.resize(count);
            hashes_out.resize(count);
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