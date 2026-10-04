#pragma once
// src/perm_stream_fat.tpp

namespace perm_stream_detail {

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR, bool DIRECTION,
             typename Sink>
    static void stream_fat_perm_list_helper(
            const Board& board,
            StreamChunk<T>& chunk,
            const u64 move,
            const u8 previousGlobalAction,
            const u8 startIndex,
            const u8 endIndex,
            Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        // A normal C followed by a normal R that commutes on this board is
        // skipped; the same state is kept through the R -> C order.
        const u8* funcIndexes = fatActionsIndexes[board.getFatXY()];
        const u32 crossMask = DIRECTION
                ? board.getFatNormalColThenRowCommutingMask(previousGlobalAction)
                : 0;

        for (u8 actn_i = startIndex; actn_i < endIndex; ++actn_i) {
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

            const u64 move_next = move | (static_cast<u64>(actn_i) << (6 * CUR_DEPTH));

            if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                emitState<T, MAX_DEPTH>(board_next, move_next, chunk, sink);

            } else if constexpr (DIRECTION) {
                u8 nextStart = 0;
                u8 nextEnd = 24;

                if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                    nextStart = static_cast<u8>(actn_i + actStruct.tillNext);
                } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                    nextEnd = static_cast<u8>(actn_i - actStruct.tillLast);
                    if (nextEnd == 255) {
                        nextEnd = 0;
                    }
                } else {
                    nextStart = 0;
                    nextEnd = 24;
                }

                stream_fat_perm_list_helper<
                        T,
                        CUR_DEPTH + 1,
                        MAX_DEPTH,
                        SECT_DIR,
                        true>(
                        board_next,
                        chunk,
                        move_next,
                        funcIndexes[actn_i],
                        nextStart,
                        nextEnd,
                        sink
                );

                stream_fat_perm_list_helper<
                        T,
                        CUR_DEPTH + 1,
                        MAX_DEPTH,
                        SECT_DIR,
                        false>(
                        board_next,
                        chunk,
                        move_next,
                        funcIndexes[actn_i],
                        24,
                        48,
                        sink
                );

            } else {
                u8 nextStart = 24;
                u8 nextEnd = 48;

                if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                    nextStart = static_cast<u8>(actn_i + actStruct.tillNext);
                } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                    nextEnd = static_cast<u8>(actn_i - actStruct.tillLast);
                    if (nextEnd == 255) {
                        nextEnd = 24;
                    }
                } else {
                    nextStart = 24;
                    nextEnd = 48;
                }

                stream_fat_perm_list_helper<
                        T,
                        CUR_DEPTH + 1,
                        MAX_DEPTH,
                        SECT_DIR,
                        true>(
                        board_next,
                        chunk,
                        move_next,
                        funcIndexes[actn_i],
                        0,
                        24,
                        sink
                );

                stream_fat_perm_list_helper<
                        T,
                        CUR_DEPTH + 1,
                        MAX_DEPTH,
                        SECT_DIR,
                        false>(
                        board_next,
                        chunk,
                        move_next,
                        funcIndexes[actn_i],
                        nextStart,
                        nextEnd,
                        sink
                );
            }
        }
    }

    template<typename T,
             i32 DEPTH,
             eSequenceDir SECT_DIR,
             typename Sink>
    static void stream_fat_perm_list(
            const Board& board_in,
            StreamChunk<T>& chunk,
            Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        if constexpr (DEPTH == 0) {
            emitState<T, 0>(board_in, 0, chunk, sink);
        } else {
            stream_fat_perm_list_helper<
                    T,
                    0,
                    DEPTH,
                    SECT_DIR,
                    true>(
                    board_in,
                    chunk,
                    0,
                    255,
                    0,
                    24,
                    sink
            );

            stream_fat_perm_list_helper<
                    T,
                    0,
                    DEPTH,
                    SECT_DIR,
                    false>(
                    board_in,
                    chunk,
                    0,
                    255,
                    24,
                    48,
                    sink
            );
        }
    }

} // namespace perm_stream_detail