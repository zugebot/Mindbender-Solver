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


    // Depth-first canonical stream for fat boards with B1B2 output, expanding
    // whole lanes with lane_rotations.hpp. Lanes are rows/columns in index
    // order, the fat pair counting as one lane at its first index; consecutive
    // same-direction moves use strictly ordered lanes (SECT_DIR), no-op moves
    // are dropped, and a normal C -> normal R pair that commutes on the board is
    // only kept in R -> C order. Unlike the slot-table stream above it never
    // emits two consecutive moves on the fat lane (such pairs equal one shorter
    // move), so it emits a subset of that stream that still covers every
    // exact-distance state (checked in testExactSolverRegression). prevDir < 0:
    // no previous move; at CUR_DEPTH 0 the caller may pass the seed's own last
    // move (ssd::buildLastMoves) so the first streamed move follows the rules too.
    template<i32 CUR_DEPTH, i32 MAX_DEPTH,
             eSequenceDir SECT_DIR,
             typename Sink>
    void stream_fat_lanes_dfs(
            const B1B2& board,
            StreamChunk<B1B2>& chunk,
            Sink& sink,
            const i32 prevDir,
            const i32 prevLane,
            const i32 prevNormalCol,
            const u32 prevAmount) {
        const u32 fatX = board.getFatX();
        const u32 fatY = board.getFatY();

        for (i32 dir = 0; dir < 2; ++dir) {
            const i32 fatLane = static_cast<i32>(dir == 0 ? fatY : fatX);

            i32 first = 0, last = 5, step = 1;
            if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                if (dir == prevDir) {
                    first = prevLane + 1;
                }
            } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                first = 5;
                last = 0;
                step = -1;
                if (dir == prevDir) {
                    first = prevLane - 1;
                }
            }

            for (i32 lane = first; step > 0 ? lane <= last : lane >= last; lane += step) {
                if (lane == fatLane + 1) {
                    continue;   // second row/column of the fat pair
                }
                const bool isFatLane = lane == fatLane;

                B1B2 shifted[5];
                u32 skip = 0;
                if (dir == 0) {
                    if (isFatLane) {
                        lane_rotations::fatRowShifts(board, static_cast<u32>(lane), fatX, shifted);
                        skip = 1u << (4 - fatX);   // amount 5 - fatX
                    } else {
                        lane_rotations::rowShifts(board, static_cast<u32>(lane), shifted);
                        if (prevNormalCol >= 0) {
                            skip = makeBoardStateView(board).doActISColMatchBatched(
                                    static_cast<u8>(prevNormalCol), static_cast<u8>(lane),
                                    static_cast<u8>(prevAmount));
                        }
                    }
                } else {
                    if (isFatLane) {
                        lane_rotations::fatColShifts(board, static_cast<u32>(lane), fatY, shifted);
                        skip = 1u << (4 - fatY);   // amount 5 - fatY
                    } else {
                        lane_rotations::colShifts(board, static_cast<u32>(lane), shifted);
                    }
                }

                const i32 normalCol = dir == 1 && !isFatLane ? lane : -1;
                for (u32 k = 0; k < 5; ++k) {
                    // `continue`, not the normal stream's `break` on a repeating lane: `skip` drops
                    // one amount of a fat lane, and a larger amount that repeats it must still run
                    if ((skip & (1u << k)) || shifted[k] == board) {
                        continue;
                    }
                    if constexpr (CUR_DEPTH + 1 == MAX_DEPTH) {
                        emitStateB1B2(shifted[k], chunk, sink);
                    } else {
                        stream_fat_lanes_dfs<CUR_DEPTH + 1, MAX_DEPTH, SECT_DIR>(
                                shifted[k], chunk, sink, dir, lane, normalCol, k + 1);
                    }
                }
            }
        }
    }

} // namespace perm_stream_detail