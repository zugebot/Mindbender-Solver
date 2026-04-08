#pragma once
// code/perm_stream_nrm.tpp

namespace perm_stream_detail {

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             typename Sink>
    void stream_perm_list_inner(
            const Board& board_in,
            StreamChunk<T>& chunk,
            StreamBuildState<T, MAX_DEPTH>& state,
            const u64 move_prev,
            Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        if constexpr (MAX_DEPTH == 0) {
            emitState<T, 0>(board_in, 0, chunk, sink);

        } else if constexpr (CUR_DEPTH == MAX_DEPTH) {
            emitState<T, MAX_DEPTH>(board_in, move_prev, chunk, sink);

        } else {
            if constexpr (CHECK_CROSS && CUR_DEPTH > 0) {
                if (state.checkRCSeq[CUR_DEPTH]) {
                    state.intersectSeq[CUR_DEPTH] = board_in.doActISColMatchBatched(
                            state.sectSeq[CUR_DEPTH - 1],
                            state.sectSeq[CUR_DEPTH],
                            state.curSeq[CUR_DEPTH - 1] - state.baseSeq[CUR_DEPTH - 1] + 1);
                }
            }

            const i32 base = state.baseSeq[CUR_DEPTH];
            u64& cur = state.curSeq[CUR_DEPTH];

            for (cur = base; cur < static_cast<u64>(base + 5); ++cur) {
                Board board_next = board_in;
                allActStructList[cur].action(board_next);

                if constexpr (CHECK_SIM) {
                    if (board_in == board_next) {
                        continue;
                    }
                }

                if constexpr (CHECK_CROSS && CUR_DEPTH > 0) {
                    if (state.checkRCSeq[CUR_DEPTH]
                        && (state.intersectSeq[CUR_DEPTH] & (1u << (cur - base)))) {
                        continue;
                    }
                }

                const u64 move_next = move_prev | (cur << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));

                stream_perm_list_inner<
                        T,
                        CUR_DEPTH + 1,
                        MAX_DEPTH,
                        CHECK_CROSS,
                        CHECK_SIM>(
                        board_next,
                        chunk,
                        state,
                        move_next,
                        sink
                );
            }
        }
    }


    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR,
             typename Sink>
    void stream_perm_list_outer(
            const Board& board_in,
            StreamChunk<T>& chunk,
            StreamBuildState<T, MAX_DEPTH>& state,
            Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        if constexpr (CUR_DEPTH == MAX_DEPTH) {
            stream_perm_list_inner<
                    T,
                    0,
                    MAX_DEPTH,
                    CHECK_CROSS,
                    CHECK_SIM>(
                    board_in,
                    chunk,
                    state,
                    0,
                    sink
            );
        } else {
            for (i32 dir = 0; dir < 2; ++dir) {
                state.dirSeq[CUR_DEPTH] = dir;

                if constexpr (SECT_DIR == eSequenceDir::NONE) {
                    static constexpr i32 SECT_START = 0;
                    static constexpr i32 SECT_END = 6;

                    for (i32 sect = SECT_START; sect < SECT_END; ++sect) {
                        state.sectSeq[CUR_DEPTH] = sect;
                        state.baseSeq[CUR_DEPTH] = (dir << 5) + sect * 5;

                        if constexpr (CUR_DEPTH > 0) {
                            const i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        stream_perm_list_outer<
                                T,
                                CUR_DEPTH + 1,
                                MAX_DEPTH,
                                CHECK_CROSS,
                                CHECK_SIM,
                                CHANGE_SECT_START,
                                SECT_DIR>(
                                board_in,
                                chunk,
                                state,
                                sink
                        );
                    }

                } else if constexpr (SECT_DIR == eSequenceDir::ASCENDING) {
                    static constexpr i32 SECT_START = 0;
                    static constexpr i32 SECT_END = 6;

                    i32 sect_start;
                    if constexpr (CHANGE_SECT_START && CUR_DEPTH != 0) {
                        sect_start = state.dirSeq[CUR_DEPTH] == state.dirSeq[CUR_DEPTH - 1]
                                             ? state.sectSeq[CUR_DEPTH - 1] + 1
                                             : SECT_START;
                    } else {
                        sect_start = SECT_START;
                    }

                    for (i32 sect = sect_start; sect < SECT_END; ++sect) {
                        state.sectSeq[CUR_DEPTH] = sect;
                        state.baseSeq[CUR_DEPTH] = (dir << 5) + sect * 5;

                        if constexpr (CUR_DEPTH > 0) {
                            const i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        stream_perm_list_outer<
                                T,
                                CUR_DEPTH + 1,
                                MAX_DEPTH,
                                CHECK_CROSS,
                                CHECK_SIM,
                                CHANGE_SECT_START,
                                SECT_DIR>(
                                board_in,
                                chunk,
                                state,
                                sink
                        );
                    }

                } else if constexpr (SECT_DIR == eSequenceDir::DESCENDING) {
                    static constexpr i32 SECT_START = 5;
                    static constexpr i32 SECT_END = 0;

                    i32 sect_start;
                    if constexpr (CHANGE_SECT_START && CUR_DEPTH != 0) {
                        sect_start = state.dirSeq[CUR_DEPTH] == state.dirSeq[CUR_DEPTH - 1]
                                             ? state.sectSeq[CUR_DEPTH - 1] - 1
                                             : SECT_START;
                    } else {
                        sect_start = SECT_START;
                    }

                    for (i32 sect = sect_start; sect >= SECT_END; --sect) {
                        state.sectSeq[CUR_DEPTH] = sect;
                        state.baseSeq[CUR_DEPTH] = (dir << 5) + sect * 5;

                        if constexpr (CUR_DEPTH > 0) {
                            const i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        stream_perm_list_outer<
                                T,
                                CUR_DEPTH + 1,
                                MAX_DEPTH,
                                CHECK_CROSS,
                                CHECK_SIM,
                                CHANGE_SECT_START,
                                SECT_DIR>(
                                board_in,
                                chunk,
                                state,
                                sink
                        );
                    }
                }
            }
        }
    }


    template<typename T,
             i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR,
             typename Sink>
    void stream_perm_list(
            const Board& board_in,
            StreamChunk<T>& chunk,
            Sink& sink) {
        static_assert(AllowedPermsType<T>, "T must be Board or B1B2");

        StreamBuildState<T, MAX_DEPTH> state;

        stream_perm_list_outer<
                T,
                0,
                MAX_DEPTH,
                CHECK_CROSS,
                CHECK_SIM,
                CHANGE_SECT_START,
                SECT_DIR>(
                board_in,
                chunk,
                state,
                sink
        );
    }

} // namespace perm_stream_detail