#pragma once
// code/perms_nrm.tpp

#include "utils/hasGetHash.hpp"


namespace perms_detail {

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM>
    void make_perm_list_inner(
            C Board &board_in,
            JVec<T> &boards_out,
            PermBuildState<T, MAX_DEPTH> &state,
            C u64 move_prev,
            i32 &count) {
        static_assert(AllowedPermsType<T>, "T must be Memory, Board, or B1B2");
        static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
        
        if constexpr (MAX_DEPTH == 0) {

            if constexpr (std::is_same_v<T, Memory>) {
                boards_out[0] = board_in.memory;
                (boards_out[0].*state.hasher)(board_in.b1, board_in.b2);
            } else if constexpr (std::is_same_v<T, Board>) {
                boards_out[0] = board_in;
                (boards_out[0].*state.hasher)();
            } else if constexpr (std::is_same_v<T, B1B2>) {
                boards_out[0] = board_in.asB1B2();
            }
            
            boards_out.resize(1);
            count = 1;

            // Base case: process and store the final board
        } else if constexpr (CUR_DEPTH == MAX_DEPTH) {

            if constexpr (std::is_same_v<T, Memory>) {
                boards_out[count] = board_in.memory;
                (boards_out[count].*state.hasher)(board_in.b1, board_in.b2);
                boards_out[count].template setNextNMove<MAX_DEPTH>(move_prev);
                
            } else if constexpr (std::is_same_v<T, Board>) {
                boards_out[count] = board_in;
                (boards_out[count].*state.hasher)();
                boards_out[count].memory.template setNextNMove<MAX_DEPTH>(move_prev);

            } else if constexpr (std::is_same_v<T, B1B2>) {
                boards_out[count] = board_in.asB1B2();
            }
            
            ++count;


        } else {

            // Handle intersection checks if needed
            if constexpr (CHECK_CROSS && CUR_DEPTH > 0) {
                if (state.checkRCSeq[CUR_DEPTH]) {
                    state.intersectSeq[CUR_DEPTH] = board_in.doActISColMatchBatched(
                            state.sectSeq[CUR_DEPTH - 1],
                            state.sectSeq[CUR_DEPTH],
                            state.curSeq[CUR_DEPTH - 1] - state.baseSeq[CUR_DEPTH - 1] + 1);
                }
            }

            // Loop over cur indices
            C i32 base = state.baseSeq[CUR_DEPTH];
            u64 &cur = state.curSeq[CUR_DEPTH];

            for (cur = base; cur < base + 5; ++cur) {

                Board board_next = board_in;
                allActStructList[cur].action(board_next);

                if constexpr (CHECK_SIM) {
                    if (board_in == board_next) { continue; }
                }

                // check intersection here
                if constexpr (CHECK_CROSS) {
                    if constexpr (CUR_DEPTH > 0) {
                        if (state.checkRCSeq[CUR_DEPTH] &&
                            state.intersectSeq[CUR_DEPTH] & (1 << (cur - base))) { continue; }
                    }
                }

                // Update move
                u64 move_next = move_prev | (cur << (MEMORY_MOVE_TYPE_BITSIZE * CUR_DEPTH));

                // Recursive call to the next depth
                make_perm_list_inner<T, CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
                        board_next, boards_out, state, move_next, count);
            }
        }
    }


    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR>
    void make_perm_list_outer(C Board &board_in,
                              JVec<T> &boards_out,
                              PermBuildState<T, MAX_DEPTH> &state,
                              i32 &count) {
        static_assert(AllowedPermsType<T>, "T must be Memory, Board, or B1B2");
        static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
        
        
        if constexpr (CUR_DEPTH == MAX_DEPTH) {
            make_perm_list_inner<T, 0, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
                    board_in, boards_out, state, 0, count);
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
                            C i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        make_perm_list_outer<T, CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS,
                                             CHECK_SIM, CHANGE_SECT_START, SECT_DIR>(
                                board_in, boards_out, state, count);
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
                            C i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        make_perm_list_outer<T, CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS,
                                             CHECK_SIM, CHANGE_SECT_START, SECT_DIR>(
                                board_in, boards_out, state, count);
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
                            C i32 prev_dir = state.dirSeq[CUR_DEPTH - 1];
                            state.checkRCSeq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                        }

                        make_perm_list_outer<T, CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS,
                                             CHECK_SIM, CHANGE_SECT_START, SECT_DIR>(
                                board_in, boards_out, state, count);
                    }
                }
            }
        }
    }


    template<typename T,
             i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, eSequenceDir SECT_DIR>
    void make_perm_list(C Board &board_in,
                        JVec<T> &boards_out,
                        C typename T::HasherPtr hasher) {
        static_assert(AllowedPermsType<T>, "T must be Memory, Board, or B1B2");
        static_assert(HasGetHash_v<T>, "T must have a getHash() method returning uint64_t");
        
        PermBuildState<T, MAX_DEPTH> state;
        state.hasher = hasher;
        i32 count = 0;

        make_perm_list_outer<T, 0, MAX_DEPTH,
                             CHECK_CROSS, CHECK_SIM,
                             CHANGE_SECT_START, SECT_DIR>(
                board_in, boards_out, state, count);
        boards_out.resize(count);
    }

}