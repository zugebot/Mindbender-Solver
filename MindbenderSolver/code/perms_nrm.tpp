#pragma once


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM>
void make_perm_list_inner(C Board &board_in,
                          JVec<Memory> &boards_out,
                          Ref<MAX_DEPTH> &ref,
                          C u64 move_prev,
                          int& count) {

    if constexpr (MAX_DEPTH == 0) {
        boards_out[0] = board_in.memory;
        (boards_out[0].*ref.hasher)(board_in.b1, board_in.b2);
        boards_out.resize(1);
        return;

    } else if constexpr (CUR_DEPTH == MAX_DEPTH) {
        // Base case: process and store the final board
        boards_out[count] = board_in.memory;
        (boards_out[count].*ref.hasher)(board_in.b1, board_in.b2);
        boards_out[count].setNextNMove<MAX_DEPTH>(move_prev);
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
        C i32 base = ref.base_seq[CUR_DEPTH];
        u64& cur = ref.cur_seq[CUR_DEPTH];

        for (cur = base; cur < base + 5; ++cur) {

            Board board_next = board_in;
            allActStructList[cur].action(board_next);

            if constexpr (CHECK_SIM) {
                if (board_in == board_next) { continue; }
            }

            // check intersection here
            if constexpr (CHECK_CROSS) {
                if constexpr (CUR_DEPTH > 0) {
                    if (ref.checkRC_seq[CUR_DEPTH] && ref.intersect_seq[CUR_DEPTH]
                        & (1 << (cur - base))) { continue; }
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


template<int CUR_DEPTH, int MAX_DEPTH, bool CHECK_CROSS,
        bool CHECK_SIM, bool CHANGE_SECT_START, bool SECT_ASCENDING>
void make_perm_list_outer(C Board &board_in,
                          JVec<Memory> &boards_out,
                          Ref<MAX_DEPTH> &ref,
                          int& count) {
    if constexpr (CUR_DEPTH == MAX_DEPTH) {
        // All depths have been processed; start the inner loops
        make_perm_list_inner<0, MAX_DEPTH, CHECK_CROSS, CHECK_SIM>(
                board_in, boards_out, ref, 0, count);

    } else {


        for (int dir = 0; dir < 2; ++dir) {
            ref.dir_seq[CUR_DEPTH] = dir;

            if constexpr (SECT_ASCENDING) { // ASCENDING (0 -> 5)
                static constexpr i32 SECT_START = 0;
                static constexpr i32 SECT_END = 6;

                i32 sect_start;
                if constexpr (CHANGE_SECT_START && CUR_DEPTH != 0) {
                    sect_start = ref.dir_seq[CUR_DEPTH] == ref.dir_seq[CUR_DEPTH - 1]
                                         ? ref.sect_seq[CUR_DEPTH - 1] + 1 : SECT_START;
                } else { sect_start = SECT_START; }

                for (int sect = sect_start; sect < SECT_END; ++sect) {
                    ref.sect_seq[CUR_DEPTH] = sect;
                    ref.base_seq[CUR_DEPTH] = dir * 30 + sect * 5;

                    // Determine do_RC_check
                    if constexpr (CUR_DEPTH > 0) {
                        C int prev_dir = ref.dir_seq[CUR_DEPTH - 1];
                        ref.checkRC_seq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                    }

                    // Recursive call to the next depth
                    make_perm_list_outer<CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS,
                        CHECK_SIM, CHANGE_SECT_START, SECT_ASCENDING>(board_in, boards_out, ref, count);
                }

            } else { // DESCENDING (5 -> 0)
                static constexpr i32 SECT_START = 5;
                static constexpr i32 SECT_END = 0;

                i32 sect_start;
                if constexpr (CHANGE_SECT_START && CUR_DEPTH != 0) {
                    sect_start = ref.dir_seq[CUR_DEPTH] == ref.dir_seq[CUR_DEPTH - 1]
                                         ? ref.sect_seq[CUR_DEPTH - 1] - 1 : SECT_START;
                } else { sect_start = SECT_START; }

                for (int sect = sect_start; sect >= SECT_END; --sect) {
                    ref.sect_seq[CUR_DEPTH] = sect;
                    ref.base_seq[CUR_DEPTH] = dir * 30 + sect * 5;

                    // Determine do_RC_check
                    if constexpr (CUR_DEPTH > 0) {
                        C int prev_dir = ref.dir_seq[CUR_DEPTH - 1];
                        ref.checkRC_seq[CUR_DEPTH] = prev_dir != dir && prev_dir != 0;
                    }

                    // Recursive call to the next depth
                    make_perm_list_outer<CUR_DEPTH + 1, MAX_DEPTH, CHECK_CROSS,
                        CHECK_SIM, CHANGE_SECT_START, SECT_ASCENDING>(board_in, boards_out, ref, count);
                }
            }

        }

    }
}


template<int MAX_DEPTH, bool CHECK_CROSS, bool CHECK_SIM,
         bool CHANGE_SECT_START, bool SECT_ASCENDING>
void make_perm_list(C Board &board_in,
                    JVec<Memory> &boards_out,
                    C Memory::HasherPtr hasher) {
    Ref<MAX_DEPTH> ref;
    ref.hasher = hasher;
    int count = 0;

    make_perm_list_outer<0, MAX_DEPTH,
                         CHECK_CROSS, CHECK_SIM,
                         CHANGE_SECT_START, SECT_ASCENDING>(
            board_in, boards_out, ref, count);
    boards_out.resize(count);
}