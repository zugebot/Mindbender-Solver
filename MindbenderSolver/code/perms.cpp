#include "perms.hpp"
#include "rotations.hpp"

#include <fstream>
#include <iostream>


template<bool CHECK_CROSS, bool CHECK_SIM>
void make_permutation_list_depth_plus_one(C JVec<Board> &boards_in, JVec<Board> &boards_out, C Board::HasherPtr hasher) {
    int count = 0;
    u8 intersects = 0;

    for (C auto &board_index: boards_in) {
        C u8 a = board_index.getMemory().getLastMove();
        C u8 a_dir = a / 30;
        C u8 a_sect = a % 30 / 5;
        C u8 a_amount = a % 5 + 1;


        for (int b_dir = 0; b_dir < 2; b_dir++) {
            C bool do_RC_check = a_dir != b_dir && a_dir != 0;

            C int b_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_start; b_sect < 6; b_sect++) {
                C int b_base = b_dir * 30 + b_sect * 5;

                if constexpr (CHECK_CROSS) {
                    if (do_RC_check) { intersects = board_index.doActISColMatchBatched(a_sect, b_sect, a_amount); }
                }

                for (int b_amount = 0; b_amount < 5; b_amount++) {
                    if constexpr (CHECK_CROSS) {
                        if (do_RC_check && intersects & (1 << (b_amount))) { continue; }
                    }

                    C int b_cur = b_base + b_amount;

                    boards_out[count] = board_index;
                    allActStructList[b_cur].action(boards_out[count]);
                    if constexpr (CHECK_SIM) {
                        if (boards_out[count].b1 == board_index.b1 && boards_out[count].b2 == board_index.b2) { continue; }
                    }

                    (boards_out[count].*hasher)();
                    boards_out[count].getMemory().setNextNMove<1>(b_cur);
                    count++;
                }
            }
        }
    }
    boards_out.resize(count);
}


C Perms::depthMap_t Perms::depthMap = {
        {1, {{1, 0}, {0, 1}}},
        {2, {{1, 1}, {0, 2}, {2, 0}}},
        {3, {{1, 2}, {2, 1}, {0, 3}, {3, 0}}},
        {4, {{2, 2}, {3, 1}, {1, 3}, {4, 0}, {0, 4}}},
        {5, {{3, 2}, {3, 2}, {4, 1}, {1, 4}, {5, 0}, {0, 5}}},
        {6, {{3, 3}, {4, 2}, {2, 4}, {5, 1}, {1, 5}}},
        {7, {{4, 3}, {3, 4}, {5, 2}, {2, 5}}},
        {8, {{4, 4}, {5, 3}, {3, 5}}},
        {9, {{4, 5}, {5, 4}}},
        {10, {{5, 5}}},
        {11, {{6, 5}}},
};


MU void Perms::reserveForDepth(MU C Board& board_in, JVec<Board> &boards_out, C u32 depth) {
    C double fraction = Board::getDuplicateEstimateAtDepth(depth);
    u64 allocSize = board_in.getFatBool() ? BOARD_FAT_MAX_MALLOC_SIZES[depth] : BOARD_PRE_MAX_MALLOC_SIZES[depth];

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


MU void Perms::reserveForDepth(MU C Board& board_in, JVec<Memory> &boards_out, C u32 depth) {
    C double fraction = Board::getDuplicateEstimateAtDepth(depth);
    u64 allocSize = board_in.getFatBool() ? BOARD_FAT_MAX_MALLOC_SIZES[depth] : BOARD_PRE_MAX_MALLOC_SIZES[depth];

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


Perms::toDepthFuncPtr_t Perms::toDepthFromLeftFuncPtrs[] = {
    &make_perm_list<0, true, true, true, true>,
    &make_perm_list<1, true, true, true, true>,
    &make_perm_list<2, true, true, true, true>,
    &make_perm_list<3, true, true, true, true>,
    &make_perm_list<4, true, true, true, true>,
    &make_perm_list<5, true, true, true, true>};


Perms::toDepthFuncPtr_t Perms::toDepthFromRightFuncPtrs[] = {
    &make_perm_list<0, true, true, true, false>,
    &make_perm_list<1, true, true, true, false>,
    &make_perm_list<2, true, true, true, false>,
    &make_perm_list<3, true, true, true, false>,
    &make_perm_list<4, true, true, true, false>,
    &make_perm_list<5, true, true, true, false>};


u32 MAKE_FAT_PERM_LIST_HELPER_CALLS = 0;
u32 MAKE_FAT_PERM_LIST_HELPER_LESS_THAN_CHECKS = 0;
u32 MAKE_FAT_PERM_LIST_HELPER_FOUND_SIMILAR = 0;


static constexpr bool tDFL1 = true;
Perms::toDepthFuncPtr_t Perms::toDepthFromLeftFatFuncPtrs[] = {
        &make_fat_perm_list<0, tDFL1>,
        &make_fat_perm_list<1, tDFL1>,
        &make_fat_perm_list<2, tDFL1>,
        &make_fat_perm_list<3, tDFL1>,
        &make_fat_perm_list<4, tDFL1>,
        &make_fat_perm_list<5, tDFL1>};


static constexpr bool tDFR1 = false;
Perms::toDepthFuncPtr_t Perms::toDepthFromRightFatFuncPtrs[] = {
        &make_fat_perm_list<0, tDFR1>,
        &make_fat_perm_list<1, tDFR1>,
        &make_fat_perm_list<2, tDFR1>,
        &make_fat_perm_list<3, tDFR1>,
        &make_fat_perm_list<4, tDFR1>,
        &make_fat_perm_list<5, tDFR1>
};



Perms::toDepthPlusOneFuncPtr_t Perms::toDepthPlusOneFuncPtr = make_permutation_list_depth_plus_one;
void Perms::getDepthPlus1Func(C JVec<Board>& boards_in, JVec<Board>& boards_out, C bool shouldResize) {
    if (shouldResize) { boards_out.resize(boards_in.size() * 60); }

    boards_out.resize(boards_out.capacity());
    C Board::HasherPtr hasher = boards_in[0].getHashFunc();
    toDepthPlusOneFuncPtr(boards_in, boards_out, hasher);
}


template<bool CHECK_CROSS, bool CHECK_SIM, u32 BUFFER_SIZE>
void make_permutation_list_depth_plus_one_buffered(
        C std::string &root_path,
        C JVec<Board> &boards_in, JVec<Board>& boards_buffer, Board::HasherPtr hasher) {

    int vector_index = 0;
    int buffer_index = 0;


    for (C auto &board_index : boards_in) {
        C u8 a = board_index.getMemory().getLastMove();
        C u8 a_dir = a / 30;
        C u8 a_sect = a % 30 / 5;
        C u8 a_amount = a % 5 + 1;


        for (int b_dir = 0; b_dir < 2; b_dir++) {
            C bool do_RC_check = a_dir != b_dir && a_dir != 0;

            C int b_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_start; b_sect < 6; b_sect++) {
                C int b_base = b_dir * 30 + b_sect * 5;

                u8 intersects = 0;
                if constexpr (CHECK_CROSS) {
                    if (do_RC_check) { intersects = board_index.doActISColMatchBatched(a_sect, b_sect, a_amount); }
                }

                for (int b_amount = 0; b_amount < 5; b_amount++) {
                    if constexpr (CHECK_CROSS) {
                        if (do_RC_check && intersects & (1 << (b_amount))) { continue; }
                    }

                    C int b_cur = b_base + b_amount;

                    boards_buffer[vector_index] = board_index;
                    allActStructList[b_cur].action(boards_buffer[vector_index]);
                    if constexpr (CHECK_SIM) {
                        if (boards_buffer[vector_index].b1 == board_index.b1 && boards_buffer[vector_index].b2 == board_index.b2) { continue; }
                    }

                    (boards_buffer[vector_index].*hasher)();
                    boards_buffer[vector_index].getMemory().setNextNMove<1>(b_cur);
                    vector_index++;

                    if EXPECT_FALSE (vector_index > BUFFER_SIZE) {
                        std::string filename = root_path + std::to_string(buffer_index) + ".bin";
                        std::cout << "writing to file '" + filename + "'.\n";
                        std::ofstream outfile(filename, std::ios::binary);
                        outfile.write(reinterpret_cast<C char *>(boards_buffer.data()), static_cast<int64_t>(boards_buffer.size() * sizeof(Board)));
                        outfile.close();
                        buffer_index++;

                        vector_index = 0;
                    }
                }
            }
        }
    }

    if EXPECT_FALSE (vector_index != 0) {
        std::string filename = root_path + std::to_string(buffer_index) + ".bin";
        std::cout << "writing to file '" + filename + "'.\n";
        std::ofstream outfile(filename, std::ios::binary);
        outfile.write(reinterpret_cast<C char *>(boards_buffer.data()), static_cast<int64_t>(vector_index * sizeof(Board)));
        outfile.close();
    }
}


Perms::toDepthPlusOneFuncBufferedPtr_t Perms::toDepthPlusOneBufferedFuncPtr = make_permutation_list_depth_plus_one_buffered;


void Perms::getDepthPlus1BufferedFunc(
        C std::string &root_path,
        C JVec<Board> &boards_in, JVec<Board> &boards_buffer, C int depth) {

    boards_buffer.resize(boards_buffer.capacity());

    C Board::HasherPtr hasher = boards_in[0].getHashFunc();

    C std::string path = root_path + std::to_string(depth + 1) + "_";
    toDepthPlusOneBufferedFuncPtr(path, boards_in, boards_buffer, hasher);
}
