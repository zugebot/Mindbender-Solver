#include "perms.hpp"
#include "rotations.hpp"

#include <array>
#include <fstream>
#include <iostream>


template<bool CHECK_CROSS, bool CHECK_SIM>
void make_permutation_list_depth_plus_one(const std::vector<Board> &boards_in, std::vector<Board> &boards_out, const Board::HasherPtr hasher) {
    int count = 0;
    u8 intersects = 0;

    for (const auto &board_index: boards_in) {
        c_u8 a = board_index.getMemory().getLastMove();
        c_u8 a_dir = a / 30;
        c_u8 a_sect = a % 30 / 5;
        c_u8 a_amount = a % 5 + 1;


        for (int b_dir = 0; b_dir < 2; b_dir++) {
            c_bool do_RC_check = a_dir != b_dir && a_dir != 0;

            c_int b_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_start; b_sect < 6; b_sect++) {
                c_int b_base = b_dir * 30 + b_sect * 5;

                if constexpr (CHECK_CROSS) {
                    if (do_RC_check) { intersects = board_index.doActISColMatchBatched(a_sect, b_sect, a_amount); }
                }

                for (int b_amount = 0; b_amount < 5; b_amount++) {
                    if constexpr (CHECK_CROSS) {
                        if (do_RC_check && intersects & (1 << (b_amount))) { continue; }
                    }

                    c_int b_cur = b_base + b_amount;

                    boards_out[count] = board_index;
                    allActionsList[b_cur](boards_out[count]);
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


const Perms::depthMap_t Perms::depthMap = {
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


MU void Perms::reserveForDepth(MU const Board& board_in, std::vector<Board> &boards_out, c_u32 depth) {
    c_double fraction = Board::getDuplicateEstimateAtDepth(depth);
    u64 allocSize = board_in.getFatBool() ? BOARD_FAT_MAX_MALLOC_SIZES[depth] : BOARD_PRE_MAX_MALLOC_SIZES[depth];

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


MU void Perms::reserveForDepth(MU const Board& board_in, std::vector<HashMem> &boards_out, c_u32 depth) {
    c_double fraction = Board::getDuplicateEstimateAtDepth(depth);
    u64 allocSize = board_in.getFatBool() ? BOARD_FAT_MAX_MALLOC_SIZES[depth] : BOARD_PRE_MAX_MALLOC_SIZES[depth];

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


Perms::toDepthFuncPtr_t Perms::toDepthFatFuncPtrs[] = {
        &make_fat_perm_list<0>,
        &make_fat_perm_list<1>,
        &make_fat_perm_list<2>,
        &make_fat_perm_list<3>,
        &make_fat_perm_list<4>,
        &make_fat_perm_list<5>};
Perms::toDepthFuncPtr_t Perms::toDepthFuncPtrs[] = {
        &make_perm_list<0>,
        &make_perm_list<1>,
        &make_perm_list<2>,
        &make_perm_list<3>,
        &make_perm_list<4>,
        &make_perm_list<5>};



void Perms::getDepthFunc(const Board& board_in, std::vector<HashMem> &boards_out, c_u32 depth, c_bool shouldResize) {
    if (depth >= PTR_LIST_SIZE) { return; }
    if (shouldResize) { reserveForDepth(board_in, boards_out, depth); }

    boards_out.resize(boards_out.capacity());
    const HashMem::HasherPtr hasher = HashMem::getHashFunc(board_in);
    if (!board_in.getFatBool()) {
        toDepthFuncPtrs[depth](board_in, boards_out, hasher);
    } else {
        toDepthFatFuncPtrs[depth](board_in, boards_out, hasher);
    }
}


Perms::toDepthPlusOneFuncPtr_t Perms::toDepthPlusOneFuncPtr = make_permutation_list_depth_plus_one;
void Perms::getDepthPlus1Func(const std::vector<Board> &boards_in, std::vector<Board> &boards_out, c_bool shouldResize) {
    if (shouldResize) { boards_out.resize(boards_in.size() * 60); }

    boards_out.resize(boards_out.capacity());
    const Board::HasherPtr hasher = boards_in[0].getHashFunc();
    toDepthPlusOneFuncPtr(boards_in, boards_out, hasher);
}


template<bool CHECK_CROSS, bool CHECK_SIM, u32 BUFFER_SIZE>
void make_permutation_list_depth_plus_one_buffered(
        const std::string &root_path,
        const std::vector<Board> &boards_in, std::vector<Board> &board_buffer, Board::HasherPtr hasher) {

    int vector_index = 0;
    int buffer_index = 0;


    for (const auto &board_index: boards_in) {
        c_u8 a = board_index.getMemory().getLastMove();
        c_u8 a_dir = a / 30;
        c_u8 a_sect = a % 30 / 5;
        c_u8 a_amount = a % 5 + 1;


        for (int b_dir = 0; b_dir < 2; b_dir++) {
            c_bool do_RC_check = a_dir != b_dir && a_dir != 0;

            c_int b_start = (b_dir == a_dir) ? a_sect + 1 : 0;
            for (int b_sect = b_start; b_sect < 6; b_sect++) {
                c_int b_base = b_dir * 30 + b_sect * 5;

                u8 intersects = 0;
                if constexpr (CHECK_CROSS) {
                    if (do_RC_check) { intersects = board_index.doActISColMatchBatched(a_sect, b_sect, a_amount); }
                }

                for (int b_amount = 0; b_amount < 5; b_amount++) {
                    if constexpr (CHECK_CROSS) {
                        if (do_RC_check && intersects & (1 << (b_amount))) { continue; }
                    }

                    c_int b_cur = b_base + b_amount;

                    board_buffer[vector_index] = board_index;
                    allActionsList[b_cur](board_buffer[vector_index]);
                    if constexpr (CHECK_SIM) {
                        if (board_buffer[vector_index].b1 == board_index.b1 && board_buffer[vector_index].b2 == board_index.b2) { continue; }
                    }

                    (board_buffer[vector_index].*hasher)();
                    board_buffer[vector_index].getMemory().setNextNMove<1>(b_cur);
                    vector_index++;

                    if EXPECT_FALSE (vector_index > BUFFER_SIZE) {
                        std::string filename = root_path + std::to_string(buffer_index) + ".bin";
                        std::cout << "writing to file '" + filename + "'.\n";
                        std::ofstream outfile(filename, std::ios::binary);
                        outfile.write(reinterpret_cast<const char *>(board_buffer.data()), (i64) (board_buffer.size() * sizeof(Board)));
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
        outfile.write(reinterpret_cast<const char *>(board_buffer.data()), (i64) (vector_index * sizeof(Board)));
        outfile.close();
    }
}
Perms::toDepthPlusOneFuncBufferedPtr_t Perms::toDepthPlusOneBufferedFuncPtr = make_permutation_list_depth_plus_one_buffered;

void Perms::getDepthPlus1BufferedFunc(
        const std::string &root_path,
        const std::vector<Board> &boards_in, std::vector<Board> &boards_out, int depth) {

    boards_out.resize(boards_out.capacity());

    const Board::HasherPtr hasher = boards_in[0].getHashFunc();

    std::string path = root_path + std::to_string(depth + 1) + "_";
    toDepthPlusOneBufferedFuncPtr(path, boards_in, boards_out, hasher);
}
