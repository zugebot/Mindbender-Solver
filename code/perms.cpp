// code/perms.cpp
#include "perms.hpp"
#include "rotations.hpp"


template class Perms<Memory>;
template class Perms<Board>;


template<typename T>
MU C typename Perms<T>::DepthMap Perms<T>::depthMap = {
        {1,  {{1, 0}, {0, 1}}},
        {2,  {{1, 1}, {0, 2}, {2, 0}}},
        {3,  {{1, 2}, {2, 1}, {0, 3}, {3, 0}}},
        {4,  {{2, 2}, {3, 1}, {1, 3}, {4, 0}, {0, 4}}},
        {5,  {{3, 2}, {2, 3}, {4, 1}, {1, 4}, {5, 0}, {0, 5}}},
        {6,  {{3, 3}, {4, 2}, {2, 4}, {5, 1}, {1, 5}}},
        {7,  {{4, 3}, {3, 4}, {5, 2}, {2, 5}}},
        {8,  {{4, 4}, {5, 3}, {3, 5}}},
        {9,  {{4, 5}, {5, 4}}},
        {10, {{5, 5}}},
        {11, {{6, 5}}},
};


template<typename T>
MU void Perms<T>::reserveForDepth(MU C Board& board_in, JVec<T>& boards_out, C u32 depth) {
    C double fraction = Board::getDuplicateEstimateAtDepth(depth);
    
    u64 allocSize = board_in.getFatBool()
                            ? BOARD_FAT_MAX_MALLOC_SIZES[depth]
                            : BOARD_PRE_MAX_MALLOC_SIZES[depth];

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromLeft::funcPtrs[] = {
        &perms_detail::make_perm_list<T, 0, true, true, true, true>,
        &perms_detail::make_perm_list<T, 1, true, true, true, true>,
        &perms_detail::make_perm_list<T, 2, true, true, true, true>,
        &perms_detail::make_perm_list<T, 3, true, true, true, true>,
        &perms_detail::make_perm_list<T, 4, true, true, true, true>,
        &perms_detail::make_perm_list<T, 5, true, true, true, true>,
};


template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromRight::funcPtrs[] = {
        &perms_detail::make_perm_list<T, 0, true, true, true, false>,
        &perms_detail::make_perm_list<T, 1, true, true, true, false>,
        &perms_detail::make_perm_list<T, 2, true, true, true, false>,
        &perms_detail::make_perm_list<T, 3, true, true, true, false>,
        &perms_detail::make_perm_list<T, 4, true, true, true, false>,
        &perms_detail::make_perm_list<T, 5, true, true, true, false>,
};


static constexpr bool ASCENDING_MOVES = true;
template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromLeft::fatFuncPtrs[] = {
        &perms_detail::make_fat_perm_list<T, 0, ASCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 1, ASCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 2, ASCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 3, ASCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 4, ASCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 5, ASCENDING_MOVES>,
};


static constexpr bool DESCENDING_MOVES = false;
template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromRight::fatFuncPtrs[] = {
        &perms_detail::make_fat_perm_list<T, 0, DESCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 1, DESCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 2, DESCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 3, DESCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 4, DESCENDING_MOVES>,
        &perms_detail::make_fat_perm_list<T, 5, DESCENDING_MOVES>,
};
