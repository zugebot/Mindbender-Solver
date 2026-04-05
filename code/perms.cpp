// code/perms.cpp
#include "perms.hpp"
#include "rotations.hpp"


template class Perms<Memory>;
template class Perms<Board>;
template class Perms<B1B2>;


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
template<eSequenceDir SECT_DIR>
MU void Perms<T>::reserveForDepth(MU C Board& board_in, JVec<T>& boards_out, C u32 depth) {
    C double fraction = Board::getDuplicateEstimateAtDepth(depth);
    
    u64 allocSize;
    if constexpr (SECT_DIR != eSequenceDir::NONE) {
        allocSize = board_in.getFatBool()
                            ? BOARD_FAT_MAX_MALLOC_SIZES[depth]
                            : BOARD_PRE_MAX_MALLOC_SIZES[depth];
    } else {
        allocSize = BOARD_SECT_NONE_PRE_MAX_MALLOC_SIZES[depth];
    }

    allocSize = static_cast<u64>(static_cast<double>(allocSize) * fraction);
    boards_out.reserve(allocSize);
}


template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromLeft::funcPtrs[] = {
        &perms_detail::make_perm_list<T, 0, false, true, true, eSequenceDir::ASCENDING>,
        &perms_detail::make_perm_list<T, 1, false, true, true, eSequenceDir::ASCENDING>,
        &perms_detail::make_perm_list<T, 2, false, true, true, eSequenceDir::ASCENDING>,
        &perms_detail::make_perm_list<T, 3, false, true, true, eSequenceDir::ASCENDING>,
        &perms_detail::make_perm_list<T, 4, false, true, true, eSequenceDir::ASCENDING>,
        &perms_detail::make_perm_list<T, 5, false, true, true, eSequenceDir::ASCENDING>,
};


// TODO: CHECK_CROSS must be OFF for fromLeft and fromRight, because it can prune too early. It may only be used on eSequenceDir::NONE.
template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromNone::funcPtrs[] = {
        &perms_detail::make_perm_list<T, 0, false, true, false, eSequenceDir::NONE>,
        &perms_detail::make_perm_list<T, 1, false, true, false, eSequenceDir::NONE>,
        &perms_detail::make_perm_list<T, 2, false, true, false, eSequenceDir::NONE>,
        &perms_detail::make_perm_list<T, 3, false, true, false, eSequenceDir::NONE>,
        &perms_detail::make_perm_list<T, 4, false, true, false, eSequenceDir::NONE>,
        &perms_detail::make_perm_list<T, 5, false, true, false, eSequenceDir::NONE>,
};

template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromRight::funcPtrs[] = {
        &perms_detail::make_perm_list<T, 0, false, true, true, eSequenceDir::DESCENDING>,
        &perms_detail::make_perm_list<T, 1, false, true, true, eSequenceDir::DESCENDING>,
        &perms_detail::make_perm_list<T, 2, false, true, true, eSequenceDir::DESCENDING>,
        &perms_detail::make_perm_list<T, 3, false, true, true, eSequenceDir::DESCENDING>,
        &perms_detail::make_perm_list<T, 4, false, true, true, eSequenceDir::DESCENDING>,
        &perms_detail::make_perm_list<T, 5, false, true, true, eSequenceDir::DESCENDING>,
};


template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromLeft::fatFuncPtrs[] = {
        &perms_detail::make_fat_perm_list<T, 0, eSequenceDir::ASCENDING>,
        &perms_detail::make_fat_perm_list<T, 1, eSequenceDir::ASCENDING>,
        &perms_detail::make_fat_perm_list<T, 2, eSequenceDir::ASCENDING>,
        &perms_detail::make_fat_perm_list<T, 3, eSequenceDir::ASCENDING>,
        &perms_detail::make_fat_perm_list<T, 4, eSequenceDir::ASCENDING>,
        &perms_detail::make_fat_perm_list<T, 5, eSequenceDir::ASCENDING>,
};


template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromRight::fatFuncPtrs[] = {
        &perms_detail::make_fat_perm_list<T, 0, eSequenceDir::DESCENDING>,
        &perms_detail::make_fat_perm_list<T, 1, eSequenceDir::DESCENDING>,
        &perms_detail::make_fat_perm_list<T, 2, eSequenceDir::DESCENDING>,
        &perms_detail::make_fat_perm_list<T, 3, eSequenceDir::DESCENDING>,
        &perms_detail::make_fat_perm_list<T, 4, eSequenceDir::DESCENDING>,
        &perms_detail::make_fat_perm_list<T, 5, eSequenceDir::DESCENDING>,
};

template<typename T>
MU typename Perms<T>::ToDepthFuncPtr Perms<T>::FromNone::fatFuncPtrs[] = {
        &perms_detail::make_fat_perm_list<T, 0, eSequenceDir::NONE>,
        &perms_detail::make_fat_perm_list<T, 1, eSequenceDir::NONE>,
        &perms_detail::make_fat_perm_list<T, 2, eSequenceDir::NONE>,
        &perms_detail::make_fat_perm_list<T, 3, eSequenceDir::NONE>,
        &perms_detail::make_fat_perm_list<T, 4, eSequenceDir::NONE>,
        &perms_detail::make_fat_perm_list<T, 5, eSequenceDir::NONE>,
};

template void Perms<Memory>::reserveForDepth<eSequenceDir::ASCENDING>(
        C Board& board_in, JVec<Memory>& boards_out, u32 depth);
template void Perms<Memory>::reserveForDepth<eSequenceDir::DESCENDING>(
        C Board& board_in, JVec<Memory>& boards_out, u32 depth);
template void Perms<Memory>::reserveForDepth<eSequenceDir::NONE>(
        C Board& board_in, JVec<Memory>& boards_out, u32 depth);

template void Perms<Board>::reserveForDepth<eSequenceDir::ASCENDING>(
        C Board& board_in, JVec<Board>& boards_out, u32 depth);
template void Perms<Board>::reserveForDepth<eSequenceDir::DESCENDING>(
        C Board& board_in, JVec<Board>& boards_out, u32 depth);
template void Perms<Board>::reserveForDepth<eSequenceDir::NONE>(
        C Board& board_in, JVec<Board>& boards_out, u32 depth);

template void Perms<B1B2>::reserveForDepth<eSequenceDir::ASCENDING>(
        C Board& board_in, JVec<B1B2>& boards_out, u32 depth);
template void Perms<B1B2>::reserveForDepth<eSequenceDir::DESCENDING>(
        C Board& board_in, JVec<B1B2>& boards_out, u32 depth);
template void Perms<B1B2>::reserveForDepth<eSequenceDir::NONE>(
        C Board& board_in, JVec<B1B2>& boards_out, u32 depth);