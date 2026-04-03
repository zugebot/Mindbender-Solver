#pragma once
// code/perms.hpp

#include "board.hpp"
#include "rotations.hpp"
#include "utils/jvec.hpp"
#include "utils/processor.hpp"

#include <array>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef USE_CUDA
// C++17 version because my GPU is ASS
template<typename T>
struct IsAllowedPermsType {
    static constexpr bool value =
            std::is_same_v<T, Memory> || std::is_same_v<T, Board>;
};

template<typename T>
constexpr bool AllowedPermsType = IsAllowedPermsType<T>::value;
#else
// C++20 concept implementation
template<typename T>
concept AllowedPermsType = std::is_same_v<T, Memory> || std::is_same_v<T, Board>;
#endif


MU static constexpr u64 BOARD_PRE_MAX_MALLOC_SIZES[8] = {
        1, 60, 2550, 104000, 4245000, 173325000, 7076687500, 288933750000,
};

MU static constexpr u64 BOARD_FAT_MAX_MALLOC_SIZES[8] = {
        1, 48, 1320, 36402, 1001168, 27513569, 0, 0,
};

namespace perms_detail {

    template<typename T, i32 MAX_DEPTH>
    struct PermBuildState {
        static_assert(AllowedPermsType<T>, "T must be Memory or Board");

        std::array<i32, MAX_DEPTH> dirSeq{};
        std::array<i32, MAX_DEPTH> sectSeq{};
        std::array<i32, MAX_DEPTH> baseSeq{};
        std::array<u64, MAX_DEPTH> curSeq{};
        std::array<bool, MAX_DEPTH> checkRCSeq{};
        std::array<u8, MAX_DEPTH> intersectSeq{};
        typename T::HasherPtr hasher{};
    };

    // ============================================================
    // Normal board permutation generation
    // ============================================================

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM>
    static void make_perm_list_inner(
            C Board& board_in,
            JVec<T>& boards_out,
            PermBuildState<T, MAX_DEPTH>& state,
            u64 move_prev,
            i32& count);

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, bool SECT_ASCENDING>
    static void make_perm_list_outer(
            C Board& board_in,
            JVec<T>& boards_out,
            PermBuildState<T, MAX_DEPTH>& state,
            i32& count);

    template<typename T,
             i32 MAX_DEPTH,
             bool CHECK_CROSS, bool CHECK_SIM,
             bool CHANGE_SECT_START, bool SECT_ASCENDING>
    static void make_perm_list(
            C Board& board_in,
            JVec<T>& boards_out,
            typename T::HasherPtr hasher);

    // ============================================================
    // Fat board permutation generation
    // ============================================================

    template<typename T,
             i32 CUR_DEPTH, i32 MAX_DEPTH,
             bool MOVES_ASCENDING, bool DIRECTION>
    static void make_fat_perm_list_helper(
            C Board& board,
            JVec<T>& boards_out,
            u32& count,
            typename T::HasherPtr hasher,
            u64 move,
            C ActStruct& lastActStruct,
            u8 startIndex,
            u8 endIndex);

    template<typename T,
             i32 DEPTH,
             bool MOVES_ASCENDING = true>
    static void make_fat_perm_list(
            C Board& board_in,
            JVec<T>& boards_out,
            typename T::HasherPtr hasher);

} // namespace perms_detail


template<typename T>
class Perms {
    static_assert(AllowedPermsType<T>, "T must be Memory or Board");

public:
    using ToDepthFuncPtr = void (*)(C Board&, JVec<T>&, typename T::HasherPtr);
    using DepthPair = std::pair<u32, u32>;
    using DepthMap = std::unordered_map<u32, std::vector<DepthPair>>;

    static constexpr u32 PTR_LIST_SIZE = 6;

    static C DepthMap depthMap;

    struct FromLeft {
        static ToDepthFuncPtr funcPtrs[PTR_LIST_SIZE];
        static ToDepthFuncPtr fatFuncPtrs[PTR_LIST_SIZE];
    };

    struct FromRight {
        static ToDepthFuncPtr funcPtrs[PTR_LIST_SIZE];
        static ToDepthFuncPtr fatFuncPtrs[PTR_LIST_SIZE];
    };

    MU static void reserveForDepth(C Board& board_in, JVec<T>& boards_out, u32 depth);

    template<bool SECT_ASCENDING = true>
    MU static void getDepthFunc(C Board& board_in, JVec<T>& boards_out, u32 depth, bool shouldResize = true);
};

template<typename T>
template<bool SECT_ASCENDING>
void Perms<T>::getDepthFunc(
        C Board& board_in,
        JVec<T>& boards_out,
        C u32 depth,
        C bool shouldResize) {

    if (depth >= PTR_LIST_SIZE) {
        return;
    }

    if (shouldResize) {
        reserveForDepth(board_in, boards_out, depth);
    }

    boards_out.resize(boards_out.capacity());

    typename T::HasherPtr hasher{};
    if constexpr (std::is_same_v<T, Memory>) {
        hasher = Memory::getHashFunc(board_in);
    } else if constexpr (std::is_same_v<T, Board>) {
        hasher = board_in.getHashFunc();
    } else {
        static_assert(AllowedPermsType<T>, "Unsupported permutation type");
    }

    if (board_in.getFatBool()) {
        constexpr auto table = SECT_ASCENDING ? FromLeft::fatFuncPtrs : FromRight::fatFuncPtrs;
        table[depth](board_in, boards_out, hasher);
    } else {
        constexpr auto table = SECT_ASCENDING ? FromLeft::funcPtrs : FromRight::funcPtrs;
        table[depth](board_in, boards_out, hasher);
    }
}

extern template class Perms<Memory>;
extern template class Perms<Board>;

#include "perms_fat.tpp"
#include "perms_nrm.tpp"