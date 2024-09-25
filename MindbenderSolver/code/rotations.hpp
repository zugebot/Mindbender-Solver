#pragma once

#include "board.hpp"

#include <array>
#include <map>

#define PERM_MACRO(name) void name(Board &board)
typedef void (*Action)(Board &);

PERM_MACRO(R_0_1); PERM_MACRO(R_0_2); PERM_MACRO(R_0_3); PERM_MACRO(R_0_4); PERM_MACRO(R_0_5);
PERM_MACRO(R_1_1); PERM_MACRO(R_1_2); PERM_MACRO(R_1_3); PERM_MACRO(R_1_4); PERM_MACRO(R_1_5);
PERM_MACRO(R_2_1); PERM_MACRO(R_2_2); PERM_MACRO(R_2_3); PERM_MACRO(R_2_4); PERM_MACRO(R_2_5);
PERM_MACRO(R_3_1); PERM_MACRO(R_3_2); PERM_MACRO(R_3_3); PERM_MACRO(R_3_4); PERM_MACRO(R_3_5);
PERM_MACRO(R_4_1); PERM_MACRO(R_4_2); PERM_MACRO(R_4_3); PERM_MACRO(R_4_4); PERM_MACRO(R_4_5);
PERM_MACRO(R_5_1); PERM_MACRO(R_5_2); PERM_MACRO(R_5_3); PERM_MACRO(R_5_4); PERM_MACRO(R_5_5);
PERM_MACRO(C_0_1); PERM_MACRO(C_0_2); PERM_MACRO(C_0_3); PERM_MACRO(C_0_4); PERM_MACRO(C_0_5);
PERM_MACRO(C_1_1); PERM_MACRO(C_1_2); PERM_MACRO(C_1_3); PERM_MACRO(C_1_4); PERM_MACRO(C_1_5);
PERM_MACRO(C_2_1); PERM_MACRO(C_2_2); PERM_MACRO(C_2_3); PERM_MACRO(C_2_4); PERM_MACRO(C_2_5);
PERM_MACRO(C_3_1); PERM_MACRO(C_3_2); PERM_MACRO(C_3_3); PERM_MACRO(C_3_4); PERM_MACRO(C_3_5);
PERM_MACRO(C_4_1); PERM_MACRO(C_4_2); PERM_MACRO(C_4_3); PERM_MACRO(C_4_4); PERM_MACRO(C_4_5);
PERM_MACRO(C_5_1); PERM_MACRO(C_5_2); PERM_MACRO(C_5_3); PERM_MACRO(C_5_4); PERM_MACRO(C_5_5);

// permutations that are special for fat boards
PERM_MACRO(R_01_1); PERM_MACRO(R_01_2); PERM_MACRO(R_01_3); PERM_MACRO(R_01_4); PERM_MACRO(R_01_5);
PERM_MACRO(R_12_1); PERM_MACRO(R_12_2); PERM_MACRO(R_12_3); PERM_MACRO(R_12_4); PERM_MACRO(R_12_5);
PERM_MACRO(R_23_1); PERM_MACRO(R_23_2); PERM_MACRO(R_23_3); PERM_MACRO(R_23_4); PERM_MACRO(R_23_5);
PERM_MACRO(R_34_1); PERM_MACRO(R_34_2); PERM_MACRO(R_34_3); PERM_MACRO(R_34_4); PERM_MACRO(R_34_5);
PERM_MACRO(R_45_1); PERM_MACRO(R_45_2); PERM_MACRO(R_45_3); PERM_MACRO(R_45_4); PERM_MACRO(R_45_5);
PERM_MACRO(C_01_1); PERM_MACRO(C_01_2); PERM_MACRO(C_01_3); PERM_MACRO(C_01_4); PERM_MACRO(C_01_5);
PERM_MACRO(C_12_1); PERM_MACRO(C_12_2); PERM_MACRO(C_12_3); PERM_MACRO(C_12_4); PERM_MACRO(C_12_5);
PERM_MACRO(C_23_1); PERM_MACRO(C_23_2); PERM_MACRO(C_23_3); PERM_MACRO(C_23_4); PERM_MACRO(C_23_5);
PERM_MACRO(C_34_1); PERM_MACRO(C_34_2); PERM_MACRO(C_34_3); PERM_MACRO(C_34_4); PERM_MACRO(C_34_5);
PERM_MACRO(C_45_1); PERM_MACRO(C_45_2); PERM_MACRO(C_45_3); PERM_MACRO(C_45_4); PERM_MACRO(C_45_5);




MU extern std::map<Action, std::string> actionToNameLookup;
MU extern std::map<std::string, Action> nameToActionLookup;

MU extern Action actions[60];
MU extern Action fatActions[25][48];
MU extern Action allActionsList[110];



#define DIF(func1) static_cast<short>(reinterpret_cast<uint64_t>( \
        ( \
        reinterpret_cast<uintptr_t>(func1) - reinterpret_cast<uintptr_t>(R_0_1) \
        ) \
 ))



class ActionHelper {
public:
    MUND static __forceinline Action getAllAction(size_t index) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myAllActions[index]);
    }

    MUND static __forceinline Action getNormalAction(c_u64 index) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myNormalActions[index]);
    }

    MUND static __forceinline Action getFatAction(c_u64 index1, c_u64 index2) {
        return (Action)(ActionHelper::smallestPtr + ActionHelper::myFatActions[index1][index2]);
    }


    MU static __forceinline void applyAllAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myAllActions[index]))(board);
    }

    MU static __forceinline void applyNormalAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myNormalActions[index]))(board);
    }


    MU static __forceinline void applyFatAction(Board& board, c_u64 index) {
        ((Action)(ActionHelper::smallestPtr + ActionHelper::myFatActions[board.getFatXY()][index]))(board);
    }



private:
    typedef short ptrType;
    static const uintptr_t smallestPtr;

    static const std::array<ptrType, 110> myAllActions;
    static const std::array<ptrType, 60> myNormalActions;
    static const std::array<std::array<ptrType, 48>, 25> myFatActions;


    static std::array<ptrType, 110> init_allActions() {
            std::array<ptrType, 110> arr{};
            for (int i = 0; i < 110; i++) {
                arr[i] = DIF(allActionsList[i]);
            }
        return arr;
    }


    static std::array<ptrType, 60> init_normalActions() {
        std::array<ptrType, 60> arr{};
        for (int i = 0; i < 60; i++) {
            arr[i] = DIF(actions[i]);
        }
        return arr;
    }

    static std::array<std::array<ptrType, 48>, 25> init_fatActions() {
        std::array<std::array<ptrType, 48>, 25> arr{};
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 48; j++) {
                arr[i][j] = DIF(fatActions[i][j]);
            }
        }
        return arr;
    }
};









