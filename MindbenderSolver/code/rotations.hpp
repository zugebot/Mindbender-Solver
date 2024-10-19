#pragma once

#include "board.hpp"
#include <map>


#define PERM_MACRO(name) void name(Board &board)

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


typedef void (*Action)(Board &);
MU extern Action allActionsList[110];


struct ActStruct {
    Action action;
    union {
        struct {
            u32 isFat :  8;
            u32 isRow :  8;
            u32 index : 16;
        };
    };

    ActStruct(Action theAction, c_u16 theIndex, c_u16 theIsRow, c_u16 theIsFat) {
        action = theAction;
        index = theIndex;
        isRow = theIsRow;
        isFat = theIsFat;
    }
};

MU extern ActStruct allActStructList[110];



MU extern u8 fatActionsIndexes[25][48];
MU extern std::map<Action, u8> actionToIndex;


extern std::string getNameFromAction(Action action);
extern Action getActionFromName(const std::string& name);
extern u8 getIndexFromAction(Action action);

extern void applyMoves(Board& board, HashMem& hashMem);
extern void applyFatMoves(Board& board, HashMem& hashMem);