#pragma once

#include "utils/processor.hpp"

#include "board.hpp"

#include <cstring>
#include <unordered_map>

/*
RED     0
GREEN   1
BLUE    2
ORANGE  3
YELLOW  4
PURPLE  5
WHITE   6
CYAN    7
 */



class BoardPair {
    C Board* board{};
    C Board* solve{};
    char name[5] = {};
    MU Board::ColorArray_t trueColors{};

public:
    BoardPair() = default;
    BoardPair(C Board* board, C Board* solve, C char nameIn[5], C u8 colors[36]) :
        board(board), solve(solve) {
        strncpy(name, nameIn, 4);
        name[4] = '\0';

        Board temp;
        trueColors = temp.setStateAndRetColors(colors);
    }

    MUND Board getStartState() C { return *board; }
    MUND Board getEndState() C { return *solve; }
    MUND std::string getName() C { return name; }
    MUND i8 getTrueColor(C u32 color) C { return trueColors[color]; }
    MUND std::string toString() C { return board->toString(solve, {true, trueColors}); }
    MUND std::string toStringReversed() C { return solve->toString(board, {true, trueColors}); }
};


#define BP(name) \
MU static C BoardPair (name)

class LevelBoardPair {
public:
    BP(p1_1);  BP(p1_2);  BP(p1_3);  BP(p1_4);  BP(p1_5);
    BP(p2_1);  BP(p2_2);  BP(p2_3);  BP(p2_4);  BP(p2_5);
    BP(p3_1);  BP(p3_2);  BP(p3_3);  BP(p3_4);  BP(p3_5);
    BP(p4_1);  BP(p4_2);  BP(p4_3);  BP(p4_4);  BP(p4_5);
    BP(p5_1);  BP(p5_2);  BP(p5_3);  BP(p5_4);  BP(p5_5);
    BP(p6_1);  BP(p6_2);  BP(p6_3);  BP(p6_4);  BP(p6_5);
    BP(p7_1);  BP(p7_2);  BP(p7_3);  BP(p7_4);  BP(p7_5);
    BP(p8_1);  BP(p8_2);  BP(p8_3);  BP(p8_4);  BP(p8_5);
    BP(p9_1);  BP(p9_2);  BP(p9_3);  BP(p9_4);  BP(p9_5);
    BP(p10_1); BP(p10_2); BP(p10_3); BP(p10_4); BP(p10_5);
    BP(p11_1); BP(p11_2); BP(p11_3); BP(p11_4); BP(p11_5);
    BP(p12_1); BP(p12_2); BP(p12_3); BP(p12_4); BP(p12_5);
    BP(p13_1); BP(p13_2); BP(p13_3); BP(p13_4); BP(p13_5);
    BP(p14_1); BP(p14_2); BP(p14_3); BP(p14_4); BP(p14_5);
    BP(p15_1); BP(p15_2); BP(p15_3); BP(p15_4); BP(p15_5);
    BP(p16_1); BP(p16_2); BP(p16_3); BP(p16_4); BP(p16_5);
    BP(p17_1); BP(p17_2); BP(p17_3); BP(p17_4); BP(p17_5);
    BP(p18_1); BP(p18_2); BP(p18_3); BP(p18_4); BP(p18_5);
    BP(p19_1); BP(p19_2); BP(p19_3); BP(p19_4); BP(p19_5);
    BP(p20_1); BP(p20_2); BP(p20_3); BP(p20_4); BP(p20_5);
};
#undef BP

class BoardLookup {
    static C std::unordered_map<std::string, C BoardPair*> boardPairDict;
    static C std::unordered_map<std::string, int> boardPairDictIndexes;
public:
    MUND static BoardPair const* getBoardPair(C std::string& name) {
        C auto pair = boardPairDict.find(name);
        if (pair == boardPairDict.end()) {
            return nullptr;
        }
        return pair->second;
    }
};


