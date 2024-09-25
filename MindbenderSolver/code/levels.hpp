#pragma once

#include "MindbenderSolver/utils/processor.hpp"
#include "board.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>

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
    Board const* board{};
    Board const* solve{};
    std::string name;

public:
    BoardPair() = default;
    BoardPair(Board const* board, Board const* solve, std::string name) :
        board(board), solve(solve), name(std::move(name)) {}

    MUND Board getInitialState() const {
        return *board;
    }

    MUND Board getSolutionState() const {
        return *solve;
    }

    MUND std::string getName() const {
        return name;
    }

    MUND std::string toString() const {
        return board->toString(solve);
    }

};


class LevelBoardPair {
public:
    MU static const BoardPair p1_1;
    MU static const BoardPair p1_2;
    MU static const BoardPair p1_3;
    MU static const BoardPair p1_4;
    MU static const BoardPair p1_5;
    MU static const BoardPair p2_1;
    MU static const BoardPair p2_2;
    MU static const BoardPair p2_3;
    MU static const BoardPair p2_4;
    MU static const BoardPair p2_5;
    MU static const BoardPair p3_1;
    MU static const BoardPair p3_2;
    MU static const BoardPair p3_3;
    MU static const BoardPair p3_4;
    MU static const BoardPair p3_5;
    MU static const BoardPair p4_1;
    MU static const BoardPair p4_2;
    MU static const BoardPair p4_3;
    MU static const BoardPair p4_4;
    MU static const BoardPair p4_5;
    MU static const BoardPair p5_1;
    MU static const BoardPair p5_2;
    MU static const BoardPair p5_3;
    MU static const BoardPair p5_4;
    MU static const BoardPair p5_5;
    MU static const BoardPair p6_1;
    MU static const BoardPair p6_2;
    MU static const BoardPair p6_3;
    MU static const BoardPair p6_4;
    MU static const BoardPair p6_5;
    MU static const BoardPair p7_1;
    MU static const BoardPair p7_2;
    MU static const BoardPair p7_3;
    MU static const BoardPair p7_4;
    MU static const BoardPair p7_5;
    MU static const BoardPair p8_1;
    MU static const BoardPair p8_2;
    MU static const BoardPair p8_3;
    MU static const BoardPair p8_4;
    MU static const BoardPair p8_5;
    MU static const BoardPair p9_1;
    MU static const BoardPair p9_2;
    MU static const BoardPair p9_3;
    MU static const BoardPair p9_4;
    MU static const BoardPair p9_5;
    MU static const BoardPair p10_1;
    MU static const BoardPair p10_2;
    MU static const BoardPair p10_3;
    MU static const BoardPair p10_4;
    MU static const BoardPair p10_5;
    MU static const BoardPair p11_1;
    MU static const BoardPair p11_2;
    MU static const BoardPair p11_3;
    MU static const BoardPair p11_4;
    MU static const BoardPair p11_5;
    MU static const BoardPair p12_1;
    MU static const BoardPair p12_2;
    MU static const BoardPair p12_3;
    MU static const BoardPair p12_4;
    MU static const BoardPair p12_5;
    MU static const BoardPair p13_1;
    MU static const BoardPair p13_2;
    MU static const BoardPair p13_3;
    MU static const BoardPair p13_4;
    MU static const BoardPair p13_5;
    MU static const BoardPair p14_1;
    MU static const BoardPair p14_2;
    MU static const BoardPair p14_3;
    MU static const BoardPair p14_4;
    MU static const BoardPair p14_5;
    MU static const BoardPair p15_1;
    MU static const BoardPair p15_2;
    MU static const BoardPair p15_3;
    MU static const BoardPair p15_4;
    MU static const BoardPair p15_5;
    MU static const BoardPair p16_1;
    MU static const BoardPair p16_2;
    MU static const BoardPair p16_3;
    MU static const BoardPair p16_4;
    MU static const BoardPair p16_5;
    MU static const BoardPair p17_1;
    MU static const BoardPair p17_2;
    MU static const BoardPair p17_3;
    MU static const BoardPair p17_4;
    MU static const BoardPair p17_5;
    MU static const BoardPair p18_1;
    MU static const BoardPair p18_2;
    MU static const BoardPair p18_3;
    MU static const BoardPair p18_4;
    MU static const BoardPair p18_5;
    MU static const BoardPair p19_1;
    MU static const BoardPair p19_2;
    MU static const BoardPair p19_3;
    MU static const BoardPair p19_4;
    MU static const BoardPair p19_5;
    MU static const BoardPair p20_1;
    MU static const BoardPair p20_2;
    MU static const BoardPair p20_3;
    MU static const BoardPair p20_4;
    MU static const BoardPair p20_5;
};


class BoardLookup {
private:
    static const std::unordered_map<std::string, const BoardPair*> boardPairDict;
public:
    MUND static BoardPair const* getBoardPair(const std::string& name) {
        auto pair = boardPairDict.find(name);
        if (pair == boardPairDict.end()) {
            return nullptr;
        }
        return pair->second;
    }
};


