#pragma once

#include "board.hpp"
#include "levels_cells.hpp"
#include "utils/processor.hpp"

#include <array>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

struct FatPos {
    bool hasFat = false;
    u8 x = 0;
    u8 y = 0;
};

struct LevelDef {
    const char* name = nullptr;
    const u8* startCells = nullptr;
    const u8* endCells = nullptr;
    FatPos startFat{};
    FatPos endFat{};
};

class BoardPair {
    Board board{};
    Board solve{};
    char name[5] = {};
    MU Board::ColorArray_t trueColors{};

    static Board makeBoard(const u8* cells, const FatPos fat) {
        if (fat.hasFat) {
            return Board(const_cast<u8*>(cells), fat.x, fat.y);
        }
        return Board(const_cast<u8*>(cells));
    }

public:
    BoardPair() = default;

    BoardPair(const char* nameIn,
              const u8* startCells,
              const u8* endCells,
              const FatPos startFat = {},
              const FatPos endFat = {})
        : board(makeBoard(startCells, startFat)),
          solve(makeBoard(endCells, endFat)) {
        std::strncpy(name, nameIn, 4);
        name[4] = '\0';

        Board temp;
        trueColors = temp.setStateAndRetColors(const_cast<u8*>(endCells));
    }

    MUND Board getStartState() C { return board; }
    MUND Board getEndState() C { return solve; }
    MUND std::string getName() C { return name; }
    MUND i8 getTrueColor(C u32 color) C { return trueColors[color]; }
    MUND std::string toString() C { return board.toString(solve, {true, trueColors}); }
    MUND std::string toStringReversed() C { return solve.toString(board, {true, trueColors}); }
};

class BoardLookup {
private:
    inline static const std::array<LevelDef, 100> LEVEL_DEFS = {{
            {"1-1",   LevelCells::b1_1,   LevelCells::s1_1,   {},           {}},
            {"1-2",   LevelCells::b1_2,   LevelCells::s1_2,   {},           {}},
            {"1-3",   LevelCells::b1_3,   LevelCells::s1_3,   {},           {}},
            {"1-4",   LevelCells::b1_4,   LevelCells::s1_4,   {},           {}},
            {"1-5",   LevelCells::b1_5,   LevelCells::s1_5,   {},           {}},

            {"2-1",   LevelCells::b2_1,   LevelCells::s2_1,   {},           {}},
            {"2-2",   LevelCells::b2_2,   LevelCells::s2_2,   {},           {}},
            {"2-3",   LevelCells::b2_3,   LevelCells::s2_3,   {},           {}},
            {"2-4",   LevelCells::b2_4,   LevelCells::s2_4,   {},           {}},
            {"2-5",   LevelCells::b2_5,   LevelCells::s2_5,   {},           {}},

            {"3-1",   LevelCells::b3_1,   LevelCells::s3_1,   {},           {}},
            {"3-2",   LevelCells::b3_2,   LevelCells::s3_2,   {},           {}},
            {"3-3",   LevelCells::b3_3,   LevelCells::s3_3,   {},           {}},
            {"3-4",   LevelCells::b3_4,   LevelCells::s3_4,   {},           {}},
            {"3-5",   LevelCells::b3_5,   LevelCells::s3_5,   {},           {}},

            {"4-1",   LevelCells::b4_1,   LevelCells::s4_1,   {},           {}},
            {"4-2",   LevelCells::b4_2,   LevelCells::s4_2,   {true, 4, 4}, {true, 2, 2}},
            {"4-3",   LevelCells::b4_3,   LevelCells::s4_3,   {},           {}},
            {"4-4",   LevelCells::b4_4,   LevelCells::s4_4,   {true, 2, 2}, {true, 2, 2}},
            {"4-5",   LevelCells::b4_5,   LevelCells::s4_5,   {},           {}},

            {"5-1",   LevelCells::b5_1,   LevelCells::s5_1,   {true, 0, 2}, {true, 3, 3}},
            {"5-2",   LevelCells::b5_2,   LevelCells::s5_2,   {},           {}},
            {"5-3",   LevelCells::b5_3,   LevelCells::s5_3,   {},           {}},
            {"5-4",   LevelCells::b5_4,   LevelCells::s5_4,   {},           {}},
            {"5-5",   LevelCells::b5_5,   LevelCells::s5_5,   {},           {}},

            {"6-1",   LevelCells::b6_1,   LevelCells::s6_1,   {true, 4, 4}, {true, 2, 2}},
            {"6-2",   LevelCells::b6_2,   LevelCells::s6_2,   {true, 1, 3}, {true, 2, 2}},
            {"6-3",   LevelCells::b6_3,   LevelCells::s6_3,   {true, 0, 1}, {true, 0, 0}},
            {"6-4",   LevelCells::b6_4,   LevelCells::s6_4,   {true, 3, 3}, {true, 2, 2}},
            {"6-5",   LevelCells::b6_5,   LevelCells::s6_5,   {true, 3, 0}, {true, 2, 2}},

            {"7-1",   LevelCells::b7_1,   LevelCells::s7_1,   {},           {}},
            {"7-2",   LevelCells::b7_2,   LevelCells::s7_2,   {},           {}},
            {"7-3",   LevelCells::b7_3,   LevelCells::s7_3,   {},           {}},
            {"7-4",   LevelCells::b7_4,   LevelCells::s7_4,   {},           {}},
            {"7-5",   LevelCells::b7_5,   LevelCells::s7_5,   {},           {}},

            {"8-1",   LevelCells::b8_1,   LevelCells::s8_1,   {},           {}},
            {"8-2",   LevelCells::b8_2,   LevelCells::s8_2,   {true, 4, 4}, {true, 2, 2}},
            {"8-3",   LevelCells::b8_3,   LevelCells::s8_3,   {},           {}},
            {"8-4",   LevelCells::b8_4,   LevelCells::s8_4,   {true, 3, 4}, {true, 2, 2}},
            {"8-5",   LevelCells::b8_5,   LevelCells::s8_5,   {},           {}},

            {"9-1",   LevelCells::b9_1,   LevelCells::s9_1,   {true, 1, 3}, {true, 2, 2}},
            {"9-2",   LevelCells::b9_2,   LevelCells::s9_2,   {},           {}},
            {"9-3",   LevelCells::b9_3,   LevelCells::s9_3,   {},           {}},
            {"9-4",   LevelCells::b9_4,   LevelCells::s9_4,   {},           {}},
            {"9-5",   LevelCells::b9_5,   LevelCells::s9_5,   {},           {}},

            {"10-1",  LevelCells::b10_1,  LevelCells::s10_1,  {},           {}},
            {"10-2",  LevelCells::b10_2,  LevelCells::s10_2,  {},           {}},
            {"10-3",  LevelCells::b10_3,  LevelCells::s10_3,  {},           {}},
            {"10-4",  LevelCells::b10_4,  LevelCells::s10_4,  {},           {}},
            {"10-5",  LevelCells::b10_5,  LevelCells::s10_5,  {},           {}},

            {"11-1",  LevelCells::b11_1,  LevelCells::s11_1,  {},           {}},
            {"11-2",  LevelCells::b11_2,  LevelCells::s11_2,  {},           {}},
            {"11-3",  LevelCells::b11_3,  LevelCells::s11_3,  {},           {}},
            {"11-4",  LevelCells::b11_4,  LevelCells::s11_4,  {},           {}},
            {"11-5",  LevelCells::b11_5,  LevelCells::s11_5,  {},           {}},

            {"12-1",  LevelCells::b12_1,  LevelCells::s12_1,  {},           {}},
            {"12-2",  LevelCells::b12_2,  LevelCells::s12_2,  {true, 1, 3}, {true, 2, 2}},
            {"12-3",  LevelCells::b12_3,  LevelCells::s12_3,  {},           {}},
            {"12-4",  LevelCells::b12_4,  LevelCells::s12_4,  {},           {}},
            {"12-5",  LevelCells::b12_5,  LevelCells::s12_5,  {},           {}},

            {"13-1",  LevelCells::b13_1,  LevelCells::s13_1,  {},           {}},
            {"13-2",  LevelCells::b13_2,  LevelCells::s13_2,  {},           {}},
            {"13-3",  LevelCells::b13_3,  LevelCells::s13_3,  {},           {}},
            {"13-4",  LevelCells::b13_4,  LevelCells::s13_4,  {true, 1, 4}, {true, 2, 2}},
            {"13-5",  LevelCells::b13_5,  LevelCells::s13_5,  {true, 2, 4}, {true, 2, 2}},

            {"14-1",  LevelCells::b14_1,  LevelCells::s14_1,  {},           {}},
            {"14-2",  LevelCells::b14_2,  LevelCells::s14_2,  {},           {}},
            {"14-3",  LevelCells::b14_3,  LevelCells::s14_3,  {},           {}},
            {"14-4",  LevelCells::b14_4,  LevelCells::s14_4,  {},           {}},
            {"14-5",  LevelCells::b14_5,  LevelCells::s14_5,  {},           {}},

            {"15-1",  LevelCells::b15_1,  LevelCells::s15_1,  {},           {}},
            {"15-2",  LevelCells::b15_2,  LevelCells::s15_2,  {true, 3, 4}, {true, 2, 2}},
            {"15-3",  LevelCells::b15_3,  LevelCells::s15_3,  {true, 1, 0}, {true, 2, 2}},
            {"15-4",  LevelCells::b15_4,  LevelCells::s15_4,  {true, 1, 4}, {true, 2, 4}},
            {"15-5",  LevelCells::b15_5,  LevelCells::s15_5,  {},           {}},

            {"16-1",  LevelCells::b16_1,  LevelCells::s16_1,  {true, 3, 4}, {true, 2, 2}},
            {"16-2",  LevelCells::b16_2,  LevelCells::s16_2,  {},           {}},
            {"16-3",  LevelCells::b16_3,  LevelCells::s16_3,  {},           {}},
            {"16-4",  LevelCells::b16_4,  LevelCells::s16_4,  {},           {}},
            {"16-5",  LevelCells::b16_5,  LevelCells::s16_5,  {true, 2, 0}, {true, 2, 2}},

            {"17-1",  LevelCells::b17_1,  LevelCells::s17_1,  {},           {}},
            {"17-2",  LevelCells::b17_2,  LevelCells::s17_2,  {true, 4, 4}, {true, 1, 3}},
            {"17-3",  LevelCells::b17_3,  LevelCells::s17_3,  {},           {}},
            {"17-4",  LevelCells::b17_4,  LevelCells::s17_4,  {true, 2, 1}, {true, 2, 2}},
            {"17-5",  LevelCells::b17_5,  LevelCells::s17_5,  {},           {}},

            {"18-1",  LevelCells::b18_1,  LevelCells::s18_1,  {true, 1, 4}, {true, 2, 2}},
            {"18-2",  LevelCells::b18_2,  LevelCells::s18_2,  {true, 0, 0}, {true, 2, 2}},
            {"18-3",  LevelCells::b18_3,  LevelCells::s18_3,  {},           {}},
            {"18-4",  LevelCells::b18_4,  LevelCells::s18_4,  {true, 1, 2}, {true, 4, 0}},
            {"18-5",  LevelCells::b18_5,  LevelCells::s18_5,  {true, 4, 3}, {true, 2, 2}},

            {"19-1",  LevelCells::b19_1,  LevelCells::s19_1,  {},           {}},
            {"19-2",  LevelCells::b19_2,  LevelCells::s19_2,  {true, 0, 3}, {true, 2, 1}},
            {"19-3",  LevelCells::b19_3,  LevelCells::s19_3,  {},           {}},
            {"19-4",  LevelCells::b19_4,  LevelCells::s19_4,  {true, 1, 4}, {true, 2, 2}},
            {"19-5",  LevelCells::b19_5,  LevelCells::s19_5,  {},           {}},

            {"20-1",  LevelCells::b20_1,  LevelCells::s20_1,  {},           {}},
            {"20-2",  LevelCells::b20_2,  LevelCells::s20_2,  {},           {}},
            {"20-3",  LevelCells::b20_3,  LevelCells::s20_3,  {true, 2, 2}, {true, 2, 2}},
            {"20-4",  LevelCells::b20_4,  LevelCells::s20_4,  {},           {}},
            {"20-5",  LevelCells::b20_5,  LevelCells::s20_5,  {},           {}},
    }};

    inline static const std::vector<BoardPair> BOARD_PAIRS = [] {
        std::vector<BoardPair> out;
        out.reserve(LEVEL_DEFS.size());

        for (const auto& def : LEVEL_DEFS) {
            out.emplace_back(def.name, def.startCells, def.endCells, def.startFat, def.endFat);
        }

        return out;
    }();

    inline static const std::unordered_map<std::string, const BoardPair*> BOARD_PAIR_DICT = [] {
        std::unordered_map<std::string, const BoardPair*> out;
        out.reserve(BOARD_PAIRS.size());

        for (const auto& pair : BOARD_PAIRS) {
            out.emplace(pair.getName(), &pair);
        }

        return out;
    }();

public:
    MUND static const BoardPair* getBoardPair(const std::string& name) {
        const auto it = BOARD_PAIR_DICT.find(name);
        if (it == BOARD_PAIR_DICT.end()) {
            return nullptr;
        }
        return it->second;
    }

    MUND static const std::vector<BoardPair>& getAllBoardPairs() {
        return BOARD_PAIRS;
    }

    MUND static const std::array<LevelDef, 100>& getAllLevelDefs() {
        return LEVEL_DEFS;
    }
};