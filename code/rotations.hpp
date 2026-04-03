#pragma once
// code/rotations.hpp

#include "board.hpp"
#include <cstring>


#define DECLARE_PERM(name) __host__ __device__ void name(B1B2& board);

// normal row moves
#define FOR_EACH_ROW_MOVE(X) \
    X(R01) X(R02) X(R03) X(R04) X(R05) \
    X(R11) X(R12) X(R13) X(R14) X(R15) \
    X(R21) X(R22) X(R23) X(R24) X(R25) \
    X(R31) X(R32) X(R33) X(R34) X(R35) \
    X(R41) X(R42) X(R43) X(R44) X(R45) \
    X(R51) X(R52) X(R53) X(R54) X(R55)

// normal column moves
#define FOR_EACH_COL_MOVE(X) \
    X(C01) X(C02) X(C03) X(C04) X(C05) \
    X(C11) X(C12) X(C13) X(C14) X(C15) \
    X(C21) X(C22) X(C23) X(C24) X(C25) \
    X(C31) X(C32) X(C33) X(C34) X(C35) \
    X(C41) X(C42) X(C43) X(C44) X(C45) \
    X(C51) X(C52) X(C53) X(C54) X(C55)

// fat row moves
#define FOR_EACH_FAT_ROW_MOVE(X) \
    X(R011) X(R012) X(R013) X(R014) X(R015) \
    X(R121) X(R122) X(R123) X(R124) X(R125) \
    X(R231) X(R232) X(R233) X(R234) X(R235) \
    X(R341) X(R342) X(R343) X(R344) X(R345) \
    X(R451) X(R452) X(R453) X(R454) X(R455)

// fat column moves
#define FOR_EACH_FAT_COL_MOVE(X) \
    X(C011) X(C012) X(C013) X(C014) X(C015) \
    X(C121) X(C122) X(C123) X(C124) X(C125) \
    X(C231) X(C232) X(C233) X(C234) X(C235) \
    X(C341) X(C342) X(C343) X(C344) X(C345) \
    X(C451) X(C452) X(C453) X(C454) X(C455)

FOR_EACH_ROW_MOVE(DECLARE_PERM)
FOR_EACH_COL_MOVE(DECLARE_PERM)
FOR_EACH_FAT_ROW_MOVE(DECLARE_PERM)
FOR_EACH_FAT_COL_MOVE(DECLARE_PERM)

#define FOR_EACH_ROW_MOVE_INFO(X) \
    X(R01,   0,  2, 5, 0) X(R02,   1,  2, 4, 1) X(R03,   2,  2, 3, 2) X(R04,   3,  2, 2, 3) X(R05,   4,  2, 1, 4) \
    X(R11,   5,  2, 5, 0) X(R12,   6,  2, 4, 1) X(R13,   7,  2, 3, 2) X(R14,   8,  2, 2, 3) X(R15,   9,  2, 1, 4) \
    X(R21,  10,  2, 5, 0) X(R22,  11,  2, 4, 1) X(R23,  12,  2, 3, 2) X(R24,  13,  2, 2, 3) X(R25,  14,  2, 1, 4) \
    X(R31,  15,  2, 5, 0) X(R32,  16,  2, 4, 1) X(R33,  17,  2, 3, 2) X(R34,  18,  2, 2, 3) X(R35,  19,  2, 1, 4) \
    X(R41,  20,  2, 5, 0) X(R42,  21,  2, 4, 1) X(R43,  22,  2, 3, 2) X(R44,  23,  2, 2, 3) X(R45,  24,  2, 1, 4) \
    X(R51,  25,  2, 5, 0) X(R52,  26,  2, 4, 1) X(R53,  27,  2, 3, 2) X(R54,  28,  2, 2, 3) X(R55,  29,  2, 1, 4)

#define FOR_EACH_COL_MOVE_INFO(X) \
    X(C01,  32,  1, 5, 0) X(C02,  33,  1, 4, 1) X(C03,  34,  1, 3, 2) X(C04,  35,  1, 2, 3) X(C05,  36,  1, 1, 4) \
    X(C11,  37,  1, 5, 0) X(C12,  38,  1, 4, 1) X(C13,  39,  1, 3, 2) X(C14,  40,  1, 2, 3) X(C15,  41,  1, 1, 4) \
    X(C21,  42,  1, 5, 0) X(C22,  43,  1, 4, 1) X(C23,  44,  1, 3, 2) X(C24,  45,  1, 2, 3) X(C25,  46,  1, 1, 4) \
    X(C31,  47,  1, 5, 0) X(C32,  48,  1, 4, 1) X(C33,  49,  1, 3, 2) X(C34,  50,  1, 2, 3) X(C35,  51,  1, 1, 4) \
    X(C41,  52,  1, 5, 0) X(C42,  53,  1, 4, 1) X(C43,  54,  1, 3, 2) X(C44,  55,  1, 2, 3) X(C45,  56,  1, 1, 4) \
    X(C51,  57,  1, 5, 0) X(C52,  58,  1, 4, 1) X(C53,  59,  1, 3, 2) X(C54,  60,  1, 2, 3) X(C55,  61,  1, 1, 4)

#define FOR_EACH_FAT_ROW_MOVE_INFO(X) \
    X(R011, 64, 0, 4, 0) X(R012, 65, 0, 3, 1) X(R013, 66, 0, 2, 2) X(R014, 67, 0, 1, 3) X(R015, 68, 0, 0, 4) \
    X(R121, 69, 0, 4, 0) X(R122, 70, 0, 3, 1) X(R123, 71, 0, 2, 2) X(R124, 72, 0, 1, 3) X(R125, 73, 0, 0, 4) \
    X(R231, 74, 0, 4, 0) X(R232, 75, 0, 3, 1) X(R233, 76, 0, 2, 2) X(R234, 77, 0, 1, 3) X(R235, 78, 0, 0, 4) \
    X(R341, 79, 0, 4, 0) X(R342, 80, 0, 3, 1) X(R343, 81, 0, 2, 2) X(R344, 82, 0, 1, 3) X(R345, 83, 0, 0, 4) \
    X(R451, 84, 0, 4, 0) X(R452, 85, 0, 3, 1) X(R453, 86, 0, 2, 2) X(R454, 87, 0, 1, 3) X(R455, 88, 0, 0, 4)

#define FOR_EACH_FAT_COL_MOVE_INFO(X) \
    X(C011,  89, 0, 4, 0) X(C012,  90, 0, 3, 1) X(C013,  91, 0, 2, 2) X(C014,  92, 0, 1, 3) X(C015,  93, 0, 0, 4) \
    X(C121,  94, 0, 4, 0) X(C122,  95, 0, 3, 1) X(C123,  96, 0, 2, 2) X(C124,  97, 0, 1, 3) X(C125,  98, 0, 0, 4) \
    X(C231,  99, 0, 4, 0) X(C232, 100, 0, 3, 1) X(C233, 101, 0, 2, 2) X(C234, 102, 0, 1, 3) X(C235, 103, 0, 0, 4) \
    X(C341, 104, 0, 4, 0) X(C342, 105, 0, 3, 1) X(C343, 106, 0, 2, 2) X(C344, 107, 0, 1, 3) X(C345, 108, 0, 0, 4) \
    X(C451, 109, 0, 4, 0) X(C452, 110, 0, 3, 1) X(C453, 111, 0, 2, 2) X(C454, 112, 0, 1, 3) X(C455, 113, 0, 0, 4)


#undef DECLARE_PERM


struct ActStruct {
    Action action{};
    std::array<char, 4> name{};
    u8 index{};
    u8 isColNotFat{};
    u8 tillNext{};
    u8 tillLast{};

    MU ActStruct() = default;

    MU ActStruct(C Action theAction, C u8 theIndex, C u8 theIsColNotFat,
                 C u8 theTillNext, C u8 theTillLast, C char* theName)
        : action(theAction),
          index(theIndex),
          isColNotFat(theIsColNotFat),
          tillNext(theTillNext),
          tillLast(theTillLast) {
        memcpy(name.data(), theName, 4);
    }
};


inline constexpr u32 NORMAL_ROW_MOVE_COUNT = 30;
inline constexpr u32 NORMAL_COL_MOVE_COUNT = 30;
inline constexpr u32 FAT_ROW_MOVE_COUNT = 25;
inline constexpr u32 FAT_COL_MOVE_COUNT = 25;

inline constexpr u32 NORMAL_MOVE_GAP_BEGIN = 30;
inline constexpr u32 NORMAL_MOVE_GAP_COUNT = 2;
inline constexpr u32 FAT_MOVE_GAP_BEGIN = 62;
inline constexpr u32 FAT_MOVE_GAP_COUNT = 2;

inline constexpr u32 TOTAL_ACT_STRUCT_COUNT = 114;
inline constexpr u32 FAT_ACTION_INDEX_ROWS = 25;
inline constexpr u32 FAT_ACTION_INDEX_COLS = 48;


MU extern ActStruct allActStructList[TOTAL_ACT_STRUCT_COUNT];
MU extern u8 fatActionsIndexes[FAT_ACTION_INDEX_ROWS][FAT_ACTION_INDEX_COLS];

extern void applyMoves(Board& board, C Memory& memory);
extern void applyFatMoves(Board& board, C Memory& memory);

extern Board makeBoardWithMoves(C Board& board, C Memory& memory);
extern Board makeBoardWithFatMoves(C Board& board, C Memory& memory);




