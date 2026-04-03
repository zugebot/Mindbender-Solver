// programs/testRotations.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "include/doctest.h"

#include "code/rotations.hpp"

#include <array>
#include <string>

namespace {

    using Cells = std::array<u8, 36>;

    // Per your documented packed layout.
    static constexpr u64 B1_UNUSED_MASK = 0x3ULL << 54;
    static constexpr u64 B2_UNUSED_MASK = 0x7FULL << 54;

    static Cells makeBaseCells() {
        return {{
                0, 1, 2, 3, 4, 5,
                6, 7, 0, 1, 2, 3,
                4, 5, 6, 7, 0, 1,
                2, 3, 4, 5, 6, 7,
                1, 0, 7, 6, 5, 4,
                3, 2, 1, 0, 7, 6
        }};
    }

    static Board makeNormalBoard() {
        Cells cells = makeBaseCells();
        return Board(cells.data());
    }

    static Board makeFatRowBoard(const u8 fatY, const u8 fatX = 2) {
        Cells cells = makeBaseCells();
        return Board(cells.data(), fatX, fatY);
    }

    static Board makeFatColBoard(const u8 fatX, const u8 fatY = 2) {
        Cells cells = makeBaseCells();
        return Board(cells.data(), fatX, fatY);
    }

    static std::array<int, 8> colorHistogram(const B1B2& board) {
        std::array<int, 8> hist{};
        for (u8 y = 0; y < 6; ++y) {
            for (u8 x = 0; x < 6; ++x) {
                ++hist[board.getColor(x, y)];
            }
        }
        return hist;
    }

    static std::string boardDump(const B1B2& board) {
        Board tmp;
        tmp.b1 = board.b1;
        tmp.b2 = board.b2;
        return tmp.toStringSingle(Board::PrintSettings(false, Board::ColorsDefault));
    }

    static std::string boardPairDump(const B1B2& lhs, const B1B2& rhs) {
        Board a;
        a.b1 = lhs.b1;
        a.b2 = lhs.b2;

        Board b;
        b.b1 = rhs.b1;
        b.b2 = rhs.b2;

        return a.toString(b, Board::PrintSettings(false, Board::ColorsDefault));
    }

    static Cells rotateRowRight(Cells cells, const int row, const int amount) {
        Cells out = cells;
        for (int x = 0; x < 6; ++x) {
            out[row * 6 + x] = cells[row * 6 + ((x - amount + 6) % 6)];
        }
        return out;
    }

    static Cells rotateColDown(Cells cells, const int col, const int amount) {
        Cells out = cells;
        for (int y = 0; y < 6; ++y) {
            out[y * 6 + col] = cells[((y - amount + 6) % 6) * 6 + col];
        }
        return out;
    }

    static Cells rotateTwoRowsRight(Cells cells, const int rowA, const int rowB, const int amount) {
        cells = rotateRowRight(cells, rowA, amount);
        cells = rotateRowRight(cells, rowB, amount);
        return cells;
    }

    static Cells rotateTwoColsDown(Cells cells, const int colA, const int colB, const int amount) {
        cells = rotateColDown(cells, colA, amount);
        cells = rotateColDown(cells, colB, amount);
        return cells;
    }

    static void requireBoardsEqual(
            const B1B2& actual,
            const B1B2& expected,
            const char* moveName,
            const char* context) {
        CHECK_MESSAGE(
                actual == expected,
                "\nMove: ", moveName,
                "\nContext: ", context,
                "\nActual vs Expected:\n", boardPairDump(actual, expected),
                "\nActual only:\n", boardDump(actual),
                "\nExpected only:\n", boardDump(expected)
        );
    }

    static Board makeExpectedBoardFromCellsAndMetadata(
            const Cells& expectedCells,
            const B1B2& metadataSource) {
        Board expected(expectedCells.data());

        if (metadataSource.getFatBool()) {
            expected.setFatXY(metadataSource.getFatX(), metadataSource.getFatY());
        } else {
            expected.setFatBool(false);
        }

        expected.setColorCount(metadataSource.getColorCount());

        expected.b1 = (expected.b1 & ~B1_UNUSED_MASK) | (metadataSource.b1 & B1_UNUSED_MASK);
        expected.b2 = (expected.b2 & ~B2_UNUSED_MASK) | (metadataSource.b2 & B2_UNUSED_MASK);

        return expected;
    }

    static void requireCellsAndMetadataEqual(
            const B1B2& actual,
            const Cells& expectedCells,
            const B1B2& expectedMetadata,
            const char* moveName,
            const char* context) {
        const Board expected = makeExpectedBoardFromCellsAndMetadata(expectedCells, expectedMetadata);
        requireBoardsEqual(actual, expected, moveName, context);
    }

    static void requireHistogramPreserved(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        const auto h1 = colorHistogram(before);
        const auto h2 = colorHistogram(after);

        CHECK_MESSAGE(
                h1 == h2,
                "\nMove: ", moveName,
                "\nColor histogram changed unexpectedly.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static void requireAlwaysPreservedMetadata(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        CHECK_MESSAGE(
                before.getFatBool() == after.getFatBool(),
                "\nMove: ", moveName,
                "\nFat bool changed unexpectedly.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );

        CHECK_MESSAGE(
                before.getColorCount() == after.getColorCount(),
                "\nMove: ", moveName,
                "\nColor count changed unexpectedly.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static void requireNormalMoveMetadataPreserved(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        requireAlwaysPreservedMetadata(before, after, moveName);

        CHECK_MESSAGE(
                before.getFatX() == after.getFatX(),
                "\nMove: ", moveName,
                "\nFat X changed unexpectedly for a normal move.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );

        CHECK_MESSAGE(
                before.getFatY() == after.getFatY(),
                "\nMove: ", moveName,
                "\nFat Y changed unexpectedly for a normal move.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static void requireFatRowMetadataPreserved(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        requireAlwaysPreservedMetadata(before, after, moveName);

        CHECK_MESSAGE(
                before.getFatY() == after.getFatY(),
                "\nMove: ", moveName,
                "\nFat Y changed unexpectedly for a fat row move.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static void requireFatColMetadataPreserved(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        requireAlwaysPreservedMetadata(before, after, moveName);

        CHECK_MESSAGE(
                before.getFatX() == after.getFatX(),
                "\nMove: ", moveName,
                "\nFat X changed unexpectedly for a fat column move.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static void requireUnusedBitsPreserved(
            const B1B2& before,
            const B1B2& after,
            const char* moveName) {
        CHECK_MESSAGE(
                (before.b1 & B1_UNUSED_MASK) == (after.b1 & B1_UNUSED_MASK),
                "\nMove: ", moveName,
                "\nUnused b1 bits changed unexpectedly.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );

        CHECK_MESSAGE(
                (before.b2 & B2_UNUSED_MASK) == (after.b2 & B2_UNUSED_MASK),
                "\nMove: ", moveName,
                "\nUnused b2 bits changed unexpectedly.",
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );
    }

    static bool isSameBoard(const B1B2& a, const B1B2& b) {
        return a == b;
    }

    static void applyAction(B1B2& board, Action action) {
        action(board);
    }

    static std::array<std::array<Action, 5>, 6> makeNormalRowGroups() {
        return {{
                {{R01, R02, R03, R04, R05}},
                {{R11, R12, R13, R14, R15}},
                {{R21, R22, R23, R24, R25}},
                {{R31, R32, R33, R34, R35}},
                {{R41, R42, R43, R44, R45}},
                {{R51, R52, R53, R54, R55}}
        }};
    }

    static std::array<std::array<Action, 5>, 6> makeNormalColGroups() {
        return {{
                {{C01, C02, C03, C04, C05}},
                {{C11, C12, C13, C14, C15}},
                {{C21, C22, C23, C24, C25}},
                {{C31, C32, C33, C34, C35}},
                {{C41, C42, C43, C44, C45}},
                {{C51, C52, C53, C54, C55}}
        }};
    }

    static std::array<std::array<Action, 5>, 5> makeFatRowGroups() {
        return {{
                {{R011, R012, R013, R014, R015}},
                {{R121, R122, R123, R124, R125}},
                {{R231, R232, R233, R234, R235}},
                {{R341, R342, R343, R344, R345}},
                {{R451, R452, R453, R454, R455}}
        }};
    }

    static std::array<std::array<Action, 5>, 5> makeFatColGroups() {
        return {{
                {{C011, C012, C013, C014, C015}},
                {{C121, C122, C123, C124, C125}},
                {{C231, C232, C233, C234, C235}},
                {{C341, C342, C343, C344, C345}},
                {{C451, C452, C453, C454, C455}}
        }};
    }

    static const auto kNormalRowGroups = makeNormalRowGroups();
    static const auto kNormalColGroups = makeNormalColGroups();
    static const auto kFatRowGroups = makeFatRowGroups();
    static const auto kFatColGroups = makeFatColGroups();

    template <size_t N>
    static int findInverseIndex(
            const B1B2& start,
            Action action,
            const std::array<Action, N>& candidates) {
        int found = -1;

        B1B2 after = start;
        applyAction(after, action);

        for (size_t i = 0; i < N; ++i) {
            B1B2 roundTrip = after;
            applyAction(roundTrip, candidates[i]);
            if (isSameBoard(roundTrip, start)) {
                if (found != -1) {
                    return -2;
                }
                found = static_cast<int>(i);
            }
        }

        return found;
    }

    static void testNormalRowMove(
            Action action,
            const char* moveName,
            const int row,
            const int amount) {
        Board before = makeNormalBoard();
        B1B2 after = before;
        applyAction(after, action);

        const Cells expectedCells = rotateRowRight(makeBaseCells(), row, amount);
        requireCellsAndMetadataEqual(after, expectedCells, before, moveName, "normal row exact rotation");

        requireHistogramPreserved(before, after, moveName);
        requireNormalMoveMetadataPreserved(before, after, moveName);
        requireUnusedBitsPreserved(before, after, moveName);

        const int inverseAmount = 6 - amount;
        B1B2 roundTrip = after;
        applyAction(roundTrip, kNormalRowGroups[row][inverseAmount - 1]);
        requireBoardsEqual(roundTrip, before, moveName, "normal row inverse round-trip");
    }

    static void testNormalColMove(
            Action action,
            const char* moveName,
            const int col,
            const int amount) {
        Board before = makeNormalBoard();
        B1B2 after = before;
        applyAction(after, action);

        const Cells expectedCells = rotateColDown(makeBaseCells(), col, amount);
        requireCellsAndMetadataEqual(after, expectedCells, before, moveName, "normal column exact rotation");

        requireHistogramPreserved(before, after, moveName);
        requireNormalMoveMetadataPreserved(before, after, moveName);
        requireUnusedBitsPreserved(before, after, moveName);

        const int inverseAmount = 6 - amount;
        B1B2 roundTrip = after;
        applyAction(roundTrip, kNormalColGroups[col][inverseAmount - 1]);
        requireBoardsEqual(roundTrip, before, moveName, "normal column inverse round-trip");
    }

    static void testFatRowMove(
            Action action,
            const char* moveName,
            const int fatYGroup,
            const int amount) {
        Board before = makeFatRowBoard(static_cast<u8>(fatYGroup));
        B1B2 after = before;
        applyAction(after, action);

        const Cells expectedCells = rotateTwoRowsRight(makeBaseCells(), fatYGroup, fatYGroup + 1, amount);
        requireCellsAndMetadataEqual(after, expectedCells, after, moveName, "fat row exact paired-row rotation");

        requireHistogramPreserved(before, after, moveName);
        requireFatRowMetadataPreserved(before, after, moveName);
        requireUnusedBitsPreserved(before, after, moveName);

        const int inverseIdx = findInverseIndex(before, action, kFatRowGroups[fatYGroup]);
        CHECK_MESSAGE(
                inverseIdx >= 0,
                "\nMove: ", moveName,
                "\nCould not find a unique inverse inside fat row group ",
                fatYGroup,
                ". Result code: ", inverseIdx,
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );

        if (inverseIdx >= 0) {
            B1B2 roundTrip = after;
            applyAction(roundTrip, kFatRowGroups[fatYGroup][inverseIdx]);
            requireBoardsEqual(roundTrip, before, moveName, "fat row inverse round-trip");
        }
    }

    static void testFatColMove(
            Action action,
            const char* moveName,
            const int fatXGroup,
            const int amount) {
        Board before = makeFatColBoard(static_cast<u8>(fatXGroup));
        B1B2 after = before;
        applyAction(after, action);

        const Cells expectedCells = rotateTwoColsDown(makeBaseCells(), fatXGroup, fatXGroup + 1, amount);
        requireCellsAndMetadataEqual(after, expectedCells, after, moveName, "fat column exact paired-column rotation");

        requireHistogramPreserved(before, after, moveName);
        requireFatColMetadataPreserved(before, after, moveName);
        requireUnusedBitsPreserved(before, after, moveName);

        const int inverseIdx = findInverseIndex(before, action, kFatColGroups[fatXGroup]);
        CHECK_MESSAGE(
                inverseIdx >= 0,
                "\nMove: ", moveName,
                "\nCould not find a unique inverse inside fat column group ",
                fatXGroup,
                ". Result code: ", inverseIdx,
                "\nBefore:\n", boardDump(before),
                "\nAfter:\n", boardDump(after)
        );

        if (inverseIdx >= 0) {
            B1B2 roundTrip = after;
            applyAction(roundTrip, kFatColGroups[fatXGroup][inverseIdx]);
            requireBoardsEqual(roundTrip, before, moveName, "fat column inverse round-trip");
        }
    }

} // namespace

TEST_CASE("rotation metadata table has expected size and key constants") {
    CHECK(TOTAL_ACT_STRUCT_COUNT == 114);
    CHECK(NORMAL_ROW_MOVE_COUNT == 30);
    CHECK(NORMAL_COL_MOVE_COUNT == 30);
    CHECK(FAT_ROW_MOVE_COUNT == 25);
    CHECK(FAT_COL_MOVE_COUNT == 25);
    CHECK(NORMAL_MOVE_GAP_BEGIN == 30);
    CHECK(NORMAL_MOVE_GAP_COUNT == 2);
    CHECK(FAT_MOVE_GAP_BEGIN == 62);
    CHECK(FAT_MOVE_GAP_COUNT == 2);
}

TEST_CASE("allActStructList key entries line up with declarations") {
    CHECK(std::string(allActStructList[0].name.data(), 3) == "R01");
    CHECK(allActStructList[0].action == R01);
    CHECK(allActStructList[0].index == 0);

    CHECK(std::string(allActStructList[29].name.data(), 3) == "R55");
    CHECK(allActStructList[29].action == R55);
    CHECK(allActStructList[29].index == 29);

    CHECK(std::string(allActStructList[32].name.data(), 3) == "C01");
    CHECK(allActStructList[32].action == C01);
    CHECK(allActStructList[32].index == 32);

    CHECK(std::string(allActStructList[61].name.data(), 3) == "C55");
    CHECK(allActStructList[61].action == C55);
    CHECK(allActStructList[61].index == 61);

    CHECK(std::string(allActStructList[64].name.data(), 4) == "R011");
    CHECK(allActStructList[64].action == R011);
    CHECK(allActStructList[64].index == 64);

    CHECK(std::string(allActStructList[88].name.data(), 4) == "R455");
    CHECK(allActStructList[88].action == R455);
    CHECK(allActStructList[88].index == 88);

    CHECK(std::string(allActStructList[89].name.data(), 4) == "C011");
    CHECK(allActStructList[89].action == C011);
    CHECK(allActStructList[89].index == 89);

    CHECK(std::string(allActStructList[113].name.data(), 4) == "C455");
    CHECK(allActStructList[113].action == C455);
    CHECK(allActStructList[113].index == 113);
}

#define DEFINE_NORMAL_ROW_TEST(name, index, isColNotFat, tillNext, tillLast)       \
TEST_CASE("rotation " #name " normal row") {                                        \
    constexpr int kRow = (index) / 5;                                               \
    constexpr int kAmount = (tillLast) + 1;                                         \
    testNormalRowMove(name, #name, kRow, kAmount);                                  \
}
FOR_EACH_ROW_MOVE_INFO(DEFINE_NORMAL_ROW_TEST)
#undef DEFINE_NORMAL_ROW_TEST

#define DEFINE_NORMAL_COL_TEST(name, index, isColNotFat, tillNext, tillLast)       \
TEST_CASE("rotation " #name " normal column") {                                     \
    constexpr int kCol = ((index) - 32) / 5;                                        \
    constexpr int kAmount = (tillLast) + 1;                                         \
    testNormalColMove(name, #name, kCol, kAmount);                                  \
}
FOR_EACH_COL_MOVE_INFO(DEFINE_NORMAL_COL_TEST)
#undef DEFINE_NORMAL_COL_TEST

#define DEFINE_FAT_ROW_TEST(name, index, isColNotFat, tillNext, tillLast)          \
TEST_CASE("rotation " #name " fat row") {                                           \
    constexpr int kFatYGroup = ((index) - 64) / 5;                                  \
    constexpr int kAmount = (tillLast) + 1;                                         \
    testFatRowMove(name, #name, kFatYGroup, kAmount);                               \
}
FOR_EACH_FAT_ROW_MOVE_INFO(DEFINE_FAT_ROW_TEST)
#undef DEFINE_FAT_ROW_TEST

#define DEFINE_FAT_COL_TEST(name, index, isColNotFat, tillNext, tillLast)          \
TEST_CASE("rotation " #name " fat column") {                                        \
    constexpr int kFatXGroup = ((index) - 89) / 5;                                  \
    constexpr int kAmount = (tillLast) + 1;                                         \
    testFatColMove(name, #name, kFatXGroup, kAmount);                               \
}
FOR_EACH_FAT_COL_MOVE_INFO(DEFINE_FAT_COL_TEST)
#undef DEFINE_FAT_COL_TEST