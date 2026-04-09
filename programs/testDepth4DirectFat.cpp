#include "code/include.hpp"
#include "utils/timer.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

struct ModeResult {
    std::string label;
    JVec<Board> boards;
    JVec<u64> hashes;
    double seconds = 0.0;

    std::vector<std::pair<u64, u64>> uniqueStateKeys;
};

static std::vector<std::pair<u64, u64>> buildUniqueStateKeys(const JVec<Board>& boards) {
    std::vector<std::pair<u64, u64>> keys;
    keys.reserve(boards.size());

    for (const Board& board : boards) {
        keys.emplace_back(board.b1, board.b2);
    }

    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    return keys;
}

static void printPreviewBoards(const ModeResult& result,
                               const Board& start,
                               const u32 previewCount) {
    const u8 fatPos = start.getFatXY();
    const u32 shown = static_cast<u32>(std::min<std::size_t>(result.boards.size(), previewCount));

    for (u32 i = 0; i < shown; ++i) {
        tcout << "  [" << i << "] "
              << result.boards[i].getMemory().asmFatStringForwards(fatPos) << '\n';
        tcout << result.boards[i].toString(start) << '\n';
    }

    if (result.boards.size() > shown) {
        tcout << "  ...\n\n";
    }
}

template<eSequenceDir DIR>
static ModeResult runMode(const char* label,
                          const Board& start,
                          const u32 depth,
                          const u32 previewCount = 6) {
    ModeResult result;
    result.label = label;

    constexpr bool shouldAlloc = true;

    Timer timer;
    Perms<Board>::getDepthFunc<DIR>(
            start,
            result.boards,
            result.hashes,
            depth,
            shouldAlloc
    );
    result.seconds = timer.getSeconds();

    result.uniqueStateKeys = buildUniqueStateKeys(result.boards);

    tcout << "=== " << result.label << " ===\n";
    tcout << "Board count: " << result.boards.size() << '\n';
    tcout << "Hash count: " << result.hashes.size() << '\n';
    tcout << "Time: " << result.seconds << "s\n";

    if (result.boards.size() != result.hashes.size()) {
        tcout << "ERROR: board/hash count mismatch\n\n";
        return result;
    }

    const std::size_t uniqueCount = result.uniqueStateKeys.size();
    const std::size_t duplicateCount = result.boards.size() - uniqueCount;

    tcout << "Unique final states: " << uniqueCount << '\n';
    tcout << "Duplicate final states: " << duplicateCount << "\n\n";

    printPreviewBoards(result, start, previewCount);
    return result;
}

static void compareModes(const ModeResult& a,
                         const ModeResult& b) {
    std::vector<std::pair<u64, u64>> onlyA;
    std::vector<std::pair<u64, u64>> onlyB;
    std::vector<std::pair<u64, u64>> both;

    std::set_difference(
            a.uniqueStateKeys.begin(), a.uniqueStateKeys.end(),
            b.uniqueStateKeys.begin(), b.uniqueStateKeys.end(),
            std::back_inserter(onlyA)
    );

    std::set_difference(
            b.uniqueStateKeys.begin(), b.uniqueStateKeys.end(),
            a.uniqueStateKeys.begin(), a.uniqueStateKeys.end(),
            std::back_inserter(onlyB)
    );

    std::set_intersection(
            a.uniqueStateKeys.begin(), a.uniqueStateKeys.end(),
            b.uniqueStateKeys.begin(), b.uniqueStateKeys.end(),
            std::back_inserter(both)
    );

    tcout << "=== Compare " << a.label << " vs " << b.label << " ===\n";
    tcout << "Common unique states: " << both.size() << '\n';
    tcout << "Only in " << a.label << ": " << onlyA.size() << '\n';
    tcout << "Only in " << b.label << ": " << onlyB.size() << "\n\n";
}

int main() {
    const u8 values[36] = {
            0, 1, 2, 3, 4, 5,
            5, 4, 3, 2, 1, 0,
            1, 3, 6, 6, 2, 4,
            4, 2, 6, 6, 3, 1,
            2, 5, 1, 4, 0, 3,
            3, 0, 4, 1, 5, 2
    };

    constexpr u8 fatX = 2;
    constexpr u8 fatY = 2;
    constexpr u32 depth = 1;
    constexpr u32 previewCount = 60;

    Board start(values, fatX, fatY);
    StateHash::refreshHashFunc(start);

    tcout << "Initial board:\n";
    tcout << start.toStringSingle({}) << '\n';
    tcout << "Fat position: (" << static_cast<u32>(start.getFatX())
          << ", " << static_cast<u32>(start.getFatY()) << ")\n";
    tcout << "Fat XY index: " << static_cast<u32>(start.getFatXY()) << '\n';
    tcout << "Color count: " << start.getColorCount() << '\n';
    tcout << "Depth: " << depth << "\n\n";

    const ModeResult asc  = runMode<eSequenceDir::ASCENDING>("ASCENDING", start, depth, previewCount);
    const ModeResult desc = runMode<eSequenceDir::DESCENDING>("DESCENDING", start, depth, previewCount);
    const ModeResult none = runMode<eSequenceDir::NONE>("NONE", start, depth, previewCount);

    compareModes(asc, desc);
    compareModes(asc, none);
    compareModes(desc, none);

    return 0;
}