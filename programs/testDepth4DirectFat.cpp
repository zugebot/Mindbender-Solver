#include "code/include.hpp"
#include "utils/timer.hpp"

#include <iostream>
#include <string>

template<eSequenceDir DIR>
static void runMode(const char* label, const Board& start, u32 depth, u32 previewCount = 12) {
    JVec<Board> boards;
    const bool shouldAlloc = true;

    Timer timer;
    Perms<Board>::getDepthFunc<DIR>(start, boards, depth, shouldAlloc);
    const double seconds = timer.getSeconds();

    tcout << "=== " << label << " ===\n";
    tcout << "Count: " << boards.size() << '\n';
    tcout << "Time: " << seconds << "s\n";

    const u8 fatPos = start.getFatXY();

    const u32 shown = static_cast<u32>(std::min<std::size_t>(boards.size(), previewCount));
    for (u32 i = 0; i < shown; ++i) {
        tcout << "  [" << i << "] "
                  << boards[i].getMemory().asmFatStringForwards(fatPos) << '\n';
    }

    if (boards.size() > shown) {
        tcout << "  ...\n";
    }

    tcout << '\n';
}

int main() {
    u8 values[36] = {
            0, 1, 2, 3, 4, 5,
            5, 4, 3, 2, 1, 0,
            1, 3, 6, 6, 2, 4,
            4, 2, 6, 6, 3, 1,
            2, 5, 1, 4, 0, 3,
            3, 0, 4, 1, 5, 2
    };
    
    constexpr u8 fatX = 2;
    constexpr u8 fatY = 2;

    Board start(values, fatX, fatY);

    constexpr u32 depth = 4;

    tcout << "Initial board:\n";
    tcout << start.toStringSingle() << '\n';
    tcout << "Fat position: (" << static_cast<u32>(start.getFatX())
              << ", " << static_cast<u32>(start.getFatY()) << ")\n";
    tcout << "Fat XY index: " << static_cast<u32>(start.getFatXY()) << '\n';
    tcout << "Color count: " << start.getColorCount() << '\n';
    tcout << "Depth: " << depth << "\n\n";

    runMode<eSequenceDir::ASCENDING>("ASCENDING", start, depth);
    runMode<eSequenceDir::DESCENDING>("DESCENDING", start, depth);
    runMode<eSequenceDir::NONE>("NONE", start, depth);

    return 0;
}