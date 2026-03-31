#include <iostream>
#include <iomanip>
#include <map>
#include <vector>
#include <array>

#include "MindbenderSolver/code/board.hpp"
#include "MindbenderSolver/code/levels.hpp"
#include "MindbenderSolver/code/perms.hpp"
#include "MindbenderSolver/code/rotations.hpp"

#include <cmath>

// -------------------------------------------------------------
// 1. Dihedral symmetry (8)
// -------------------------------------------------------------
static inline void dihedral_map(int sym, int x, int y, int &ox, int &oy) {
    switch (sym) {
        case 0: ox = x;        oy = y;        break;
        case 1: ox = 5 - y;    oy = x;        break;
        case 2: ox = 5 - x;    oy = 5 - y;    break;
        case 3: ox = y;        oy = 5 - x;    break;
        case 4: ox = 5 - x;    oy = y;        break;
        case 5: ox = x;        oy = 5 - y;    break;
        case 6: ox = y;        oy = x;        break;
        case 7: ox = 5 - y;    oy = 5 - x;    break;
        default: ox = x;       oy = y;        break;
    }
}

// -------------------------------------------------------------
// 2. Board <-> 36-array
// -------------------------------------------------------------
static inline void boardToArray(const Board &b, uint8_t cells[36]) {
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
            cells[y * 6 + x] = b.getColor((uint8_t)x, (uint8_t)y);
}

static inline Board arrayToBoard(const uint8_t cells[36]) {
    Board nb;
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x)
            nb.setColor((uint8_t)x, (uint8_t)y, cells[y * 6 + x]);
    return nb;
}

// -------------------------------------------------------------
// 3. Count color multiplicities
// -------------------------------------------------------------
static inline void countColors(const uint8_t cells[36], int count[8], int &colorCnt)
{
    for (int i = 0; i < 8; i++) count[i] = 0;

    for (int i = 0; i < 36; i++)
        count[cells[i]]++;

    colorCnt = 0;
    for (int i = 0; i < 8; i++)
        if (count[i] > 0)
            colorCnt++;
}

// -------------------------------------------------------------
// 4. Generate only valid color permutations
// -------------------------------------------------------------
static void genColorPermutations(
    const int count[8],
    std::vector<std::array<uint8_t,8>> &perms)
{
    // bucket colors by multiplicity
    std::map<int, std::vector<int>> buckets;
    for (int c = 0; c < 8; c++) {
        if (count[c] > 0) {
            buckets[count[c]].push_back(c);
        }
    }

    // start with identity permutation
    std::array<uint8_t,8> base{};
    for (int i = 0; i < 8; i++) base[i] = (uint8_t)i;

    perms.clear();
    perms.push_back(base);

    // multiply through each bucket
    for (auto &kv : buckets) {
        const std::vector<int> &group = kv.second;
        int n = group.size();
        if (n <= 1) continue;

        std::vector<std::array<uint8_t,8>> next;
        std::vector<int> idx = group;

        do {
            for (auto &p : perms) {
                auto np = p;
                for (int i = 0; i < n; i++)
                    np[group[i]] = (uint8_t)idx[i];
                next.push_back(np);
            }
        } while (std::next_permutation(idx.begin(), idx.end()));

        perms.swap(next);
    }
}

// -------------------------------------------------------------
// 5. Apply color permutation
// -------------------------------------------------------------
static inline void applyColorPerm(
    uint8_t out[36],
    const uint8_t in[36],
    const std::array<uint8_t,8> &perm)
{
    for (int i = 0; i < 36; i++)
        out[i] = perm[in[i]];
}

#define USE_DIHEDRAL
#define USE_TORUS_SHIFT
#define USE_COLOR_SWAP

// -------------------------------------------------------------
// 6. Canonical key WITH EXTERNAL colorPerms
// -------------------------------------------------------------
static std::pair<uint64_t,uint64_t>
canonicalKey(const Board &b, const std::vector<std::array<uint8_t,8>> &colorPerms)
{
    uint8_t base[36];
    boardToArray(b, base);

    uint64_t bestB1 = 0, bestB2 = 0;
    bool first = true;

    // loop bounds depend on flags
#if defined(USE_DIHEDRAL)
    const int SYM_MAX = 8;
#else
    const int SYM_MAX = 1;
#endif

#if defined(USE_TORUS_SHIFT)
    const int SHIFT_Y_MAX = 6;
    const int SHIFT_X_MAX = 6;
#else
    const int SHIFT_Y_MAX = 1;
    const int SHIFT_X_MAX = 1;
#endif

    for (int sym = 0; sym < SYM_MAX; ++sym) {
        for (int sy = 0; sy < SHIFT_Y_MAX; ++sy) {
            for (int sx = 0; sx < SHIFT_X_MAX; ++sx) {

                uint8_t tmp[36];
                uint8_t tmp2[36];

                // ---------- geometric transforms (dihedral + torus) ----------
                for (int y = 0; y < 6; ++y) {
                    for (int x = 0; x < 6; ++x) {
                        int rx = x;
                        int ry = y;

#if defined(USE_DIHEDRAL)
                        dihedral_map(sym, x, y, rx, ry);
#endif

                        int nx = rx;
                        int ny = ry;
#if defined(USE_TORUS_SHIFT)
                        nx = (rx + sx) % 6;
                        ny = (ry + sy) % 6;
#endif
                        tmp[ny * 6 + nx] = base[y * 6 + x];
                    }
                }

                // ---------- color permutations (optional) ----------
#if defined(USE_COLOR_SWAP)
                if (!colorPerms.empty()) {
                    for (const auto &perm : colorPerms) {
                        applyColorPerm(tmp2, tmp, perm);
                        Board nb = arrayToBoard(tmp2);
                        uint64_t cb1 = nb.b1;
                        uint64_t cb2 = nb.b2;
                        if (first || cb1 < bestB1 || (cb1 == bestB1 && cb2 < bestB2)) {
                            bestB1 = cb1;
                            bestB2 = cb2;
                            first  = false;
                        }
                    }
                } else {
                    // no perms provided -> treat as identity
                    std::memcpy(tmp2, tmp, 36);
                    Board nb = arrayToBoard(tmp2);
                    uint64_t cb1 = nb.b1;
                    uint64_t cb2 = nb.b2;
                    if (first || cb1 < bestB1 || (cb1 == bestB1 && cb2 < bestB2)) {
                        bestB1 = cb1;
                        bestB2 = cb2;
                        first  = false;
                    }
                }
#else
                // color swapping disabled -> just use raw tmp as tmp2
                std::memcpy(tmp2, tmp, 36);
                {
                    Board nb = arrayToBoard(tmp2);
                    uint64_t cb1 = nb.b1;
                    uint64_t cb2 = nb.b2;
                    if (first || cb1 < bestB1 || (cb1 == bestB1 && cb2 < bestB2)) {
                        bestB1 = cb1;
                        bestB2 = cb2;
                        first  = false;
                    }
                }
#endif
            }
        }
    }

    return {bestB1, bestB2};
}

// -------------------------------------------------------------
//  MAIN
// -------------------------------------------------------------
int main(int argc, char** argv) {
    std::cout.setf(std::ios::unitbuf);

    const char* id = (argc >= 2 ? argv[1] : "7-4");
    const auto* pair = BoardLookup::getBoardPair(id);
    if (!pair) {
        std::cerr << "Unknown puzzle ID\n";
        return 1;
    }

    Board start = pair->getStartState();
    int maxDepth = (argc >= 3 ? std::stoi(argv[2]) : 4);

    // ---------------------------------------------------------
    // PRECOMPUTE COLOR PERMUTATIONS (once!)
    // ---------------------------------------------------------
    uint8_t base[36];
    boardToArray(start, base);

    int count[8], colorCnt;
    countColors(base, count, colorCnt);

    std::vector<std::array<uint8_t,8>> colorPerms;
    genColorPermutations(count, colorPerms);

    // print perm stats
    std::cout << "Puzzle: " << id << "\n";
    std::cout << "MaxDepth = " << maxDepth << "\n\n";

    std::cout << "Color multiplicities:\n";
    for (int i = 0; i < 8; i++)
        if (count[i] > 0)
            std::cout << "  Color " << i << ": " << count[i] << "\n";

    std::cout << "\nNumber of color permutations: " << colorPerms.size() << "\n\n";

    std::cout << "Permutations:\n";
    for (size_t i = 0; i < colorPerms.size(); i++) {
        const auto &p = colorPerms[i];
        std::cout << "  Perm " << i << ":  ";
        for (int c = 0; c < 8; c++)
            std::cout << (int)p[c] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";

    // ---------------------------------------------------------
    // TABLE HEADER
    // ---------------------------------------------------------
    std::cout << std::setw(6) << "Depth"
              << " | " << std::setw(12) << "BFS Raw"
              << " | " << std::setw(12) << "MyPruning"
              << " | " << std::setw(18) << "MyPruning+Canon"
              << "\n";

    std::cout << "--------------------------------------------------------------\n";

    // ---------------------------------------------------------
    // DEPTH LOOP
    // ---------------------------------------------------------
    for (int depth = 0; depth <= maxDepth; depth++) {

        // (1) RAW BFS (estimated)
        uint64_t rawCount = (uint64_t)std::pow(60, depth);

        // (2) pruning
        JVec<Board> prunedBoards;
        Perms<Board>::getDepthFunc(start, prunedBoards, depth, /*prune=*/true);
        uint64_t prunedCount = prunedBoards.size();

        // (3) PRUNING + CANON
        std::map<std::pair<uint64_t,uint64_t>, bool> localSet;
        for (const Board &b : prunedBoards)
            localSet[ canonicalKey(b, colorPerms) ] = true;

        uint64_t canonCount = localSet.size();

        std::cout << std::setw(6) << depth
                  << " | " << std::setw(12) << rawCount
                  << " | " << std::setw(12) << prunedCount
                  << " | " << std::setw(18) << canonCount
                  << "\n";
    }

    return 0;
}
