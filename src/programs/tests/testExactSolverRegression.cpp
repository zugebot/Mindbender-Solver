#include "src/chuzzles/include.hpp"
#include "src/chuzzles/intersection.hpp"
#include "src/solver/frontier_builder.hpp"
#include "src/solver/inram/right_frontier_table.hpp"
#include "src/solver/inram/solver_frontier.hpp"
#include "src/solver/ssd/common.hpp"

#include <algorithm>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {

bool testThreadedIntersectionCoversTail() {
    JVec<B1B2> left;
    left.resize(10);
    JVec<u64> leftHashes;
    leftHashes.resize(10);
    for (std::size_t i = 0; i < left.size(); ++i) {
        left[i].b1 = static_cast<u64>(i + 1);
        left[i].b2 = static_cast<u64>(100 + i);
    }

    // With four workers, the old range calculation extended the first chunk
    // through indices [0,6), then gave a later chunk an end beyond size 10.
    // process_chunk() silently discarded that chunk, including index 9.
    for (std::size_t i = 0; i < 6; ++i) leftHashes[i] = 1;
    for (std::size_t i = 6; i < 9; ++i) leftHashes[i] = 2;
    leftHashes[9] = 3;

    JVec<B1B2> right;
    right.resize(1);
    JVec<u64> rightHashes;
    rightHashes.resize(1);
    right[0] = left[9];
    rightHashes[0] = 3;

    const auto single = intersection(left, leftHashes, right, rightHashes);
    const auto threaded = intersection_threaded(
            left, leftHashes, right, rightHashes, 4);

    return single.size() == 1 && threaded.size() == 1 &&
           *threaded[0].first == left[9] && *threaded[0].second == right[0];
}

bool testExactIndexRejectsHashCollisions() {
    JVec<B1B2> states;
    states.resize(3);
    JVec<u64> hashes;
    hashes.resize(3);

    states[0] = B1B2{1, 10};
    states[1] = B1B2{2, 20};
    states[2] = B1B2{3, 30};
    hashes[0] = hashes[1] = hashes[2] = 0x123456789abcdef0ULL;

    RightFrontierTableB1B2 index;
    Board unusedRoot;
    index.buildFromUniqueStates<false>(
            std::move(states), std::move(hashes), unusedRoot);

    JVec<B1B2> queries;
    queries.resize(2);
    JVec<u64> queryHashes;
    queryHashes.resize(2);
    queries[0] = B1B2{2, 20};
    queries[1] = B1B2{4, 40};
    queryHashes[0] = queryHashes[1] = 0x123456789abcdef0ULL;

    JVec<B1B2> matches;
    JVec<u64> matchHashes;
    RightFrontierTableB1B2::ProbeStats stats;
    index.collectMatches<5, true>(
            queries, queryHashes, matches, matchHashes, &stats);

    return matches.size() == 1 && matches[0] == queries[0] &&
           stats.hashHits == 2 && stats.exactMatches == 1;
}

bool testExactLayerRetainsLongerReachability() {
    const BoardPair* pair = BoardLookup::getBoardPair("7-1");
    if (pair == nullptr) {
        return false;
    }

    JVec<B1B2> depth1;
    JVec<u64> depth1Hashes;
    JVec<B1B2> depth3;
    JVec<u64> depth3Hashes;

    FrontierBuilderB1B2 builder(pair->getStartState(), false);
    builder.buildExactNoneDepth(1, depth1, depth1Hashes);
    builder.buildExactNoneDepth(3, depth3, depth3Hashes);

    if (depth1.empty()) {
        return false;
    }

    return std::find(depth3.begin(), depth3.end(), depth1[0]) != depth3.end();
}


bool testFatCanonicalGeneratorsCoverKnownMiddleSegment() {
    const BoardPair* pair = BoardLookup::getBoardPair("6-2");
    if (pair == nullptr) {
        return false;
    }

    // The missing 40-solution family contains this valid prefix. After its
    // first five moves, both canonical fat generators must retain the board
    // reached by the consecutive R04/R54 segment. The old code combined two
    // opposing ordering rules and removed every representative of this state.
    const std::vector<u8> moves = Memory::parseFatMoveString(
            "R11 R342 C05 R15 C25 R04 R54 C345 C54 R235 C11 C41"
    );
    if (moves.size() != 12) {
        return false;
    }

    Board segmentStart = pair->getStartState();
    for (std::size_t i = 0; i < 5; ++i) {
        allActStructList[moves[i]].action(segmentStart);
    }

    Board target = pair->getStartState();
    for (std::size_t i = 0; i < 7; ++i) {
        allActStructList[moves[i]].action(target);
    }
    const B1B2 targetState = target.asB1B2();

    struct TargetSink {
        B1B2 target{};
        bool found = false;

        void operator()(JVec<B1B2>& states,
                        JVec<u64>&,
                        const u32 count) {
            for (u32 i = 0; i < count; ++i) {
                if (states[i] == target) {
                    found = true;
                    return;
                }
            }
        }
    } sink{targetState, false};

    PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 2>(
            segmentStart, sink, 1024
    );

    JVec<B1B2> materializedStates;
    JVec<u64> materializedHashes;
    Perms<B1B2>::getDepthFunc<eSequenceDir::ASCENDING>(
            segmentStart, materializedStates, materializedHashes, 2
    );
    const bool materializedFound =
            std::find(materializedStates.begin(), materializedStates.end(), targetState)
            != materializedStates.end();

    return sink.found && materializedFound;
}

bool testStreamCoversExactDistanceShell(const char* puzzle) {
    // Every state at exact BFS distance 3 from the root must be produced by
    // the canonical stream. The old fat column->row commuting mask pruned
    // pairs that do not commute and lost ~8% of these states on 6-2; 1-1
    // (2 colors) exercises the normal stream's CHECK_CROSS pruning hardest.
    const BoardPair* pair = BoardLookup::getBoardPair(puzzle);
    if (pair == nullptr) {
        return false;
    }
    const Board root = pair->getStartState();

    std::vector<B1B2> closer;
    for (u32 depth = 0; depth < 3; ++depth) {
        JVec<B1B2> states;
        JVec<u64> hashes;
        FrontierBuilderB1B2 builder(root, false);
        builder.buildExactNoneDepth(depth, states, hashes);
        closer.insert(closer.end(), states.begin(), states.end());
    }
    std::sort(closer.begin(), closer.end());

    JVec<B1B2> depth3;
    JVec<u64> depth3Hashes;
    FrontierBuilderB1B2 builder(root, false);
    builder.buildExactNoneDepth(3, depth3, depth3Hashes);

    struct CollectSink {
        std::vector<B1B2> states;

        void operator()(JVec<B1B2>& chunk,
                        JVec<u64>&,
                        const u32 count) {
            states.insert(states.end(), chunk.begin(), chunk.begin() + count);
        }
    } sink;
    PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 3>(root, sink, 1024);
    std::sort(sink.states.begin(), sink.states.end());

    for (const B1B2& state : depth3) {
        if (std::binary_search(closer.begin(), closer.end(), state)) {
            continue;
        }
        if (!std::binary_search(sink.states.begin(), sink.states.end(), state)) {
            return false;
        }
    }
    return true;
}

bool testFatColThenRowMaskIsExact() {
    // The C -> R pruning mask must flag exactly the pairs where applying the
    // row move before the column move gives the same board. Flagging a pair
    // that does not commute loses states; the old version did so for ~89%.
    std::mt19937_64 rng(12345);
    for (int trial = 0; trial < 2000; ++trial) {
        u8 cells[36];
        const int colors = 2 + trial % 4;
        for (u8& cell : cells) {
            cell = static_cast<u8>(rng() % colors);
        }
        const Board board(cells, static_cast<u8>(rng() % 5), static_cast<u8>(rng() % 5));
        const u8* relative = fatActionsIndexes[board.getFatXY()];

        for (int colSlot = 24; colSlot < 48; ++colSlot) {
            const u8 col = relative[colSlot];
            if (col < 32 || col >= 62) {
                continue;
            }
            Board afterCol = board;
            allActStructList[col].action(afterCol);
            const u32 mask = afterCol.getFatNormalColThenRowCommutingMask(col);

            for (int rowSlot = 0; rowSlot < 24; ++rowSlot) {
                const u8 row = relative[rowSlot];
                const bool flagged = (mask >> rowSlot) & 1U;
                if (row >= 30) {
                    if (flagged) {
                        return false;
                    }
                    continue;
                }
                Board colThenRow = afterCol;
                allActStructList[row].action(colThenRow);
                Board rowThenCol = board;
                allActStructList[row].action(rowThenCol);
                allActStructList[col].action(rowThenCol);
                if (flagged != (colThenRow == rowThenCol)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool testLaneRotationsMatchActions() {
    // lane_rotations.hpp must reproduce every normal row/column action.
    std::mt19937_64 rng(3);
    for (int trial = 0; trial < 2000; ++trial) {
        u8 cells[36];
        for (u8& cell : cells) {
            cell = static_cast<u8>(rng() % (2 + trial % 7));
        }
        const Board board(cells);
        B1B2 shifted[5];
        for (u32 lane = 0; lane < 6; ++lane) {
            lane_rotations::rowShifts(board, lane, shifted);
            for (u32 n = 1; n <= 5; ++n) {
                B1B2 expected = board;
                allActStructList[lane * 5 + n - 1].action(expected);
                if (!(expected == shifted[n - 1])) {
                    return false;
                }
            }
            lane_rotations::colShifts(board, lane, shifted);
            for (u32 n = 1; n <= 5; ++n) {
                B1B2 expected = board;
                allActStructList[32 + lane * 5 + n - 1].action(expected);
                if (!(expected == shifted[n - 1])) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool testFatLaneRotationsMatchActions() {
    // Every slot of fatActionsIndexes must match the corresponding lane shift,
    // and the fat lanes must omit exactly amount (5 - fatX) / (5 - fatY).
    std::mt19937_64 rng(5);
    for (int trial = 0; trial < 1000; ++trial) {
        u8 cells[36];
        for (u8& cell : cells) {
            cell = static_cast<u8>(rng() % (2 + trial % 6));
        }
        const u8 fatX = static_cast<u8>(trial % 5);
        const u8 fatY = static_cast<u8>((trial / 5) % 5);
        const Board board(cells, fatX, fatY);
        const u8* relative = fatActionsIndexes[board.getFatXY()];

        bool rowAmountListed[6] = {}, colAmountListed[6] = {};
        for (int slot = 0; slot < 48; ++slot) {
            const u32 action = relative[slot];
            B1B2 expected = board;
            allActStructList[action].action(expected);
            B1B2 shifted[5];
            u32 amount;
            if (action < 30) {
                lane_rotations::rowShifts(board, action / 5, shifted);
                amount = action % 5 + 1;
            } else if (action < 62) {
                lane_rotations::colShifts(board, (action - 32) / 5, shifted);
                amount = (action - 32) % 5 + 1;
            } else if (action < 89) {
                lane_rotations::fatRowShifts(board, (action - 64) / 5, fatX, shifted);
                amount = (action - 64) % 5 + 1;
                rowAmountListed[amount] = true;
            } else {
                lane_rotations::fatColShifts(board, (action - 89) / 5, fatY, shifted);
                amount = (action - 89) % 5 + 1;
                colAmountListed[amount] = true;
            }
            if (!(expected == shifted[amount - 1])) {
                return false;
            }
        }
        for (u32 amount = 1; amount <= 5; ++amount) {
            if (rowAmountListed[amount] == (amount == 5U - fatX)
                || colAmountListed[amount] == (amount == 5U - fatY)) {
                return false;
            }
        }
    }
    return true;
}

bool testFatDfsStreamWithinSlotTableStream() {
    // The lane-based fat stream drops only consecutive moves on the fat lane,
    // so its states must be a subset of the slot-table generator's states.
    struct CollectSink {
        std::vector<B1B2> states;

        void operator()(JVec<B1B2>& chunk, JVec<u64>&, const u32 count) {
            states.insert(states.end(), chunk.begin(), chunk.begin() + count);
        }
    };

    for (const char* name : {"6-2", "4-4", "13-4"}) {
        const Board root = BoardLookup::getBoardPair(name)->getStartState();
        CollectSink sink;
        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 3>(root, sink, 1024);

        JVec<B1B2> boards;
        JVec<u64> hashes;
        boards.reserve(BOARD_FAT_MAX_MALLOC_SIZES[3]);
        boards.resize(BOARD_FAT_MAX_MALLOC_SIZES[3]);
        hashes.reserve(BOARD_FAT_MAX_MALLOC_SIZES[3]);
        hashes.resize(BOARD_FAT_MAX_MALLOC_SIZES[3]);
        perms_detail::make_fat_perm_list<B1B2, 3, eSequenceDir::ASCENDING>(root, boards, hashes);

        std::vector<B1B2> reference(boards.begin(), boards.end());
        std::sort(sink.states.begin(), sink.states.end());
        sink.states.erase(std::unique(sink.states.begin(), sink.states.end()), sink.states.end());
        std::sort(reference.begin(), reference.end());
        reference.erase(std::unique(reference.begin(), reference.end()), reference.end());
        if (!std::includes(reference.begin(), reference.end(), sink.states.begin(), sink.states.end())) {
            return false;
        }
    }
    return true;
}

bool testDfsStreamMatchesMaterializedPerms() {
    // The depth-first B1B2 stream must emit the same set of states as the
    // independent materialized generator with the same rules, and nothing the
    // generator does not: a sub-multiset of it. (Not the same multiset: the DFS
    // stops a lane at its period, so it repeats fewer states on lanes whose
    // pattern repeats; the materialized list still visits those amounts.)
    std::vector<Board> roots;
    for (const char* name : {"1-1", "9-3", "10-3", "7-4"}) {
        const BoardPair* pair = BoardLookup::getBoardPair(name);
        roots.push_back(pair->getStartState());
        roots.push_back(pair->getEndState());
    }

    struct CollectSink {
        std::vector<B1B2> states;

        void operator()(JVec<B1B2>& chunk, JVec<u64>&, const u32 count) {
            states.insert(states.end(), chunk.begin(), chunk.begin() + count);
        }
    };

    for (const Board& root : roots) {
        CollectSink sink;
        PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 3>(root, sink, 1024);

        JVec<B1B2> boards;
        JVec<u64> hashes;
        boards.reserve(BOARD_PRE_MAX_MALLOC_SIZES[3]);
        boards.resize(BOARD_PRE_MAX_MALLOC_SIZES[3]);
        hashes.reserve(BOARD_PRE_MAX_MALLOC_SIZES[3]);
        hashes.resize(BOARD_PRE_MAX_MALLOC_SIZES[3]);
        perms_detail::make_perm_list<B1B2, 3, true, true, true, eSequenceDir::ASCENDING>(root, boards, hashes);

        std::vector<B1B2> expected(boards.begin(), boards.end());
        std::sort(sink.states.begin(), sink.states.end());
        std::sort(expected.begin(), expected.end());
        if (!std::includes(expected.begin(), expected.end(), sink.states.begin(), sink.states.end())) {
            return false;   // the stream emitted a state (or a repeat) the generator does not
        }
        std::vector<B1B2> streamedSet(sink.states);
        streamedSet.erase(std::unique(streamedSet.begin(), streamedSet.end()), streamedSet.end());
        expected.erase(std::unique(expected.begin(), expected.end()), expected.end());
        if (streamedSet != expected) {
            return false;   // a state the generator reaches is missing from the stream
        }
    }
    return true;
}

bool testLastMovesCoverNextLayer() {
    // ssd::buildLastMoves (the first-move rule, GitHub issue 1), normal and 2x2
    // boards, depths 1..3. The frontier builder's layer d holds every endpoint of
    // a d-move sequence without no-op moves (deduplicated within the layer only),
    // so it also holds shallower states; the rule is about states at distance
    // exactly d, i.e. layer d minus layers d-1 and d-2. For those: (1) the coded
    // last move equals a brute force over every parent and every move (the
    // lexicographically smallest path's last move), and (2) streaming each state
    // of layer d one move with its last move as the previous one reaches every
    // state at distance exactly d+1.
    for (const char* name : {"1-1", "9-3", "10-3", "6-2", "17-4"}) {
        const BoardPair* pair = BoardLookup::getBoardPair(name);
        if (pair == nullptr) {
            return false;
        }
        const Board root = pair->getStartState();
        std::vector<JVec<B1B2>> layers(5);
        layers[0].resize(1);
        layers[0][0] = root.asB1B2();
        for (u32 d = 1; d <= 4; ++d) {
            ssd::buildSeeds(root, d, layers[d]);
        }
        std::vector<std::vector<B1B2>> sorted(5);
        for (u32 d = 0; d <= 4; ++d) {
            sorted[d].assign(layers[d].begin(), layers[d].end());
            std::sort(sorted[d].begin(), sorted[d].end());
        }
        auto shallower = [&](const u32 d, const B1B2& s) {   // in layer d-1 or d-2
            for (u32 k = d >= 2 ? d - 2 : 0; k < d; ++k) {
                if (std::binary_search(sorted[k].begin(), sorted[k].end(), s)) {
                    return true;
                }
            }
            return false;
        };
        std::map<B1B2, u32> above{{root.asB1B2(), 0}};   // brute-force codes of the layer above
        for (u32 d = 1; d <= 3; ++d) {
            const JVec<B1B2>& layer = layers[d];
            std::vector<u8> moves;
            ssd::buildLastMoves(root, d, layer, moves, 4);
            if (moves.size() != layer.size()) {
                return false;
            }
            std::map<B1B2, u32> codes;
            for (std::size_t i = 0; i < layer.size(); ++i) {
                codes.emplace(layer[i], 0xFFFFFFFFu);
            }
            for (const auto& [parent, code] : above) {
                if (code == 0xFFFFFFFFu) {
                    continue;
                }
                perm_stream_detail::forEachFirstMove(parent, perm_stream_detail::NO_PREV_MOVE,
                                                     [&](const u8 move, const B1B2& child) {
                    auto it = codes.find(child);
                    if (it != codes.end()) {
                        it->second = std::min(it->second, (code << 6) | move);
                    }
                });
            }
            for (std::size_t i = 0; i < layer.size(); ++i) {
                if (shallower(d, layer[i])) {
                    continue;
                }
                const u32 c = codes.at(layer[i]);
                if (c == 0xFFFFFFFFu || static_cast<u8>(c & 63) != moves[i]) {
                    std::cerr << "last moves: " << name << " depth " << d << " state " << i << ": brute force "
                              << (c == 0xFFFFFFFFu ? -1 : static_cast<int>(c & 63)) << " vs " << static_cast<int>(moves[i]) << '\n';
                    return false;
                }
            }

            struct CollectSink {
                std::vector<B1B2> states;
                void operator()(JVec<B1B2>& chunk, JVec<u64>&, const u32 count) {
                    states.insert(states.end(), chunk.begin(), chunk.begin() + count);
                }
            } sink;
            for (std::size_t i = 0; i < layer.size(); ++i) {
                PermStream<B1B2>::streamDepth<eSequenceDir::ASCENDING, 1>(makeBoardFromState(layer[i]), sink, 64, moves[i]);
            }
            std::sort(sink.states.begin(), sink.states.end());
            const JVec<B1B2>& next = layers[d + 1];
            std::size_t exact = 0;
            for (std::size_t i = 0; i < next.size(); ++i) {
                if (shallower(d + 1, next[i])) {
                    continue;
                }
                ++exact;
                if (!std::binary_search(sink.states.begin(), sink.states.end(), next[i])) {
                    std::cerr << "last moves: " << name << " depth " << d << ": distance-" << d + 1 << " state " << i
                              << " not reached (" << sink.states.size() << " streamed)\n";
                    return false;
                }
            }
            if (exact == 0) {
                return false;
            }
            above.swap(codes);
        }
    }
    return true;
}

bool testGetColorMatchesPackedLayout() {
    // getShiftAmount used to read rows {2,1,0,5,4,3} and shift by >= 64 (UB);
    // at -O3 the unrolled loop below folded most reads to 0.
    u8 cells[36];
    for (int i = 0; i < 36; ++i) {
        cells[i] = static_cast<u8>((i * 5 + i / 6) % 8);
    }
    const Board board(cells);

    for (u8 y = 0; y < 6; ++y) {
        for (u8 x = 0; x < 6; ++x) {
            if (board.getColor(x, y) != board.getColor(static_cast<u64>(x), static_cast<u64>(y))) {
                return false;
            }
        }
    }

    // setState renumbers the colors present in ascending value order.
    bool present[8] = {};
    for (const u8 c : cells) {
        present[c] = true;
    }
    u8 adjusted[8] = {};
    u8 next = 0;
    for (u8 c = 0; c < 8; ++c) {
        if (present[c]) {
            adjusted[c] = next++;
        }
    }
    for (u8 y = 0; y < 6; ++y) {
        for (u8 x = 0; x < 6; ++x) {
            if (board.getColor(x, y) != adjusted[cells[6 * y + x]]) {
                return false;
            }
        }
    }
    return true;
}

bool testSmallEndToEndExactSolver() {
    const BoardPair* base = BoardLookup::getBoardPair("7-1");
    if (base == nullptr) {
        return false;
    }

    Board start = base->getStartState();
    Board goal = start;
    const u8 targetMoves[3] = {0, 34, 23};
    for (const u8 action : targetMoves) {
        allActStructList[action].action(goal);
    }

    BoardPair pair("regression", start, goal);

    std::set<std::string> expected;
    for (u64 d0 = 0; d0 < 2; ++d0) {
        for (u64 s0 = 0; s0 < 6; ++s0) {
            for (u64 h0 = 0; h0 < 5; ++h0) {
                const u8 a0 = static_cast<u8>((d0 << 5) + s0 * 5 + h0);
                Board b0 = start;
                allActStructList[a0].action(b0);
                if (b0 == start) continue;

                for (u64 d1 = 0; d1 < 2; ++d1) {
                    for (u64 s1 = 0; s1 < 6; ++s1) {
                        for (u64 h1 = 0; h1 < 5; ++h1) {
                            const u8 a1 = static_cast<u8>((d1 << 5) + s1 * 5 + h1);
                            Board b1 = b0;
                            allActStructList[a1].action(b1);
                            if (b1 == b0) continue;

                            for (u64 d2 = 0; d2 < 2; ++d2) {
                                for (u64 s2 = 0; s2 < 6; ++s2) {
                                    for (u64 h2 = 0; h2 < 5; ++h2) {
                                        const u8 a2 = static_cast<u8>(
                                                (d2 << 5) + s2 * 5 + h2);
                                        Board b2 = b1;
                                        allActStructList[a2].action(b2);
                                        if (b2 == b1 || b2 != goal) continue;

                                        Memory memory;
                                        memory.setNextNMove<3>(
                                                static_cast<u64>(a0) |
                                                (static_cast<u64>(a1) << 6U) |
                                                (static_cast<u64>(a2) << 12U));
                                        expected.insert(memory.asmStringForwards());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    BoardSolverFrontier solver(&pair);
    solver.setWriteSolutionsToFile(false);
    const int found = solver.findSolutionsFrontierThreaded<1, 1, 1, false>(
            4, BoardSolverFrontier::SearchDirection::Forward);

    return found == static_cast<int>(expected.size());
}

} // namespace

int main() {
    const bool intersectionOk = testThreadedIntersectionCoversTail();
    const bool indexOk = testExactIndexRejectsHashCollisions();
    const bool layerOk = testExactLayerRetainsLongerReachability();
    const bool fatGeneratorsOk = testFatCanonicalGeneratorsCoverKnownMiddleSegment();
    const bool fatShellOk = testStreamCoversExactDistanceShell("6-2");
    const bool normalShellOk = testStreamCoversExactDistanceShell("1-1");
    const bool fatMaskOk = testFatColThenRowMaskIsExact();
    const bool laneOk = testLaneRotationsMatchActions();
    const bool dfsOk = testDfsStreamMatchesMaterializedPerms();
    const bool fatLaneOk = testFatLaneRotationsMatchActions();
    const bool fatDfsOk = testFatDfsStreamWithinSlotTableStream();
    const bool getColorOk = testGetColorMatchesPackedLayout();
    const bool solverOk = testSmallEndToEndExactSolver();
    const bool lastMovesOk = testLastMovesCoverNextLayer();

    std::cout << "threaded intersection tail coverage: "
              << (intersectionOk ? "PASS" : "FAIL") << '\n';
    std::cout << "exact index collision validation: "
              << (indexOk ? "PASS" : "FAIL") << '\n';
    std::cout << "exact-depth layer semantics: "
              << (layerOk ? "PASS" : "FAIL") << '\n';
    std::cout << "fat canonical middle coverage: "
              << (fatGeneratorsOk ? "PASS" : "FAIL") << '\n';
    std::cout << "fat stream exact distance-3 coverage: "
              << (fatShellOk ? "PASS" : "FAIL") << '\n';
    std::cout << "normal stream exact distance-3 coverage: "
              << (normalShellOk ? "PASS" : "FAIL") << '\n';
    std::cout << "fat col->row commuting mask exact: "
              << (fatMaskOk ? "PASS" : "FAIL") << '\n';
    std::cout << "lane rotations match actions: "
              << (laneOk ? "PASS" : "FAIL") << '\n';
    std::cout << "DFS stream matches materialized perms: "
              << (dfsOk ? "PASS" : "FAIL") << '\n';
    std::cout << "fat lane rotations match actions: "
              << (fatLaneOk ? "PASS" : "FAIL") << '\n';
    std::cout << "fat DFS stream within slot-table stream: "
              << (fatDfsOk ? "PASS" : "FAIL") << '\n';
    std::cout << "getColor matches packed layout: "
              << (getColorOk ? "PASS" : "FAIL") << '\n';
    std::cout << "small end-to-end exact solver: "
              << (solverOk ? "PASS" : "FAIL") << '\n';
    std::cout << "seed last moves (brute force, next layer covered): "
              << (lastMovesOk ? "PASS" : "FAIL") << '\n';

    return intersectionOk && indexOk && layerOk && fatGeneratorsOk
           && fatShellOk && normalShellOk && fatMaskOk && laneOk && dfsOk
           && fatLaneOk && fatDfsOk && getColorOk && solverOk && lastMovesOk ? 0 : 1;
}
