#include "code/include.hpp"
#include "code/board_hash_segments.hpp"
#include "code/right_cache_index.hpp"

#include <algorithm>

constexpr u32 RIGHT_PREFIX_BITS = 20;
constexpr Memory::HashMode HASH_MODE = Memory::HashMode::Auto;

struct MeetState {
    B1B2 state{};
    u64 hash = 0;

    MUND u64 getHash() C { return hash; }

    FORCEINLINE bool operator<(C MeetState& other) C {
        if (hash != other.hash) {
            return hash < other.hash;
        }
        if (state.b1 != other.state.b1) {
            return state.b1 < other.state.b1;
        }
        return state.b2 < other.state.b2;
    }
};

MUND static u64 computeStateHash(C B1B2& state) {
#ifndef __CUDA_ARCH__
    switch (Memory::getHashModeOverride()) {
        case Memory::HashMode::Hash2:
            return (getSegment2bits(state.b1) << 18) | getSegment2bits(state.b2);
        case Memory::HashMode::Hash3:
            return (getSegment3bits(state.b1) << 30) | getSegment3bits(state.b2);
        case Memory::HashMode::Hash4:
            return prime_func1(state.b2, state.b1);
        case Memory::HashMode::Auto:
            break;
    }
#endif

    C u64 colorCount = state.getColorCount();
    if (state.getFatBool() || colorCount > 3) {
        return prime_func1(state.b2, state.b1);
    }
    if (colorCount <= 2) {
        return (getSegment2bits(state.b1) << 18) | getSegment2bits(state.b2);
    }
    return (getSegment3bits(state.b1) << 30) | getSegment3bits(state.b2);
}

MUND static MeetState makeMeetState(C B1B2& state) {
    return {state, computeStateHash(state)};
}

MUND static MeetState makeMeetState(C Board& board) {
    return makeMeetState(board.asB1B2());
}

static void compressBoardsToMeetStates(C JVec<Board>& src, JVec<MeetState>& dst) {
    dst.resize(src.size());
    for (u64 i = 0; i < src.size(); ++i) {
        dst[i] = makeMeetState(src[i]);
    }
}

static void buildPrefixFrontier(C Board& start, JVec<Board>& out, C u32 depth) {
    out.clear();

    if (depth == 0) {
        out.resize(1);
        out[0] = start;
        return;
    }

    Perms<Board>::getDepthFunc<true>(start, out, depth, false);
}

MUND static u8 getInverseMove(C u8 move) {
    return static_cast<u8>(move + allActStructList[move].tillNext - 1 - allActStructList[move].tillLast);
}

static void applyMovesReverse(Board& board, C Memory& memory) {
    for (i32 i = static_cast<i32>(memory.getMoveCount()) - 1; i >= 0; --i) {
        C u8 move = memory.getMove(static_cast<u8>(i));
        allActStructList[getInverseMove(move)].action(board);
    }
}

static Board makeBoardWithMovesReverse(C Board& board, C Memory& memory) {
    Board temp = board;
    applyMovesReverse(temp, memory);
    return temp;
}

struct RightCache {
    JVec<MeetState> rightFrontier;
    right_cache_idx::PrefixIndex rightPrefixIndex;
    double rightBuildTime = 0.0;
    double rightSortTime = 0.0;
    double rightCompressTime = 0.0;
    double rightIndexTime = 0.0;
};

struct TrialTimings {
    double leftBuildTime = 0.0;
    double leftCompressTime = 0.0;
    double meetTime = 0.0;
    double totalTime = 0.0;
    u32 attempts = 0;
    u64 leftSize = 0;
    bool found = false;
    B1B2 middleState{};
    std::string moves;
};

struct BenchmarkResult {
    RightCache rightCache;
    double avgLeftBuildTime = 0.0;
    double avgLeftCompressTime = 0.0;
    double avgMeetTime = 0.0;
    double avgTotalTime = 0.0;
    double avgAttempts = 0.0;
    u64 leftSize = 0;
    u32 solvedTrials = 0;
    B1B2 sampleMiddle{};
    std::string sampleMoves;
};

struct LeftLoopScratch {
    JVec<Board> leftPrefixes;
    JVec<Board> generatedBoards;
    JVec<MeetState> leftFrontier;

    void init(C Board& start, C u32 leftLoopDepth, C u32 leftDepth) {
        Perms<Board>::reserveForDepth(start, leftPrefixes, leftLoopDepth);
        Perms<Board>::reserveForDepth(start, generatedBoards, leftDepth);
    }
};

static RightCache buildRightCache(C Board& start, C Board& goal, C u32 rightDepth) {
    RightCache cache;
    JVec<Board> generatedBoards;

    std::cout << "[RC] Generating..." << std::endl;
    C Timer timerRightBuild;
    Perms<Board>::getDepthFunc<true>(goal, generatedBoards, rightDepth, true);
    cache.rightBuildTime = timerRightBuild.getSeconds();

    std::cout << "[RC] Size: " << generatedBoards.size() << std::endl;

    std::cout << "[RC] Compressing..." << std::endl;
    C Timer timerRightCompress;
    compressBoardsToMeetStates(generatedBoards, cache.rightFrontier);
    cache.rightCompressTime = timerRightCompress.getSeconds();

    std::cout << "[RC] Sorting..." << std::endl;
    C Timer timerRightSort;
    std::sort(cache.rightFrontier.begin(), cache.rightFrontier.end());
    cache.rightSortTime = timerRightSort.getSeconds();

    std::cout << "[RC] Indexing..." << std::endl;
    C Timer timerRightIndex;
    cache.rightPrefixIndex.build(cache.rightFrontier, {RIGHT_PREFIX_BITS});
    cache.rightIndexTime = timerRightIndex.getSeconds();

    C auto& stats = cache.rightPrefixIndex.stats;
    C double singlePct = stats.nonEmptyBuckets == 0
                                 ? 0.0
                                 : (100.0 * static_cast<double>(stats.singleBucketCount)
                                    / static_cast<double>(stats.nonEmptyBuckets));
    C double multiPct = stats.nonEmptyBuckets == 0
                                ? 0.0
                                : (100.0 * static_cast<double>(stats.multiBucketCount)
                                   / static_cast<double>(stats.nonEmptyBuckets));

    std::cout << "[RC] Prefix bits: " << cache.rightPrefixIndex.getPrefixBits() << std::endl;
    std::cout << "[RC] Total prefix buckets: " << cache.rightPrefixIndex.getBucketCount() << std::endl;
    std::cout << "[RC] Non-empty prefix buckets: " << stats.nonEmptyBuckets
              << " (" << stats.occupancyPct << "%)" << std::endl;
    std::cout << "[RC] Avg boards/bucket (all): " << stats.avgBoardsPerBucket << std::endl;
    std::cout << "[RC] Avg boards/bucket (non-empty): " << stats.avgBoardsPerNonEmptyBucket << std::endl;
    std::cout << "[RC] Max bucket size: " << stats.maxBucketSize << std::endl;
    std::cout << "[RC] Single-board non-empty buckets: " << stats.singleBucketCount
              << " (" << singlePct << "%)" << std::endl;
    std::cout << "[RC] Multi-board non-empty buckets: " << stats.multiBucketCount
              << " (" << multiPct << "%)" << std::endl;
    std::cout << "[RC] Collision boards (non-first entries): " << stats.collisionBoards << std::endl;

    return cache;
}

static bool tryMeet(C JVec<MeetState>& left,
                    C RightCache& cache,
                    B1B2& middleOut) {
    for (u64 li = 0; li < left.size(); ++li) {
        C u64 leftHash = left[li].hash;

        C auto [rangeBegin, rangeEnd] = cache.rightPrefixIndex.getRange(leftHash);
        if (rangeBegin == rangeEnd) {
            continue;
        }

        C MeetState* first = cache.rightFrontier.begin() + rangeBegin;
        C MeetState* last = cache.rightFrontier.begin() + rangeEnd;

        C MeetState key{ {}, leftHash };
        C MeetState* it = std::lower_bound(first, last, key);

        while (it != last && it->hash == leftHash) {
            if (it->state == left[li].state) {
                middleOut = it->state;
                return true;
            }
            ++it;
        }
    }

    return false;
}

static bool findPathToMiddle(C Board& root,
                             C B1B2& target,
                             C u32 depth,
                             C bool reverseSide,
                             Memory& out) {
    JVec<Memory> frontier;
    Perms<Memory>::reserveForDepth(root, frontier, depth);
    Perms<Memory>::getDepthFunc<true>(root, frontier, depth, reverseSide);

    Board targetBoard;
    targetBoard.b1 = target.b1;
    targetBoard.b2 = target.b2;

    for (u64 i = 0; i < frontier.size(); ++i) {
        C Board candidate = reverseSide
                                    ? makeBoardWithMovesReverse(root, frontier[i])
                                    : makeBoardWithMoves(root, frontier[i]);

        if (candidate == targetBoard) {
            out = frontier[i];
            return true;
        }
    }

    return false;
}

static std::string recoverSolutionString(C Board& start,
                                         C Board& goal,
                                         C B1B2& middle,
                                         C u32 leftTotalDepth,
                                         C u32 rightDepth) {
    Memory leftPath;
    Memory rightPath;

    if (!findPathToMiddle(start, middle, leftTotalDepth, false, leftPath)) {
        return {};
    }
    if (!findPathToMiddle(goal, middle, rightDepth, true, rightPath)) {
        return {};
    }

    return leftPath.asmString(&rightPath);
}

static TrialTimings runSingleTrial(C Board& start,
                                   C Board& goal,
                                   C RightCache& cache,
                                   LeftLoopScratch& scratch,
                                   C u32 leftLoopDepth,
                                   C u32 leftDepth,
                                   C u32 rightDepth) {
    TrialTimings out;
    C Timer timerTotal;

    buildPrefixFrontier(start, scratch.leftPrefixes, leftLoopDepth);

    for (u64 prefixIdx = 0; prefixIdx < scratch.leftPrefixes.size(); ++prefixIdx) {
        ++out.attempts;

        scratch.generatedBoards.clear();
        C Timer timerLeftBuild;
        Perms<Board>::getDepthFunc<true>(scratch.leftPrefixes[prefixIdx],
                                         scratch.generatedBoards,
                                         leftDepth,
                                         false);
        out.leftBuildTime += timerLeftBuild.getSeconds();

        C Timer timerLeftCompress;
        compressBoardsToMeetStates(scratch.generatedBoards, scratch.leftFrontier);
        out.leftCompressTime += timerLeftCompress.getSeconds();

        C Timer timerMeet;
        if (tryMeet(scratch.leftFrontier, cache, out.middleState)) {
            out.meetTime += timerMeet.getSeconds();
            out.leftSize = scratch.leftFrontier.size();
            out.found = true;
            out.moves = recoverSolutionString(start, goal, out.middleState,
                                              leftLoopDepth + leftDepth,
                                              rightDepth);
            out.totalTime = timerTotal.getSeconds();
            return out;
        }
        out.meetTime += timerMeet.getSeconds();
    }

    out.leftSize = scratch.leftFrontier.size();
    out.totalTime = timerTotal.getSeconds();
    return out;
}

static BenchmarkResult runBenchmark(C std::string& levelName,
                                    C u32 leftLoopDepth,
                                    C u32 leftDepth,
                                    C u32 rightDepth,
                                    C u32 runCount) {
    BenchmarkResult result;
    C auto pair = BoardLookup::getBoardPair(levelName);
    if (pair == nullptr) {
        return result;
    }

    C Board start = pair->getStartState();
    C Board goal = pair->getEndState();

    result.rightCache = buildRightCache(start, goal, rightDepth);

    LeftLoopScratch scratch;
    scratch.init(start, leftLoopDepth, leftDepth);

    for (u32 run = 0; run < runCount; ++run) {
        C TrialTimings trial = runSingleTrial(start, goal, result.rightCache, scratch,
                                              leftLoopDepth, leftDepth, rightDepth);

        result.avgLeftBuildTime += trial.leftBuildTime;
        result.avgLeftCompressTime += trial.leftCompressTime;
        result.avgMeetTime += trial.meetTime;
        result.avgTotalTime += trial.totalTime;
        result.avgAttempts += static_cast<double>(trial.attempts);
        result.leftSize = trial.leftSize;

        if (trial.found) {
            result.solvedTrials++;
            result.sampleMiddle = trial.middleState;
            if (result.sampleMoves.empty()) {
                result.sampleMoves = trial.moves;
            }
        }
    }

    if (runCount > 0) {
        C double divisor = static_cast<double>(runCount);
        result.avgLeftBuildTime /= divisor;
        result.avgLeftCompressTime /= divisor;
        result.avgMeetTime /= divisor;
        result.avgTotalTime /= divisor;
        result.avgAttempts /= divisor;
    }

    return result;
}

int main() {
    constexpr u32 LEFT_LOOP = 2;
    constexpr u32 LEFT_DEPTH = 3;
    constexpr u32 RIGHT_DEPTH_LOCAL = 5;
    constexpr u32 RUN_COUNT = 1;

    Memory::setHashModeOverride(HASH_MODE);

    C std::string levelName = "5-5";
    C auto pair = BoardLookup::getBoardPair(levelName);
    if (pair == nullptr) {
        std::cout << "Level not found: " << levelName << std::endl;
        return -2;
    }

    C BenchmarkResult result = runBenchmark(levelName, LEFT_LOOP, LEFT_DEPTH, RIGHT_DEPTH_LOCAL, RUN_COUNT);

    std::cout << "Runs: " << RUN_COUNT << std::endl;
    std::cout << "Left loop depth: " << LEFT_LOOP << std::endl;
    std::cout << "Avg attempts: " << result.avgAttempts << std::endl;
    std::cout << "Left size (depth " << LEFT_DEPTH << "): " << result.leftSize << std::endl;
    std::cout << "Right size (depth " << RIGHT_DEPTH_LOCAL << "): " << result.rightCache.rightFrontier.size() << std::endl;
    std::cout << "Right build time (one-time): " << result.rightCache.rightBuildTime << std::endl;
    std::cout << "Right sort time (one-time): " << result.rightCache.rightSortTime << std::endl;
    std::cout << "Right compress time (one-time): " << result.rightCache.rightCompressTime << std::endl;
    std::cout << "Right index time (one-time): " << result.rightCache.rightIndexTime << std::endl;
    std::cout << "Avg left build time: " << result.avgLeftBuildTime << std::endl;
    std::cout << "Avg left compress time: " << result.avgLeftCompressTime << std::endl;
    std::cout << "Avg meet time: " << result.avgMeetTime << std::endl;
    std::cout << "Avg total time: " << result.avgTotalTime << std::endl;

    if (result.solvedTrials > 0) {
        std::cout << "Solved runs: " << result.solvedTrials << "/" << RUN_COUNT << std::endl;
        std::cout << "Sample moves: " << result.sampleMoves << std::endl;
        std::cout << "Found Sol: 0" << std::endl;
        return 0;
    }

    std::cout << "Found Sol: -1" << std::endl;
    return -1;
}