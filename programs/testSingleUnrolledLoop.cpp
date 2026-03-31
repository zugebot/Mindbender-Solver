#include "code/include.hpp"

struct HashRange {
    u64 hash;
    size_t begin;
    size_t end;
};

struct RightCache {
    JVec<Board> rightFrontier;
    std::vector<HashRange> rightHashRanges;
    double rightBuildTime = 0.0;
    double rightSortTime = 0.0;
};

struct TrialTimings {
    double leftBuildTime = 0.0;
    double meetTime = 0.0;
    double totalTime = 0.0;
    u32 attempts = 0;
    u64 leftSize = 0;
    bool found = false;
    std::string moves;
};

struct BenchmarkResult {
    RightCache rightCache;
    double avgLeftBuildTime = 0.0;
    double avgMeetTime = 0.0;
    double avgTotalTime = 0.0;
    double avgAttempts = 0.0;
    u64 leftSize = 0;
    u32 solvedTrials = 0;
    std::string sampleMoves;
};

struct LeftLoopScratch {
    JVec<Memory> leftFrontier;
    std::vector<Board> boardStack;
    std::vector<u8> prefixMoves;
    u32 prefixHitDepth = 0;

    void init(C Board& start, C u32 leftLoopDepth, C u32 leftDepth) {
        Perms<Memory>::reserveForDepth(start, leftFrontier, leftDepth);
        boardStack.resize(leftLoopDepth + 1);
        prefixMoves.resize(leftLoopDepth);
        prefixHitDepth = 0;
    }
};

template<typename F>
static bool forEachLeftPrefix(C Board& start,
                              C u32 leftLoopDepth,
                              LeftLoopScratch& scratch,
                              F&& callback) {
    if (scratch.boardStack.size() < leftLoopDepth + 1) {
        scratch.boardStack.resize(leftLoopDepth + 1);
    }
    if (scratch.prefixMoves.size() < leftLoopDepth) {
        scratch.prefixMoves.resize(leftLoopDepth);
    }
    scratch.boardStack[0] = start;

    const auto rec = [&](auto&& self, C u32 depth) -> bool {
        if (depth == leftLoopDepth) {
            return callback(scratch.boardStack[depth], scratch.prefixMoves, depth);
        }

        C Board& cur = scratch.boardStack[depth];
        Board& next = scratch.boardStack[depth + 1];
        for (int move = 0; move < 62; ++move) {
            if (move == 30 || move == 31) {
                continue;
            }

            next = cur;
            allActStructList[move].action(next);
            scratch.prefixMoves[depth] = static_cast<u8>(move);
            if (self(self, depth + 1)) {
                return true;
            }
        }
        return false;
    };

    return rec(rec, 0);
}

static RightCache buildRightCache(C Board& start, C Board& goal, C u32 rightDepth) {
    RightCache cache;

    std::cout << "[RC] Generating..." << std::endl;

    C Timer timerRightBuild;
    Perms<Board>::getDepthFunc<true>(goal, cache.rightFrontier, rightDepth, true);
    cache.rightBuildTime = timerRightBuild.getSeconds();

    std::cout << "[RC] Size: " << cache.rightFrontier.size() << std::endl;
    std::cout << "[RC] Sorting..." << std::endl;
    C Timer timerRightSort;
#ifdef BOOST_FOUND
    boost::sort::block_indirect_sort(cache.rightFrontier.begin(), cache.rightFrontier.end());
#else
    BoardSorter<Board> sorter;
    C u8 colorCount = start.getColorCount();
    sorter.sortBoards(cache.rightFrontier, rightDepth, colorCount);
#endif
    cache.rightSortTime = timerRightSort.getSeconds();

    std::cout << "[RC] Caching..." << std::endl;
    cache.rightHashRanges.reserve(cache.rightFrontier.size());
    for (size_t i = 0; i < cache.rightFrontier.size();) {
        C u64 hash = cache.rightFrontier[i].getHash();
        size_t j = i + 1;
        while (j < cache.rightFrontier.size() && cache.rightFrontier[j].getHash() == hash) {
            ++j;
        }
        cache.rightHashRanges.push_back({hash, i, j});
        i = j;
    }

    C size_t boardCount = cache.rightFrontier.size();
    C size_t bucketCount = cache.rightHashRanges.size();
    size_t maxBucketSize = 0;
    size_t singleBucketCount = 0;
    size_t multiBucketCount = 0;
    size_t collisionBoards = 0;

    for (C HashRange& range : cache.rightHashRanges) {
        C size_t bucketSize = range.end - range.begin;
        if (bucketSize > maxBucketSize) {
            maxBucketSize = bucketSize;
        }
        if (bucketSize == 1) {
            ++singleBucketCount;
        } else {
            ++multiBucketCount;
            collisionBoards += bucketSize - 1;
        }
    }

    C double avgBoardsPerBucket = bucketCount == 0
            ? 0.0
            : static_cast<double>(boardCount) / static_cast<double>(bucketCount);
    C double singletonPct = bucketCount == 0
            ? 0.0
            : (100.0 * static_cast<double>(singleBucketCount) / static_cast<double>(bucketCount));
    C double multiBucketPct = bucketCount == 0
            ? 0.0
            : (100.0 * static_cast<double>(multiBucketCount) / static_cast<double>(bucketCount));

    std::cout << "[RC] Bucket count: " << bucketCount << std::endl;
    std::cout << "[RC] Avg boards/bucket: " << avgBoardsPerBucket << std::endl;
    std::cout << "[RC] Max bucket size: " << maxBucketSize << std::endl;
    std::cout << "[RC] Single-board buckets: " << singleBucketCount << " (" << singletonPct << "%)" << std::endl;
    std::cout << "[RC] Multi-board buckets: " << multiBucketCount << " (" << multiBucketPct << "%)" << std::endl;
    std::cout << "[RC] Collision boards (non-first entries): " << collisionBoards << std::endl;


    return cache;
}

static bool tryMeet(C Board& leftRoot,
                    C JVec<Memory>& left,
                    C RightCache& cache,
                    C Memory*& leftOut,
                    C Memory*& rightOut) {
    for (size_t li = 0; li < left.size(); ++li) {
        C u64 leftHash = left[li].getHash();
        C auto it = std::lower_bound(
                cache.rightHashRanges.begin(), cache.rightHashRanges.end(), leftHash,
                [](C HashRange& range, u64 value) { return range.hash < value; });
        if (it == cache.rightHashRanges.end() || it->hash != leftHash) {
            continue;
        }

        C Board leftMid = makeBoardWithMoves(leftRoot, left[li]);
        for (size_t rj = it->begin; rj < it->end; ++rj) {
            C Board& rightMid = cache.rightFrontier[rj];
            if (leftMid == rightMid) {
                leftOut = &left[li];
                rightOut = &cache.rightFrontier[rj].getMemory();
                return true;
            }
        }
    }
    return false;
}

static TrialTimings runSingleTrial(C Board& start,
                                   C RightCache& cache,
                                   LeftLoopScratch& scratch,
                                   C u32 leftLoopDepth,
                                   C u32 leftDepth) {
    TrialTimings out;
    C Timer timerTotal;

    C Memory* leftHit = nullptr;
    C Memory* rightHit = nullptr;
    scratch.prefixHitDepth = 0;

    C bool found = forEachLeftPrefix(start, leftLoopDepth, scratch,
                                     [&](C Board& leftStart, C std::vector<u8>& prefixMoves, C u32 prefixDepth) {
        out.attempts++;

        C Timer timerLeftBuild;
        Perms<Memory>::getDepthFunc<true>(leftStart, scratch.leftFrontier, leftDepth, false);
        out.leftBuildTime += timerLeftBuild.getSeconds();

        C Timer timerMeet;
        if (tryMeet(leftStart, scratch.leftFrontier, cache, leftHit, rightHit)) {
            out.meetTime += timerMeet.getSeconds();
            scratch.prefixHitDepth = prefixDepth;
            return true;
        }
        out.meetTime += timerMeet.getSeconds();

        return false;
    });

    out.leftSize = scratch.leftFrontier.size();
    out.totalTime = timerTotal.getSeconds();

    if (found && leftHit != nullptr) {
        out.found = true;
        std::string prefixStr;
        for (u32 i = 0; i < scratch.prefixHitDepth; ++i) {
            if (i != 0) {
                prefixStr += " ";
            }
            prefixStr += Memory::formatMoveString(scratch.prefixMoves[i], true);
        }
        C std::string tail = leftHit->asmString(rightHit);
        if (prefixStr.empty()) {
            out.moves = tail;
        } else if (tail.empty()) {
            out.moves = prefixStr;
        } else {
            out.moves = prefixStr + " " + tail;
        }
    }

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
        C TrialTimings trial = runSingleTrial(start, result.rightCache, scratch, leftLoopDepth, leftDepth);
        result.avgLeftBuildTime += trial.leftBuildTime;
        result.avgMeetTime += trial.meetTime;
        result.avgTotalTime += trial.totalTime;
        result.avgAttempts += static_cast<double>(trial.attempts);
        result.leftSize = trial.leftSize;
        if (trial.found) {
            result.solvedTrials++;
            if (result.sampleMoves.empty()) {
                result.sampleMoves = trial.moves;
            }
        }
    }

    if (runCount > 0) {
        C double divisor = static_cast<double>(runCount);
        result.avgLeftBuildTime /= divisor;
        result.avgMeetTime /= divisor;
        result.avgTotalTime /= divisor;
        result.avgAttempts /= divisor;
    }

    return result;
}

int main() {
    constexpr u32 LEFT_LOOP = 0;
    constexpr u32 LEFT_DEPTH = 2;
    constexpr u32 RIGHT_DEPTH = 5;
    constexpr u32 RUN_COUNT = 1;

    C std::string levelName = "5-3";
    C auto pair = BoardLookup::getBoardPair(levelName);
    if (pair == nullptr) {
        std::cout << "Level not found: " << levelName << std::endl;
        return -2;
    }

    C BenchmarkResult result = runBenchmark(levelName, LEFT_LOOP, LEFT_DEPTH, RIGHT_DEPTH, RUN_COUNT);

    std::cout << "Runs: " << RUN_COUNT << std::endl;
    std::cout << "Left loop depth: " << LEFT_LOOP << std::endl;
    std::cout << "Avg attempts: " << result.avgAttempts << std::endl;
    std::cout << "Left size (depth " << LEFT_DEPTH << "): " << result.leftSize << std::endl;
    std::cout << "Right size (depth " << RIGHT_DEPTH << "): " << result.rightCache.rightFrontier.size() << std::endl;
    std::cout << "Right build time (one-time): " << result.rightCache.rightBuildTime << std::endl;
    std::cout << "Right sort time (one-time): " << result.rightCache.rightSortTime << std::endl;
    std::cout << "Right mid cache time (one-time): 0" << std::endl;
    std::cout << "Avg left build time: " << result.avgLeftBuildTime << std::endl;
    std::cout << "Avg left sort time: 0" << std::endl;
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