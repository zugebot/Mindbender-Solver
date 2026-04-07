#include "code/include.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct BenchPair {
    Board lhs;
    Board rhs;
};

struct KuhnSampleCase {
    std::size_t pairIndex = 0;
    Board lhs;
    Board rhs;
    i32 exactScore = 0;
    bool exactReject5 = false;
    bool oneMoveFast = false;
    bool oneMoveExact = false;
};

static Board makeBaseBoard() {
    return Board({
            2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2,
            2, 2, 6, 2, 2, 2,
            2, 2, 2, 6, 2, 2,
            2, 2, 2, 2, 6, 2,
            2, 2, 2, 2, 2, 2
    });
}

static u8 randomNormalMoveIndex(std::mt19937_64& rng) {
    static std::uniform_int_distribution<int> dist(0, 59);
    const int m = dist(rng);
    return static_cast<u8>(m < 30 ? m : m + 2);
}

static void applyRandomMoves(Board& board,
                             std::mt19937_64& rng,
                             int moveCount) {
    for (int i = 0; i < moveCount; ++i) {
        const u8 moveIndex = randomNormalMoveIndex(rng);
        allActStructList[moveIndex].action(board);
    }
}

static std::vector<BenchPair> generatePairs(std::size_t pairCount,
                                            int maxMovesPerSide,
                                            std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> lenDist(0, maxMovesPerSide);

    std::vector<BenchPair> pairs;
    pairs.reserve(pairCount);

    const Board base = makeBaseBoard();

    for (std::size_t i = 0; i < pairCount; ++i) {
        Board lhs = base;
        Board rhs = base;

        applyRandomMoves(lhs, rng, lenDist(rng));
        applyRandomMoves(rhs, rng, lenDist(rng));

        pairs.push_back({lhs, rhs});
    }

    return pairs;
}

static double percent(std::size_t part, std::size_t whole) {
    if (whole == 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(part) / static_cast<double>(whole);
}

static double ratePerSecond(std::size_t count, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(count) / seconds;
}

static std::string buildDiffMaskString(const Board& lhs, const Board& rhs) {
    std::string out;

    for (int row = 0; row < 6; ++row) {
        for (int col = 0; col < 6; ++col) {
            out.push_back(lhs.getColor(static_cast<u8>(col), static_cast<u8>(row))
                                          != rhs.getColor(static_cast<u8>(col), static_cast<u8>(row))
                                  ? 'X'
                                  : '.');

            if (col != 5) {
                out.push_back(' ');
            }
        }

        out.push_back('\n');
    }

    return out;
}

static bool canBeSolvedIn1MoveExact(const Board& lhs, const Board& rhs) {
    if (lhs == rhs) {
        return true;
    }

    for (u32 i = 0; i < TOTAL_ACT_STRUCT_COUNT; ++i) {
        const Action action = allActStructList[i].action;
        if (action == nullptr) {
            continue;
        }

        Board temp = lhs;
        action(temp);

        if (temp == rhs) {
            return true;
        }
    }

    return false;
}

static void printSampleList(const std::string& title,
                            const std::vector<KuhnSampleCase>& samples) {
    tcout << "\n================ " << title << " ================\n";

    if (samples.empty()) {
        tcout << "No samples collected.\n";
        return;
    }

    for (std::size_t i = 0; i < samples.size(); ++i) {
        const KuhnSampleCase& s = samples[i];

        tcout << "\nSample #" << (i + 1)
              << "  pairIndex=" << s.pairIndex << '\n';

        tcout << "exactScore=" << s.exactScore
              << "  exactReject5=" << (s.exactReject5 ? "true" : "false")
              << "  oneMoveFast=" << (s.oneMoveFast ? "true" : "false")
              << "  oneMoveExact=" << (s.oneMoveExact ? "true" : "false")
              << '\n';

        tcout << "diff mask (X=different, .=same):\n";
        tcout << buildDiffMaskString(s.lhs, s.rhs);

        tcout << "lhs:\n" << s.lhs.toBlandString();
        tcout << "rhs:\n" << s.rhs.toBlandString();
    }
}

int main() {
    constexpr std::size_t PAIR_COUNT = 1'000'000;
    constexpr int MAX_MOVES_PER_SIDE = 12;
    constexpr std::uint64_t RNG_SEED = 0xC0FFEEULL;
    constexpr std::size_t MAX_SAMPLES_PER_BUCKET = 8;

    Board hashBoard = makeBaseBoard();
    StateHash::refreshB1B2(hashBoard);
    StateHash::refreshBoard(hashBoard);
    StateHash::refreshMemory(hashBoard);

    tcout << std::fixed << std::setprecision(6);

    tcout << "Generating benchmark pairs...\n";
    Timer genTimer;
    std::vector<BenchPair> pairs = generatePairs(PAIR_COUNT, MAX_MOVES_PER_SIDE, RNG_SEED);
    const double genSeconds = genTimer.getSeconds();
    tcout << "Generated " << pairs.size() << " pairs in " << genSeconds << " sec\n";
    tcout << "Generation rate: " << ratePerSecond(pairs.size(), genSeconds) << " pairs/sec\n\n";

    double exactScoreSeconds = 0.0;
    double exactRejectSeconds = 0.0;
    double oneMoveFastSeconds = 0.0;
    double oneMoveExactSeconds = 0.0;
    double combinedSeconds = 0.0;

    std::uint64_t exactScoreSum = 0;
    std::size_t reject5Count = 0;

    std::size_t sameOneMove = 0;
    std::size_t fastTrueExactFalse = 0;
    std::size_t exactTrueFastFalse = 0;

    std::vector<KuhnSampleCase> oneMoveAgreeTrueSamples;
    std::vector<KuhnSampleCase> oneMoveFastFalsePositiveSamples;
    std::vector<KuhnSampleCase> oneMoveFastFalseNegativeSamples;
    std::vector<KuhnSampleCase> exactScoreLe3Samples;
    std::vector<KuhnSampleCase> exactScoreGe6Samples;

    {
        Timer timer;
        volatile std::uint64_t sink = 0;

        for (const BenchPair& p : pairs) {
            sink += static_cast<std::uint64_t>(p.lhs.getExactRowColLowerBound(p.rhs));
        }

        exactScoreSeconds = timer.getSeconds();
        tcout << "Kuhn exact lower bound time: " << exactScoreSeconds << " sec\n";
        tcout << "Kuhn exact lower bound rate: "
              << ratePerSecond(pairs.size(), exactScoreSeconds) << " evals/sec\n";
        tcout << "Kuhn exact lower bound sink: " << sink << "\n\n";
    }

    {
        Timer timer;
        volatile std::uint64_t sink = 0;

        for (const BenchPair& p : pairs) {
            sink += static_cast<std::uint64_t>(p.lhs.getExactRowColLowerBoundTill<5>(p.rhs));
        }

        exactRejectSeconds = timer.getSeconds();
        tcout << "Kuhn reject>5 time: " << exactRejectSeconds << " sec\n";
        tcout << "Kuhn reject>5 rate: "
              << ratePerSecond(pairs.size(), exactRejectSeconds) << " evals/sec\n";
        tcout << "Kuhn reject>5 sink: " << sink << "\n\n";
    }

    {
        Timer timer;
        volatile std::uint64_t sink = 0;

        for (const BenchPair& p : pairs) {
            sink += static_cast<std::uint64_t>(p.lhs.couldBeSolvedIn1Move(p.rhs));
        }

        oneMoveFastSeconds = timer.getSeconds();
        tcout << "Fast couldBeSolvedIn1Move time: " << oneMoveFastSeconds << " sec\n";
        tcout << "Fast couldBeSolvedIn1Move rate: "
              << ratePerSecond(pairs.size(), oneMoveFastSeconds) << " evals/sec\n";
        tcout << "Fast couldBeSolvedIn1Move sink: " << sink << "\n\n";
    }

    {
        Timer timer;
        volatile std::uint64_t sink = 0;

        for (const BenchPair& p : pairs) {
            sink += static_cast<std::uint64_t>(canBeSolvedIn1MoveExact(p.lhs, p.rhs));
        }

        oneMoveExactSeconds = timer.getSeconds();
        tcout << "Exact one-move check time: " << oneMoveExactSeconds << " sec\n";
        tcout << "Exact one-move check rate: "
              << ratePerSecond(pairs.size(), oneMoveExactSeconds) << " evals/sec\n";
        tcout << "Exact one-move check sink: " << sink << "\n\n";
    }

    {
        Timer timer;
        volatile std::uint64_t sink = 0;

        for (std::size_t i = 0; i < pairs.size(); ++i) {
            const BenchPair& p = pairs[i];

            const i32 exactScore = p.lhs.getExactRowColLowerBound(p.rhs);
            const bool exactReject5 = p.lhs.getExactRowColLowerBoundTill<5>(p.rhs);
            const bool oneMoveFast = p.lhs.couldBeSolvedIn1Move(p.rhs);
            const bool oneMoveExact = canBeSolvedIn1MoveExact(p.lhs, p.rhs);

            exactScoreSum += static_cast<std::uint64_t>(exactScore);
            reject5Count += static_cast<std::size_t>(exactReject5);

            KuhnSampleCase sample{
                    i,
                    p.lhs,
                    p.rhs,
                    exactScore,
                    exactReject5,
                    oneMoveFast,
                    oneMoveExact
            };

            if (oneMoveFast == oneMoveExact) {
                ++sameOneMove;

                if (oneMoveFast && oneMoveAgreeTrueSamples.size() < MAX_SAMPLES_PER_BUCKET) {
                    oneMoveAgreeTrueSamples.push_back(sample);
                }
            } else if (oneMoveFast && !oneMoveExact) {
                ++fastTrueExactFalse;

                if (oneMoveFastFalsePositiveSamples.size() < MAX_SAMPLES_PER_BUCKET) {
                    oneMoveFastFalsePositiveSamples.push_back(sample);
                }
            } else {
                ++exactTrueFastFalse;

                if (oneMoveFastFalseNegativeSamples.size() < MAX_SAMPLES_PER_BUCKET) {
                    oneMoveFastFalseNegativeSamples.push_back(sample);
                }
            }

            if (exactScore <= 3 && exactScoreLe3Samples.size() < MAX_SAMPLES_PER_BUCKET) {
                exactScoreLe3Samples.push_back(sample);
            }

            if (exactScore >= 6 && exactScoreGe6Samples.size() < MAX_SAMPLES_PER_BUCKET) {
                exactScoreGe6Samples.push_back(sample);
            }

            sink += static_cast<std::uint64_t>(exactScore);
            sink += static_cast<std::uint64_t>(exactReject5);
            sink += static_cast<std::uint64_t>(oneMoveFast);
            sink += static_cast<std::uint64_t>(oneMoveExact);
        }

        combinedSeconds = timer.getSeconds();
        tcout << "Combined analysis loop time: " << combinedSeconds << " sec\n";
        tcout << "Combined analysis rate: "
              << ratePerSecond(pairs.size(), combinedSeconds) << " pairs/sec\n";
        tcout << "Combined analysis sink: " << sink << "\n\n";
    }

    tcout << "================ SPEED SUMMARY ================\n";
    tcout << "pair count: " << pairs.size() << '\n';
    tcout << "kuhn exact score time:   " << exactScoreSeconds << " sec\n";
    tcout << "kuhn reject>5 time:      " << exactRejectSeconds << " sec\n";
    tcout << "fast one-move time:      " << oneMoveFastSeconds << " sec\n";
    tcout << "exact one-move time:     " << oneMoveExactSeconds << " sec\n";

    if (oneMoveFastSeconds < oneMoveExactSeconds) {
        tcout << "fast one-move is faster by "
              << (oneMoveExactSeconds / oneMoveFastSeconds) << "x\n";
    } else {
        tcout << "exact one-move is faster by "
              << (oneMoveFastSeconds / oneMoveExactSeconds) << "x\n";
    }

    tcout << "\n================ KUHN SUMMARY ================\n";
    tcout << "avg exact score: "
          << (static_cast<double>(exactScoreSum) / static_cast<double>(pairs.size())) << '\n';
    tcout << "reject>5 count: " << reject5Count
          << " (" << percent(reject5Count, pairs.size()) << "%)\n";

    tcout << "\n================ ONE-MOVE CHECK AGREEMENT ================\n";
    tcout << "same result: " << sameOneMove
          << " (" << percent(sameOneMove, pairs.size()) << "%)\n";
    tcout << "fast=true, exact=false: " << fastTrueExactFalse
          << " (" << percent(fastTrueExactFalse, pairs.size()) << "%)\n";
    tcout << "fast=false, exact=true: " << exactTrueFastFalse
          << " (" << percent(exactTrueFastFalse, pairs.size()) << "%)\n";

    if (fastTrueExactFalse == 0 && exactTrueFastFalse == 0) {
        tcout << "Fast one-move check matched exact one-move check on all tested pairs.\n";
    }

    printSampleList("SAMPLES: oneMoveFast && oneMoveExact", oneMoveAgreeTrueSamples);
    printSampleList("SAMPLES: oneMoveFast && !oneMoveExact", oneMoveFastFalsePositiveSamples);
    printSampleList("SAMPLES: !oneMoveFast && oneMoveExact", oneMoveFastFalseNegativeSamples);
    printSampleList("SAMPLES: exactScore <= 3", exactScoreLe3Samples);
    printSampleList("SAMPLES: exactScore >= 6", exactScoreGe6Samples);

    return 0;
}