#pragma once
// src/solver/bounds/calibrate.hpp
//
// Picks this run's fastest pruning mode by timing every eligible mode on real
// seeds. Their work counts toward the search (nothing is redone).
//
// Each block runs ONE mode on every thread, the way the run will use it:
// timing the modes seed by seed side by side let their cache and memory
// traffic disturb each other (16-4: pdbparent timed at 1.0-1.1x vs 1.5x in
// full runs). A round takes the next chunk of seed positions and gives each
// mode every k-th of them (k = modes still timed), one mode after another in a
// rotating order. Neighbouring positions cost about the same, so the modes are
// compared round by round (log of the time ratio), which cancels both
// seed-to-seed cost and changes in machine load. After MIN_ROUNDS a mode
// clearly slower than the best is dropped; timing goes on until the rest are
// told apart (2 standard errors) or known to ~1%, at least MIN_SECONDS, within
// a budget of 3% of the pass's estimated search time (20 s to maxSeconds).
//
// The positions should come from a visiting order spread over the whole seed
// list (see the probe's seedAt): timing only the first 1% of a sorted list
// rated 16-4's pdbparent 1.11x vs off where full runs show 1.45-1.5x.

#include "src/utils/parallel_for.hpp"
#include "src/utils/timer.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace bounds {

    constexpr std::size_t CALIBRATION_BLOCK_PER_THREAD = 32;   // seeds per thread in one block
    constexpr int CALIBRATION_MIN_ROUNDS = 4;
    constexpr double CALIBRATION_MIN_SECONDS = 5.0;

    // Positions a run needs left for a calibration to be worth it.
    inline std::size_t calibrationPositionsNeeded(const u32 threads, const std::size_t modes) {
        return static_cast<std::size_t>(threads) * CALIBRATION_BLOCK_PER_THREAD * modes * CALIBRATION_MIN_ROUNDS * 4;
    }

    struct Calibration {
        std::size_t best = 0;   // index into the modes; 0 is the baseline (no pruning)
        std::string report;     // the log line's detail
    };

    // names[0] must be the baseline mode. runSeed(thread, position, modeIndex)
    // processes the seed at that visiting position in that mode. Positions
    // [firstPos, limitPos) are used in order, each at most once, and firstPos
    // is advanced past the ones used.
    template<typename RunSeed>
    Calibration calibrateModes(const std::vector<const char*>& names, const u32 threads, std::size_t& firstPos,
                               const std::size_t limitPos, const double seedsInPass, const double maxSeconds,
                               RunSeed&& runSeed) {
        const std::size_t nM = names.size();
        const std::size_t block = static_cast<std::size_t>(threads) * CALIBRATION_BLOCK_PER_THREAD;
        // perRound[m][r]: mean ns per seed of mode m in round r (< 0: not timed)
        std::vector<std::vector<double>> perRound(nM);
        std::vector<int> droppedAt(nM, 0);
        auto alive = [&](const std::size_t m) { return droppedAt[m] == 0; };
        // mean and standard error of log(time m / time b) over rounds timing both
        auto logRatio = [&](const std::size_t m, const std::size_t b) {
            std::vector<double> d;
            for (std::size_t r = 0; r < perRound[m].size() && r < perRound[b].size(); ++r) {
                if (perRound[m][r] > 0.0 && perRound[b][r] > 0.0) {
                    d.push_back(std::log(perRound[m][r] / perRound[b][r]));
                }
            }
            double mu = 0.0, ss = 0.0;
            for (const double v : d) {
                mu += v;
            }
            mu /= static_cast<double>(std::max<std::size_t>(1, d.size()));
            for (const double v : d) {
                ss += (v - mu) * (v - mu);
            }
            const double n = static_cast<double>(d.size());
            return std::pair<double, double>{mu, d.size() < 2 ? 1e300 : std::sqrt(ss / (n - 1.0) / n)};
        };
        auto meanOf = [&](const std::size_t m) {
            double s = 0.0;
            std::size_t n = 0;
            for (const double v : perRound[m]) {
                if (v > 0.0) {
                    s += v;
                    ++n;
                }
            }
            return s / static_cast<double>(std::max<std::size_t>(1, n));
        };
        const Timer timer;
        double budget = maxSeconds;
        int rounds = 0;
        std::size_t timedSeeds = 0;
        const char* why = "budget";
        std::size_t best = 0;
        while (true) {
            std::vector<std::size_t> live;
            for (std::size_t m = 0; m < nM; ++m) {
                if (alive(m)) {
                    live.push_back(m);
                }
            }
            const std::size_t k = live.size();
            if (firstPos + block * k > limitPos) {
                why = "seed limit";
                break;
            }
            const std::size_t base = firstPos;
            for (std::size_t m = 0; m < nM; ++m) {
                perRound[m].push_back(-1.0);
            }
            for (std::size_t j = 0; j < k; ++j) {
                const std::size_t slot = (j + static_cast<std::size_t>(rounds)) % k;
                const std::size_t m = live[slot];
                std::atomic<u64> nanos = 0;
                parallelFor(threads, block, [&](const u32 t, const std::size_t i) {
                    const auto t0 = std::chrono::steady_clock::now();
                    runSeed(t, base + i * k + slot, m);
                    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - t0).count();
                    nanos.fetch_add(static_cast<u64>(ns), std::memory_order_relaxed);
                });
                perRound[m].back() = static_cast<double>(nanos.load()) / static_cast<double>(block);
            }
            firstPos += block * k;
            timedSeeds += block * k;
            ++rounds;

            // best = the lowest geometric-mean time among the modes still timed
            best = live[0];
            for (const std::size_t m : live) {
                if (m != best && logRatio(m, best).first < 0.0) {
                    best = m;
                }
            }
            if (rounds < CALIBRATION_MIN_ROUNDS) {
                continue;
            }
            bool settled = true;
            std::size_t others = 0;
            for (const std::size_t m : live) {
                if (m == best) {
                    continue;
                }
                const auto [mu, se] = logRatio(m, best);
                if (mu > std::log(1.05) && mu > 3.0 * se) {
                    droppedAt[m] = rounds;
                    continue;
                }
                ++others;
                if (mu <= 2.0 * se && se > 0.01) {
                    settled = false;   // not told apart yet, and not known to ~1%
                }
            }
            const double passEstimate = meanOf(best) * seedsInPass / threads / 1e9;
            budget = std::min(maxSeconds, std::max(20.0, 0.03 * passEstimate));
            if (timer.getSeconds() < CALIBRATION_MIN_SECONDS) {
                continue;
            }
            if (others == 0) {
                why = "one mode left";
                break;
            }
            if (settled) {
                why = "settled";
                break;
            }
            if (timer.getSeconds() >= budget) {
                break;
            }
        }
        // the baseline's time relative to the best, from the rounds that timed both
        const double baseOverBest = best == 0 ? 1.0 : std::exp(logRatio(0, best).first);
        const std::size_t fastest = best;
        if (best != 0 && baseOverBest < 1.03) {
            best = 0;   // under 3%: not worth it
        }
        std::ostringstream out;
        out << timedSeeds << " seeds, " << rounds << " rounds of " << block << "-seed blocks per mode in "
            << timer.getSeconds() << "s (" << why << ", budget " << budget << "s); time vs " << names[fastest] << ":";
        for (std::size_t m = 0; m < nM; ++m) {
            const auto [mu, se] = logRatio(m, fastest);
            out << ' ' << names[m] << '=';
            if (m == fastest) {
                out << "1 (" << meanOf(m) / 1000.0 << " us/seed)";
            } else {
                out << std::exp(mu) << " +-" << 100.0 * se << '%';
            }
            if (!alive(m)) {
                out << " (dropped after " << droppedAt[m] << ")";
            }
        }
        out << " -> " << names[best] << " (" << baseOverBest << "x vs " << names[0] << ")";
        return {best, out.str()};
    }

} // namespace bounds
