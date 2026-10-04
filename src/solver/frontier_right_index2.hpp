#pragma once
// src/solver/frontier_right_index2.hpp

#include "src/solver/frontier_builder.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>

// Exact, memory-lean right-frontier index.
//
// The input lanes are already sorted by (hash, state).  Keep those lanes as
// the sole source of truth.  Presence filters and a prefix directory are only
// rejection accelerators; every hit is validated against the full B1B2 state.
class RightFrontierIndexB1B2 {
public:
    struct BucketStats {
        std::size_t stateCount = 0;
        std::size_t bucketCount = 0;

        u32 minBucketSize = 0;
        u32 maxBucketSize = 0;
        double avgBucketSize = 0.0;

        std::size_t singletonBucketCount = 0;
        std::size_t multiBucketCount = 0;
        std::size_t statesInSingletonBuckets = 0;
        std::size_t statesInMultiBuckets = 0;
        u64 collisionPairs = 0;

        u32 p50BucketSize = 0;
        u32 p90BucketSize = 0;
        u32 p99BucketSize = 0;

        u64 bucketsSize1 = 0;
        u64 bucketsSize2 = 0;
        u64 bucketsSize3 = 0;
        u64 bucketsSize4 = 0;
        u64 bucketsSize5to8 = 0;
        u64 bucketsSize9to16 = 0;
        u64 bucketsSize17to32 = 0;
        u64 bucketsSize33to64 = 0;
        u64 bucketsSize65plus = 0;

        u64 statesSize1 = 0;
        u64 statesSize2 = 0;
        u64 statesSize3 = 0;
        u64 statesSize4 = 0;
        u64 statesSize5to8 = 0;
        u64 statesSize9to16 = 0;
        u64 statesSize17to32 = 0;
        u64 statesSize33to64 = 0;
        u64 statesSize65plus = 0;
    };

    struct ProbeStats {
        u64 leftStateCount = 0;
        u64 hashHits = 0;
        u64 hashMisses = 0;
        u64 bucketsVisited = 0;
        u64 bucketStatesScanned = 0;
        u64 equalityChecks = 0;
        u64 exactMatches = 0;

        u64 highFilterRejects = 0;
        u64 midFilterRejects = 0;
        u64 lowFilterRejects = 0;
        u64 extraFilterRejects = 0;
        u64 prefixRejects = 0;
        u64 filterPasses = 0;
    };

private:
    static constexpr u32 HASH_PREFIX_BITS = 23;
    static constexpr std::size_t HASH_PREFIX_BUCKET_COUNT =
            static_cast<std::size_t>(1) << HASH_PREFIX_BITS;
    static constexpr u32 HASH_PREFIX_SHIFT = 64U - HASH_PREFIX_BITS;

    static constexpr u32 HASH_PRESENCE_HIGH_BITS = 29;
    static constexpr std::size_t HASH_PRESENCE_HIGH_BUCKET_COUNT =
            static_cast<std::size_t>(1) << HASH_PRESENCE_HIGH_BITS;
    static constexpr std::size_t HASH_PRESENCE_HIGH_WORD_COUNT =
            HASH_PRESENCE_HIGH_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_HIGH_SHIFT = 64U - HASH_PRESENCE_HIGH_BITS;

    static constexpr u32 HASH_PRESENCE_MID_BITS = 28;
    static constexpr std::size_t HASH_PRESENCE_MID_BUCKET_COUNT =
            static_cast<std::size_t>(1) << HASH_PRESENCE_MID_BITS;
    static constexpr std::size_t HASH_PRESENCE_MID_WORD_COUNT =
            HASH_PRESENCE_MID_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_MID_SHIFT =
            (64U - HASH_PRESENCE_MID_BITS) / 2U;
    static constexpr u64 HASH_PRESENCE_MID_MASK =
            static_cast<u64>(HASH_PRESENCE_MID_BUCKET_COUNT - 1ULL);

    static constexpr u32 HASH_PRESENCE_LOW_BITS = 27;
    static constexpr std::size_t HASH_PRESENCE_LOW_BUCKET_COUNT =
            static_cast<std::size_t>(1) << HASH_PRESENCE_LOW_BITS;
    static constexpr std::size_t HASH_PRESENCE_LOW_WORD_COUNT =
            HASH_PRESENCE_LOW_BUCKET_COUNT / 64ULL;
    static constexpr u64 HASH_PRESENCE_LOW_MASK =
            static_cast<u64>(HASH_PRESENCE_LOW_BUCKET_COUNT - 1ULL);

    static constexpr u32 HASH_PRESENCE_EXTRA_BITS = 26;
    static constexpr std::size_t HASH_PRESENCE_EXTRA_BUCKET_COUNT =
            static_cast<std::size_t>(1) << HASH_PRESENCE_EXTRA_BITS;
    static constexpr std::size_t HASH_PRESENCE_EXTRA_WORD_COUNT =
            HASH_PRESENCE_EXTRA_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_EXTRA_SHIFT = 13U;
    static constexpr u64 HASH_PRESENCE_EXTRA_MASK =
            static_cast<u64>(HASH_PRESENCE_EXTRA_BUCKET_COUNT - 1ULL);

    JVec<B1B2> states_{};
    JVec<u64> hashes_{};
    JVec<u32> prefixStarts_{};
    JVec<u64> highPresenceWords_{};
    JVec<u64> midPresenceWords_{};
    JVec<u64> lowPresenceWords_{};
    JVec<u64> extraPresenceWords_{};
    BucketStats stats_{};

    template<typename T>
    MUND static u64 liveBytes(const JVec<T>& lane) {
        return static_cast<u64>(lane.size()) * static_cast<u64>(sizeof(T));
    }

    template<typename T>
    MUND static u64 reservedBytes(const JVec<T>& lane) {
        return static_cast<u64>(lane.capacity()) * static_cast<u64>(sizeof(T));
    }

    MUND static u32 hashPrefix(const u64 hash) {
        return static_cast<u32>(hash >> HASH_PREFIX_SHIFT);
    }

    MUND static u32 highBucket(const u64 hash) {
        return static_cast<u32>(hash >> HASH_PRESENCE_HIGH_SHIFT);
    }

    MUND static u32 midBucket(const u64 hash) {
        return static_cast<u32>((hash >> HASH_PRESENCE_MID_SHIFT) & HASH_PRESENCE_MID_MASK);
    }

    MUND static u32 lowBucket(const u64 hash) {
        return static_cast<u32>(hash & HASH_PRESENCE_LOW_MASK);
    }

    MUND static u32 extraBucket(const u64 hash) {
        return static_cast<u32>((hash >> HASH_PRESENCE_EXTRA_SHIFT) & HASH_PRESENCE_EXTRA_MASK);
    }

    MU static void setBit(JVec<u64>& words, const u32 bucket) {
        words[static_cast<std::size_t>(bucket >> 6U)] |=
                1ULL << static_cast<u32>(bucket & 63U);
    }

    MUND static bool hasBit(const JVec<u64>& words, const u32 bucket) {
        return (words[static_cast<std::size_t>(bucket >> 6U)] &
                (1ULL << static_cast<u32>(bucket & 63U))) != 0ULL;
    }

    MU void buildPrefixStarts() {
        prefixStarts_.clear();
        prefixStarts_.resize(HASH_PREFIX_BUCKET_COUNT + 1ULL);

        std::size_t index = 0;
        for (u32 prefix = 0; prefix < HASH_PREFIX_BUCKET_COUNT; ++prefix) {
            prefixStarts_[prefix] = static_cast<u32>(index);
            while (index < hashes_.size() && hashPrefix(hashes_[index]) == prefix) {
                ++index;
            }
        }
        prefixStarts_[HASH_PREFIX_BUCKET_COUNT] = static_cast<u32>(hashes_.size());
    }

    MU void buildPresenceFilters() {
        highPresenceWords_.clear();
        midPresenceWords_.clear();
        lowPresenceWords_.clear();
        extraPresenceWords_.clear();

        highPresenceWords_.resize(HASH_PRESENCE_HIGH_WORD_COUNT);
        midPresenceWords_.resize(HASH_PRESENCE_MID_WORD_COUNT);
        lowPresenceWords_.resize(HASH_PRESENCE_LOW_WORD_COUNT);
        extraPresenceWords_.resize(HASH_PRESENCE_EXTRA_WORD_COUNT);

        std::fill(highPresenceWords_.begin(), highPresenceWords_.end(), 0ULL);
        std::fill(midPresenceWords_.begin(), midPresenceWords_.end(), 0ULL);
        std::fill(lowPresenceWords_.begin(), lowPresenceWords_.end(), 0ULL);
        std::fill(extraPresenceWords_.begin(), extraPresenceWords_.end(), 0ULL);

        for (const u64 hash : hashes_) {
            setBit(highPresenceWords_, highBucket(hash));
            setBit(midPresenceWords_, midBucket(hash));
            setBit(lowPresenceWords_, lowBucket(hash));
            setBit(extraPresenceWords_, extraBucket(hash));
        }
    }

    MU static void addHistogram(const u32 count, BucketStats& stats) {
        if (count == 1) {
            ++stats.bucketsSize1; stats.statesSize1 += 1;
        } else if (count == 2) {
            ++stats.bucketsSize2; stats.statesSize2 += 2;
        } else if (count == 3) {
            ++stats.bucketsSize3; stats.statesSize3 += 3;
        } else if (count == 4) {
            ++stats.bucketsSize4; stats.statesSize4 += 4;
        } else if (count <= 8) {
            ++stats.bucketsSize5to8; stats.statesSize5to8 += count;
        } else if (count <= 16) {
            ++stats.bucketsSize9to16; stats.statesSize9to16 += count;
        } else if (count <= 32) {
            ++stats.bucketsSize17to32; stats.statesSize17to32 += count;
        } else if (count <= 64) {
            ++stats.bucketsSize33to64; stats.statesSize33to64 += count;
        } else {
            ++stats.bucketsSize65plus; stats.statesSize65plus += count;
        }
    }

    MU void buildStats() {
        stats_ = {};
        stats_.stateCount = states_.size();
        if (hashes_.empty()) {
            return;
        }

        u64 total = 0;
        std::size_t begin = 0;
        while (begin < hashes_.size()) {
            std::size_t end = begin + 1;
            while (end < hashes_.size() && hashes_[end] == hashes_[begin]) {
                ++end;
            }

            const u32 count = static_cast<u32>(end - begin);
            ++stats_.bucketCount;
            total += count;

            if (stats_.bucketCount == 1) {
                stats_.minBucketSize = count;
                stats_.maxBucketSize = count;
            } else {
                stats_.minBucketSize = std::min(stats_.minBucketSize, count);
                stats_.maxBucketSize = std::max(stats_.maxBucketSize, count);
            }

            if (count == 1) {
                ++stats_.singletonBucketCount;
                ++stats_.statesInSingletonBuckets;
            } else {
                ++stats_.multiBucketCount;
                stats_.statesInMultiBuckets += count;
                stats_.collisionPairs +=
                        (static_cast<u64>(count) * static_cast<u64>(count - 1U)) / 2ULL;
            }

            addHistogram(count, stats_);
            begin = end;
        }

        stats_.avgBucketSize = static_cast<double>(total) /
                               static_cast<double>(stats_.bucketCount);

        // Do not allocate one u32 per hash bucket merely to report percentiles.
        // Hash collisions are diagnostic only; correctness always comes from
        // the subsequent full-state equality check.
        stats_.p50BucketSize = stats_.minBucketSize;
        stats_.p90BucketSize = stats_.maxBucketSize == 1 ? 1 : stats_.minBucketSize;
        stats_.p99BucketSize = stats_.maxBucketSize == 1 ? 1 : stats_.maxBucketSize;
    }

public:
    MU void clear() {
        JVec<B1B2>{}.swap(states_);
        JVec<u64>{}.swap(hashes_);
        JVec<u32>{}.swap(prefixStarts_);
        JVec<u64>{}.swap(highPresenceWords_);
        JVec<u64>{}.swap(midPresenceWords_);
        JVec<u64>{}.swap(lowPresenceWords_);
        JVec<u64>{}.swap(extraPresenceWords_);
        stats_ = {};
    }

    template<bool BUILD_BUCKET_STATS = true>
    MU void buildFromUniqueStates(JVec<B1B2>&& states,
                                  JVec<u64>&& hashes,
                                  const Board&) {
        clear();
        if (states.size() != hashes.size()) {
            throw std::runtime_error("right-frontier state/hash lane size mismatch");
        }
        if (hashes.size() > static_cast<std::size_t>(std::numeric_limits<u32>::max())) {
            throw std::runtime_error("right frontier exceeds 32-bit prefix index capacity");
        }
        if (!std::is_sorted(hashes.begin(), hashes.end())) {
            throw std::runtime_error("right-frontier hashes must be sorted");
        }

        states_.swap(states);
        hashes_.swap(hashes);
        buildPrefixStarts();
        buildPresenceFilters();
        if constexpr (BUILD_BUCKET_STATS) {
            buildStats();
        } else {
            stats_.stateCount = states_.size();
            std::size_t buckets = 0;
            for (std::size_t i = 0; i < hashes_.size();) {
                ++buckets;
                const u64 hash = hashes_[i++];
                while (i < hashes_.size() && hashes_[i] == hash) ++i;
            }
            stats_.bucketCount = buckets;
        }
    }

    MUND std::size_t size() const { return states_.size(); }
    MUND std::size_t rangeCount() const { return stats_.bucketCount; }
    MUND const BucketStats& stats() const { return stats_; }

    MU void printStats() const {
        const u64 statesLive = liveBytes(states_);
        const u64 hashesLive = liveBytes(hashes_);
        const u64 prefixLive = liveBytes(prefixStarts_);
        const u64 filtersLive = liveBytes(highPresenceWords_) + liveBytes(midPresenceWords_)
                              + liveBytes(lowPresenceWords_) + liveBytes(extraPresenceWords_);
        const u64 totalLive = statesLive + hashesLive + prefixLive + filtersLive;

        tcout << "right frontier stats (exact-state):\n";
        tcout << "    total states: " << states_.size() << '\n';
        tcout << "    unique primary hashes: " << stats_.bucketCount << '\n';
        tcout << "    exact state validation: always\n";
        tcout << "    memory states: " << bytesFormatted<1000>(statesLive) << '\n';
        tcout << "    memory hashes: " << bytesFormatted<1000>(hashesLive) << '\n';
        tcout << "    memory prefix: " << bytesFormatted<1000>(prefixLive) << '\n';
        tcout << "    memory filters: " << bytesFormatted<1000>(filtersLive) << '\n';
        tcout << "    memory total: " << bytesFormatted<1000>(totalLive) << '\n';
        if (stats_.bucketCount != 0) {
            tcout << "    min bucket size: " << stats_.minBucketSize << '\n';
            tcout << "    max bucket size: " << stats_.maxBucketSize << '\n';
            tcout << "    avg bucket size: " << stats_.avgBucketSize << '\n';
            tcout << "    primary-hash collision buckets: " << stats_.multiBucketCount << '\n';
        }
    }

    template<int RIGHT_FRONTIER_DEPTH, bool COLLECT_PROBE_STATS = true>
    MU void collectMatches(const JVec<B1B2>& leftStates,
                           const JVec<u64>& leftHashes,
                           JVec<B1B2>& outUniqueMatches,
                           JVec<u64>& outUniqueMatchHashes,
                           ProbeStats* probeStats = nullptr) const {
        (void) RIGHT_FRONTIER_DEPTH;
        outUniqueMatches.clear();
        outUniqueMatchHashes.clear();

        if (leftStates.size() != leftHashes.size()) {
            throw std::runtime_error("left probe state/hash lane size mismatch");
        }
        if constexpr (COLLECT_PROBE_STATS) {
            if (probeStats == nullptr) {
                throw std::runtime_error("probe statistics requested with null destination");
            }
            *probeStats = {};
            probeStats->leftStateCount = static_cast<u64>(leftStates.size());
        }
        if (leftStates.empty() || states_.empty()) {
            return;
        }

        frontier_recovery_detail::reserveStateHashLanes(
                outUniqueMatches,
                outUniqueMatchHashes,
                std::min<std::size_t>(leftStates.size(), 256ULL));

        std::size_t writeIndex = 0;
        for (std::size_t lhsIndex = 0; lhsIndex < leftStates.size(); ++lhsIndex) {
            const B1B2& lhs = leftStates[lhsIndex];
            const u64 lhsHash = leftHashes[lhsIndex];

            if (!hasBit(highPresenceWords_, highBucket(lhsHash))) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->highFilterRejects;
                }
                continue;
            }
            if (!hasBit(midPresenceWords_, midBucket(lhsHash))) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->midFilterRejects;
                }
                continue;
            }
            if (!hasBit(lowPresenceWords_, lowBucket(lhsHash))) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->lowFilterRejects;
                }
                continue;
            }
            if (!hasBit(extraPresenceWords_, extraBucket(lhsHash))) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->extraFilterRejects;
                }
                continue;
            }
            if constexpr (COLLECT_PROBE_STATS) {
                ++probeStats->filterPasses;
            }

            const u32 prefix = hashPrefix(lhsHash);
            const u32 begin = prefixStarts_[prefix];
            const u32 end = prefixStarts_[prefix + 1U];
            if (begin == end) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->prefixRejects;
                }
                continue;
            }

            const u64* hashBegin = hashes_.data() + begin;
            const u64* hashEnd = hashes_.data() + end;
            const u64* found = std::lower_bound(hashBegin, hashEnd, lhsHash);
            if (found == hashEnd || *found != lhsHash) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses; ++probeStats->prefixRejects;
                }
                continue;
            }

            if constexpr (COLLECT_PROBE_STATS) {
                ++probeStats->hashHits;
                ++probeStats->bucketsVisited;
            }

            std::size_t stateIndex = begin + static_cast<std::size_t>(found - hashBegin);
            bool exact = false;
            while (stateIndex < end && hashes_[stateIndex] == lhsHash) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->bucketStatesScanned;
                    ++probeStats->equalityChecks;
                }
                if (states_[stateIndex] == lhs) {
                    exact = true;
                    break;
                }
                ++stateIndex;
            }

            if (!exact) {
                continue;
            }

            frontier_recovery_detail::ensureWritableTail(
                    outUniqueMatches,
                    outUniqueMatchHashes,
                    writeIndex + 1ULL);
            outUniqueMatches[writeIndex] = lhs;
            outUniqueMatchHashes[writeIndex] = lhsHash;
            ++writeIndex;

            if constexpr (COLLECT_PROBE_STATS) {
                ++probeStats->exactMatches;
            }
        }

        outUniqueMatches.resize(writeIndex);
        outUniqueMatchHashes.resize(writeIndex);
        if (!outUniqueMatches.empty()) {
            frontier_recovery_detail::normalizeBucketsByState(
                    outUniqueMatches, outUniqueMatchHashes);
            compactUniqueSortedStatesInPlace(
                    outUniqueMatches, outUniqueMatchHashes);
        }
    }
};
