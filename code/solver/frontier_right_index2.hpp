#pragma once
// code/solver/frontier_right_index2.hpp

#include "code/solver/frontier_builder.hpp"

#include <algorithm>
#include <cstdlib>

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
    static constexpr std::size_t HASH_PREFIX_BUCKET_COUNT = (static_cast<std::size_t>(1) << HASH_PREFIX_BITS);
    static constexpr u32 HASH_PREFIX_SHIFT = 64U - HASH_PREFIX_BITS;
    static constexpr u32 PREFIX_LINEAR_SCAN_THRESHOLD = 32;

    static constexpr u32 HASH_PRESENCE_HIGH_BITS = 29;
    static constexpr std::size_t HASH_PRESENCE_HIGH_BUCKET_COUNT =
            (static_cast<std::size_t>(1) << HASH_PRESENCE_HIGH_BITS);
    static constexpr std::size_t HASH_PRESENCE_HIGH_WORD_COUNT =
            HASH_PRESENCE_HIGH_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_HIGH_SHIFT = 64U - HASH_PRESENCE_HIGH_BITS;

    static constexpr u32 HASH_PRESENCE_MID_BITS = 28;
    static constexpr std::size_t HASH_PRESENCE_MID_BUCKET_COUNT =
            (static_cast<std::size_t>(1) << HASH_PRESENCE_MID_BITS);
    static constexpr std::size_t HASH_PRESENCE_MID_WORD_COUNT =
            HASH_PRESENCE_MID_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_MID_SHIFT = (64U - HASH_PRESENCE_MID_BITS) / 2U;
    static constexpr u64 HASH_PRESENCE_MID_MASK =
            static_cast<u64>(HASH_PRESENCE_MID_BUCKET_COUNT - 1ULL);

    static constexpr u32 HASH_PRESENCE_LOW_BITS = 27;
    static constexpr std::size_t HASH_PRESENCE_LOW_BUCKET_COUNT =
            (static_cast<std::size_t>(1) << HASH_PRESENCE_LOW_BITS);
    static constexpr std::size_t HASH_PRESENCE_LOW_WORD_COUNT =
            HASH_PRESENCE_LOW_BUCKET_COUNT / 64ULL;
    static constexpr u64 HASH_PRESENCE_LOW_MASK =
            static_cast<u64>(HASH_PRESENCE_LOW_BUCKET_COUNT - 1ULL);

    // Fourth filter. Uses a different interior slice than the existing high/mid/low filters.
    // This is intentionally smaller than the others to limit added memory pressure.
    static constexpr u32 HASH_PRESENCE_EXTRA_BITS = 26;
    static constexpr std::size_t HASH_PRESENCE_EXTRA_BUCKET_COUNT =
            (static_cast<std::size_t>(1) << HASH_PRESENCE_EXTRA_BITS);
    static constexpr std::size_t HASH_PRESENCE_EXTRA_WORD_COUNT =
            HASH_PRESENCE_EXTRA_BUCKET_COUNT / 64ULL;
    static constexpr u32 HASH_PRESENCE_EXTRA_SHIFT = 13U;
    static constexpr u64 HASH_PRESENCE_EXTRA_MASK =
            static_cast<u64>(HASH_PRESENCE_EXTRA_BUCKET_COUNT - 1ULL);

    static constexpr u32 META_PAYLOAD_BITS = 29;
    static constexpr u32 META_COUNT_BITS = 3;
    static constexpr u32 META_COUNT_SHIFT = META_PAYLOAD_BITS;
    static constexpr u32 META_PAYLOAD_MASK = (1u << META_PAYLOAD_BITS) - 1u;
    static constexpr u32 META_COUNT_MASK = ((1u << META_COUNT_BITS) - 1u) << META_COUNT_SHIFT;

    static constexpr u32 META_COUNTCODE_SINGLETON = 1u;
    static constexpr u32 META_MAX_DIRECT_COLLISION_COUNT = 7u;

    JVec<B1B2> states_{};
    JVec<u64> hashes_{};

    JVec<u64> entryHashes_{};
    JVec<u32> entryMeta_{};

    JVec<u32> collisionIndices_{};
    JVec<u32> prefixStarts_{};

    JVec<u64> highPresenceWords_{};
    JVec<u64> midPresenceWords_{};
    JVec<u64> lowPresenceWords_{};
    JVec<u64> extraPresenceWords_{};

    B1B2 goalState_{};
    bool hasGoalState_ = false;
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

    MUND static u32 hashPresenceHighBucket(const u64 hash) {
        return static_cast<u32>(hash >> HASH_PRESENCE_HIGH_SHIFT);
    }

    MUND static u32 hashPresenceMidBucket(const u64 hash) {
        return static_cast<u32>((hash >> HASH_PRESENCE_MID_SHIFT) & HASH_PRESENCE_MID_MASK);
    }

    MUND static u32 hashPresenceLowBucket(const u64 hash) {
        return static_cast<u32>(hash & HASH_PRESENCE_LOW_MASK);
    }

    MUND static u32 hashPresenceExtraBucket(const u64 hash) {
        return static_cast<u32>((hash >> HASH_PRESENCE_EXTRA_SHIFT) & HASH_PRESENCE_EXTRA_MASK);
    }

    MU static void setPresenceBit(JVec<u64>& words,
                                  const u32 bucket) {
        words[static_cast<std::size_t>(bucket >> 6U)] |=
                (1ULL << static_cast<u32>(bucket & 63U));
    }

    MUND static bool getPresenceBit(const JVec<u64>& words,
                                    const u32 bucket) {
        return (words[static_cast<std::size_t>(bucket >> 6U)] &
                (1ULL << static_cast<u32>(bucket & 63U))) != 0ULL;
    }

    MUND static u32 metaPayload(const u32 meta) {
        return meta & META_PAYLOAD_MASK;
    }

    MUND static u32 metaCountCode(const u32 meta) {
        return (meta >> META_COUNT_SHIFT) & ((1u << META_COUNT_BITS) - 1u);
    }

    MUND static u32 makeMeta(const u32 payload,
                             const u32 countCode) {
        return (payload & META_PAYLOAD_MASK) | (countCode << META_COUNT_SHIFT);
    }

    MUND u32 entryCountAt(const std::size_t entryIndex) const {
        return metaCountCode(entryMeta_[entryIndex]);
    }

    MU static void addBucketToHistogram(const u32 bucketSize,
                                        BucketStats& s) {
        if (bucketSize == 1) {
            ++s.bucketsSize1;
            s.statesSize1 += 1;
        } else if (bucketSize == 2) {
            ++s.bucketsSize2;
            s.statesSize2 += 2;
        } else if (bucketSize == 3) {
            ++s.bucketsSize3;
            s.statesSize3 += 3;
        } else if (bucketSize == 4) {
            ++s.bucketsSize4;
            s.statesSize4 += 4;
        } else if (bucketSize <= 8) {
            ++s.bucketsSize5to8;
            s.statesSize5to8 += bucketSize;
        } else if (bucketSize <= 16) {
            ++s.bucketsSize9to16;
            s.statesSize9to16 += bucketSize;
        } else if (bucketSize <= 32) {
            ++s.bucketsSize17to32;
            s.statesSize17to32 += bucketSize;
        } else if (bucketSize <= 64) {
            ++s.bucketsSize33to64;
            s.statesSize33to64 += bucketSize;
        } else {
            ++s.bucketsSize65plus;
            s.statesSize65plus += bucketSize;
        }
    }

    MU void buildPrefixStarts() {
        prefixStarts_.clear();
        prefixStarts_.resize(HASH_PREFIX_BUCKET_COUNT + 1);

        u32 entryIndex = 0;
        for (u32 prefix = 0; prefix < HASH_PREFIX_BUCKET_COUNT; ++prefix) {
            prefixStarts_[prefix] = entryIndex;

            while (entryIndex < entryHashes_.size()
                   && hashPrefix(entryHashes_[entryIndex]) == prefix) {
                ++entryIndex;
            }
        }

        prefixStarts_[HASH_PREFIX_BUCKET_COUNT] = static_cast<u32>(entryHashes_.size());
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

        for (std::size_t i = 0; i < entryHashes_.size(); ++i) {
            const u64 hash = entryHashes_[i];

            setPresenceBit(highPresenceWords_, hashPresenceHighBucket(hash));
            setPresenceBit(midPresenceWords_, hashPresenceMidBucket(hash));
            setPresenceBit(lowPresenceWords_, hashPresenceLowBucket(hash));
            setPresenceBit(extraPresenceWords_, hashPresenceExtraBucket(hash));
        }
    }

    MUND i64 findEntryIndex(const u64 hash) const {
        if (entryHashes_.empty() || prefixStarts_.empty()) {
            return -1;
        }

        const u32 prefix = hashPrefix(hash);
        const u32 begin = prefixStarts_[prefix];
        const u32 end = prefixStarts_[prefix + 1];

        if (begin == end) {
            return -1;
        }

        const u32 span = end - begin;

        if (span <= PREFIX_LINEAR_SCAN_THRESHOLD) {
            for (u32 i = begin; i < end; ++i) {
                const u64 cur = entryHashes_[i];
                if (cur == hash) {
                    return static_cast<i64>(i);
                }
                if (cur > hash) {
                    return -1;
                }
            }
            return -1;
        }

        i64 lo = static_cast<i64>(begin);
        i64 hi = static_cast<i64>(end) - 1;

        while (lo <= hi) {
            const i64 mid = lo + ((hi - lo) >> 1);
            const u64 midHash = entryHashes_[static_cast<std::size_t>(mid)];

            if (midHash < hash) {
                lo = mid + 1;
            } else if (midHash > hash) {
                hi = mid - 1;
            } else {
                return mid;
            }
        }

        return -1;
    }

    MU void buildStats() {
        stats_ = {};

        stats_.stateCount = states_.size();
        stats_.bucketCount = entryHashes_.size();

        if (entryHashes_.empty()) {
            return;
        }

        JVec<u32> bucketSizes;
        bucketSizes.resize(entryHashes_.size());

        u64 totalBucketSize = 0;
        u32 minBucketSize = entryCountAt(0);
        u32 maxBucketSize = minBucketSize;

        for (std::size_t i = 0; i < entryHashes_.size(); ++i) {
            const u32 bucketSize = entryCountAt(i);
            bucketSizes[i] = bucketSize;

            totalBucketSize += bucketSize;

            if (bucketSize < minBucketSize) {
                minBucketSize = bucketSize;
            }
            if (bucketSize > maxBucketSize) {
                maxBucketSize = bucketSize;
            }

            if (bucketSize == 1) {
                ++stats_.singletonBucketCount;
                ++stats_.statesInSingletonBuckets;
            } else {
                ++stats_.multiBucketCount;
                stats_.statesInMultiBuckets += bucketSize;
                stats_.collisionPairs += (static_cast<u64>(bucketSize) * static_cast<u64>(bucketSize - 1)) / 2ULL;
            }

            addBucketToHistogram(bucketSize, stats_);
        }

        stats_.minBucketSize = minBucketSize;
        stats_.maxBucketSize = maxBucketSize;
        stats_.avgBucketSize = static_cast<double>(totalBucketSize) / static_cast<double>(entryHashes_.size());

        std::sort(bucketSizes.begin(), bucketSizes.end());

        auto percentileValue = [&](double p) -> u32 {
            if (bucketSizes.empty()) {
                return 0;
            }

            const std::size_t idx = static_cast<std::size_t>(p * static_cast<double>(bucketSizes.size() - 1));
            return bucketSizes[idx];
        };

        stats_.p50BucketSize = percentileValue(0.50);
        stats_.p90BucketSize = percentileValue(0.90);
        stats_.p99BucketSize = percentileValue(0.99);
    }

public:
    MU void clear() {
        states_.clear();
        hashes_.clear();
        entryHashes_.clear();
        entryMeta_.clear();
        collisionIndices_.clear();
        prefixStarts_.clear();
        highPresenceWords_.clear();
        midPresenceWords_.clear();
        lowPresenceWords_.clear();
        extraPresenceWords_.clear();
        goalState_ = {};
        hasGoalState_ = false;
        stats_ = {};
    }

    template<bool BUILD_BUCKET_STATS = true>
    MU void buildFromUniqueStates(JVec<B1B2>&& states,
                                  JVec<u64>&& hashes,
                                  const Board& goalBoard) {
        clear();

        states_.swap(states);
        hashes_.swap(hashes);

        goalState_ = goalBoard.asB1B2();
        hasGoalState_ = true;
        stats_ = {};

        if (states_.empty()) {
            return;
        }

        u32 uniqueHashCount = 1;
        u32 collidedStateCount = 0;

        {
            u32 begin = 0;
            u64 currentHash = hashes_[0];

            for (u32 i = 1; i < hashes_.size(); ++i) {
                const u64 nextHash = hashes_[i];
                if (nextHash != currentHash) {
                    const u32 count = i - begin;
                    if (count > 1) {
                        collidedStateCount += count;
                    }
                    if (count > META_MAX_DIRECT_COLLISION_COUNT) {
                        tcout << "fatal: right frontier hash bucket exceeded max supported collision count of "
                              << META_MAX_DIRECT_COLLISION_COUNT
                              << " for hash "
                              << currentHash
                              << " with count "
                              << count
                              << '\n';
                        std::abort();
                    }

                    ++uniqueHashCount;
                    begin = i;
                    currentHash = nextHash;
                }
            }

            const u32 tailCount = static_cast<u32>(hashes_.size()) - begin;
            if (tailCount > 1) {
                collidedStateCount += tailCount;
            }
            if (tailCount > META_MAX_DIRECT_COLLISION_COUNT) {
                tcout << "fatal: right frontier hash bucket exceeded max supported collision count of "
                      << META_MAX_DIRECT_COLLISION_COUNT
                      << " for hash "
                      << currentHash
                      << " with count "
                      << tailCount
                      << '\n';
                std::abort();
            }
        }

        entryHashes_.resize(uniqueHashCount);
        entryMeta_.resize(uniqueHashCount);
        collisionIndices_.resize(collidedStateCount);

        u32 entryWrite = 0;
        u32 collisionWrite = 0;
        u32 begin = 0;
        u64 currentHash = hashes_[0];

        for (u32 i = 1; i < hashes_.size(); ++i) {
            const u64 nextHash = hashes_[i];
            if (nextHash != currentHash) {
                const u32 count = i - begin;

                entryHashes_[entryWrite] = currentHash;

                if (count == 1) {
                    entryMeta_[entryWrite] = makeMeta(begin, META_COUNTCODE_SINGLETON);
                } else {
                    entryMeta_[entryWrite] = makeMeta(collisionWrite, count);
                    for (u32 j = begin; j < i; ++j) {
                        collisionIndices_[collisionWrite++] = j;
                    }
                }

                ++entryWrite;
                begin = i;
                currentHash = nextHash;
            }
        }

        {
            const u32 count = static_cast<u32>(hashes_.size()) - begin;

            entryHashes_[entryWrite] = currentHash;

            if (count == 1) {
                entryMeta_[entryWrite] = makeMeta(begin, META_COUNTCODE_SINGLETON);
            } else {
                entryMeta_[entryWrite] = makeMeta(collisionWrite, count);
                for (u32 j = begin; j < hashes_.size(); ++j) {
                    collisionIndices_[collisionWrite++] = j;
                }
            }
        }

        buildPrefixStarts();
        buildPresenceFilters();

        if constexpr (BUILD_BUCKET_STATS) {
            buildStats();
        } else {
            stats_ = {};
            stats_.stateCount = states_.size();
            stats_.bucketCount = entryHashes_.size();
        }
    }

    MUND const JVec<B1B2>& states() const {
        return states_;
    }

    MUND const JVec<u64>& hashes() const {
        return hashes_;
    }

    MUND std::size_t size() const {
        return states_.size();
    }

    MUND std::size_t rangeCount() const {
        return entryHashes_.size();
    }

    MUND const BucketStats& stats() const {
        return stats_;
    }

    MU void printStats() const {
        const u64 statesLive = liveBytes(states_);
        const u64 hashesLive = liveBytes(hashes_);
        const u64 entryHashesLive = liveBytes(entryHashes_);
        const u64 entryMetaLive = liveBytes(entryMeta_);
        const u64 collisionsLive = liveBytes(collisionIndices_);
        const u64 prefixLive = liveBytes(prefixStarts_);
        const u64 highPresenceLive = liveBytes(highPresenceWords_);
        const u64 midPresenceLive = liveBytes(midPresenceWords_);
        const u64 lowPresenceLive = liveBytes(lowPresenceWords_);
        const u64 extraPresenceLive = liveBytes(extraPresenceWords_);

        const u64 statesReserved = reservedBytes(states_);
        const u64 hashesReserved = reservedBytes(hashes_);
        const u64 entryHashesReserved = reservedBytes(entryHashes_);
        const u64 entryMetaReserved = reservedBytes(entryMeta_);
        const u64 collisionsReserved = reservedBytes(collisionIndices_);
        const u64 prefixReserved = reservedBytes(prefixStarts_);
        const u64 highPresenceReserved = reservedBytes(highPresenceWords_);
        const u64 midPresenceReserved = reservedBytes(midPresenceWords_);
        const u64 lowPresenceReserved = reservedBytes(lowPresenceWords_);
        const u64 extraPresenceReserved = reservedBytes(extraPresenceWords_);

        const u64 totalLive =
                statesLive + hashesLive + entryHashesLive + entryMetaLive + collisionsLive +
                prefixLive + highPresenceLive + midPresenceLive + lowPresenceLive + extraPresenceLive;
        const u64 totalReserved =
                statesReserved + hashesReserved + entryHashesReserved + entryMetaReserved + collisionsReserved +
                prefixReserved + highPresenceReserved + midPresenceReserved + lowPresenceReserved + extraPresenceReserved;

        tcout << "right frontier stats:\n";
        tcout << "    total states: " << stats_.stateCount << '\n';
        tcout << "    total buckets: " << stats_.bucketCount << '\n';
        tcout << "    prefix bits: " << HASH_PREFIX_BITS << '\n';
        tcout << "    prefix table entries: " << prefixStarts_.size() << '\n';
        tcout << "    high presence bits: " << HASH_PRESENCE_HIGH_BITS << '\n';
        tcout << "    high presence words: " << highPresenceWords_.size() << '\n';
        tcout << "    mid presence bits: " << HASH_PRESENCE_MID_BITS << '\n';
        tcout << "    mid presence words: " << midPresenceWords_.size() << '\n';
        tcout << "    low presence bits: " << HASH_PRESENCE_LOW_BITS << '\n';
        tcout << "    low presence words: " << lowPresenceWords_.size() << '\n';
        tcout << "    extra presence bits: " << HASH_PRESENCE_EXTRA_BITS << '\n';
        tcout << "    extra presence words: " << extraPresenceWords_.size() << '\n';

        tcout << "    memory states: " << bytesFormatted<1000>(statesLive)
              << " live, " << bytesFormatted<1000>(statesReserved) << " rsv\n";
        tcout << "    memory hashes: " << bytesFormatted<1000>(hashesLive)
              << " live, " << bytesFormatted<1000>(hashesReserved) << " rsv\n";
        tcout << "    memory entry hashes: " << bytesFormatted<1000>(entryHashesLive)
              << " live, " << bytesFormatted<1000>(entryHashesReserved) << " rsv\n";
        tcout << "    memory entry meta: " << bytesFormatted<1000>(entryMetaLive)
              << " live, " << bytesFormatted<1000>(entryMetaReserved) << " rsv\n";
        tcout << "    memory collisions: " << bytesFormatted<1000>(collisionsLive)
              << " live, " << bytesFormatted<1000>(collisionsReserved) << " rsv\n";
        tcout << "    memory prefix: " << bytesFormatted<1000>(prefixLive)
              << " live, " << bytesFormatted<1000>(prefixReserved) << " rsv\n";
        tcout << "    memory high presence: " << bytesFormatted<1000>(highPresenceLive)
              << " live, " << bytesFormatted<1000>(highPresenceReserved) << " rsv\n";
        tcout << "    memory mid presence: " << bytesFormatted<1000>(midPresenceLive)
              << " live, " << bytesFormatted<1000>(midPresenceReserved) << " rsv\n";
        tcout << "    memory low presence: " << bytesFormatted<1000>(lowPresenceLive)
              << " live, " << bytesFormatted<1000>(lowPresenceReserved) << " rsv\n";
        tcout << "    memory extra presence: " << bytesFormatted<1000>(extraPresenceLive)
              << " live, " << bytesFormatted<1000>(extraPresenceReserved) << " rsv\n";
        tcout << "    memory total:  " << bytesFormatted<1000>(totalLive)
              << " live, " << bytesFormatted<1000>(totalReserved) << " rsv\n";

        tcout << "    min bucket size: " << stats_.minBucketSize << '\n';
        tcout << "    max bucket size: " << stats_.maxBucketSize << '\n';
        tcout << "    avg bucket size: " << stats_.avgBucketSize << '\n';
        tcout << "    p50 bucket size: " << stats_.p50BucketSize << '\n';
        tcout << "    p90 bucket size: " << stats_.p90BucketSize << '\n';
        tcout << "    p99 bucket size: " << stats_.p99BucketSize << '\n';

        tcout << "    singleton buckets: " << stats_.singletonBucketCount
              << " (" << (stats_.bucketCount
                                  ? (100.0 * static_cast<double>(stats_.singletonBucketCount) / static_cast<double>(stats_.bucketCount))
                                  : 0.0)
              << "%)\n";

        tcout << "    multi buckets: " << stats_.multiBucketCount
              << " (" << (stats_.bucketCount
                                  ? (100.0 * static_cast<double>(stats_.multiBucketCount) / static_cast<double>(stats_.bucketCount))
                                  : 0.0)
              << "%)\n";

        tcout << "    states in singleton buckets: " << stats_.statesInSingletonBuckets
              << " (" << (stats_.stateCount
                                  ? (100.0 * static_cast<double>(stats_.statesInSingletonBuckets) / static_cast<double>(stats_.stateCount))
                                  : 0.0)
              << "%)\n";

        tcout << "    states in multi buckets: " << stats_.statesInMultiBuckets
              << " (" << (stats_.stateCount
                                  ? (100.0 * static_cast<double>(stats_.statesInMultiBuckets) / static_cast<double>(stats_.stateCount))
                                  : 0.0)
              << "%)\n";

        tcout << "    collision pairs inside buckets: " << stats_.collisionPairs << '\n';

        tcout << "    bucket histogram:\n";
        tcout << "        size 1:      " << stats_.bucketsSize1 << " buckets, " << stats_.statesSize1 << " states\n";
        tcout << "        size 2:      " << stats_.bucketsSize2 << " buckets, " << stats_.statesSize2 << " states\n";
        tcout << "        size 3:      " << stats_.bucketsSize3 << " buckets, " << stats_.statesSize3 << " states\n";
        tcout << "        size 4:      " << stats_.bucketsSize4 << " buckets, " << stats_.statesSize4 << " states\n";
        tcout << "        size 5..8:   " << stats_.bucketsSize5to8 << " buckets, " << stats_.statesSize5to8 << " states\n";
        tcout << "        size 9..16:  " << stats_.bucketsSize9to16 << " buckets, " << stats_.statesSize9to16 << " states\n";
        tcout << "        size 17..32: " << stats_.bucketsSize17to32 << " buckets, " << stats_.statesSize17to32 << " states\n";
        tcout << "        size 33..64: " << stats_.bucketsSize33to64 << " buckets, " << stats_.statesSize33to64 << " states\n";
        tcout << "        size 65+:    " << stats_.bucketsSize65plus << " buckets, " << stats_.statesSize65plus << " states\n";
    }

    template<int RIGHT_FRONTIER_DEPTH, bool COLLECT_PROBE_STATS = true>
    MU void collectMatches(const JVec<B1B2>& leftStates,
                           const JVec<u64>& leftHashes,
                           JVec<B1B2>& outUniqueMatches,
                           JVec<u64>& outUniqueMatchHashes,
                           ProbeStats* probeStats = nullptr) const {
        outUniqueMatches.clear();
        outUniqueMatchHashes.clear();

        if constexpr (COLLECT_PROBE_STATS) {
            *probeStats = {};
            probeStats->leftStateCount = static_cast<u64>(leftStates.size());
        }

        if (leftStates.empty() || states_.empty()) {
            return;
        }

        frontier_recovery_detail::reserveStateHashLanes(
                outUniqueMatches,
                outUniqueMatchHashes,
                std::min<std::size_t>(leftStates.size(), 256)
        );

        std::size_t writeIndex = 0;

        for (std::size_t lhsIndex = 0; lhsIndex < leftStates.size(); ++lhsIndex) {
            const B1B2& lhs = leftStates[lhsIndex];
            const u64 lhsHash = leftHashes[lhsIndex];

            const u32 highBucket = hashPresenceHighBucket(lhsHash);
            if (!getPresenceBit(highPresenceWords_, highBucket)) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses;
                    ++probeStats->highFilterRejects;
                }
                continue;
            }

            const u32 midBucket = hashPresenceMidBucket(lhsHash);
            if (!getPresenceBit(midPresenceWords_, midBucket)) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses;
                    ++probeStats->midFilterRejects;
                }
                continue;
            }

            const u32 lowBucket = hashPresenceLowBucket(lhsHash);
            if (!getPresenceBit(lowPresenceWords_, lowBucket)) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses;
                    ++probeStats->lowFilterRejects;
                }
                continue;
            }

            const u32 extraBucket = hashPresenceExtraBucket(lhsHash);
            if (!getPresenceBit(extraPresenceWords_, extraBucket)) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses;
                    ++probeStats->extraFilterRejects;
                }
                continue;
            }

            if constexpr (COLLECT_PROBE_STATS) {
                ++probeStats->filterPasses;
            }

            const i64 entryIndex = findEntryIndex(lhsHash);
            if (entryIndex < 0) {
                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->hashMisses;
                    ++probeStats->prefixRejects;
                }
                continue;
            }

            if constexpr (COLLECT_PROBE_STATS) {
                ++probeStats->hashHits;
                ++probeStats->bucketsVisited;
            }

            const std::size_t entryIdx = static_cast<std::size_t>(entryIndex);
            const u32 meta = entryMeta_[entryIdx];
            const u32 countCode = metaCountCode(meta);
            const u32 payload = metaPayload(meta);

            if (countCode == META_COUNTCODE_SINGLETON) {
                const u32 stateIndex = payload;

                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->bucketStatesScanned;
                    ++probeStats->equalityChecks;
                }

                if (lhs == states_[stateIndex]) {
                    frontier_recovery_detail::ensureWritableTail(
                            outUniqueMatches,
                            outUniqueMatchHashes,
                            writeIndex + 1
                    );

                    outUniqueMatches[writeIndex] = lhs;
                    outUniqueMatchHashes[writeIndex] = lhsHash;
                    ++writeIndex;

                    if constexpr (COLLECT_PROBE_STATS) {
                        ++probeStats->exactMatches;
                    }
                }

                continue;
            }

            const u32 collisionBegin = payload;
            const u32 collisionEnd = collisionBegin + countCode;

            if constexpr (COLLECT_PROBE_STATS) {
                probeStats->bucketStatesScanned += static_cast<u64>(countCode);
            }

            for (u32 i = collisionBegin; i < collisionEnd; ++i) {
                const u32 stateIndex = collisionIndices_[i];

                if constexpr (COLLECT_PROBE_STATS) {
                    ++probeStats->equalityChecks;
                }

                if (lhs == states_[stateIndex]) {
                    frontier_recovery_detail::ensureWritableTail(
                            outUniqueMatches,
                            outUniqueMatchHashes,
                            writeIndex + 1
                    );

                    outUniqueMatches[writeIndex] = lhs;
                    outUniqueMatchHashes[writeIndex] = lhsHash;
                    ++writeIndex;

                    if constexpr (COLLECT_PROBE_STATS) {
                        ++probeStats->exactMatches;
                    }
                    break;
                }
            }
        }

        outUniqueMatches.resize(writeIndex);
        outUniqueMatchHashes.resize(writeIndex);

        if (!outUniqueMatches.empty()) {
            frontier_recovery_detail::normalizeBucketsByState(outUniqueMatches, outUniqueMatchHashes);
            compactUniqueSortedStatesInPlace(outUniqueMatches, outUniqueMatchHashes);
        }
    }
};