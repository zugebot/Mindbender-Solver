#pragma once

#include "code/board.hpp"
#include "utils/jvec.hpp"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace right_cache_idx {

constexpr u32 DEFAULT_PREFIX_BITS = 20;
constexpr u32 MAX_PREFIX_BITS = 24;

struct PrefixConfig {
    u32 bits = DEFAULT_PREFIX_BITS;
};

struct PrefixStats {
    u32 nonEmptyBuckets = 0;
    u32 maxBucketSize = 0;
    u64 singleBucketCount = 0;
    u64 multiBucketCount = 0;
    u64 collisionBoards = 0;
    double avgBoardsPerBucket = 0.0;
    double avgBoardsPerNonEmptyBucket = 0.0;
    double occupancyPct = 0.0;
};

struct PrefixIndex {
    std::vector<u32> offsets;
    PrefixStats stats;
    PrefixConfig config;
    u32 bucketCount = 0;

    MUND static u32 getPrefix(C u64 hash, C u32 bits) {
        return static_cast<u32>(hash >> (64 - bits));
    }

    void build(C JVec<Board>& sortedBoards, PrefixConfig cfg = {}) {
        if (cfg.bits == 0 || cfg.bits > MAX_PREFIX_BITS) {
            cfg.bits = DEFAULT_PREFIX_BITS;
        }
        config = cfg;
        bucketCount = 1u << config.bits;

        C u64 boardCount64 = sortedBoards.size();
        if (boardCount64 > static_cast<u64>(std::numeric_limits<u32>::max())) {
            throw std::runtime_error("right cache too large for u32 prefix offsets");
        }

        C u32 boardCount = static_cast<u32>(boardCount64);
        offsets.assign(bucketCount + 1, boardCount);

        u32 i = 0;
        for (u32 prefix = 0; prefix < bucketCount; ++prefix) {
            offsets[prefix] = i;
            while (i < boardCount && getPrefix(sortedBoards[i].getHash(), config.bits) == prefix) {
                ++i;
            }
        }
        offsets[bucketCount] = boardCount;

        stats = {};
        stats.avgBoardsPerBucket = static_cast<double>(boardCount) / static_cast<double>(bucketCount);

        for (u32 prefix = 0; prefix < bucketCount; ++prefix) {
            C u32 bucketSize = offsets[prefix + 1] - offsets[prefix];
            if (bucketSize == 0) {
                continue;
            }

            ++stats.nonEmptyBuckets;
            if (bucketSize > stats.maxBucketSize) {
                stats.maxBucketSize = bucketSize;
            }
            if (bucketSize == 1) {
                ++stats.singleBucketCount;
            } else {
                ++stats.multiBucketCount;
                stats.collisionBoards += static_cast<u64>(bucketSize - 1);
            }
        }

        if (stats.nonEmptyBuckets > 0) {
            stats.avgBoardsPerNonEmptyBucket =
                    static_cast<double>(boardCount) / static_cast<double>(stats.nonEmptyBuckets);
            stats.occupancyPct =
                    100.0 * static_cast<double>(stats.nonEmptyBuckets) / static_cast<double>(bucketCount);
        }
    }

    MUND std::pair<u32, u32> getRange(C u64 hash) C {
        C u32 prefix = getPrefix(hash, config.bits);
        return {offsets[prefix], offsets[prefix + 1]};
    }

    MUND u32 getPrefixBits() C { return config.bits; }
    MUND u32 getBucketCount() C { return bucketCount; }
};

} // namespace right_cache_idx

