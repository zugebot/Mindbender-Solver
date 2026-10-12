#pragma once
// src/solver/ssd/fingerprint_index.hpp
//
// The compact fingerprint index of the goal side (Bloom blocks + 16-bit
// fingerprints per bucket), in RAM-sized partition ranges, with large pages when
// allowed; built from the disk table or filled in RAM by the no-table modes.

#include "src/solver/ssd/common.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // Fingerprint mode: compact in-RAM index, one left pass per key range
    // ------------------------------------------------------------------
    //
    // Everything is laid out per partition, so any contiguous partition range
    // can be loaded on its own:
    //   fp16.bin    : u16 low key bits of every table state, in key order
    //   fpdir32.bin : per partition, (bucketsPerPartition + 1) u32 offsets
    //                 relative to the partition start (~4 states per bucket)
    //   bloomp.bin  : per partition, a split-block Bloom filter (10 bits/state)
    //   fp16meta.txt: "<bucketBitsPerPartition> <bloomBlocksPerPartition> <states>"
    //
    // A probe passes its partition range check, then the Bloom block (one
    // cache line, ~1.3% pass), then the bucket's fingerprints. Survivors are
    // confirmed against the full states on disk.

    constexpr u32 BLOOM_BITS_PER_STATE = 10;
    constexpr u32 TARGET_STATES_PER_BUCKET = 4;
    constexpr u32 BLOOM_SALT[8] = {0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU,
                                   0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U};

    // Independent of the partition/bucket (top) and fingerprint (low) key bits.
    FORCEINLINE u64 bloomHash(const u64 key) {
        return mixKey(key ^ 0x5851f42d4c957f2dULL);
    }

    // Split-block Bloom blocks: 8 x u32 (32 bytes, never straddling a cache
    // line); a key sets one bit in each word of its block.
    FORCEINLINE void bloomInsertAtomic(u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        for (u32 i = 0; i < 8; ++i) {
            __atomic_fetch_or(&block[i], 1U << ((low * BLOOM_SALT[i]) >> 27), __ATOMIC_RELAXED);
        }
    }

    // For a block only one thread writes.
    FORCEINLINE void bloomInsert(u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        for (u32 i = 0; i < 8; ++i) {
            block[i] |= 1U << ((low * BLOOM_SALT[i]) >> 27);
        }
    }

    FORCEINLINE bool bloomTest(const u32* block, const u64 h) {
        const u32 low = static_cast<u32>(h);
        u32 missing = 0;
        for (u32 i = 0; i < 8; ++i) {
            missing |= ~block[i] & (1U << ((low * BLOOM_SALT[i]) >> 27));
        }
        return missing == 0;
    }

    FORCEINLINE u64 bloomBlockInPartition(const u64 h, const u64 blocksPerPartition) {
        return ((h >> 32) * blocksPerPartition) >> 32;
    }

    // u32 storage over-allocated so `aligned` can start on a 64-byte boundary.
    struct AlignedWords {
        std::vector<u32> storage;
        u32* aligned = nullptr;

        void allocate(const u64 words) {
            storage.assign(words + 16, 0);
            const auto address = reinterpret_cast<std::uintptr_t>(storage.data());
            aligned = reinterpret_cast<u32*>((address + 63) & ~static_cast<std::uintptr_t>(63));
        }
    };

    inline bool g_indexBuiltWithTable = false;   // set by the in-RAM depth-6 build

    inline bool enableLockMemoryPrivilege() {
        static const bool enabled = [] {
            HANDLE token = nullptr;
            if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
                return false;
            }
            TOKEN_PRIVILEGES privileges{};
            privileges.PrivilegeCount = 1;
            privileges.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
            bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &privileges.Privileges[0].Luid)
                      && AdjustTokenPrivileges(token, FALSE, &privileges, 0, nullptr, nullptr)
                      && GetLastError() == ERROR_SUCCESS;
            CloseHandle(token);
            return ok;
        }();
        return enabled;
    }

    template<typename T>
    class BigArray {
        T* data_ = nullptr;
        u64 size_ = 0;
        u64 capacity_ = 0;
        bool large_ = false;

        void release() {
            if (data_ != nullptr) {
                VirtualFree(data_, 0, MEM_RELEASE);
            }
            data_ = nullptr;
            size_ = 0;
            capacity_ = 0;
            large_ = false;
        }

    public:
        BigArray() = default;
        BigArray(const BigArray&) = delete;
        BigArray& operator=(const BigArray&) = delete;
        ~BigArray() { release(); }

        // Storage for `count` elements, reusing the current block when it is
        // big enough (contents are then NOT cleared: callers overwrite them).
        // Allocating once, early, and reusing it across passes avoids
        // re-committing (and re-zeroing) many GB per pass and keeps a large-page
        // block once one was obtained.
        void ensure(const u64 count) {
            if (data_ != nullptr && count <= capacity_) {
                size_ = count;
                return;
            }
            allocate(count);
        }

        // Zero-filled, page-aligned storage for `count` elements.
        void allocate(const u64 count) {
            release();
            size_ = count;
            capacity_ = count;
            const u64 bytes = std::max<u64>(1, count * sizeof(T));
            if (g_largePagesWanted && enableLockMemoryPrivilege()) {
                const u64 page = GetLargePageMinimum();
                if (page != 0) {
                    data_ = static_cast<T*>(VirtualAlloc(nullptr, (bytes + page - 1) / page * page,
                                                         MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                                                         PAGE_READWRITE));
                    large_ = data_ != nullptr;
                }
            }
            if (data_ == nullptr) {
                data_ = static_cast<T*>(VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
            }
            if (data_ == nullptr) {
                throw std::runtime_error("VirtualAlloc failed for probe index");
            }
        }

        MUND T* data() { return data_; }
        MUND const T* data() const { return data_; }
        MUND u64 size() const { return size_; }
        MUND bool largePages() const { return large_; }
        FORCEINLINE T& operator[](const u64 i) { return data_[i]; }
        FORCEINLINE const T& operator[](const u64 i) const { return data_[i]; }
    };

    struct FingerprintMeta {
        u32 bucketBits = 0;           // buckets per partition = 2^bucketBits
        u64 bloomBlocksPerPartition = 0;
        u64 stateCount = 0;
        u32 partitionBegin = 0;                 // index covers partitions [begin, limit);
        u32 partitionLimit = PARTITION_COUNT;   // its files start at partition `begin`

        MUND u64 bucketsPerPartition() const { return 1ULL << bucketBits; }
        MUND u32 bucketShift() const { return PARTITION_SHIFT - bucketBits; }
        MUND u64 indexBytes(const u64 states, const u32 partitions) const {
            return states * 2 + partitions * ((bucketsPerPartition() + 1) * 4 + bloomBlocksPerPartition * 32);
        }
    };

    fs::path fp16Path(const fs::path& dir) { return dir / "fp16.bin"; }
    fs::path fpDir32Path(const fs::path& dir) { return dir / "fpdir32.bin"; }
    fs::path bloomPartPath(const fs::path& dir) { return dir / "bloomp.bin"; }
    fs::path fp16MetaPath(const fs::path& dir) { return dir / "fp16meta.txt"; }

    // The files of one compact index: the depth-6 table's (next to its states)
    // or the index-only depth-7 table's (no states; per-partition counts kept).
    struct IndexFiles {
        fs::path fp16, dir32, bloom, meta, counts;
    };

    inline IndexFiles d6Files(const fs::path& dir) {
        return {fp16Path(dir), fpDir32Path(dir), bloomPartPath(dir), fp16MetaPath(dir), {}};
    }

    inline IndexFiles d7Files(const fs::path& dir) {
        return {dir / "d7_fp16.bin", dir / "d7_fpdir32.bin", dir / "d7_bloomp.bin", dir / "d7_meta.txt",
                dir / "d7_counts.bin"};
    }

    inline FingerprintMeta readFingerprintMeta(const fs::path& metaPath) {
        FingerprintMeta meta;
        std::ifstream in(metaPath);
        in >> meta.bucketBits >> meta.bloomBlocksPerPartition >> meta.stateCount;
        if (!(in >> meta.partitionLimit)) {
            meta.partitionLimit = PARTITION_COUNT;
        }
        if (!(in >> meta.partitionBegin)) {
            meta.partitionBegin = 0;
        }
        return meta;
    }

    inline void buildFingerprintIndex(const fs::path& dir, const std::vector<u64>& counts, const u32 threads) {
        Timer timer;
        u64 n = 0, maxCount = 0;
        for (const u64 c : counts) {
            n += c;
            maxCount = std::max(maxCount, c);
        }
        FingerprintMeta meta;
        meta.stateCount = n;
        while (meta.bucketBits < PARTITION_SHIFT
               && (maxCount >> meta.bucketBits) > TARGET_STATES_PER_BUCKET) {
            ++meta.bucketBits;
        }
        meta.bloomBlocksPerPartition = std::max<u64>(1, (maxCount * BLOOM_BITS_PER_STATE + 255) / 256);
        const u64 bucketsPerPartition = meta.bucketsPerPartition();
        const u32 shift = meta.bucketShift();

        std::FILE* fpOut = std::fopen(fp16Path(dir).string().c_str(), "wb");
        std::FILE* dirOut = std::fopen(fpDir32Path(dir).string().c_str(), "wb");
        std::FILE* bloomOut = std::fopen(bloomPartPath(dir).string().c_str(), "wb");
        if (fpOut == nullptr || dirOut == nullptr || bloomOut == nullptr) {
            throw std::runtime_error("failed to open fingerprint index files");
        }

        AlignedWords bloom;
        std::vector<u32> starts(bucketsPerPartition + 1);
        g_progress.phase("d6-index", PARTITION_COUNT, "fingerprint index of the depth-6 table");
        for (u32 p = 0; p < PARTITION_COUNT; ++p, g_progress.add()) {
            const std::vector<B1B2> states = readAll<B1B2>(sortedPath(dir, p));
            std::vector<u64> keys(states.size());
            bloom.allocate(meta.bloomBlocksPerPartition * 8);
            parallelFor(threads, (states.size() + 65535) / 65536, [&](u32, const std::size_t c) {
                const std::size_t end = std::min(states.size(), (c + 1) * 65536);
                for (std::size_t i = c * 65536; i < end; ++i) {
                    keys[i] = keyOf(states[i]);
                    const u64 h = bloomHash(keys[i]);
                    bloomInsertAtomic(bloom.aligned + bloomBlockInPartition(h, meta.bloomBlocksPerPartition) * 8, h);
                }
            });

            std::vector<u16> fps(states.size());
            u64 next = 0;
            for (std::size_t i = 0; i < states.size(); ++i) {
                const u64 bucket = (keys[i] >> shift) & (bucketsPerPartition - 1);
                while (next <= bucket) {
                    starts[next++] = static_cast<u32>(i);
                }
                fps[i] = static_cast<u16>(keys[i]);
            }
            while (next <= bucketsPerPartition) {
                starts[next++] = static_cast<u32>(states.size());
            }

            if (std::fwrite(fps.data(), sizeof(u16), fps.size(), fpOut) != fps.size()
                || std::fwrite(starts.data(), sizeof(u32), starts.size(), dirOut) != starts.size()
                || std::fwrite(bloom.aligned, sizeof(u32), meta.bloomBlocksPerPartition * 8, bloomOut)
                   != meta.bloomBlocksPerPartition * 8) {
                throw std::runtime_error("fingerprint index write failed");
            }
        }
        std::fclose(fpOut);
        std::fclose(dirOut);
        std::fclose(bloomOut);
        std::ofstream(fp16MetaPath(dir)) << meta.bucketBits << ' ' << meta.bloomBlocksPerPartition
                                         << ' ' << n << '\n';

        tcout << "BUILD fingerprint index: " << n << " states, 2^" << meta.bucketBits
              << " buckets/partition, " << meta.indexBytes(n, PARTITION_COUNT) / 1e9 << " GB ("
              << static_cast<double>(meta.indexBytes(n, PARTITION_COUNT)) / static_cast<double>(n)
              << " B/state) in " << timer.getSeconds() << "s\n";
    }

    constexpr u32 MAX_PROBE_GROUP = 128;

    // One batch of in-range probes and its stage-1 results.
    struct ProbeBatch {
        B1B2 states[MAX_PROBE_GROUP];
        u64 keys[MAX_PROBE_GROUP];
        u64 h[MAX_PROBE_GROUP];
        const u32* blocks[MAX_PROBE_GROUP];
        u32 count = 0;
    };

    // The index for partitions [pBegin, pEnd), loaded with three contiguous reads.
    struct FingerprintRange {
        FingerprintMeta meta;
        u32 pBegin = 0, pEnd = 0;
        std::vector<u64> partitionFirst;   // first fp index of each partition in range, + end
        BigArray<u16> fingerprints;
        BigArray<u32> starts;              // (pEnd - pBegin) * (bucketsPerPartition + 1)
        BigArray<u32> bloom;               // page-aligned, so blocks never straddle lines

        void load(const IndexFiles& files, const FingerprintMeta& m, const std::vector<u64>& counts,
                  const u32 begin, const u32 end) {
            meta = m;
            pBegin = begin;
            pEnd = end;
            u64 before = 0;
            for (u32 p = 0; p < pBegin; ++p) {
                before += counts[p];
            }
            partitionFirst.assign(1, 0);
            for (u32 p = pBegin; p < pEnd; ++p) {
                partitionFirst.push_back(partitionFirst.back() + counts[p]);
            }
            const u64 parts = pEnd - pBegin;
            const u64 dirPer = meta.bucketsPerPartition() + 1;
            const u64 bloomPer = meta.bloomBlocksPerPartition * 8;

            fingerprints.ensure(partitionFirst.back());
            starts.ensure(parts * dirPer);
            bloom.ensure(parts * bloomPer);
            readRange(files.fp16, before * sizeof(u16), fingerprints.data(), fingerprints.size() * sizeof(u16));
            const u64 filePart = pBegin - meta.partitionBegin;   // files start at partitionBegin
            readRange(files.dir32, filePart * dirPer * sizeof(u32), starts.data(), starts.size() * sizeof(u32));
            readRange(files.bloom, filePart * bloomPer * sizeof(u32), bloom.data(), parts * bloomPer * sizeof(u32));
        }

        // Allocates the buffers for the largest pass up front, before the left
        // seeds and per-pass work fragment memory.
        void reserve(const FingerprintMeta& m, const u64 maxStates, const u64 maxPartitions) {
            fingerprints.ensure(maxStates);
            starts.ensure(maxPartitions * (m.bucketsPerPartition() + 1));
            bloom.ensure(maxPartitions * m.bloomBlocksPerPartition * 8);
        }

        MUND bool largePages() const {
            return fingerprints.largePages() && starts.largePages() && bloom.largePages();
        }

        // Chunked fread with a 64-bit seek: a single ifstream::read of more
        // than 2 GB fails on MinGW/Windows.
        static void readRange(const fs::path& path, const u64 offset, void* dst, const u64 bytes) {
            std::FILE* f = std::fopen(path.string().c_str(), "rb");
            if (f == nullptr || _fseeki64(f, static_cast<long long>(offset), SEEK_SET) != 0) {
                throw std::runtime_error("failed to open/seek " + path.string());
            }
            constexpr u64 CHUNK = 1ULL << 30;
            char* out = static_cast<char*>(dst);
            for (u64 done = 0; done < bytes;) {
                const u64 n = std::min(CHUNK, bytes - done);
                if (std::fread(out + done, 1, n, f) != n) {
                    std::fclose(f);
                    throw std::runtime_error("failed to read " + path.string());
                }
                done += n;
            }
            std::fclose(f);
        }

        MUND u64 bytes() const {
            return fingerprints.size() * 2 + starts.size() * 4
                   + (pEnd - pBegin) * meta.bloomBlocksPerPartition * 32;
        }

        // Stage 1 of a probe batch: Bloom hashes and block addresses, with the
        // block lines prefetched. Splitting it from finish() lets a caller
        // issue the next batch's fetches before testing the current batch.
        void issue(ProbeBatch& b) const {
            for (u32 i = 0; i < b.count; ++i) {
                const u64 part = (b.keys[i] >> PARTITION_SHIFT) - pBegin;
                b.h[i] = bloomHash(b.keys[i]);
                b.blocks[i] = bloom.data()
                              + (part * meta.bloomBlocksPerPartition
                                 + bloomBlockInPartition(b.h[i], meta.bloomBlocksPerPartition)) * 8;
                __builtin_prefetch(b.blocks[i]);
            }
        }

        // Stages 2-4: Bloom test, then bucket offsets, then fingerprints; each
        // stage prefetches the next stage's lines for the whole batch. Returns
        // the number of probes that passed the Bloom filter.
        u32 finish(const ProbeBatch& b, std::vector<KeyedState>& candidates) const {
            const u64 bucketsPer = meta.bucketsPerPartition();
            const u64 dirPer = bucketsPer + 1;
            const u32 shift = meta.bucketShift();
            const u64* keys = b.keys;
            const B1B2* probes = b.states;
            const u32 count = b.count;

            u32 pass[MAX_PROBE_GROUP];
            u32 passCount = 0;
            for (u32 i = 0; i < count; ++i) {
                if (bloomTest(b.blocks[i], b.h[i])) {
                    pass[passCount++] = i;
                }
            }

            const u32* dirSlot[MAX_PROBE_GROUP];
            u64 base[MAX_PROBE_GROUP], lo[MAX_PROBE_GROUP], hi[MAX_PROBE_GROUP];
            for (u32 j = 0; j < passCount; ++j) {
                const u64 key = keys[pass[j]];
                const u64 part = (key >> PARTITION_SHIFT) - pBegin;
                base[j] = partitionFirst[part];
                dirSlot[j] = &starts[part * dirPer + ((key >> shift) & (bucketsPer - 1))];
                __builtin_prefetch(dirSlot[j]);
            }
            for (u32 j = 0; j < passCount; ++j) {
                lo[j] = base[j] + dirSlot[j][0];
                hi[j] = base[j] + dirSlot[j][1];
                if (lo[j] < hi[j]) {
                    __builtin_prefetch(&fingerprints[lo[j]]);
                }
            }
            for (u32 j = 0; j < passCount; ++j) {
                const u32 i = pass[j];
                const u16 fp = static_cast<u16>(keys[i]);
                for (u64 k = lo[j]; k < hi[j]; ++k) {
                    if (fingerprints[k] == fp) {
                        candidates.push_back({keys[i], probes[i]});
                        break;
                    }
                }
            }
            return passCount;
        }

        // Unpipelined probe of up to MAX_PROBE_GROUP states (issue + finish).
        u32 probeGroup(const B1B2* probes, const u64* keys, const u32 count,
                       std::vector<KeyedState>& candidates) const {
            ProbeBatch b;
            b.count = count;
            for (u32 i = 0; i < count; ++i) {
                b.states[i] = probes[i];
                b.keys[i] = keys[i];
            }
            issue(b);
            return finish(b, candidates);
        }
    };

} // namespace ssd
