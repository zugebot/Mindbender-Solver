#pragma once
// src/solver/ssd/probe_sink.hpp
//
// Probe variants (batch size, pipelining), the per-machine probe profile,
// the sink that looks streamed left states up in the index (with the optional
// leaf-level bound checks), and the run's RAM budget.

#include "src/solver/ssd/fingerprint_index.hpp"
#include "src/solver/bounds/pattern_db.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // Probe variants and per-machine calibration
    // ------------------------------------------------------------------
    struct ProbeVariant {
        u32 group = PROBE_GROUP;
        bool pipeline = false;

        MUND std::string name() const { return std::to_string(group) + (pipeline ? "/pipelined" : "/plain"); }
    };

    inline const std::vector<ProbeVariant>& probeVariants() {
        static const std::vector<ProbeVariant> v = {{32, false}, {32, true}, {64, false}, {64, true},
                                                    {128, false}, {128, true}};
        return v;
    }

    // Collects in-range left states into batches and probes them with the
    // chosen variant; pipelined batches overlap their fetches with the
    // previous batch's tests.
    // LEAF_BOUND: in-range states whose colour-mix bound exceeds `limit`
    // cannot be within the table depth of the right root and are not probed.
    template<bool LEAF_BOUND, bool LEAF_PDB = false>
    struct ProbeSinkT {
        const FingerprintRange& index;
        ProbeVariant variant;
        ProbeBatch batches[2];
        u32 cur = 0;
        bool prevPending = false;
        std::vector<KeyedState> found;
        u64 streamed = 0, inRange = 0, bloomPasses = 0, boundSkipped = 0;
        const lane_bound::LaneTables* tables = nullptr;
        int limit = 0;

        ProbeSinkT(const FingerprintRange& idx, const ProbeVariant v) : index(idx), variant(v) {}
        ProbeSinkT(const FingerprintRange& idx, const ProbeVariant v, const lane_bound::LaneTables* tabs, const int lim)
            : index(idx), variant(v), tables(tabs), limit(lim) {}

        void submit() {
            ProbeBatch& c = batches[cur];
            index.issue(c);
            if (!variant.pipeline) {
                bloomPasses += index.finish(c, found);
                c.count = 0;
                return;
            }
            if (prevPending) {
                ProbeBatch& prev = batches[cur ^ 1];
                bloomPasses += index.finish(prev, found);
                prev.count = 0;
            }
            prevPending = true;
            cur ^= 1;
        }

        void drain() {
            if (batches[cur].count != 0) {
                submit();
            }
            if (prevPending) {
                ProbeBatch& prev = batches[cur ^ 1];
                bloomPasses += index.finish(prev, found);
                prev.count = 0;
                prevPending = false;
            }
        }

        void operator()(JVec<B1B2>& chunkStates, JVec<u64>& chunkHashes, const u32 count) {
            streamed += count;
            for (u32 i = 0; i < count; ++i) {
                const u64 key = mixKey(chunkHashes[i]);
                const u32 p = static_cast<u32>(key >> PARTITION_SHIFT);
                if (p < index.pBegin || p >= index.pEnd) {
                    continue;
                }
                ++inRange;
                if constexpr (LEAF_BOUND) {
                    if (lane_bound::bound(chunkStates[i], *tables) > limit) {
                        ++boundSkipped;
                        continue;
                    }
                }
                if constexpr (LEAF_PDB) {
                    bool isFar = false;
                    for (u32 k = 0; k < pattern_db::g_tableCount && !isFar; ++k) {
                        isFar = pattern_db::g_tables[k]->farLeaf(chunkStates[i]);
                    }
                    if (isFar) {
                        ++boundSkipped;
                        continue;
                    }
                }
                ProbeBatch& b = batches[cur];
                b.states[b.count] = chunkStates[i];
                b.keys[b.count] = key;
                if (++b.count == variant.group) {
                    submit();
                }
            }
        }
    };

    // "<cpu brand>|<RAM GB>|<threads>": a calibration result belongs to one machine setup.
    inline std::string machineKey(const u32 threads) {
        char brand[49] = {};
        unsigned int regs[4];
        for (unsigned int leaf = 0; leaf < 3; ++leaf) {
            if (__get_cpuid(0x80000002 + leaf, &regs[0], &regs[1], &regs[2], &regs[3])) {
                std::memcpy(brand + leaf * 16, regs, 16);
            }
        }
        std::string cpu = brand;
        while (!cpu.empty() && cpu.front() == ' ') cpu.erase(cpu.begin());
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        const u64 ramGb = (status.ullTotalPhys + 512ULL * 1024 * 1024) / (1024ULL * 1024 * 1024);
        return cpu + "|" + std::to_string(ramGb) + "GB|" + std::to_string(threads) + "t";
    }

    inline fs::path probeProfilePath() {
        const char* local = std::getenv("LOCALAPPDATA");
        return fs::path(local != nullptr ? local : ".") / "MindbenderSolver" / "probe_profile.txt";
    }

    // Line format: "<machine key>\t<group>\t<pipeline>\t<ns/state best>\t<ns/state baseline>"
    inline bool readCachedVariant(const std::string& key, ProbeVariant& out) {
        std::ifstream in(probeProfilePath());
        std::string line;
        while (std::getline(in, line)) {
            const auto tab = line.find('\t');
            if (tab == std::string::npos || line.substr(0, tab) != key) {
                continue;
            }
            std::istringstream rest(line.substr(tab + 1));
            int g = 0, pl = 0;
            if (rest >> g >> pl && g > 0 && g <= static_cast<int>(MAX_PROBE_GROUP)) {
                out = {static_cast<u32>(g), pl != 0};
                return true;
            }
        }
        return false;
    }

    inline void writeCachedVariant(const std::string& key, const ProbeVariant& v, const double bestNs, const double baseNs) {
        const fs::path path = probeProfilePath();
        std::vector<std::string> lines;
        {
            std::ifstream in(path);
            std::string line;
            while (std::getline(in, line)) {
                if (line.rfind(key + "\t", 0) != 0 && !line.empty()) {
                    lines.push_back(line);
                }
            }
        }
        std::ostringstream entry;
        entry << key << '\t' << v.group << '\t' << (v.pipeline ? 1 : 0) << '\t' << bestNs << '\t' << baseNs;
        lines.push_back(entry.str());
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        std::ofstream out(path, std::ios::trunc);
        for (const std::string& l : lines) {
            out << l << '\n';
        }
    }

    // RAM for the index (or the depth-7 build's key buffers), measured on this
    // machine when the run starts: free physical memory minus a reserve for
    // the OS and other programs (max(3 GB, 10% of RAM)) and minus `otherBytes`
    // this run will also allocate (seeds, candidate buffers). At least 2 GB.
    // (Previously a flat 60% of free memory, which left ~8 GB idle and cost
    // whole extra passes, e.g. 12-1's 13.1 GB index ran in 2 passes.)
    // --ram-gb overrides it.
    inline u64 defaultRamBudgetBytes(const u64 otherBytes = 0) {
        MEMORYSTATUSEX status{};
        status.dwLength = sizeof(status);
        GlobalMemoryStatusEx(&status);
        const u64 reserve = std::max<u64>(3'000'000'000ULL, status.ullTotalPhys / 10);
        const u64 used = reserve + otherBytes;
        const u64 avail = status.ullAvailPhys > used ? status.ullAvailPhys - used : 0;
        return std::max<u64>(2'000'000'000ULL, avail);
    }

} // namespace ssd
