#pragma once
// src/solver/ssd/work_units.hpp
//
// Middles files for work units (--shard k/N, --middles-out, --recover-from):
// written atomically with a header and an end line; read back with checks
// that the shards are 0..N-1 of one search, each exactly once and complete.

#include "src/solver/ssd/common.hpp"

namespace ssd {

    // ------------------------------------------------------------------
    // Middles files (work units)
    // ------------------------------------------------------------------
    inline void writeMiddlesFile(const std::string& path, const Config& cfg, const u64 seedsInShard,
                          const std::vector<B1B2>& middles) {
        const std::string tmp = path + ".part";
        {
            std::ofstream out(tmp, std::ios::trunc);
            out << "# mindbender middles v1\n"
                << "puzzle " << cfg.puzzle << "\n"
                << "split " << cfg.seedDepth << " " << cfg.streamDepth << " " << cfg.tableDepth << "\n"
                << "reverse " << (cfg.reverse ? 1 : 0) << "\n"
                << "shard " << cfg.shardIndex << " " << cfg.shardCount << "\n"
                << "seeds " << seedsInShard << "\n"
                << "count " << middles.size() << "\n";
            out << std::hex;
            for (const B1B2& m : middles) {
                out << m.b1 << " " << m.b2 << "\n";
            }
            out << std::dec << "end\n";
            if (!out) {
                throw std::runtime_error("failed to write " + tmp);
            }
        }
        fs::rename(tmp, path);   // only a complete file ever has the final name
        tcout << "middles written: " << path << " (" << middles.size() << " middles, shard " << cfg.shardIndex
              << "/" << cfg.shardCount << ")\n";
    }

    // Reads shards 0..N-1 of one search (each exactly once) and returns the
    // union of their middles; refuses mixed puzzles, splits or incomplete files.
    inline std::vector<B1B2> readMiddlesFiles(const std::vector<std::string>& paths, const Config& cfg) {
        std::vector<B1B2> all;
        std::vector<int> seen;
        u32 shardCount = 0;
        for (const std::string& path : paths) {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("cannot open " + path);
            }
            std::string line, puzzle;
            u32 s0 = 0, s1 = 0, s2 = 0, rev = 0, k = 0, n = 0;
            u64 count = 0;
            std::getline(in, line);
            if (line.rfind("# mindbender middles v1", 0) != 0) {
                throw std::runtime_error(path + ": not a middles file");
            }
            std::string key;
            in >> key >> puzzle >> key >> s0 >> s1 >> s2 >> key >> rev >> key >> k >> n >> key >> line >> key >> count;
            if (!in || puzzle != cfg.puzzle || s0 != cfg.seedDepth || s1 != cfg.streamDepth || s2 != cfg.tableDepth
                || (rev != 0) != cfg.reverse) {
                throw std::runtime_error(path + ": a different puzzle or split than this run");
            }
            if (shardCount == 0) {
                shardCount = n;
                seen.assign(n, 0);
            }
            if (n != shardCount || k >= n || seen[k]++ != 0) {
                throw std::runtime_error(path + ": shard " + std::to_string(k) + "/" + std::to_string(n)
                                         + " does not fit the other files (each shard exactly once)");
            }
            std::vector<B1B2> mine;
            in >> std::hex;
            for (u64 i = 0; i < count; ++i) {
                B1B2 m;
                in >> m.b1 >> m.b2;
                mine.push_back(m);
            }
            in >> std::dec >> line;
            if (!in || line != "end" || mine.size() != count) {
                throw std::runtime_error(path + ": incomplete");
            }
            all.insert(all.end(), mine.begin(), mine.end());
            tcout << "  " << path << ": shard " << k << "/" << n << ", " << count << " middles\n";
        }
        for (u32 k = 0; k < shardCount; ++k) {
            if (seen[k] == 0) {
                throw std::runtime_error("shard " + std::to_string(k) + "/" + std::to_string(shardCount) + " is missing");
            }
        }
        std::sort(all.begin(), all.end());
        all.erase(std::unique(all.begin(), all.end()), all.end());
        tcout << "merged " << paths.size() << " shard files: " << all.size() << " unique middles\n";
        return all;
    }

} // namespace ssd
