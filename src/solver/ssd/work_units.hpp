#pragma once
// src/solver/ssd/work_units.hpp
//
// Middles files for work units (--shard k/N, --middles-out, --recover-from):
// written atomically with a header and an end line; read back with checks
// that the shards are 0..N-1 of one search, each exactly once and complete.
// With --d7-groups G the header also says "d7groups G": unit k covers goal partition group
// k / (N/G) and seed shard k % (N/G), so units 0..N-1 again cover the whole search.

#include "src/solver/ssd/common.hpp"

#include <fstream>
#include <sstream>

namespace ssd {

    // ------------------------------------------------------------------
    // Middles files (work units)

    // The direction a middles file was searched in (its "reverse" line): 1 reverse, 0 forward,
    // -1 unreadable. --recover-from with --direction auto follows it.
    inline int middlesFileReverse(const std::string& path) {
        std::ifstream in(path);
        std::string line;
        for (int i = 0; i < 16 && std::getline(in, line); ++i) {
            if (line.rfind("reverse ", 0) == 0) {
                return line.size() > 8 && line[8] == '1' ? 1 : 0;
            }
        }
        return -1;
    }
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
                << "shard " << cfg.shardIndex << " " << cfg.shardCount << "\n";
            if (cfg.d7Groups > 1) {
                out << "d7groups " << cfg.d7Groups << "\n";   // older readers: only ever G = 1
            }
            if (cfg.blockJoin) {
                out << "block 1\n";   // middles at left depth seed + stream + 1
            }
            out << "seeds " << seedsInShard << "\n"
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
        u32 shardCount = 0, groups = 0;
        for (const std::string& path : paths) {
            std::ifstream in(path);
            if (!in) {
                throw std::runtime_error("cannot open " + path);
            }
            std::string line, puzzle;
            u32 s0 = 0, s1 = 0, s2 = 0, rev = 0, k = 0, n = 0, g = 1, block = 0;
            u64 count = 0;
            std::getline(in, line);
            if (line.rfind("# mindbender middles v1", 0) != 0) {
                throw std::runtime_error(path + ": not a middles file");
            }
            // "key value" lines up to and including "count"
            bool haveCount = false, haveShard = false;
            while (!haveCount && std::getline(in, line)) {
                std::istringstream kv(line);
                std::string key;
                kv >> key;
                if (key == "puzzle") kv >> puzzle;
                else if (key == "split") kv >> s0 >> s1 >> s2;
                else if (key == "reverse") kv >> rev;
                else if (key == "shard") haveShard = static_cast<bool>(kv >> k >> n);
                else if (key == "d7groups") kv >> g;
                else if (key == "block") kv >> block;
                else if (key == "count") haveCount = static_cast<bool>(kv >> count);
            }
            if (!haveCount || !haveShard || puzzle != cfg.puzzle || s0 != cfg.seedDepth || s1 != cfg.streamDepth
                || s2 != cfg.tableDepth || (rev != 0) != cfg.reverse || (block != 0) != cfg.blockJoin) {
                throw std::runtime_error(path + ": a different puzzle or split than this run");
            }
            if (shardCount == 0) {
                shardCount = n;
                groups = g;
                seen.assign(n, 0);
            }
            if (g == 0 || n % g != 0 || g != groups) {
                throw std::runtime_error(path + ": d7groups " + std::to_string(g) + " does not fit the other files");
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
            tcout << "  " << path << ": shard " << k << "/" << n
                  << (g > 1 ? " (goal partition group " + std::to_string(k / (n / g)) + " of " + std::to_string(g)
                              + ", seed shard " + std::to_string(k % (n / g)) + "/" + std::to_string(n / g) + ")"
                            : std::string())
                  << ", " << count << " middles\n";
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
