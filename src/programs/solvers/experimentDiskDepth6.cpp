// src/programs/solvers/experimentDiskDepth6.cpp
//
// The SSD meet-in-the-middle solver. The code is in src/solver/ssd/ (see its
// README.md) and the pruning bounds in src/solver/bounds/.
//
// Experiment: <S, 3, T> meet-in-the-middle with the right frontier on disk
// (S = --seed-depth, default 3; T = --table-depth 6 (default) or 7).
// <3,3,6> = 12 moves, <4,3,6> = 13, <5,3,6> = 14, <5,3,7> = 15.
//
//   Build:  exact depth-5 frontier in RAM -> expand every state by one move ->
//           route each child to one of 256 partition files by the top 8 bits of
//           a mixed state key -> sort + de-dupe each partition independently.
//           --ram-build: regenerate the children per RAM-sized group of
//           partitions, sort them in memory and write the table + index in one
//           go (no raw spill, no re-read). Correct, but on a drive whose
//           sustained write speed has collapsed it is slower, because the disk
//           build's writes are absorbed by the OS file cache.
//   Probe (default): load a compact per-partition index (Bloom filter + u16
//           fingerprints + bucket offsets, ~4.25 B/state) for as many partitions
//           as fit --ram-gb, stream the left side once per such range, and
//           confirm fingerprint hits against the full states on disk.
//   Probe (--batched): load a contiguous range of sorted partitions that fits the RAM
//           budget, index it with a direct-address bucket table, then regenerate
//           the whole left side (exact depth-S seeds, canonical depth-3 stream)
//           and probe only states whose key falls in that range. Repeat per range.
//   Depth 7 (--table-depth 7): the depth-7 states (~46 G for a fat goal,
//           ~750 GB as states) are never stored. Every child of the depth-6
//           table is kept only as a 64-bit key, a few partitions per pass over
//           the depth-6 table, then sorted, de-duped and turned into the same
//           compact index (Bloom + u16 fingerprints + bucket offsets, ~3.4
//           B/state). A fingerprint hit L is confirmed exactly by checking that
//           one of L's one-move neighbours is in the exact depth-6 table.
//           --d7-partitions N [--d7-first-partition F] builds and probes only
//           key partitions [F, F+N), a cheap way to check results against known
//           solutions: the middles and solutions whose middle falls in those
//           partitions must match exactly.
//   No table (default for depth 6 when no table is on disk yet): nothing is
//           written. The depth-5 frontier stays in RAM; each probe pass builds
//           its slice of the fingerprint index in RAM straight from the keys of
//           the regenerated depth-6 children, and the fingerprint hits are
//           confirmed by regenerating depth 6 once more. The drive's (slow)
//           write speed never matters. --disk-table forces the on-disk table.
//           With --table-depth 7 the same holds one level deeper: each pass
//           fills its index slice straight from the regenerated depth-7
//           descendants of the in-RAM depth-5 frontier (raw fingerprints in
//           counted buckets, de-duplicated per bucket afterwards).
//   Recover: full solutions through every middle via BoardSolverFrontier.
//
// All disk I/O is sequential. With --solutions, both the middle states and the
// recovered solution set are checked against a complete known solution file.
//
// Usage:
//   experimentDiskDepth6 <puzzle> [--out X:/mindbender_d6] [--batch 320000000]
//                        [--threads N] [--seed-depth S] [--reverse] [--rebuild]
//                        [--no-recover] [--batched] [--ram-gb GB] [--no-large-pages]
//                        [--solutions file]
//                        [--table-depth 6|7 [--d7-partitions N [--d7-first-partition F]]]
//                        [--progress-file path.json [--progress-interval SECONDS]]
//                        [--probe-variant G:P] [--no-calibrate] [--calibrate-seconds S]
//
// Probe self-calibration: the first ~20 s of the first probe pass run the real
// seeds through every probe variant (batch size G = 32/64/128 lookups, P =
// pipelined: the next batch's memory fetches are issued before the current
// batch is tested), interleaved seed by seed so they share the same memory and
// CPU conditions. The fastest variant then runs the rest of the search. That
// work is part of the search (nothing is repeated). Results are cached per
// machine in %LOCALAPPDATA%/MindbenderSolver/probe_profile.txt; --no-calibrate
// uses the cached best, --probe-variant forces one.
//
// --bound auto|off|leaf|both: an admissible lane lower bound
// (--bound-kind rotation, the default: per row / column, the fewest cells wrong
// over its 6 rotations; or mix: colour multisets only)
// (src/solver/bounds/lane_bound.hpp) skips left states that cannot be within
// the table depth of the right root ("leaf": before their lookup) and, on
// normal boards, whole last stream levels under nodes that cannot have such a
// child ("both": inside PermStream). auto audits the bound on this run's own
// seeds (fast tables vs the reference definition, |dh| <= 1 on real edges,
// and pruned vs full streams keeping identical wanted leaves), then times
// every eligible mode on real seeds and keeps the fastest: blocks of seeds,
// one mode on all threads per block, rotating; clearly slower modes are
// dropped and timing goes on until the rest are told apart, up to 3% of the
// search (20 s minimum, --bound-calibrate-seconds maximum, default 300).
// Any audit failure turns it off. 2x2 boards: off (the halved bound prunes
// nothing).
// --pdb auto|off|on adds grouped pattern databases (src/solver/bounds/pattern_db.hpp):
// up to 3 merged-colour classes solved exactly at startup (2 bits per entry),
// asked at the same parent nodes ("pdb") and optionally at leaves
// ("pdbleaf"); audited the same way and timed with the other modes. auto:
// normal boards, 14+ moves (seed depth >= 5), when colour-mix leaves room.
// --stream-depth 4 (default 3): a 4-move left stream from the seeds, so
// <5,4,6> = 15 moves; the parent prune then runs at every level above the
// leaves (a node with r moves below is dropped when h > depth + r).
// --stream-depth 5: <5,5,6> = 16 moves with the one-pass depth-6 goal side
// (normal boards; the bound audits sample fewer seeds at this depth).
// --recover-stream-depth D (default 5): recovery streams every per-middle half of D+ moves
// instead of storing all its paths (left depth 10 = two depth-5 halves; stored, the per-middle
// one alone is ~30 GB). Lower it to test the streamed path against known solutions.
// Depth-7 goal side (no table): --no-goal-filter, --d7-counts-ram and
// --d7-partitions N [--d7-first-partition F], see ssd/no_table_d7.hpp.
// "parent" / "pdbparent" prune only at parents, with no per-leaf check: on
// puzzles where the leaf check skips little (16-4: 39%) it costs more than
// the lookups it saves, while a parent check covers ~40 leaves.
//
// Work units (searches split across runs or machines):
//   --shard k/N         search only the left seeds with index = k (mod N); the
//                       N shards together are exactly the full search
//   --middles-out FILE  write the confirmed middles (text, with the puzzle,
//                       split and shard in the header and an "end" line)
//   --recover-from FILE (repeatable) skip the search: merge the middles of
//                       shards 0..N-1 of one search (each exactly once, same
//                       puzzle and split), then recover and validate as usual
//
// --progress-file: a background thread rewrites a small JSON status (phase,
// done/total units, elapsed) every few seconds for python/scheduler; workers
// only add to an atomic counter once per seed / partition. Log lines are
// flushed as they are written, so a redirected log can be followed live.
//                        [--d7-sample N [--d7-pass-budget STATES]]

#include "src/solver/ssd/batched_probe.hpp"
#include "src/solver/ssd/cli.hpp"
#include "src/solver/ssd/disk_depth6.hpp"
#include "src/solver/ssd/disk_depth7.hpp"
#include "src/solver/ssd/no_table_d6.hpp"
#include "src/solver/ssd/no_table_d7.hpp"
#include "src/solver/ssd/ram_build.hpp"
#include "src/solver/ssd/recovery.hpp"
#include "src/solver/ssd/work_units.hpp"

using namespace ssd;

int main(const int argc, char** argv) {
    const Config cfg = parseArgs(argc, argv);
    g_largePagesWanted = cfg.largePages;
    tcout << std::unitbuf;   // flush every line: a redirected log can be followed live
    g_progress.start(cfg.progressFile, cfg.progressInterval);
    const BoardPair* pair = BoardLookup::getBoardPair(cfg.puzzle);
    if (pair == nullptr) {
        tcout << "Unknown puzzle: " << cfg.puzzle << '\n';
        return 1;
    }

    const Board start = pair->getStartState();
    const Board goal = pair->getEndState();
    const Board& leftRoot = cfg.reverse ? goal : start;
    const Board& rightRoot = cfg.reverse ? start : goal;
    g_rightRoot = rightRoot.asB1B2();
    const fs::path dir = cfg.outRoot / (cfg.puzzle + (cfg.reverse ? "_start_d6" : "_goal_d6"));
    const u32 leftDepth = cfg.seedDepth + cfg.streamDepth;
    const u32 totalDepth = leftDepth + cfg.tableDepth;

    if (cfg.streamDepth != LEFT_STREAM_DEPTH && cfg.batched) {
        throw std::runtime_error("--batched supports only --stream-depth 3");
    }
    tcout << "puzzle " << cfg.puzzle << " <" << cfg.seedDepth << "," << cfg.streamDepth << ","
          << cfg.tableDepth << "> = "
          << totalDepth << " moves, " << (cfg.reverse ? "reverse" : "forward")
          << ", threads " << cfg.threads << ", table dir " << dir.string() << '\n';

    const Timer total;
    if (cfg.batched && cfg.shardCount > 1) {
        throw std::runtime_error("--batched does not support --shard");
    }
    std::vector<B1B2> middles;
    double probeSeconds = 0.0;
    double buildSeconds = 0.0;
    if (!cfg.recoverFrom.empty()) {
        tcout << "recovering from " << cfg.recoverFrom.size() << " shard file(s); no search\n";
        middles = readMiddlesFiles(cfg.recoverFrom, cfg);
        tcout << "unique middle states (" << leftDepth << " from left, " << cfg.tableDepth << " from table side): "
              << middles.size() << " (merged shards)\n";
    } else {
    std::vector<u64> counts = cfg.rebuild ? std::vector<u64>{} : loadPartitionCounts(dir);
    // No table unless one is already on disk (with its index) or the run needs it.
    const bool noTable = cfg.tableDepth == 6 && !cfg.reverse && !cfg.batched && !cfg.diskTable && !cfg.ramBuild
                         && cfg.d7Sample == 0 && (counts.empty() || !fs::exists(fp16MetaPath(dir)));
    // Depth 7: no table unless a complete depth-7 index is already on disk (6-4).
    const bool noTable7 = cfg.tableDepth == 7 && !cfg.reverse && !cfg.diskTable
                          && cfg.d7Sample == 0 && !fs::exists(d7Files(dir).meta);
    if (noTable || noTable7) {
        tcout << "no-table mode: nothing is written to " << dir.string() << "\n";
    } else if (counts.empty()) {
        const Timer buildTimer;
        counts = cfg.ramBuild ? buildDepth6InRam(rightRoot, dir, cfg, cfg.threads)
                              : buildDepth6OnDisk(rightRoot, dir, cfg.threads);
        buildSeconds = buildTimer.getSeconds();
        tcout << "BUILD depth-6 table: " << buildSeconds << "s\n";
    } else {
        u64 n = 0;
        for (const u64 c : counts) n += c;
        tcout << "reusing depth-6 table: " << n << " states (" << gbOf(n) << " GB)\n";
    }

    if (cfg.d7Sample > 0) {
        sampleDepth7(rightRoot, dir, counts, cfg.threads, cfg.d7Sample, cfg.d7PassBudget);
        return 0;
    }

    const Timer probeTimer;
    if (cfg.d7Subset()) {
        tcout << "depth-7 TEST SUBSET: only key partitions [" << cfg.d7FirstPartition << "," << cfg.d7End()
              << ") of " << PARTITION_COUNT << "\n";
    }
    middles = noTable
            ? probeNoTable(leftRoot, rightRoot, cfg, cfg.threads)
            : noTable7
            ? probeNoTableDepth7(leftRoot, rightRoot, cfg, cfg.threads)
            : cfg.tableDepth == 7
            ? probeDepth7(leftRoot, rightRoot, dir, counts, cfg, cfg.threads)
            : cfg.batched
            ? probeAllBatches(leftRoot, dir, counts, cfg, cfg.threads)
            : probeFingerprint(leftRoot, dir, counts, cfg, cfg.threads);
    probeSeconds = probeTimer.getSeconds();
    tcout << "unique middle states (" << leftDepth << " from left, " << cfg.tableDepth << " from table side): "
          << middles.size() << '\n';
    tcout << "PROBE phase: " << probeSeconds << "s\n";
    if (!cfg.middlesOut.empty()) {
        writeMiddlesFile(cfg.middlesOut, cfg, 0, middles);
    }
    if (cfg.shardCount > 1) {
        tcout << "SHARD " << cfg.shardIndex << "/" << cfg.shardCount << " complete: " << middles.size()
              << " middles (a partial search: no solution validation)\n";
    }
    }

    int status = 0;
    std::vector<std::string> known;
    if (!cfg.solutionsFile.empty()) {
        known = readSolutionLines(cfg.solutionsFile);
        if (cfg.d7Subset()) {
            // Only solutions whose middle falls in the probed partitions can be found.
            std::vector<std::string> kept;
            std::vector<u32> middleParts;
            for (const std::string& line : known) {
                const u32 part = static_cast<u32>(keyOf(middleOf(*pair, line, leftDepth)) >> PARTITION_SHIFT);
                middleParts.push_back(part);
                if (part >= cfg.d7FirstPartition && part < cfg.d7End()) {
                    kept.push_back(line);
                }
            }
            std::sort(middleParts.begin(), middleParts.end());
            middleParts.erase(std::unique(middleParts.begin(), middleParts.end()), middleParts.end());
            tcout << "known middles lie in partitions:";
            for (const u32 part : middleParts) {
                tcout << ' ' << part;
            }
            tcout << "\ntest subset: " << kept.size() << " of " << known.size()
                  << " known solutions have their middle in partitions [" << cfg.d7FirstPartition << ","
                  << cfg.d7End() << ")\n";
            known = kept;
        }
        // The middle sits leftDepth moves from the left root; counted from the
        // true start that is leftDepth (forward) or the table depth (reverse) moves in.
        const std::vector<B1B2> expected = expectedMiddles(*pair, known, cfg.reverse ? cfg.tableDepth : leftDepth);
        std::vector<B1B2> missing, extra;
        std::set_difference(expected.begin(), expected.end(), middles.begin(), middles.end(),
                            std::back_inserter(missing));
        std::set_difference(middles.begin(), middles.end(), expected.begin(), expected.end(),
                            std::back_inserter(extra));
        tcout << "middle validation vs " << cfg.solutionsFile << ": expected " << expected.size()
              << ", missing " << missing.size() << ", extra " << extra.size()
              << (missing.empty() && extra.empty() ? "  -> EXACT MATCH\n" : "  -> MISMATCH\n");
        status = missing.empty() && extra.empty() ? 0 : 2;
    }

    double recoverSeconds = 0.0;
    if (cfg.recover && !middles.empty()) {
        g_progress.phase("recovery", middles.size(), "full solutions through the middles");
        const Timer recoverTimer;
        BoardSolverFrontier solver(pair);
        fs::create_directories(dir / "solutions");
        solver.setWriteDirectory((dir / "solutions").string());
        const int found = recoverSolutions(solver, cfg, middles);
        recoverSeconds = recoverTimer.getSeconds();
        tcout << "RECOVERY: " << found << " solutions of " << totalDepth << " moves in "
              << recoverSeconds << "s (written under " << (dir / "solutions" / "levels").string() << ")\n";

        if (!known.empty()) {
            const std::set<std::string> knownSet(known.begin(), known.end());
            const bool same = knownSet == solver.getExpandedSolutions();
            tcout << "solution validation: known " << knownSet.size() << ", recovered "
                  << solver.getExpandedSolutions().size()
                  << (same ? "  -> IDENTICAL SET\n" : "  -> SETS DIFFER\n");
            if (!same) {
                status = 2;
            }
        }
    }

    tcout << "TOTAL: " << total.getSeconds() << "s (build " << buildSeconds << "s, probe "
          << probeSeconds << "s, recovery " << recoverSeconds << "s)\n";
    g_progress.phase("done", 0, "exit " + std::to_string(status));
    g_progress.stop();
    return status;
}
