# src/solver/ssd/

The meet-in-the-middle solver for 12-16 moves, `experimentDiskDepth6`
(`src/programs/solvers/experimentDiskDepth6.cpp` is its `main` and documents
every flag). A search `<S, T2, T>` streams every exact depth-S left seed `T2`
more moves (`--stream-depth`) and looks each state up in the goal side's
depth-`T` set; the pruning bounds that skip most of that work are in
`src/solver/bounds/`.

Headers, in dependency order (each includes what it uses; all in `namespace ssd`):

| File | What |
|---|---|
| `common.hpp` | partition keys (`mixKey`), `Config`, `KeyedState`, partition file helpers, left seeds, run-wide globals |
| `cli.hpp` | command-line flags -> `Config` |
| `disk_depth6.hpp` | the depth-6 goal table on disk: 256 sorted partition files (`--disk-table`) |
| `ram_build.hpp` | `--ram-build`: the same table sorted in RAM |
| `batched_probe.hpp` | `--batched`: the oldest probe, exact states against RAM-sized batches |
| `fingerprint_index.hpp` | the compact goal-side index (Bloom blocks + 16-bit fingerprints), large pages |
| `probe_sink.hpp` | probe variants and the per-machine profile, the lookup sink, the RAM budget |
| `probe.hpp` | the probe loop: seeds in a strided order, bound audit + timing, one pass per index range |
| `no_table_d6.hpp` | no-table depth 6 (default for `--table-depth 6`): index slices from regenerated keys |
| `no_table_d7.hpp` | no-table depth 7 (default for `--table-depth 7`): count pass, owner-routed fills |
| `disk_depth7.hpp` | the index-only depth-7 table on disk (6-4's) and `--d7-sample` |
| `work_units.hpp` | middles files for `--shard`, `--middles-out`, `--recover-from` |
| `recovery.hpp` | middles -> full solutions, known-solution checks (`--solutions`) |

The program is one translation unit: `main` includes these headers once.
Docs: `CLAUDE.md` (roadmap, measured facts) and `docs/changes-*.md` (history).
