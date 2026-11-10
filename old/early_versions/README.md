# Early versions (2023 to January 2024)

The solvers written before this repository existed. None of it is built and
nothing outside this folder uses it. Copied here on 2026-11-10 from backups,
with the original file dates kept. Compiled `.class` files, IDE folders and
zip snapshots were left out.

| Folder | Last changed | Copied from | What it is |
|---|---|---|---|
| `java_2023-01/` | 2023-01-22 | `Y:\Dump\Lexar\Java\Chuzzle Solver` (also on F:) | Version 2, Java: the two-ended beam search, with the answer files it wrote (`levels/`) |
| `java_2023-04_2x2_prototype/` | 2023-04-20 | `C:\Users\jerrin\IdeaProjects\Chuzzle Fat Solver` | The 2x2 move table: 48 moves for each spot the piece can sit in. Prints the table; never searched |
| `cpp_2023-05_port/` | 2023-05 to 2023-09 | `C:\Users\jerrin\CLionProjects\Chuzzle Solver C++` | Version 3: the straight C++ port of version 2. It never worked |
| `java_2023-07_github/` | 2023-07-12 | github.com/zugebot/Chuzzle-Solver | The copy uploaded to GitHub, a cleanup of `java_2023-01` |
| `java_2024-01/` | 2024-01-05 | `Y:\Dev\eclipse-workspace\ChuzzleSolver` (also on F:); before that `D:\IdeaProjects\Chuzzle Solver` | The last Java version, with its answer files. The C++ rewrite (this repository) started three days later |

Version 1 (Python, January 2021) was not found on any drive; the oldest Python
files on this PC are from June 2021.

Each Java folder compiles on its own (checked 2026-11-10 with JDK 21):

```
javac -d out $(find . -name '*.java')
```

## How they worked

- **Search.** One move from the start, then one from the goal, in turn, up to
  20 moves (14 in `java_2024-01`). After each step the two newest layers are
  compared in a hash map; the first length with a match is written out as
  `levels/<puzzle>_c<moves>.txt`, every pair found, one per line.
- **Beam.** Each side keeps only its best-scoring boards: 300,000 in 2023,
  100,000 in `java_2024-01`. That is why none of these answers is a proof.
- **Score.** `getScore1` counts cells already matching the goal. `getScore2`
  takes, for each lane index, the best of the 6 rotations (row and column
  together) and sums them: the measure the rotation lane bound uses today
  (`src/solver/bounds/lane_bound.hpp`), but used to rank boards, not to prune.
  The 2023 sort key is `getScore2 * 36 + getScore1`; `java_2024-01` adds
  the two instead. `java_2024-01` also has
  `getScore3`, a greedy count of the rows and columns that cover every wrong
  cell, and an unfinished IDA* search (`IDA/`).
- **Canonical order.** Two moves in a row on the same axis may not go to a
  lower lane (`highSect`), as in the C++ solver; `java_2024-01` reads the
  allowed moves from a table indexed by the last move's axis and lane.
- **Boards.** `byte[6][6]`, colours as letters `R O Y G C B P W` (indices 0-7);
  `-` marks the 2x2 piece's cells. Each move is packed in one byte: 2 bits
  row or column, 3 bits lane, 3 bits shift.

## The 2x2 piece

The search never moved it. Each run listed the lanes it could move
(`allowed`), and the piece was first moved into place by hand: `toAdd` holds
those moves (written in their own notation, e.g. `R31 U21`, `U32 L21`) and
`toAddDepth` their count. So the 2x2 boards in `Levels.java` are the game's
boards after those hand moves: 4-2's goal and the starts of 5-1, 6-1, 6-2,
6-5, 8-2, 8-4 and 9-1 are each today's board (`src/chuzzles/levels_cells.hpp`)
after two 2x2 moves. Every other board matches today's data.

## Their answers against the proven ones

Replayed on 2026-11-10 against `build/levels_final/` (48 puzzles have an answer
file in one of the two `levels/` folders):

- 39 puzzles without a 2x2: 38 at the proven shortest length; 5-5 one move
  long (11, shortest 10). Few of the shortest solutions were found, e.g. 45 of
  4-5's 57,663.
- 9 puzzles with a 2x2, all longer than the shortest, counting the hand moves:
  4-2 10 (9), 4-4 8 (7), 5-1 11 (8), 6-1 15 (12), 6-2 17 (12), 6-5 12 (10),
  8-2 10 (8), 8-4 10 (8), 9-1 11 (8). The searched part of every one is valid
  on the boards in its own `Levels.java`.
- `java_2024-01/levels/10-1_c9.txt` is really 7-3: `Main.java` names the run
  "10-1" but loads 7-3's boards. Its 60 answers solve 7-3 in 9 moves, the
  shortest.
- `java_2023-01/levels/7-3_c10.txt` is an earlier 10-move run of 7-3, before
  `7-3_c9.txt` found 9.

Answer files use 0-indexed lanes: `R25` = row 3 shifted right 5, `C04` =
column 1 shifted down 4.
