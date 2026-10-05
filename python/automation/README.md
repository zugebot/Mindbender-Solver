# Chuzzle 1 automation tools

Scripts that drive the real Chuzzle Deluxe / Christmas Edition window (or its
save files), gathered from older standalone projects so they can be combined
into one demo app. `legacy/` holds them **as they were** (copied, not
modified); the originals are still in `C:\Users\jerrin\PycharmProjects`.

| Folder | Original project | What it does |
|---|---|---|
| `legacy/speed_resetter/` | `PycharmProjects/ChuzzleResetter` | Hotkey (space) Speed Chuzzle reset: deletes `SAVEGAME-SPEEDCHUZZLE.DAT` and clicks through the menus at window-relative positions. Includes the PyInstaller `.spec`. |
| `legacy/mindbender_macro/` | `PycharmProjects/ChuzzlePuzzleMacro` | Plays a mindbender solution by dragging pieces in the real window (`ChuzzleWindow.solve("5-3")`); profile creation; button positions as window-relative fractions. `levels/` are its old one-solution-per-level files. |
| `legacy/speed_save_reader/` | `PycharmProjects/Chuzzle 1 Editor` | Reads the Speed Chuzzle save format (score, scrambles, bottle, speed bar). Includes sample saves and hex dumps. |
| `legacy/screen_bot/` | `PycharmProjects/PythonOldCode/chuzzles2.py` | Early bot: screen-grabs the board, simulates row/column moves, clicks. |

Related code already in this repo: `python/viewer/` (mouse-route studio),
`mouse/` (C mouse-path DP library), `src/programs/DecompMindbenderScramble.cpp`
(the game's scramble), and `build/levels_final/` (complete optimal solutions —
the source a unified "show the solution" demo should use instead of
`mindbender_macro/levels/`).

## Common pieces to unify

All three clicking tools reimplement the same things: find the window by title
(`win32gui.EnumWindows`), map window-relative fractions to screen pixels, and
click/drag with `pyautogui`. A combined app needs one shared window/coordinate
module, the button and board-cell positions, and actions built on top
(reset Speed Chuzzle, open a mindbender, play/show a solution).
