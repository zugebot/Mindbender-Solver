# Chuzzle Demo app

One pygame window that combines the old Chuzzle 1 tools: launch any Chuzzle
exe on the PC, browse every
mindbender and its optimal solutions, play a solution in the **real game**,
reset Speed Chuzzle, and track every place the game keeps save files.

```bash
python python/demo_app/run_demo.py
```

Needs Python 3.10+ and `pygame` (no pywin32/pyautogui; Windows calls go through
ctypes). For the best mouse routes, build the mouse-path DLL once (the app
falls back to plain drags without it):

```bash
C:/mingw64/bin/gcc -O3 -shared -Imouse/include mouse/src/chuzzle_mouse.c -o python/viewer/libchuzzle_mouse.dll -lm
```

(or build the `ChuzzleMouse` CMake target, which writes the same file).

## Launcher tab

Every Chuzzle exe on the PC, with a **Launch** and an **Open folder** button
each. A running game shows as *running* (with its window title) and its button
becomes **Bring to front**; a game started here becomes the app's game (bottom
bar, auto-play) as soon as its window opens.

- Each row: the edition (the shortest part of the folder path that tells the
  installs apart, plus the exe name when it is not `Chuzzle.exe`), the product
  version from the exe's version info, its build date (linker timestamp), size,
  and "PopCap DRM wrapper" for builds whose exe is PopCap's DRM loader with the
  game appended (the 2006 Deluxe 1.0).
- Games start from their own folder, like their shortcuts, so they find their
  data and keep saving where they always do.
- **Re-check list** (instant, also at every start): the cached exes that still
  exist plus the usual install folders (`saves.DEFAULT_GAME_SEARCH_ROOTS`).
- **Scan all drives**: walks every fixed drive once (minutes on big drives;
  skips Windows, Temp, recycle bins, `.git`, `node_modules` and similar) and
  caches the result in `user_data/launcher_cache.json`. It runs by itself the
  first time the app starts.

## Solutions tab

- **Grid**: 20 worlds x 5 puzzles. Green = solved, with its optimal move
  count; red = unsolved (no solution of 12 moves or fewer). A white dot marks
  the 2x2 piece.
- **Preview**: the board from `src/chuzzles/levels.hpp`, stepped or animated
  through a solution from `build/levels_final` (arrow keys / space work too).
  Solutions are ranked by mouse travel with the `python/viewer` library and
  cached in `user_data/route_rank_cache.json`. Moves are shown 1-indexed:
  `R14` = row 1 (top) shifted right 4, `C453` = columns 4+5 (2x2 piece) down 3.
- **Real game**: start Chuzzle (Deluxe 1.0/1.01, also when PopCap's wrapper
  runs it as `pop*.tmp`, or the Christmas Edition), open the mindbender,
  press *Find game*, then *Play solution in game* (or *Step in game*, one
  move per press).
  - **Input: background** (default): mouse messages are posted straight to the
    game window, so your real mouse stays free and the game needn't be in
    front. Before each move it unpauses the game if it shows "GAME PAUSED"
    (fake focus + a click on the empty pause band), and it waits while your
    cursor is over the game window, since real moves there would mix in.
  - **Input: real cursor**: moves the real mouse; hands off. Alt-tabbing
    away stops it.
  - Each press is held still for ~60 ms (a few frames) before the drag starts.
    Holding **Esc** or pressing *Stop* stops either mode at once and releases
    the button.
  - *Navigate in game* clicks the world arrows and level buttons. Tell it which
    world the game is showing first; it can't read the screen.
- **Calibrate** if clicks land off-target: hover the centre of the top-left
  chuzzle and press F9, then the bottom-right one and press F9. The result is
  saved per game exe in `user_data/calibration.json`. Without calibration
  the app uses the 800x600 layout from `python/deluxe_clone` (board at
  (255, 60), 80 px chuzzles), scaled to the window's client area.
## Speed Chuzzle tab

Shows the current profile of the game folder in use, its
`SAVEGAME-SPEEDCHUZZLE.DAT` (size, last write) and the Speed high scores from
`HSSpeedChuzzle.cfg`. *Reset Speed Chuzzle save* moves that file into
`user_data/save_backups/stashed`, keeping the newest 20 (the old resetter
deleted it), so the next Speed game starts fresh; the tab lists the stash.

## Saves tab

Scans every place Chuzzle 1 writes saves, shows profiles, file sizes and
times, highlights the folder the running (or last played) game uses (`>`)
and the `CurrentProfile` from `config.cfg` (`*`). It polls the files every
2 s and lists changes live. *Back up selected* copies a profile or a whole
location into `user_data/save_backups`. *Restore latest backup* asks for a
second click and backs up the current files first.

| Location | Written by |
|---|---|
| `<folder of Chuzzle.exe>\Profiles` and `\backup` | Portable/older builds, e.g. `E:\Games\Chuzzle Editions\Chuzzle.Deluxe\Chuzzle.Deluxe` |
| `%LOCALAPPDATA%\VirtualStore\<exe folder>\Profiles` | Those builds run from Program Files without admin |
| `%ProgramData%\PopCap Games\Chuzzle\Profiles` and `\backup` | PopCap installer version |
| `Documents\my games\Chuzzle\Profiles` (config one level up) | Christmas Edition |
| `%ProgramData%\Steam\Chuzzle*\userdata` | Steam PopCap builds (convention, unverified) |
| `%APPDATA%\Chuzzle2\_cloud\<user>` | Chuzzle 2's Chuzzle 1 modes |
| `Documents\my games\Backup_Profiles` | Manual backups |

Game folders are searched under `E:\Games\Chuzzle Editions`, PopCap Games in
Program Files, and Steam's `common` folder (`saves.DEFAULT_GAME_SEARCH_ROOTS`).
The running game's exe is always added.

## Layout

| File | Role |
|---|---|
| `chuzzle_demo/levels.py` | boards from `src/chuzzles`, solutions from `build/levels_final`, 1-indexed display |
| `chuzzle_demo/board_sim.py` | applies moves (incl. the 2x2 piece); replays all ~100k stored solutions |
| `chuzzle_demo/routes.py` | solution -> drags via the mouse library (or simple drags); ranking + cache |
| `chuzzle_demo/game_window.py` | window lookup by process, DPI-aware client coords, calibration, mouse input |
| `chuzzle_demo/player.py` | worker thread that performs drags with the safety stops |
| `chuzzle_demo/saves.py` | save-location discovery, watcher, backup / restore / stash |
| `chuzzle_demo/launcher.py` | finds every Chuzzle exe (drive scan + cache), version / build date / DRM wrapper, launching |
| `chuzzle_demo/ui.py` | the pygame window |

Tests: `python -m unittest discover -s python/demo_app/tests`.
`user_data/` is git-ignored.
