# Chuzzle Suite

One pygame window that combines the old Chuzzle 1 tools: launch any Chuzzle
exe on the PC, browse every
mindbender and its optimal solutions, play a solution in the **real game**,
run a tool-assisted speedrun of Mind Bender for its trophies, reset Speed
Chuzzle, track every place the game keeps save files, and watch the solver queue (with a small always-on-top view).

```bash
python python/chuzzle_suite/run_suite.py
```

As a Windows program (Start menu entry with the app's icon, no console window; output goes to
`user_data/chuzzle_suite.log`):

```bash
python python/chuzzle_suite/install_shortcut.py
```

Then Start menu -> Chuzzle Suite -> right-click -> Pin to taskbar (Windows 10 lets no program pin
itself). The shortcut and the app share one AppUserModelID (`paths.APP_ID`), so the pinned icon
and the open window are a single taskbar button, also when the app was started from a console.
Run the script again after moving the repo or changing Python (`--desktop` adds a desktop
shortcut, `--remove` deletes them; it needs pywin32, the app itself does not). The icon is drawn
by `assets/make_icon.py` (`assets/chuzzle_suite.ico`, `.png`).

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

- **Refresh** (top right of the grid, or F5): reloads `build/levels_final` and the bounds in
  `build/README.md` without restarting, e.g. after the solver queue solves a puzzle. The log
  names newly solved puzzles; their solutions are ranked again.
- **Grid**: 20 worlds x 5 puzzles. Green = solved, with its optimal move
  count; red = unsolved (no solution of 12 moves or fewer). A white dot marks
  the 2x2 piece; a yellow dot marks a puzzle whose goal the game also accepts
  with the piece elsewhere.
- **Goal switch** (*< Goal* / *Goal >*, bottom right of the preview, only on those
  puzzles): the game counts a board as solved when every colour matches, so the
  2x2 piece may also end on another 2x2 block of its colour. Each such placement
  has its own solutions (`build/levels_final/game_accepts/`, e.g. `4-4@r2c3` =
  piece at rows 2-3, columns 3-4) and is labelled "not the official goal" in
  yellow. It plays in the game like any other solution.
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
    **Cursor sprite** (on by default, *show a cursor sprite* toggles it): since
    the real cursor doesn't move, a pointer sprite follows the posted mouse
    (yellow with a ring while the button is held) so recordings show the
    moves. It is its own always-on-top, click-through window: your mouse and
    the game are unaffected. Record with display or region capture; a capture
    of the game window alone does not include it.
  - **Input: real cursor**: moves the real mouse; hands off. Alt-tabbing
    away stops it.
  - Moves between drags follow a gentle arc (alternating sides) with a
    hand-like speed curve: quick start, slower approach. The drags themselves
    follow the solution's routes exactly.
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
## TAS tab

A tool-assisted speedrun of Mind Bender in the **real game**, for its four trophies, with the
real mouse (the successor of the 2023 PUZZLER run, `python/automation/legacy/mindbender_macro`).

1. Make a new profile in the game (or use one whose Mind Bender save is stashed, below), open
   Mind Bender: level 1, puzzle 1 shows, scrambled. Press *Find game* in the bottom bar.
2. Pick the trophy, the puzzles per level and how far it goes; tick the trophies the profile
   already has (or *Read profile*).
3. *Start run*, then once more within 3 s (it takes over the mouse). Hands off; holding **Esc**
   stops it at once (also Stop, the game window closing or losing the foreground), and the button
   is always released.

Options:

- **Trophy to earn**: PUZZLER (reach level 10: levels 1-9, ends with the slide into 10), PSYCHIC
  (all 5 puzzles of levels 1-10), MENTALIST MASTER (reach level 20: levels 1-19), MENTALIST
  SUPREMO (all 100). The trophy sets the defaults of the next two.
- **Puzzles per level**: the 3 fastest (3 solved open the next level) or all 5. PSYCHIC and
  SUPREMO need all 5.
- **Last level played**: shorter practice runs (no target trophy; the timer stops at the end).
- **The profile already has**: those trophies do not pop up (the game skips them). A pop-up costs
  about 5.6 s: it takes a click only after 4.0-4.3 s, the click opens the trophy room, then
  "Return to Game". On the way, PSYCHIC also earns PUZZLER, MENTALIST MASTER earns PUZZLER (and
  PSYCHIC with all 5), SUPREMO earns all four. *every other trophy too*: the profile has all 21
  others, so the last Mind Bender trophy also brings TROPHY COLLECTOR. *Read profile* sets these
  from the profile's `INFO.CFG` (`T<n>` keys).
- **LiveSplit key**: numpad 1 before the first input, and again when the target trophy is earned
  (when the slide into level 10/20 ends, or at the last solve), before its pop-up.
- **Check the board on screen** (on by default; off for record attempts): before the start the
  board must be 1-1 scrambled, and before each level arrow every puzzle played on the level must
  show its solved star (the green check). Otherwise the run stops instead of playing on a wrong
  board. It costs a screen capture per level.
- **Mind Bender save**: the profile's `SAVEGAME-MINDBENDER.DAT` (level shown, puzzles solved).
  The game resumes it, so a run needs a profile without one. *Stash the save* (two clicks) moves
  it into `user_data/save_backups/stashed` like the Speed Chuzzle reset. The game writes it when
  it leaves Mind Bender, so stash it from the main menu.

Both versions, with and without pop-ups: run on a new profile (all pop-ups), go to the main menu,
*Stash the save*, *Read profile* (it now has them), run again (no pop-ups).

**Puzzle choice** (`tas_plan.py`): the cursor jumps, so mouse travel costs nothing; what costs
time is drags and clicks. Each level plays the puzzles with the fewest moves; on a tie the set
with puzzle 1 wins, because entering a level loads puzzle 1 by itself. Each puzzle uses its
shortest goal, including a 2x2 piece placement the game also accepts (4-4 in 6 moves instead of
7). The PUZZLER run is 27 puzzles, 178 moves (the 2023 picks on levels 5, 8 and 9 were 10 moves
and a click more). A solved puzzle's banner opens the next puzzle when that is the next one
played; otherwise its star is clicked.

**Input** (`tas_runner.py`): every input goes through one `SendInput` stream (absolute cursor
moves, button and key events) so the press, moves and release reach the game in the order sent;
`SetCursorPos` is synchronous while button events are queued, so mixing them (as in 2023) needs
waits. A drag is: press on the lane, a 4 px nudge along it (the game starts a drag after 2 px and
counts the offset from that move), the jump to one nudge past the target cell, release.

**Speed**: the game handles every queued message at the start of each 10 ms update, and nothing
in a run waits for an animation (no scramble animation, the banner is live at the solve, input
works during the level slide). The only limit is Windows merging cursor moves the game has not
read yet: the nudge and the jump must arrive as two messages. *Measure speed* (two clicks; the
game must show a scrambled puzzle of 2+ moves) replays that puzzle's solution over and over,
reading the board after each try: first with an F24 key press between nudge and jump and no wait,
else the shortest wait that solves every try (plus a third). The result is kept per game exe in
`user_data/tas_timing.json`; *Reset timing* goes back to the default (a 25 ms wait).

The game's behaviour was read in the decomp of the 2008 Christmas exe (`X:/chuzzle-decomp`; the
Mind Bender logic is the same in the 2005 and 2007 builds) and checked there with its replay
harness (the original exe, scripted input at exact frames, a sandboxed profile): whole PUZZLER,
PSYCHIC and MENTALIST MASTER runs of the runner's inputs, and these timings:

| After | Until | Updates |
|---|---|---|
| arrow click into level 10 / 20 | the pop-up takes a press | 432 |
| the solve that earns 22 / 24 | the pop-up takes a press | ~400 |
| that press | "Return to Game" works (pointer on it an update before) | ~62 |
| "Return to Game" | the board takes drags | ~62 |
| a solving release | the star's green check is drawn | 4 |

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

## Queue tab

The solver scheduler (`python/scheduler/mbq.py`) inside the app. While the tab
(or the mini view) is on screen, a background thread asks the daemon for
`GET /status` every 2.5 s; the rest of the app never waits for it. The lines
are `mbq status`'s: the running job (puzzle, moves, split such as `<5,4,7>`,
gpu), its phase with a progress bar, elapsed time, time left and ETA, the
job's notes, then the queued jobs, one line each. "SOLVER ON HOLD since ..."
and "queue is PAUSED" show above them.

- **Hold solver / Unhold solver**: `POST /hold` / `/unhold` (freeze the solver
  in place, then let it carry on).
- **Pause queue / Resume queue**: `POST /pause` / `/resume`. Pause asks for a
  second click within 3 s: it stops the queue from starting the next job.
- **Mini view**: the small window below. **Open dashboard**: the daemon's page.
- When the daemon is not reachable (refused or timed out) the tab says
  "scheduler not running" and the buttons are off.
- The address is `queue_client.DEFAULT_URL` (`http://127.0.0.1:8766`); the
  environment variable `CHUZZLE_SUITE_MBQ_URL` points the app at another server,
  e.g. a fake one for tests (`tests/test_queue.py` has one).

## Title bar and mini view

The window is borderless (`pygame.NOFRAME`) with the app's own dark title bar:
pin, minimize, maximize (greyed: the layout has a fixed size) and close. The
content keeps its 1200x840 layout below the 32 px bar.

- **Pin**: keeps the window on top of other windows (`SetWindowPos`
  `HWND_TOPMOST`). A pinned pin is upright and blue; unpinned, it is a tilted
  outline. Remembered in `user_data/settings.json`. It applies to whichever
  view is showing.
- Dragging the empty part of the bar moves the window (Windows does the drag).
  The window has `WS_MINIMIZEBOX | WS_SYSMENU`, so its taskbar button minimizes
  and restores it and Alt+Space opens the system menu. Close (the X, Alt+F4,
  the taskbar) goes through the app's normal quit.
- **Mini view** (440x150): the running job, phase, progress bar, time left and
  ETA, the number queued, hold and pause state, and Hold / Pause buttons. Its
  title bar has pin, expand (back to the full view) and close. Switching views
  keeps the window's top-right corner in place (moved inside the screen if it
  would stick out).

## Layout

| File | Role |
|---|---|
| `chuzzle_suite/levels.py` | boards from `src/chuzzles`, solutions from `build/levels_final`, 1-indexed display |
| `chuzzle_suite/board_sim.py` | applies moves (incl. the 2x2 piece); replays all ~100k stored solutions |
| `chuzzle_suite/routes.py` | solution -> drags via the mouse library (or simple drags); ranking + cache |
| `chuzzle_suite/game_window.py` | window lookup by process, DPI-aware client coords, calibration, mouse input |
| `chuzzle_suite/player.py` | worker thread that performs drags with the safety stops |
| `chuzzle_suite/cursor_overlay.py` | the cursor sprite: a click-through layered window that shows the posted pointer |
| `chuzzle_suite/saves.py` | save-location discovery, watcher, backup / restore / stash; a profile's trophies and Mind Bender save |
| `chuzzle_suite/launcher.py` | finds every Chuzzle exe (drive scan + cache), version / build date / DRM wrapper, launching |
| `chuzzle_suite/queue_client.py` | the scheduler's HTTP API: status poller, commands, `mbq status` wording |
| `chuzzle_suite/title_bar.py` | the app's own title bar (drawing, hit-testing, icons) |
| `chuzzle_suite/window_frame.py` | Win32 for the borderless window: pin on top, dragging, taskbar styles, placement |
| `chuzzle_suite/tas_plan.py` | the TAS plan: puzzle choice, step order, trophy pop-ups, straight drags |
| `chuzzle_suite/tas_runner.py` | plays a plan with SendInput, screen checks, speed measurement, timing per exe |
| `chuzzle_suite/settings.py` | `user_data/settings.json` (the pin, the TAS options) |
| `chuzzle_suite/ui.py` | the pygame window |

Tests: `python -m unittest discover -s python/chuzzle_suite/tests` (`test_tas.py`: the planner,
and the runner against a fake game that follows the decomp's input rules).
`user_data/` is git-ignored.
