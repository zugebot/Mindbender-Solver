# python/chuzzle_suite/chuzzle_suite/paths.py
"""Locations inside the repository that Chuzzle Suite reads."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEVELS_HPP = REPO_ROOT / "src" / "chuzzles" / "levels.hpp"
LEVELS_CELLS_HPP = REPO_ROOT / "src" / "chuzzles" / "levels_cells.hpp"
LEVELS_FINAL_DIR = REPO_ROOT / "build" / "levels_final"
GAME_ACCEPTS_DIR = LEVELS_FINAL_DIR / "game_accepts"   # placements the game also accepts
VIEWER_DIR = REPO_ROOT / "python" / "viewer"
BUILD_CONFIG = REPO_ROOT / "build" / "config.json"
BUILD_README = REPO_ROOT / "build" / "README.md"   # "6-4: at least 15 moves" lines

APP_DIR = Path(__file__).resolve().parents[1]
USER_DATA_DIR = APP_DIR / "user_data"          # git-ignored: calibration, settings
SAVE_BACKUP_DIR = USER_DATA_DIR / "save_backups"
LOG_FILE = USER_DATA_DIR / "chuzzle_suite.log"   # stdout/stderr when started without a console

# The app as a Windows program: the shortcut (install_shortcut.py) and the running window share
# this AppUserModelID, so a pinned taskbar icon and the open window are one button.
APP_ID = "Mindbender.ChuzzleSuite"
ICON_ICO = APP_DIR / "assets" / "chuzzle_suite.ico"   # made by assets/make_icon.py
ICON_PNG = APP_DIR / "assets" / "chuzzle_suite.png"   # the window's own icon
RUN_SUITE = APP_DIR / "run_suite.py"


def relaunch_command() -> str:
    """How the taskbar starts the app again (a pinned running window): pythonw, no console."""
    import sys
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return f'"{pythonw if pythonw.is_file() else exe}" "{RUN_SUITE}"'
