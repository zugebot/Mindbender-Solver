# python/demo_app/chuzzle_demo/paths.py
"""Locations inside the repository that the demo app reads."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEVELS_HPP = REPO_ROOT / "src" / "chuzzles" / "levels.hpp"
LEVELS_CELLS_HPP = REPO_ROOT / "src" / "chuzzles" / "levels_cells.hpp"
LEVELS_FINAL_DIR = REPO_ROOT / "build" / "levels_final"
VIEWER_DIR = REPO_ROOT / "python" / "viewer"
BUILD_CONFIG = REPO_ROOT / "build" / "config.json"

APP_DIR = Path(__file__).resolve().parents[1]
USER_DATA_DIR = APP_DIR / "user_data"          # git-ignored: calibration, settings
SAVE_BACKUP_DIR = USER_DATA_DIR / "save_backups"
