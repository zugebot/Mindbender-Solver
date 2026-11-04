# python/chuzzle_suite/chuzzle_suite/settings.py
"""Small app settings kept across restarts in user_data/settings.json
(e.g. whether the window is pinned on top)."""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import paths

SETTINGS_FILE = paths.USER_DATA_DIR / "settings.json"


def load(path: Path | None = None) -> dict:
    try:
        data = json.loads((path or SETTINGS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def get(key: str, default=None, path: Path | None = None):
    return load(path).get(key, default)


def put(key: str, value, path: Path | None = None) -> None:
    """Set one key, keeping the others (written atomically)."""
    path = path or SETTINGS_FILE
    data = load(path)
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)
