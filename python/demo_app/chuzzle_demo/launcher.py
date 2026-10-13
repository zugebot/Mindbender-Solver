# python/demo_app/chuzzle_demo/launcher.py
"""Every Chuzzle game exe on this PC, for the Launcher tab.

A full scan walks every fixed drive once (minutes on big drives) and caches
what it finds in user_data/launcher_cache.json; each start of the app re-checks
the cached paths and the usual install folders (saves.DEFAULT_GAME_SEARCH_ROOTS)
in a moment. Games are started from their own folder, as their shortcuts do,
so they find their data and save where they always do."""

from __future__ import annotations

import ctypes
import json
import os
import struct
import subprocess
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from . import paths, saves

CACHE_FILE = paths.USER_DATA_DIR / "launcher_cache.json"
DRM_MARKER = b"!popcapdrmprotec"   # PopCap's DRM wrapper: the real game is appended to a loader
# folders a full scan skips (system files, caches, source trees: never a game install)
SKIP_DIR_NAMES = {"windows", "$recycle.bin", "system volume information", "recovery", "$windows.~bt",
                  "$windows.~ws", "winsxs", "node_modules", ".git", "__pycache__", ".gradle", ".m2",
                  "site-packages", ".cache", "temp", "tmp", "cmake-build-debug", "cmake-build-release"}
SKIP_EXE_NAMES = saves.SKIP_EXE_NAMES | {"unins000.exe"}

kernel32 = ctypes.windll.kernel32
version = ctypes.windll.version


@dataclass
class GameExe:
    path: Path
    label: str              # edition, from the folder name
    product_version: str    # from the exe's version resource ("" if it has none)
    build_date: str         # linker timestamp (YYYY-MM-DD)
    size: int
    drm_wrapped: bool       # PopCap DRM loader with the game appended

    def describe(self) -> str:
        parts = [f"v{self.product_version}" if self.product_version else "no version info",
                 f"built {self.build_date}" if self.build_date else ""]
        if self.drm_wrapped:
            parts.append("PopCap DRM wrapper")
        return ", ".join(p for p in parts if p)


def _product_version(path: Path) -> str:
    size = version.GetFileVersionInfoSizeW(str(path), None)
    if not size:
        return ""
    buf = ctypes.create_string_buffer(size)
    if not version.GetFileVersionInfoW(str(path), 0, size, buf):
        return ""
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    # the first language/codepage pair names the string table
    if version.VerQueryValueW(buf, "\\VarFileInfo\\Translation", ctypes.byref(ptr), ctypes.byref(length)) and length.value >= 4:
        lang, cp = struct.unpack("<HH", ctypes.string_at(ptr.value, 4))
        key = f"\\StringFileInfo\\{lang:04x}{cp:04x}\\ProductVersion"
        if version.VerQueryValueW(buf, key, ctypes.byref(ptr), ctypes.byref(length)) and length.value:
            return ctypes.wstring_at(ptr.value, length.value).rstrip("\0").strip()
    if version.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length)) and length.value >= 52:
        ms, ls = struct.unpack_from("<II", ctypes.string_at(ptr.value, 52), 16)   # dwProductVersionMS/LS
        return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"
    return ""


def _build_date(data: bytes) -> str:
    try:
        pe = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe:pe + 4] != b"PE\0\0":
            return ""
        stamp = struct.unpack_from("<I", data, pe + 8)[0]
        return time.strftime("%Y-%m-%d", time.gmtime(stamp)) if stamp else ""
    except struct.error:
        return ""


def describe_exe(path: Path) -> GameExe:
    data = path.read_bytes()
    return GameExe(path=path, label=saves._edition_label(path), product_version=_product_version(path),
                   build_date=_build_date(data), size=len(data), drm_wrapped=DRM_MARKER in data)


def is_candidate(name: str) -> bool:
    low = name.lower()
    return low.endswith(".exe") and "chuzzle" in low and low not in SKIP_EXE_NAMES


def fixed_drives() -> list[Path]:
    mask = kernel32.GetLogicalDrives()
    drives = []
    for i in range(26):
        if mask & (1 << i):
            root = f"{chr(65 + i)}:\\"
            if kernel32.GetDriveTypeW(root) == 3:   # DRIVE_FIXED
                drives.append(Path(root))
    return drives


def full_scan(progress: Callable[[str, int], None] | None = None,
              roots: list[Path] | None = None) -> list[Path]:
    """Every Chuzzle*.exe under `roots` (default: all fixed drives). progress(folder, folders
    so far) is called every 2000 folders."""
    found: list[Path] = []
    folders = 0
    for root in roots or fixed_drives():
        for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
            dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIR_NAMES]
            folders += 1
            if progress and folders % 2000 == 0:
                progress(dirpath, folders)
            for name in filenames:
                if is_candidate(name):
                    found.append(Path(dirpath) / name)
    return sorted(set(found))


def load_cache() -> tuple[list[Path], float | None]:
    try:
        data = json.loads(CACHE_FILE.read_text())
        return [Path(p) for p in data.get("exes", [])], data.get("scanned_at")
    except (OSError, ValueError):
        return [], None


def save_cache(exes: list[Path], scanned_at: float) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({"scanned_at": scanned_at, "exes": [str(p) for p in exes]}, indent=1))


def quick_list(cached: list[Path]) -> list[Path]:
    """Cached exes that still exist plus whatever the usual install folders hold."""
    exes = {p for p in cached if p.is_file()}
    exes.update(p for p in saves.find_game_exes() if is_candidate(p.name))
    return sorted(exes)


def unique_labels(exes: list[Path]) -> dict[Path, str]:
    """The shortest tail of each exe's folder path that tells it apart from the others
    ("PopCap Games\\Chuzzle", "Chuzzles\\Chuzzle"), plus the exe name when it is not
    Chuzzle.exe."""
    labels: dict[Path, str] = {}
    for p in exes:
        parts = p.parent.parts[1:] or p.parent.parts
        for n in range(1, len(parts) + 1):
            tail = parts[-n:]
            # exes in the same folder are told apart by their name, not their folder
            if not any(q.parent != p.parent and q.parent.parts[1:][-n:] == tail for q in exes):
                break
        label = "\\".join(tail)
        if p.name.lower() != "chuzzle.exe":
            label += f"  ({p.name})"
        labels[p] = label
    return labels


def describe_all(exes: list[Path]) -> list[GameExe]:
    labels = unique_labels(exes)
    out = []
    for p in exes:
        try:
            g = describe_exe(p)
        except OSError:
            continue
        g.label = labels[p]
        out.append(g)
    out.sort(key=lambda g: (str(g.path.parent).lower(), g.path.name.lower()))
    return out


def launch(exe: Path) -> subprocess.Popen:
    """Start a game from its own folder (its data and saves are relative to it)."""
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen([str(exe)], cwd=str(exe.parent), creationflags=flags, close_fds=True)
