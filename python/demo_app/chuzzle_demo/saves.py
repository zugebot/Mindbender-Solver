# python/demo_app/chuzzle_demo/saves.py
"""Finds and tracks every place Chuzzle 1 keeps save data.

Where the game writes depends on the build (see docs/changes-2026-10-04.md):

- Portable / older builds (e.g. Deluxe 1.01): `<folder of Chuzzle.exe>\\Profiles`
  next to the exe, plus a `backup` folder there.
- Same builds installed under Program Files without admin rights: Windows
  redirects the writes to `%LOCALAPPDATA%\\VirtualStore\\<that path>`.
- PopCap installer version: `%ProgramData%\\PopCap Games\\Chuzzle\\Profiles`.
- Christmas Edition: `Documents\\my games\\Chuzzle\\Profiles`.
- Steam PopCap builds: `%ProgramData%\\Steam\\Chuzzle*` (convention; unverified here).
- Chuzzle 2's Chuzzle 1 modes: `%APPDATA%\\Chuzzle2\\_cloud\\<user>\\chuzzle1_*.save`.
- Manual backups: `Documents\\my games\\Backup_Profiles`.

Profiles hold SAVEGAME-*.DAT and INFO.CFG; the Profiles folder itself holds
config.cfg (Profiles=, CurrentProfile=, DX::ScreenWidth...) and HS*.cfg.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

SAVE_SUFFIXES = (".dat", ".cfg", ".save")
DEFAULT_GAME_SEARCH_ROOTS = [
    Path(r"E:\Games\Chuzzle Editions"),
    Path(r"C:\Program Files (x86)\PopCap Games"),
    Path(r"C:\Program Files\PopCap Games"),
    Path(r"C:\Program Files (x86)\Steam\steamapps\common"),
    Path(r"C:\Program Files\Steam\steamapps\common"),
]
SKIP_EXE_NAMES = {"popuninstall.exe", "uninstall.exe"}


@dataclass
class SaveFile:
    path: Path
    size: int
    mtime: float


@dataclass
class Profile:
    name: str
    path: Path
    files: list[SaveFile] = field(default_factory=list)

    @property
    def last_modified(self) -> float:
        return max((f.mtime for f in self.files), default=0.0)


@dataclass
class SaveLocation:
    label: str          # what writes here, e.g. "Deluxe 1.01 (next to exe)"
    kind: str           # install | virtualstore | programdata | documents | steam | chuzzle2 | backup
    root: Path          # the Profiles-style folder
    exe: Path | None = None
    profiles: list[Profile] = field(default_factory=list)
    shared_files: list[SaveFile] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)

    @property
    def exists(self) -> bool:
        return self.root.is_dir()

    @property
    def current_profile(self) -> str | None:
        return self.config.get("CurrentProfile")

    @property
    def last_modified(self) -> float:
        times = [p.last_modified for p in self.profiles] + [f.mtime for f in self.shared_files]
        return max(times, default=0.0)

    def profile(self, name: str) -> Profile | None:
        return next((p for p in self.profiles if p.name.lower() == name.lower()), None)


# --------------------------------------------------------------------------
# Windows folders
# --------------------------------------------------------------------------

def _known_folder(guid: str, fallback: Path) -> Path:
    """SHGetKnownFolderPath, so a redirected Documents folder is found."""
    try:
        class GUID(ctypes.Structure):
            _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                        ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

        parts = guid.strip("{}").split("-")
        data4 = bytes.fromhex(parts[3] + parts[4])
        g = GUID(int(parts[0], 16), int(parts[1], 16), int(parts[2], 16),
                 (ctypes.c_ubyte * 8)(*data4))
        out = ctypes.c_wchar_p()
        if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out)) == 0:
            result = Path(out.value)
            ctypes.windll.ole32.CoTaskMemFree(out)
            return result
    except Exception:
        pass
    return fallback


def documents_dir() -> Path:
    return _known_folder("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}", Path.home() / "Documents")


def _env_path(name: str, fallback: str) -> Path:
    return Path(os.environ.get(name, fallback))


def virtualstore_mirror(folder: Path) -> Path | None:
    """Where UAC redirects writes to `folder` (only for Program Files / Windows
    paths on the system drive)."""
    try:
        drive, rest = folder.drive, folder.relative_to(folder.anchor)
    except ValueError:
        return None
    if not drive or not rest.parts:
        return None
    if rest.parts[0].lower() not in {"program files", "program files (x86)", "windows", "programdata"}:
        return None
    return _env_path("LOCALAPPDATA", str(Path.home() / "AppData/Local")) / "VirtualStore" / rest


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def find_game_exes(search_roots: list[Path] | None = None, max_depth: int = 4) -> list[Path]:
    exes: list[Path] = []
    for root in search_roots or DEFAULT_GAME_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            here = Path(dirpath)
            if len(here.parts) - base_depth >= max_depth:
                dirnames[:] = []
            # skip asset folders; they never hold the exe
            dirnames[:] = [d for d in dirnames if d.lower() not in {"images", "sounds", "music", "cached", "fonts"}]
            for name in filenames:
                low = name.lower()
                if low.endswith(".exe") and "chuzzle" in low and low not in SKIP_EXE_NAMES:
                    exes.append(here / name)
    return sorted(set(exes))


def _edition_label(exe: Path) -> str:
    parent = exe.parent
    for part in reversed(parent.parts):
        if "chuzzle" in part.lower():
            return part
    return parent.name


def candidate_locations(game_exes: list[Path]) -> list[SaveLocation]:
    docs = documents_dir()
    program_data = _env_path("ProgramData", r"C:\ProgramData")
    app_data = _env_path("APPDATA", str(Path.home() / "AppData/Roaming"))

    out: list[SaveLocation] = []
    seen_dirs: set[Path] = set()
    for exe in game_exes:
        if exe.parent in seen_dirs:
            continue
        seen_dirs.add(exe.parent)
        label = _edition_label(exe)
        out.append(SaveLocation(f"{label} (next to exe)", "install", exe.parent / "Profiles", exe))
        out.append(SaveLocation(f"{label} (game backup)", "backup", exe.parent / "backup", exe))
        mirror = virtualstore_mirror(exe.parent)
        if mirror is not None:
            out.append(SaveLocation(f"{label} (UAC VirtualStore)", "virtualstore", mirror / "Profiles", exe))

    out.append(SaveLocation("PopCap installer version", "programdata",
                            program_data / "PopCap Games" / "Chuzzle" / "Profiles"))
    out.append(SaveLocation("PopCap installer version (backup)", "backup",
                            program_data / "PopCap Games" / "Chuzzle" / "backup"))
    # config.cfg and the HS*.cfg high scores sit one level above Profiles here
    out.append(SaveLocation("Christmas Edition (my games)", "documents",
                            docs / "my games" / "Chuzzle" / "Profiles"))
    for steam_dir in sorted((program_data / "Steam").glob("Chuzzle*")) if (program_data / "Steam").is_dir() else []:
        out.append(SaveLocation(f"Steam ({steam_dir.name})", "steam", steam_dir / "userdata"))
    out.append(SaveLocation("Chuzzle 2 (Chuzzle 1 modes)", "chuzzle2", app_data / "Chuzzle2" / "_cloud"))
    out.append(SaveLocation("Manual backups", "backup", docs / "my games" / "Backup_Profiles"))
    return out


def _save_files(folder: Path) -> list[SaveFile]:
    files = []
    try:
        for entry in os.scandir(folder):
            if entry.is_file() and entry.name.lower().endswith(SAVE_SUFFIXES):
                st = entry.stat()
                files.append(SaveFile(Path(entry.path), st.st_size, st.st_mtime))
    except OSError:
        pass
    return sorted(files, key=lambda f: f.path.name.lower())


def read_config(path: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                config[key.strip()] = value.strip()
    except OSError:
        pass
    return config


def scan_location(location: SaveLocation) -> SaveLocation:
    location.profiles = []
    location.shared_files = []
    location.config = {}
    if not location.exists:
        return location
    location.shared_files = _save_files(location.root)
    if location.kind == "documents":
        location.shared_files += _save_files(location.root.parent)
    for config_path in (location.root / "config.cfg", location.root.parent / "config.cfg"):
        if config_path.is_file():
            location.config = read_config(config_path)
            break
    try:
        subdirs = sorted((Path(e.path) for e in os.scandir(location.root) if e.is_dir()),
                         key=lambda p: p.name.lower())
    except OSError:
        subdirs = []
    for sub in subdirs:
        files = _save_files(sub)
        if files:
            location.profiles.append(Profile(sub.name, sub, files))
    return location


def discover(search_roots: list[Path] | None = None, extra_exes: list[Path] | None = None,
             include_missing: bool = False) -> list[SaveLocation]:
    exes = find_game_exes(search_roots)
    for exe in extra_exes or []:
        if exe not in exes:
            exes.append(exe)
    locations = [scan_location(loc) for loc in candidate_locations(exes)]
    if not include_missing:
        locations = [loc for loc in locations if loc.exists and (loc.profiles or loc.shared_files)]
    return locations


def active_location(locations: list[SaveLocation], exe: Path | None,
                    window_title: str = "") -> SaveLocation | None:
    """The Profiles folder a running game uses: next to its exe if present,
    then its VirtualStore mirror, then my games for the Christmas Edition,
    then the most recently written shared location (ProgramData / my games).
    With no game running, the most recently written game folder."""
    if exe is not None:
        for kind in ("install", "virtualstore"):
            for loc in locations:
                if loc.kind == kind and loc.exe is not None and loc.exe.parent == exe.parent and loc.exists:
                    return loc
    if "christmas" in window_title.lower():
        docs = next((loc for loc in locations if loc.kind == "documents"), None)
        if docs is not None:
            return docs
    if exe is not None:
        shared = [loc for loc in locations if loc.kind in ("programdata", "documents") and loc.profiles]
    else:   # game not running: whichever game folder was written last
        shared = [loc for loc in locations
                  if loc.kind in ("install", "virtualstore", "programdata", "documents") and loc.profiles]
    return max(shared, key=lambda loc: loc.last_modified, default=None)


# --------------------------------------------------------------------------
# Change tracking
# --------------------------------------------------------------------------

@dataclass
class SaveEvent:
    when: float
    change: str         # added | changed | removed
    path: Path


class SaveWatcher:
    """Polls the save files of the given locations and reports changes."""

    def __init__(self, locations: list[SaveLocation]):
        self.locations = locations
        self._snapshot = self._take()
        self.events: list[SaveEvent] = []

    def set_locations(self, locations: list[SaveLocation]) -> None:
        self.locations = locations
        self._snapshot = self._take()

    def _take(self) -> dict[Path, tuple[int, float]]:
        snap = {}
        for loc in self.locations:
            folders = [loc.root] + [p.path for p in scan_location(loc).profiles]
            if loc.kind == "documents":
                folders.append(loc.root.parent)
            for folder in folders:
                for f in _save_files(folder):
                    snap[f.path] = (f.size, f.mtime)
        return snap

    def poll(self) -> list[SaveEvent]:
        now = time.time()
        snap = self._take()
        new = []
        for path, stamp in snap.items():
            old = self._snapshot.get(path)
            if old is None:
                new.append(SaveEvent(now, "added", path))
            elif old != stamp:
                new.append(SaveEvent(now, "changed", path))
        for path in self._snapshot.keys() - snap.keys():
            new.append(SaveEvent(now, "removed", path))
        self._snapshot = snap
        self.events.extend(new)
        del self.events[:-200]
        return new


# --------------------------------------------------------------------------
# Backups (never deletes the game's files; moves/copies into user_data)
# --------------------------------------------------------------------------

def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text).strip("_")


def backup_folder(source: Path, label: str, backup_root: Path = paths.SAVE_BACKUP_DIR) -> Path:
    dest = backup_root / _slug(label) / _slug(source.name) / time.strftime("%Y%m%d-%H%M%S")
    dest.mkdir(parents=True, exist_ok=False)
    for f in _save_files(source):
        shutil.copy2(f.path, dest / f.path.name)
    return dest


def list_backups(source: Path, label: str, backup_root: Path = paths.SAVE_BACKUP_DIR) -> list[Path]:
    folder = backup_root / _slug(label) / _slug(source.name)
    return sorted((p for p in folder.iterdir() if p.is_dir()), reverse=True) if folder.is_dir() else []


def restore_backup(backup: Path, target: Path, label: str,
                   backup_root: Path = paths.SAVE_BACKUP_DIR) -> Path:
    """Copies a backup over `target`, after backing up what is there now.
    Returns the safety backup's folder."""
    safety = backup_folder(target, label + "_before_restore", backup_root)
    for f in _save_files(backup):
        shutil.copy2(f.path, target / f.path.name)
    return safety


def stash_file(path: Path, label: str, keep: int = 20,
               backup_root: Path = paths.SAVE_BACKUP_DIR) -> Path | None:
    """Moves one save file out of the game's folder (e.g. to reset Speed
    Chuzzle), keeping the newest `keep` stashed copies."""
    if not path.is_file():
        return None
    folder = stash_dir(label, backup_root)
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"{time.strftime('%Y%m%d-%H%M%S')}_{int(time.time() * 1000) % 1000:03d}_{path.name}"
    shutil.move(str(path), dest)
    stashed = sorted(folder.glob(f"*_{path.name}"))
    for old in stashed[:-keep]:
        old.unlink()
    return dest


def stash_dir(label: str, backup_root: Path = paths.SAVE_BACKUP_DIR) -> Path:
    return backup_root / "stashed" / _slug(label)


def list_stashed(label: str, backup_root: Path = paths.SAVE_BACKUP_DIR) -> list[Path]:
    """Stashed files for `label`, newest first."""
    folder = stash_dir(label, backup_root)
    return sorted((p for p in folder.iterdir() if p.is_file()), reverse=True) if folder.is_dir() else []
