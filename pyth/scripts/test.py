from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


TARGET_DIR = Path(r"C:\Program Files")
FOLLOW_SYMLINKS = False


def iter_files_safe(root: Path) -> Iterable[Path]:
    stack = [root]

    while stack:
        current = stack.pop()

        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=FOLLOW_SYMLINKS):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=FOLLOW_SYMLINKS):
                            yield Path(entry.path)
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
        except (PermissionError, FileNotFoundError, OSError):
            continue


def get_folder_size_bytes(folder: Path) -> int:
    total = 0

    for file_path in iter_files_safe(folder):
        try:
            total += file_path.stat(follow_symlinks=FOLLOW_SYMLINKS).st_size
        except (PermissionError, FileNotFoundError, OSError):
            continue

    return total


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:,.2f} {unit}"
        size /= 1024.0

    return f"{num_bytes} B"


def main() -> None:
    if not TARGET_DIR.exists() or not TARGET_DIR.is_dir():
        print(f"Target directory does not exist or is not a folder: {TARGET_DIR}")
        return

    subfolders: list[Path] = []
    try:
        with os.scandir(TARGET_DIR) as entries:
            for entry in entries:
                try:
                    if entry.is_dir(follow_symlinks=FOLLOW_SYMLINKS):
                        subfolders.append(Path(entry.path))
                except (PermissionError, FileNotFoundError, OSError):
                    continue
    except (PermissionError, FileNotFoundError, OSError) as exc:
        print(f"Failed to read target directory: {exc}")
        return

    results: list[tuple[Path, int]] = []

    for folder in subfolders:
        print(f"Scanning: {folder}")
        size_bytes = get_folder_size_bytes(folder)
        results.append((folder, size_bytes))

    results.sort(key=lambda item: item[1], reverse=True)

    print("\nLargest folders:\n")
    for folder, size_bytes in results:
        print(f"{format_size(size_bytes):>12}  {folder.name}")


if __name__ == "__main__":
    main()