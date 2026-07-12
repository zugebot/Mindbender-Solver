from __future__ import annotations

import fnmatch
import zipfile
from pathlib import Path


# Paths are relative to the repository root.
# Each entry may be either a file or a directory.
INCLUDE_PATHS = [
    # ".git",
    # ".idea",
    # "archive",
    "build/levels",
    "build/levels_final",
    # "build/models",
    # "build/train",
    "build/config.json",
    # "build/config_ml_torch.json",
    # "build/config_ml_training_data.json",
    "build/README.md",

    # "cmake-build-*",
    # "decomp",
    "docs",
    "include",
    "mouse",
    "pretty-printers",
    # "pretty-printers/__pycache__",
    "python",
    # "python/viewer/cache",
    "scripts",
    "src",
    # "venv",
    ".gdbinit",
    ".gitignore",
    "CMakeLists.txt",
    "CMakePresets.json",
]

# Paths and patterns are also relative to the repository root.
#
# A directory entry skips that directory and everything inside it.
# Glob patterns are supported.
SKIP_PATHS = [
    ".git",
    ".idea",
    "archive",
    # "build/levels",
    # "build/levels_final",
    "build/models",
    "build/train",
    # "build/config.json",
    "build/config_ml_torch.json",
    "build/config_ml_training_data.json",
    # "build/README.md",

    "cmake-build-*",
    "decomp",
    # "docs",
    # "include",
    # "mouse",
    # "pretty-printers",
    "pretty-printers/__pycache__",
    # "python",
    "python/viewer/cache",
    "__pycache__",
    "*.pyc",
    "project.zip",
    # "scripts",
    # "src",
    "venv",
    # ".gdbinit",
    # ".gitignore",
    # "CMakeLists.txt",
    # "CMakePresets.json",
]

ZIP_NAME = "project.zip"


def get_repo_root() -> Path:
    """Return the repository root, assuming this script is in /scripts."""
    return Path(__file__).resolve().parent.parent


def normalize_relative_path(path: Path) -> str:
    """Convert a relative path to a forward-slash path for ZIP files."""
    return path.as_posix()


def matches_skip_pattern(relative_path: Path) -> bool:
    """
    Return True when a path should be skipped.

    Skip entries can match:
    - A full repository-relative path, such as "src/generated"
    - A file or directory name anywhere, such as "__pycache__"
    - A glob pattern, such as "*.pyc" or "cmake-build-*"
    """
    relative_string = normalize_relative_path(relative_path)

    for raw_pattern in SKIP_PATHS:
        pattern = raw_pattern.replace("\\", "/").strip("/")
        if not pattern:
            continue

        # Match the complete repository-relative path.
        if fnmatch.fnmatch(relative_string, pattern):
            return True

        # Match any individual path component.
        if any(fnmatch.fnmatch(part, pattern) for part in relative_path.parts):
            return True

        # Treat a non-glob path as a directory prefix.
        has_glob = any(character in pattern for character in "*?[")

        if not has_glob and (
                relative_string == pattern
                or relative_string.startswith(pattern + "/")
        ):
            return True

    return False


def add_directory(
        archive: zipfile.ZipFile,
        directory: Path,
        repo_root: Path,
        output_zip: Path,
) -> None:
    """Recursively add a directory to the ZIP while respecting SKIP_PATHS."""
    for path in sorted(directory.rglob("*")):
        relative_path = path.relative_to(repo_root)

        if path.resolve() == output_zip.resolve():
            continue

        if matches_skip_pattern(relative_path):
            continue

        if path.is_file():
            archive.write(
                path,
                arcname=normalize_relative_path(relative_path),
            )


def create_project_zip() -> Path:
    repo_root = get_repo_root()
    output_zip = repo_root / ZIP_NAME

    if output_zip.exists():
        output_zip.unlink()

    added_paths = 0

    with zipfile.ZipFile(
            output_zip,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
    ) as archive:
        for included_path_string in INCLUDE_PATHS:
            included_path = repo_root / included_path_string
            relative_path = included_path.relative_to(repo_root)

            if not included_path.exists():
                print(f"Warning: included path does not exist: {relative_path}")
                continue

            if matches_skip_pattern(relative_path):
                print(f"Skipped included path due to skip rules: {relative_path}")
                continue

            if included_path.is_file():
                if included_path.resolve() != output_zip.resolve():
                    archive.write(
                        included_path,
                        arcname=normalize_relative_path(relative_path),
                    )
                    added_paths += 1

            elif included_path.is_dir():
                before_count = len(archive.infolist())
                add_directory(
                    archive=archive,
                    directory=included_path,
                    repo_root=repo_root,
                    output_zip=output_zip,
                )
                added_paths += len(archive.infolist()) - before_count

    print(f"Created: {output_zip}")
    print(f"Files added: {added_paths}")

    return output_zip


if __name__ == "__main__":
    create_project_zip()