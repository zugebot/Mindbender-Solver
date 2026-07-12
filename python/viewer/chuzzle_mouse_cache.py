from __future__ import annotations
# python/viewer/chuzzle_mouse_cache.py

import hashlib
import json
from pathlib import Path
from typing import Any

from chuzzle_mouse_domain import (
    CachedScoreFile,
    cached_score_file_from_dict,
    cached_score_file_to_dict,
)


class ChuzzleMouseCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_file(path: str | Path) -> str:
        file_path = Path(path)
        return hashlib.sha256(file_path.read_bytes()).hexdigest()

    @staticmethod
    def hash_options(options: dict[str, Any]) -> str:
        normalized = json.dumps(options, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(normalized).hexdigest()

    def cache_path_for_source(self, source_file: str | Path) -> Path:
        source_path = Path(source_file)
        return self.cache_dir / f"{source_path.name}.scorecache.json"

    def load(self, source_file: str | Path) -> CachedScoreFile | None:
        cache_path = self.cache_path_for_source(source_file)
        if not cache_path.exists():
            return None
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return cached_score_file_from_dict(data)
        except Exception:
            return None

    def save(self, cached: CachedScoreFile) -> None:
        cache_path = self.cache_path_for_source(cached.source_file)
        data = cached_score_file_to_dict(cached)
        cache_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def is_valid(self, cached: CachedScoreFile, source_file: str | Path, options: dict[str, Any]) -> bool:
        source_path = Path(source_file)
        if Path(cached.source_file).name != source_path.name:
            return False

        current_source_hash = self.hash_file(source_path)
        current_options_hash = self.hash_options(options)

        return (
                cached.source_hash == current_source_hash
                and cached.options_hash == current_options_hash
        )