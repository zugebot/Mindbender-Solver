# python/scheduler/levels_fat.py
"""Which puzzles have the 2x2 piece (read from src/chuzzles/levels.hpp)."""

import re
from pathlib import Path

LEVELS_HPP = Path(__file__).resolve().parents[2] / "src" / "chuzzles" / "levels.hpp"


def fat_puzzles() -> set[str]:
    text = re.sub(r"//[^\n]*", "", LEVELS_HPP.read_text(errors="replace"))
    out = set()
    for m in re.finditer(r'\{"(\d+-\d+)"\s*,[^,]*,[^,]*,\s*(\{[^}]*\})', text):
        if "true" in m.group(2):
            out.add(m.group(1))
    return out
