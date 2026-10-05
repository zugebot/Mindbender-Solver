# Jerrin Shirks

import os
import re
from .puzzle import Puzzle


class Puzzles:
    CELL_RE = re.compile(r"INIT\s+([bs]\d+_\d+)\s*\[36]\s*=\s*{([^}]*)};")
    LEVEL_DEF_RE = re.compile(
        r"{\s*\"(\d+-\d+)\"\s*,\s*LevelCells::(b\d+_\d+)\s*,\s*LevelCells::(s\d+_\d+)\s*,\s*({[^}]*})\s*,\s*({[^}]*})\s*}"
    )

    def __init__(self):
        self.puzzles = []
        self.load_puzzles()

    @staticmethod
    def _parse_fat(fat_text):
        fat_text = fat_text.strip()
        if fat_text == "{}":
            return None

        fat_match = re.match(r"{\s*true\s*,\s*(\d+)\s*,\s*(\d+)\s*}", fat_text)
        if not fat_match:
            return None
        return int(fat_match.group(1)), int(fat_match.group(2))

    @staticmethod
    def _to_rows(cells):
        return [cells[i:i + 6] for i in range(0, 36, 6)]

    def _parse_cells(self, levels_cells_path):
        with open(levels_cells_path, "r") as f:
            content = f.read()

        result = {}
        for symbol, values in self.CELL_RE.findall(content):
            nums = [int(v.strip()) for v in values.split(",") if v.strip()]
            if len(nums) == 36:
                result[symbol] = nums
        return result

    def load_puzzles(self):
        root = os.path.dirname(os.path.dirname(__file__))
        levels_path = os.path.join(root, "data", "levels.hpp")
        levels_cells_path = os.path.join(root, "data", "levels_cells.hpp")

        cell_map = self._parse_cells(levels_cells_path)

        with open(levels_path, "r") as f:
            levels_content = f.read()

        self.puzzles = []
        for name, start_key, end_key, start_fat_text, end_fat_text in self.LEVEL_DEF_RE.findall(levels_content):
            if start_key not in cell_map or end_key not in cell_map:
                continue

            self.puzzles.append(Puzzle(
                name=name,
                initial_cells=self._to_rows(cell_map[start_key]),
                solved_cells=self._to_rows(cell_map[end_key]),
                start_fat=self._parse_fat(start_fat_text),
                end_fat=self._parse_fat(end_fat_text),
            ))
