# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.button import Button


class MindBenderMiniPanel:
    def __init__(self, menu):
        self.menu = menu
        self.game = menu.game
        self.textures = menu.textures

        self.mini_coords = [
            (20, 325),
            (61, 353),
            (107, 363),
            (153, 353),
            (195, 330),
        ]

        self._mini_solution_cache = {}
        self._mini_scaled_textures = None
        self._mini_scaled_cell_size = None
        self.buttons = []

    def build_buttons(self):
        self.buttons = []

        for i, coords in enumerate(self.mini_coords):
            btn = Button(
                game=self.game,
                coords=coords,
                textures=self.textures.MindBenderMini,
                data={
                    "hitbox": self.textures.MindBenderMini_Alpha[0],
                    "sounds": self.menu.buttonSoundLowOnlyClick,
                    "func": {"clickup": (self.menu.setPuzzle, i)}
                }
            )
            self.buttons.append(btn)

        return self.buttons

    def _get_mini_scaled_textures(self, cell_size):
        if self._mini_scaled_textures is not None and self._mini_scaled_cell_size == cell_size:
            return self._mini_scaled_textures

        self._mini_scaled_cell_size = cell_size
        self._mini_scaled_textures = [
            p.transform.smoothscale(tex, (cell_size, cell_size))
            for tex in self.textures.MiniChuzzles
        ]
        return self._mini_scaled_textures

    def _build_mini_solution_surface(self, puzzle_index):
        if puzzle_index < 0 or puzzle_index >= len(self.menu.puzzles.puzzles):
            return None

        mini_w = self.textures.MindBenderMini[0].get_width()
        mini_h = self.textures.MindBenderMini[0].get_height()
        preview_size = max(12, int(min(mini_w, mini_h) * 0.62))
        cell_size = max(1, preview_size // 6)
        grid_size = cell_size * 6

        preview = p.Surface((preview_size, preview_size), p.SRCALPHA)
        puzzle = self.menu.puzzles.puzzles[puzzle_index]
        textures = self._get_mini_scaled_textures(cell_size)

        offset_x = (preview_size - grid_size) // 2
        offset_y = (preview_size - grid_size) // 2

        for y in range(6):
            for x in range(6):
                color = puzzle.solved_cells[y][x]
                preview.blit(textures[color], (offset_x + x * cell_size, offset_y + y * cell_size))

        return preview

    def _get_mini_solution_surface(self, puzzle_index):
        if puzzle_index not in self._mini_solution_cache:
            self._mini_solution_cache[puzzle_index] = self._build_mini_solution_surface(puzzle_index)
        return self._mini_solution_cache[puzzle_index]

    def draw(self, screen):
        level_start = self.menu.level * 5

        select_texture = self.textures.MindbBenderMini_Select
        solved_texture = self.textures.MindBenderSolvedSmall
        check_texture = self.textures.MindBenderMini_Check

        mini_w = self.textures.MindBenderMini[0].get_width()
        mini_h = self.textures.MindBenderMini[0].get_height()

        for i, coords in enumerate(self.mini_coords):
            absolute_index = level_start + i
            cx = coords[0] + mini_w // 2 + 1
            cy = coords[1] + mini_h // 2

            mini_solution = self._get_mini_solution_surface(absolute_index)
            if mini_solution is not None:
                mini_solution_rect = mini_solution.get_rect(center=(cx, cy))
                screen.blit(mini_solution, mini_solution_rect)

            if i == self.menu.puzzleIndex:
                select_rect = select_texture.get_rect(center=(cx - 1, cy - 1))
                screen.blit(select_texture, select_rect)

            if absolute_index in self.menu.solved_indices:
                check_rect = check_texture.get_rect(center=(cx - 1, cy - 1))
                screen.blit(check_texture, check_rect)

        if self.menu.current_index in self.menu.solved_indices and not self.menu.popup.replaying_solved_puzzle:
            preview_center = (75 + 54 + 2, 203 + 56)
            preview_rect = solved_texture.get_rect(center=preview_center)
            screen.blit(solved_texture, preview_rect)