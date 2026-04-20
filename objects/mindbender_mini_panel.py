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
            (20, 323),
            (61, 352),
            (107, 360),
            (154, 351),
            (196, 327),
        ]

        self._mini_solution_cache = {}
        self.buttons = []

        # flat colors for the 8 puzzle colors
        self._mini_solution_colors = {
            0: (178, 8, 8),       # red
            1: (8, 174, 8),       # green
            2: (8, 8, 255),       # blue
            3: (255, 105, 2),     # orange
            4: (255, 230, 0),     # yellow
            5: (207, 7, 207),     # purple
            6: (255, 255, 255),   # white
            7: (87, 197, 221),    # cyan
        }

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

    def _color_for_cell(self, color_index):
        return self._mini_solution_colors.get(int(color_index), (255, 0, 255))

    def _build_mini_solution_surface(self, puzzle_index):
        if puzzle_index < 0 or puzzle_index >= len(self.menu.puzzles.puzzles):
            return None

        mini_w = self.textures.MindBenderMini[0].get_width()
        mini_h = self.textures.MindBenderMini[0].get_height()

        preview_size = max(12, int(min(mini_w, mini_h) * 0.62))

        # padding only on right/bottom
        pad_left = 0
        pad_top = 0
        pad_right = 1
        pad_bottom = 1

        gap = 1

        inner_w = preview_size - pad_left - pad_right
        inner_h = preview_size - pad_top - pad_bottom
        if inner_w <= 0 or inner_h <= 0:
            return None

        cell_size = max(1, min(
            (inner_w - gap * (6 - 1)) // 6,
            (inner_h - gap * (6 - 1)) // 6
        ))

        grid_w = cell_size * 6 + gap * (6 - 1)
        grid_h = cell_size * 6 + gap * (6 - 1)

        preview = p.Surface((preview_size, preview_size), p.SRCALPHA)
        puzzle = self.menu.puzzles.puzzles[puzzle_index]

        offset_x = pad_left + (inner_w - grid_w) // 2
        offset_y = pad_top + (inner_h - grid_h) // 2

        fat_tl = getattr(puzzle, "end_fat", None)

        for y in range(6):
            for x in range(6):
                color_index = puzzle.solved_cells[y][x]
                color = self._color_for_cell(color_index)

                draw_x = offset_x + x * (cell_size + gap)
                draw_y = offset_y + y * (cell_size + gap)
                draw_w = cell_size
                draw_h = cell_size

                if fat_tl is not None:
                    fat_x, fat_y = fat_tl

                    # TL
                    if x == fat_x and y == fat_y:
                        draw_w += gap
                        draw_h += gap

                    # TR
                    elif x == fat_x + 1 and y == fat_y:
                        draw_h += gap

                    # BL
                    elif x == fat_x and y == fat_y + 1:
                        draw_w += gap

                    # BR
                    elif x == fat_x + 1 and y == fat_y + 1:
                        pass

                cell_rect = p.Rect(draw_x, draw_y, draw_w, draw_h)
                p.draw.rect(preview, (*color, 174), cell_rect)

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
            preview_center = (132, 256)
            preview_rect = solved_texture.get_rect(center=preview_center)
            screen.blit(solved_texture, preview_rect)