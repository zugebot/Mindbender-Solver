# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.point import Point
from tools.sprite_object import spriteObject


class MatchBoard(spriteObject):
    def __init__(self, menu, board):
        super().__init__(menu)

        self.board = board
        self.settings = self.game.settings

        self.size = Point(20, 20) * self.game.settings.boardSize
        self.base_surface = self.game.instance.Surface(self.size.tuple()).convert_alpha()
        self.base_surface.fill((0, 0, 0, 0))

        self.matchThis_pos = (75, 201)
        self.gameModeTitle_pos = (57, 110)

        self._preview_cache_key = None
        self._preview_cache_surface = None
        self._mini_fat_cache = {}

        raw_frames = self.game.textures.MatchThis
        if isinstance(raw_frames, list):
            frames = [frame for frame in raw_frames if isinstance(frame, p.Surface)]
        else:
            frames = [raw_frames] if isinstance(raw_frames, p.Surface) else []

        if not frames:
            self.match_this_frames = []
        elif len(frames) >= 5:
            self.match_this_frames = frames[:5]
        else:
            self.match_this_frames = frames + [frames[-1]] * (5 - len(frames))

        frame_count = len(self.match_this_frames)
        if frame_count <= 1:
            self.match_this_sequence = [0] if frame_count == 1 else []
        else:
            self.match_this_sequence = list(range(frame_count)) + list(range(frame_count - 2, 0, -1))

        self.match_this_sequence_index = 0
        self.match_this_tick = 0
        self.match_this_tick_rate = 8

    def invalidate_cache(self):
        self._preview_cache_key = None
        self._preview_cache_surface = None

    def _board_preview_key(self):
        should_tint_solved = (
            self.menu.current_index in self.menu.solved_indices
            and not self.menu.popup.replaying_solved_puzzle
        )

        cells = []
        for chuzzle in self.board:
            cells.append((
                int(chuzzle.position.x),
                int(chuzzle.position.y),
                int(chuzzle.color),
                bool(chuzzle.fatTL),
                bool(chuzzle.fatTR),
                bool(chuzzle.fatBL),
                bool(chuzzle.fatBR),
            ))

        cells.sort()

        return (
            tuple(cells),
            bool(self.settings.colorblindMode),
            bool(should_tint_solved),
            int(self.game.settings.boardSize),
        )

    def _get_mini_fat_texture(self, color):
        key = (int(color), bool(self.settings.colorblindMode))
        cached = self._mini_fat_cache.get(key)
        if cached is not None:
            return cached

        texture = self.game.textures.ChuzzleHuge[color][0]
        texture = p.transform.scale(texture, (38, 38)).convert_alpha()
        self._mini_fat_cache[key] = texture
        return texture

    def _build_preview_surface(self):
        surface = self.base_surface.copy()
        use_colorblind = self.settings.colorblindMode

        for y in range(self.board.chuzCount.y):
            for x in range(self.board.chuzCount.x):
                point = Point(x, y) * 18 + Point(1, 2)
                chuzzle = self.board.getPos(Point(x, y))
                if chuzzle is None:
                    continue

                color = chuzzle.color

                if chuzzle.fatTL:
                    texture = self._get_mini_fat_texture(color)
                    fat_pos = point + Point(0, -1)
                    surface.blit(texture, fat_pos.tuple())

                    if use_colorblind:
                        symbol_index = self.game.textures.colorsMainBlindSmall.get(str(color))
                        if symbol_index is not None and symbol_index != -1:
                            symbol = self.game.textures.ColorblindSymbolsSmall[symbol_index]
                            symbol_rect = symbol.get_rect(center=(fat_pos.x + 19, fat_pos.y + 19))
                            surface.blit(symbol, symbol_rect.topleft)
                    continue

                if chuzzle.fatTR or chuzzle.fatBL or chuzzle.fatBR:
                    continue

                if use_colorblind:
                    symbol_index = self.game.textures.colorsMainBlindSmall.get(str(color))
                    if symbol_index == -1:
                        texture = self.game.textures.MiniChuzzles[color]
                        surface.blit(texture, point.tuple())
                    elif symbol_index is not None:
                        texture = self.game.textures.ColorblindSymbolsSmall[symbol_index]
                        tex_rect = texture.get_rect(center=(point.x + 9, point.y + 9))
                        surface.blit(texture, tex_rect.topleft)
                else:
                    texture = self.game.textures.MiniChuzzles[color]
                    surface.blit(texture, point.tuple())

        should_tint_solved = (
            self.menu.current_index in self.menu.solved_indices
            and not self.menu.popup.replaying_solved_puzzle
        )

        if should_tint_solved:
            solved_tint = p.Surface(surface.get_size(), p.SRCALPHA)
            solved_tint.fill((0, 255, 41, 255))
            surface.blit(solved_tint, (0, 0), special_flags=p.BLEND_RGBA_MULT)

        new_size = self.size / self.game.settings.boardSize * 6
        return p.transform.scale(surface, new_size.tuple()).convert_alpha()

    def _get_preview_surface(self):
        key = self._board_preview_key()
        if key != self._preview_cache_key or self._preview_cache_surface is None:
            self._preview_cache_key = key
            self._preview_cache_surface = self._build_preview_surface()
        return self._preview_cache_surface

    def render(self, screen):
        preview = self._get_preview_surface()
        screen.blit(preview, self.matchThis_pos)

        game_mode_title = self.game.textures.GameModeTitle
        if isinstance(game_mode_title, list):
            if len(game_mode_title) > 0 and isinstance(game_mode_title[0], p.Surface):
                screen.blit(game_mode_title[0], self.gameModeTitle_pos)
        elif isinstance(game_mode_title, p.Surface):
            screen.blit(game_mode_title, self.gameModeTitle_pos)

        if self.match_this_frames:
            self.match_this_tick += 1
            if self.match_this_tick >= self.match_this_tick_rate:
                self.match_this_tick = 0
                self.match_this_sequence_index = (self.match_this_sequence_index + 1) % len(self.match_this_sequence)

            badge_frame_index = self.match_this_sequence[self.match_this_sequence_index]
            badge = self.match_this_frames[badge_frame_index]
            preview_rect = preview.get_rect(topleft=self.matchThis_pos)
            badge_rect = badge.get_rect(midbottom=(preview_rect.centerx - 5, preview_rect.top - 6))
            screen.blit(badge, badge_rect)