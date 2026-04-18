# Jerrin Shirks

# native imports
import random as r
import pygame as p

# custom imports
from tools.point import Point
from tools.sprite_object import spriteObject


class MatchBoard(spriteObject):
    def __init__(self, menu, board):
        super().__init__(menu)

        self.board = board
        self.settings = self.game.settings

        # 120 base
        self.size = Point(20, 20) * self.game.settings.boardSize
        self.surface = self.game.instance.Surface(self.size.tuple()).convert_alpha()
        self.surface.fill((0, 0, 0, 0))
        self.colors = [[r.randint(0, 8) for _ in range(self.game.settings.boardSize)] for _ in range(self.game.settings.boardSize)]

        self.preview_pos = (75, 203)

        # MatchThis.png is a 5-state strip: 0 -> 1 -> 2 -> 3 -> 4 -> repeat.
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
            # Fallback: repeat the last available frame so indexing remains stable.
            self.match_this_frames = frames + [frames[-1]] * (5 - len(frames))

        # Ping-pong sequence for 5 states: 0 1 2 3 4 3 2 1 -> repeat.
        frame_count = len(self.match_this_frames)
        if frame_count <= 1:
            self.match_this_sequence = [0] if frame_count == 1 else []
        else:
            self.match_this_sequence = list(range(frame_count)) + list(range(frame_count - 2, 0, -1))

        self.match_this_sequence_index = 0
        self.match_this_tick = 0
        self.match_this_tick_rate = 8

    def render(self, screen):
        surface = self.surface.copy()

        use_colorblind = self.settings.colorblindMode

        for y in range(self.board.chuzCount.y):
            for x in range(self.board.chuzCount.x):
                point = Point(x, y) * 18 + Point(0, 1)
                chuzzle = self.board[y * self.game.settings.boardSize + x]
                color = chuzzle.color

                # Fat chuzzles should always show the mini fat texture.
                if chuzzle.fatTL:
                    texture = self.game.textures.ChuzzleHuge[color][0]
                    texture = p.transform.smoothscale(texture, (38, 38))
                    fat_pos = point + Point(0, -1)
                    surface.blit(texture, fat_pos.tuple())

                    # Fat chuzzles still get a colorblind symbol, centered on the 2x2 mini fat.
                    if use_colorblind:
                        symbol_index = self.game.textures.colorsMainBlindSmall.get(str(color))
                        if symbol_index is not None and symbol_index != -1:
                            symbol = self.game.textures.ColorblindSymbolsSmall[symbol_index]
                            symbol_rect = symbol.get_rect(center=(fat_pos.x + 19, fat_pos.y + 19))
                            surface.blit(symbol, symbol_rect.topleft)
                    continue

                # Skip non-main fat corners so only the TL draws the 2x2 mini fat.
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

        newSize = self.size / self.game.settings.boardSize * 6
        surface = p.transform.smoothscale(surface, newSize.tuple())
        screen.blit(surface, self.preview_pos)

        if self.match_this_frames:
            self.match_this_tick += 1
            if self.match_this_tick >= self.match_this_tick_rate:
                self.match_this_tick = 0
                self.match_this_sequence_index = (self.match_this_sequence_index + 1) % len(self.match_this_sequence)

            badge_frame_index = self.match_this_sequence[self.match_this_sequence_index]
            badge = self.match_this_frames[badge_frame_index]
            preview_rect = surface.get_rect(topleft=self.preview_pos)
            badge_rect = badge.get_rect(midbottom=(preview_rect.centerx - 5, preview_rect.top - 6))
            screen.blit(badge, badge_rect)