# Jerrin Shirks

# native imports
import atexit
import time
from pathlib import Path

import pygame as p

# custom imports
from engine.constants import MOUSE_CLICK_UP, MOUSE_CLICK_DOWN
from tools.menu import Menu
from tools.button import Button
from tools.image import Image
from objects.board import Board
from objects.matchthis import MatchBoard
from engine.chuzzle_list import ChuzzleList
from puzzles.puzzles import Puzzles
from engine.mindbender_stats import MindBenderStats
from objects.mindbender_mini_panel import MindBenderMiniPanel
from objects.mindbender_solved_popup import MindBenderSolvedPopup
from objects.mindbender_route_guide import MindBenderRouteGuide


class MindBender(Menu):
    MOUSE_CLICK_UP = MOUSE_CLICK_UP
    MOUSE_CLICK_DOWN = MOUSE_CLICK_DOWN

    def __init__(self, game):
        super().__init__(game)

        self.puzzle = None
        self.puzzles = Puzzles()
        self.level = 0
        self.puzzleIndex = 0
        self.world_number = 1
        self.level_number = 1

        self.board = Board(self, gamemode="mindbender")
        self.solution_board = ChuzzleList(game=self.game)
        self.solution_board.fill()
        self.matchBoard = MatchBoard(menu=self, board=self.solution_board)
        self.left_world_star_btn = None
        self.center_world_star = None
        self.right_world_star_btn = None

        self.level_font = p.font.SysFont("arial", 26, bold=True)
        self.star_number_font = p.font.SysFont("arial", 36, bold=True)

        self.stats_font = p.font.SysFont("arial", 20, bold=True)
        self.stats_shadow_color = (40, 20, 60)
        self.stats_text_color = (230, 255, 210)

        self.solved_indices = set()
        self.current_index = 0

        self.stats_file = Path(__file__).resolve().parent.parent / "data" / "mindbender_stats.json"
        self.stats = MindBenderStats(self.stats_file)

        self.current_puzzle_timer = None
        self.current_world_timer = None

        self.current_puzzle_recorded = False
        self.current_world_any3_recorded = False
        self.current_world_all5_recorded = False

        atexit.register(self.stats.save)

        board_size = self.board.surfaceSize
        self.board_rect = p.Rect(self.board.boardOffset.x, self.board.boardOffset.y, board_size.x, board_size.y)

        self.popup = MindBenderSolvedPopup(self)
        self.mini_panel = MindBenderMiniPanel(self)

        self.route_guide = MindBenderRouteGuide(
            self,
            Path(__file__).resolve().parent.parent / "data" / "puzzler.export.json",
        )

        self.populate()

        self._hidden_left_star = p.Surface(self.textures.MindBenderSmallStarLeft[0].get_size(), p.SRCALPHA)
        self._dim_alpha_locked = 128
        self._dim_right_star = [self._with_alpha(tex, self._dim_alpha_locked)
                                for tex in self.textures.MindBenderSmallStarRight]
        self._show_left_world_star = True
        self._can_go_next_world = False

        self.updatePuzzle()

    def _now(self):
        return time.time()

    def _current_puzzle_key(self):
        return f"world_{self.world_number}_puzzle_{self.level_number}"

    def _current_world_key(self):
        return f"world_{self.world_number}"

    def _reset_current_puzzle_timer(self):
        self.current_puzzle_timer = None
        self.current_puzzle_recorded = False

    def _reset_current_world_timer(self):
        self.current_world_timer = None
        self.current_world_any3_recorded = False
        self.current_world_all5_recorded = False

    def _reset_all_active_timers(self):
        self._reset_current_puzzle_timer()
        self._reset_current_world_timer()

    def _ensure_timers_started_from_board_click(self, mousePos, mouseState):
        if mouseState != MOUSE_CLICK_DOWN:
            return
        if self.popup.visible:
            return
        if not self.board_rect.collidepoint(mousePos.tuple()):
            return
        if self.current_index in self.solved_indices and not self.popup.replaying_solved_puzzle:
            return

        now = self._now()

        if self.current_puzzle_timer is None:
            self.current_puzzle_timer = {
                "started_at": now,
                "world_number": self.world_number,
                "level_number": self.level_number,
                "puzzle_index": self.current_index,
            }
            self.current_puzzle_recorded = False

        if self.current_world_timer is None:
            self.current_world_timer = {
                "started_at": now,
                "world_number": self.world_number,
                "level": self.level,
            }
            self.current_world_any3_recorded = False
            self.current_world_all5_recorded = False

    def _record_current_puzzle_if_needed(self):
        if self.current_puzzle_timer is None:
            return
        if self.current_puzzle_recorded:
            return
        if self.current_puzzle_timer["puzzle_index"] != self.current_index:
            return

        elapsed = self._now() - self.current_puzzle_timer["started_at"]
        self.stats.record_puzzle(
            world_number=self.world_number,
            level_number=self.level_number,
            puzzle_index=self.current_index,
            elapsed_seconds=elapsed
        )
        self.current_puzzle_recorded = True

    def _record_world_exit_if_needed(self, old_level, new_level, reason):
        if self.current_world_timer is None:
            return
        if self.current_world_timer["level"] != old_level:
            return
        if new_level <= old_level:
            return

        elapsed = self._now() - self.current_world_timer["started_at"]
        solved_count = self._solved_count_in_world(old_level)
        world_number = old_level + 1

        if reason == "right_star" and solved_count >= 3 and not self.current_world_any3_recorded:
            self.stats.record_world_any3(
                world_number=world_number,
                solved_count_when_leaving=solved_count,
                reason=reason,
                elapsed_seconds=elapsed
            )
            self.current_world_any3_recorded = True

        if solved_count >= 5 and not self.current_world_all5_recorded:
            self.stats.record_world_all5(
                world_number=world_number,
                solved_count_when_leaving=solved_count,
                reason=reason,
                elapsed_seconds=elapsed
            )
            self.current_world_all5_recorded = True

        self._reset_current_world_timer()

    def _solved_count_in_world(self, level):
        world_start = level * 5
        world_end = world_start + 5
        return sum(1 for idx in self.solved_indices if world_start <= idx < world_end)

    def saveGame(self):
        self.stats.save()
        self.game.popMenu()

    def setPuzzle(self, value):
        old_index = self.current_index
        self.puzzleIndex = max(0, min(4, value))
        self.updatePuzzle()

        if self.current_index != old_index:
            self._reset_current_puzzle_timer()
        else:
            self._reset_route_guide_for_current_puzzle()

    def setLevel(self, value):
        old_level = self.level
        self.level = max(0, min(19, value))
        self.updatePuzzle()

        if self.level != old_level:
            self._reset_current_puzzle_timer()
            self._reset_current_world_timer()
        else:
            self._reset_route_guide_for_current_puzzle()

    def incrementLevel(self):
        if not self.can_advance_world():
            return

        old_level = self.level
        self._record_world_exit_if_needed(old_level, old_level + 1, reason="right_star")

        self.puzzleIndex = 0
        self.setLevel(self.level + 1)

    def decrementLevel(self):
        self._reset_current_puzzle_timer()
        self._reset_current_world_timer()
        self.puzzleIndex = 0
        self.setLevel(self.level - 1)

    def playNextPuzzle(self):
        old_level = self.level

        if self.puzzleIndex < 4:
            self.setPuzzle(self.puzzleIndex + 1)
        else:
            self._record_world_exit_if_needed(old_level, (self.level + 1) % 20, reason="play_next_puzzle")
            self.setLevel((self.level + 1) % 20)
            self.setPuzzle(0)

    def playNextUnsolvedPuzzle(self):
        total = len(self.puzzles.puzzles)
        if total <= 0:
            return

        world_start = (self.current_index // 5) * 5
        local_index = self.current_index % 5

        for step in range(1, 6):
            candidate_local = (local_index + step) % 5
            candidate_index = world_start + candidate_local
            if candidate_index < total and candidate_index not in self.solved_indices:
                self.setPuzzle(candidate_local)
                return

        next_world_level = self.level + 1
        max_level = max(0, (total - 1) // 5)
        if next_world_level <= max_level:
            old_level = self.level
            self._record_world_exit_if_needed(old_level, next_world_level, reason="play_next_unsolved")
            self.puzzleIndex = 0
            self.setLevel(next_world_level)
            return

        self.updatePuzzle()

    def _board_signature(self, chuzzle_list):
        return tuple(sorted(
            (int(ch.position.x), int(ch.position.y), int(ch.color),
             int(ch.fatTL), int(ch.fatTR), int(ch.fatBL), int(ch.fatBR))
            for ch in chuzzle_list
        ))

    def _is_board_solved(self):
        return self._board_signature(self.board.chuzzles) == self._board_signature(self.solution_board)

    def _with_alpha(self, surface, alpha):
        tinted = surface.copy()
        tinted.set_alpha(alpha)
        return tinted

    def _sync_world_star_state(self):
        self._show_left_world_star = self.level > 0
        self._can_go_next_world = self.level < 19 and self.can_advance_world()

        if self._show_left_world_star:
            self.left_world_star_btn.clickable = True
            self.left_world_star_btn.data["sounds"] = self.buttonSoundLow
            self.left_world_star_btn.data["textures"]["default"] = self.textures.MindBenderSmallStarLeft[0]
            self.left_world_star_btn.data["textures"]["hover"] = self.textures.MindBenderSmallStarLeft[1]
            self.left_world_star_btn.data["textures"]["clickdown"] = self.textures.MindBenderSmallStarLeft[2]
            self.left_world_star_btn.data["hitbox"] = self.textures.MindBenderSmallStarLeft_Alpha[0]
            if self.left_world_star_btn.image is self._hidden_left_star:
                self.left_world_star_btn.image = self.textures.MindBenderSmallStarLeft[0]
        else:
            self.left_world_star_btn.clickable = False
            self.left_world_star_btn.data["sounds"] = self.buttonSoundLow
            self.left_world_star_btn.data["textures"]["default"] = self._hidden_left_star
            self.left_world_star_btn.data["textures"]["hover"] = self._hidden_left_star
            self.left_world_star_btn.data["textures"]["clickdown"] = self._hidden_left_star
            self.left_world_star_btn.data["hitbox"] = self._hidden_left_star
            self.left_world_star_btn.image = self._hidden_left_star

        if self._can_go_next_world:
            self.right_world_star_btn.clickable = True
            self.right_world_star_btn.data["sounds"] = self.buttonSoundLow
            self.right_world_star_btn.data["textures"]["default"] = self.textures.MindBenderSmallStarRight[0]
            self.right_world_star_btn.data["textures"]["hover"] = self.textures.MindBenderSmallStarRight[1]
            self.right_world_star_btn.data["textures"]["clickdown"] = self.textures.MindBenderSmallStarRight[2]
            self.right_world_star_btn.data["hitbox"] = self.textures.MindBenderSmallStarRight_Alpha[0]
            self.right_world_star_btn.image = self.textures.MindBenderSmallStarRight[0]
        else:
            self.right_world_star_btn.clickable = False
            self.right_world_star_btn.data["sounds"] = self.buttonSoundLow
            self.right_world_star_btn.data["textures"]["default"] = self._dim_right_star[0]
            self.right_world_star_btn.data["textures"]["hover"] = self._dim_right_star[0]
            self.right_world_star_btn.data["textures"]["clickdown"] = self._dim_right_star[0]
            self.right_world_star_btn.data["hitbox"] = self.textures.MindBenderSmallStarRight_Alpha[0]
            self.right_world_star_btn.image = self._dim_right_star[0]

    def _reset_route_guide_for_current_puzzle(self):
        self.route_guide.reset_for_puzzle(self.world_number, self.level_number)
    
    def _draw_world_star_numbers(self, screen):
        current_world = self.level + 1
        prev_world = current_world - 1
        next_world = current_world + 1

        def draw_number(value, center, color=(180, 255, 140), shadow=(60, 20, 90), alpha=255):
            text = self.star_number_font.render(str(value), True, color)
            shadow_text = self.star_number_font.render(str(value), True, shadow)
            if alpha < 255:
                text.set_alpha(alpha)
                shadow_text.set_alpha(alpha)
            text_rect = text.get_rect(center=center)
            shadow_rect = shadow_text.get_rect(center=(center[0] + 2, center[1] + 2))
            screen.blit(shadow_text, shadow_rect)
            screen.blit(text, text_rect)

        if self._show_left_world_star:
            draw_number(prev_world, self.left_world_star_btn.rect.center)

        draw_number(current_world, (self.center_world_star.rect.centerx, self.center_world_star.rect.centery + 6))

        if current_world < 20:
            next_alpha = 255 if self._can_go_next_world else self._dim_alpha_locked
            draw_number(next_world, self.right_world_star_btn.rect.center, alpha=next_alpha)

    def updatePuzzle(self):
        if not self.puzzles.puzzles:
            return

        index = self.level * 5 + self.puzzleIndex
        index = max(0, min(len(self.puzzles.puzzles) - 1, index))

        self.level = index // 5
        self.puzzleIndex = index % 5
        self.current_index = index
        self.world_number = self.level + 1
        self.level_number = self.puzzleIndex + 1

        self.puzzle = self.puzzles.puzzles[index]

        showing_solved_state = index in self.solved_indices and not self.popup.suppress_replay_popup_once

        if showing_solved_state:
            self.board.chuzzles.load_from_array(self.puzzle.solved_cells)
            if self.puzzle.end_fat is not None:
                self.board.chuzzles.placeFatChuzzle(*self.puzzle.end_fat)
        else:
            self.board.chuzzles.load_from_array(self.puzzle.initial_cells)
            if self.puzzle.start_fat is not None:
                self.board.chuzzles.placeFatChuzzle(*self.puzzle.start_fat)

        self.solution_board.load_from_array(self.puzzle.solved_cells)
        if self.puzzle.end_fat is not None:
            self.solution_board.placeFatChuzzle(*self.puzzle.end_fat)

        if index in self.solved_indices and not self.popup.suppress_replay_popup_once:
            self.popup.replaying_solved_puzzle = False
            self.popup.show_replay_popup()
        else:
            self.popup.hide()
            if index not in self.solved_indices:
                self.popup.replaying_solved_puzzle = False

        self._sync_world_star_state()
        self.popup.suppress_replay_popup_once = False
        self.route_guide.on_puzzle_changed(self.world_number, self.level_number)

    def _available_world_count(self):
        total = len(self.puzzles.puzzles)
        return max(1, (total + 4) // 5)

    def _go_to_world_first_puzzle(self, world_number):
        world_index = max(0, min(self._available_world_count() - 1, world_number - 1))
        self.level = world_index
        self.puzzleIndex = 0
        self.popup.clear_transient_state()
        self.updatePuzzle()

    def _reset_current_puzzle(self):
        self.solved_indices.discard(self.current_index)
        self._reset_current_puzzle_timer()
        self.popup.clear_transient_state()
        self.updatePuzzle()
        self._reset_route_guide_for_current_puzzle()

    def _reset_current_world(self):
        world_start = self.level * 5
        world_end = world_start + 5
        self.solved_indices = {idx for idx in self.solved_indices if not (world_start <= idx < world_end)}
        self.puzzleIndex = 0
        self._reset_current_puzzle_timer()
        self._reset_current_world_timer()
        self.popup.clear_transient_state()
        self.updatePuzzle()
        self._reset_route_guide_for_current_puzzle()

    def _reset_all_progress(self):
        self.solved_indices.clear()
        self.level = 0
        self.puzzleIndex = 0
        self._reset_all_active_timers()
        self.popup.clear_transient_state()
        self.updatePuzzle()
        self._reset_route_guide_for_current_puzzle()

    def _jump_world_and_clear_all_progress(self, world_number):
        self.solved_indices.clear()
        self._reset_all_active_timers()
        self._go_to_world_first_puzzle(world_number)
        self._reset_route_guide_for_current_puzzle()

    def onBoardMoveCommitted(self, move_data: dict):
        self.route_guide.on_board_move_committed(move_data)

    def onKeyDown(self, event):
        e = event.key
        mods = event.mod
        shift_held = bool(mods & p.KMOD_SHIFT)
        left_ctrl_held = bool(mods & p.KMOD_LCTRL)

        if e == p.K_q:
            if self.left_world_star_btn.clickable:
                self.decrementLevel()
            return

        if e == p.K_e:
            if self.right_world_star_btn.clickable:
                self.incrementLevel()
            return

        if e == p.K_t:
            self._print_best_world_any3_sum(1, 9)
            return

        if e == p.K_s:
            if self.current_index not in self.solved_indices:
                self.solved_indices.add(self.current_index)
                self._record_current_puzzle_if_needed()
                self.popup.start_popup_animation()
            return

        if e == p.K_u:
            if self.current_index in self.solved_indices:
                self.solved_indices.discard(self.current_index)
                self.popup.clear_transient_state()
                self._reset_route_guide_for_current_puzzle()
                self.updatePuzzle()
            return

        if e == p.K_r:
            if left_ctrl_held and shift_held:
                self._reset_all_progress()
                return
            if shift_held:
                self._reset_current_world()
                return
            self._reset_current_puzzle()
            return

        world_key_map = {
            p.K_1: 1,
            p.K_2: 2,
            p.K_3: 3,
            p.K_4: 4,
            p.K_5: 5,
            p.K_6: 6,
            p.K_7: 7,
            p.K_8: 8,
            p.K_9: 9,
            p.K_0: 0,
        }
        if e in world_key_map:
            world_key = world_key_map[e]
            if shift_held:
                if world_key == 0:
                    world_key = 20
                else:
                    world_key += 10
            self._jump_world_and_clear_all_progress(world_key)
            return

        if e == p.K_b:
            x = int(input("x: "))
            y = int(input("y: "))
            self.board.chuzzles.getPos((x, y)).locked = True
        if e == p.K_n:
            for chuzzle in self.board.chuzzles:
                chuzzle.locked = True
        if e == p.K_m:
            for chuzzle in self.board.chuzzles:
                chuzzle.locked = False

        if e == p.K_j:
            x = int(input("x: "))
            y = int(input("y: "))
            self.board.chuzzles.placeFatChuzzle(x, y)
        if e == p.K_k:
            self.board.chuzzles.placeFatChuzzle()
        if e == p.K_l:
            self.board.chuzzles.deleteFatChuzzle()

    def run(self, mousePos, mouseState):
        self._ensure_timers_started_from_board_click(mousePos, mouseState)

        if self.popup.visible:
            for sprite in self.sprites:
                if sprite is self.board:
                    continue
                sprite.testMouseStates(mousePos.tuple(), mouseState)
        else:
            super().run(mousePos, mouseState)

        self.popup.update_interaction(mousePos, mouseState, self.buttonSoundLow)

        if mouseState == MOUSE_CLICK_UP:
            self.popup.handle_click_up(mousePos)

    def populate(self):
        bkg = self.game.instance.image.load("textures/Backdrops/bkg1.jpg").convert_alpha()
        self.sprites.append(Image(image=bkg))

        self.sprites.append(self.board)

        self.sprites.append(Button(game=self.game, coords=(0, 553),
                                   textures=self.textures.MainMenuButton, data={
            "hitbox": self.textures.MainMenuButton_Alpha[0],
            "func": {"clickup": self.saveGame},
            "sounds": self.buttonSoundLow
        }))

        self.sprites.append(Image((0, 125), self.textures.MindBenderGadget))

        for btn in self.mini_panel.build_buttons():
            self.sprites.append(btn)

        self.sprites.append(Image(image=self.textures.MindBenderMarquee))
        self.sprites.append(self.matchBoard)

        self.left_world_star_btn = Button(game=self.game, coords=(23, 455),
                                          textures=self.textures.MindBenderSmallStarLeft, data={
            "hitbox": self.textures.MindBenderSmallStarLeft_Alpha[0],
            "sounds": self.buttonSoundLow,
            "func": {"clickup": self.decrementLevel}
        })
        self.sprites.append(self.left_world_star_btn)

        self.center_world_star = Image((78, 410), self.textures.MindBenderBigStar)
        self.sprites.append(self.center_world_star)

        self.right_world_star_btn = Button(game=self.game, coords=(170, 455),
                                           textures=self.textures.MindBenderSmallStarRight, data={
            "hitbox": self.textures.MindBenderSmallStarRight_Alpha[0],
            "sounds": self.buttonSoundLow,
            "func": {"clickup": self.incrementLevel}
        })
        self.sprites.append(self.right_world_star_btn)

        self.sprites.append(self.route_guide.toggle_button)

    def update(self, screen, default=False):
        super().update(screen, default=default)

        if self.current_index not in self.solved_indices and self._is_board_solved():
            self.solved_indices.add(self.current_index)
            self._record_current_puzzle_if_needed()
            self.popup.start_popup_animation()

        if self.popup.replaying_solved_puzzle and self._is_board_solved():
            self.popup.replaying_solved_puzzle = False
            self._record_current_puzzle_if_needed()
            self.popup.start_popup_animation()

        self._sync_world_star_state()

        self.mini_panel.draw(screen)
        self.popup.draw(screen)
        self._draw_world_star_numbers(screen)
        self._draw_bottom_stats(screen)
        self.route_guide.draw(screen)

    def _solved_count_in_current_world(self):
        return self._solved_count_in_world(self.level)

    def can_advance_world(self):
        return self._solved_count_in_current_world() >= 3
    
    def _sum_best_world_any3_times(self, start_world: int, end_world: int):
        total = 0.0
        found_any = False

        for world_number in range(start_world, end_world + 1):
            world_key = f"world_{world_number}"
            best = self.stats.get_best_seconds("worlds_any3", world_key)
            if best is None:
                continue
            total += best
            found_any = True

        return total if found_any else None

    def _print_best_world_any3_sum(self, start_world: int, end_world: int):
        total = self._sum_best_world_any3_times(start_world, end_world)
        if total is None:
            print(f"No recorded 3-Solves times for worlds {start_world}-{end_world}")
            return

        formatted = MindBenderStats.format_elapsed(total)
        print(f"Best 3-Solves total for worlds {start_world}-{end_world}: {formatted} ({total:.3f}s)")

    def _draw_bottom_stats(self, screen):
        puzzle_key = self._current_puzzle_key()
        world_key = self._current_world_key()

        puzzle_best = self.stats.get_best_seconds("puzzles", puzzle_key)
        world_any3_best = self.stats.get_best_seconds("worlds_any3", world_key)
        world_all5_best = self.stats.get_best_seconds("worlds_all5", world_key)

        lines = [
            f"Level {self.world_number}-{self.level_number}: {MindBenderStats.format_elapsed(puzzle_best)}",
            f"3-Solves: {MindBenderStats.format_elapsed(world_any3_best)}",
            f"5-Solves: {MindBenderStats.format_elapsed(world_all5_best)}",
        ]

        start_x = 147
        start_y = 572
        width = 850 - start_x
        line_gap = 230
        pad_x = 10
        pad_y = 6

        rendered = []
        for text_value in lines:
            shadow = self.stats_font.render(text_value, True, self.stats_shadow_color)
            text = self.stats_font.render(text_value, True, self.stats_text_color)
            rendered.append((shadow, text))

        bg_left = start_x - pad_x
        bg_top = start_y - pad_y

        last_text = rendered[-1][1]
        bg_right = start_x + (len(lines) - 1) * line_gap + last_text.get_width() + pad_x
        bg_bottom = start_y + last_text.get_height() + pad_y

        bg_rect = p.Rect(bg_left, bg_top, width, bg_bottom - bg_top)

        bg_surface = p.Surface((bg_rect.width, bg_rect.height), p.SRCALPHA)
        bg_surface.fill((20, 20, 28, 165))
        screen.blit(bg_surface, bg_rect.topleft)

        for i, (shadow, text) in enumerate(rendered):
            x = start_x + i * line_gap
            screen.blit(shadow, (x + 2, start_y + 2))
            screen.blit(text, (x, start_y))