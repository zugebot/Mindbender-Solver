# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from tools.button import Button


class MindBenderSolvedPopup:
    def __init__(self, menu):
        self.menu = menu
        self.game = menu.game
        self.textures = menu.textures

        self.visible = False
        self.animating = False
        self.frames = 30
        self.frame = 0
        self.mode = None  # "next" | "replay"

        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False

        self.popup_badge_x_offset = -1
        self.popup_badge_y_offset = -14
        self.popup_prompt_gap = 3

        # Use the real visible board size, not the padded render surface size.
        board_size = getattr(self.menu.board, "boardSizePx", self.menu.board.surfaceSize)
        board_rect = p.Rect(
            self.menu.board.boardOffset.x,
            self.menu.board.boardOffset.y,
            board_size.x,
            board_size.y
        )

        bar_inset_left = 1
        bar_inset_right = -1
        bar_h = 127
        bar_y_offset = -8

        self.popup_bar_rect = p.Rect(
            board_rect.x + bar_inset_left,
            board_rect.centery - bar_h // 2 + bar_y_offset,
            board_rect.width - bar_inset_left - bar_inset_right + 3,
            bar_h
        )

        solved_center = (
            self.popup_bar_rect.centerx + self.popup_badge_x_offset,
            self.popup_bar_rect.centery + self.popup_badge_y_offset
        )
        solved_rect = self.textures.Matched.get_rect(center=solved_center)

        next_default = self.textures.ClickHereNextPuzzle[0]
        replay_default = self.textures.ClickHereToPlayAgain[0]

        next_prompt_top = min(
            solved_rect.bottom + self.popup_prompt_gap,
            self.popup_bar_rect.bottom - next_default.get_height() - 2
        )
        replay_prompt_top = min(
            solved_rect.bottom + self.popup_prompt_gap,
            self.popup_bar_rect.bottom - replay_default.get_height() - 2
        )

        next_prompt_x = self.popup_bar_rect.centerx - (next_default.get_width() // 2) - 1
        replay_prompt_x = self.popup_bar_rect.centerx - (replay_default.get_width() // 2) - 1

        self.next_prompt_btn = Button(
            game=self.game,
            coords=(next_prompt_x, next_prompt_top),
            textures=self.textures.ClickHereNextPuzzle,
            data={
                "hitbox": "all",
                "sounds": self.menu.buttonSoundLow,
                "func": {"clickup": self._handle_next_click}
            }
        )

        self.replay_prompt_btn = Button(
            game=self.game,
            coords=(replay_prompt_x, replay_prompt_top),
            textures=self.textures.ClickHereToPlayAgain,
            data={
                "hitbox": "all",
                "sounds": self.menu.buttonSoundLow,
                "func": {"clickup": self._handle_replay_click}
            }
        )

    def _active_prompt_button(self):
        if self.mode == "next":
            return self.next_prompt_btn
        if self.mode == "replay":
            return self.replay_prompt_btn
        return None

    def _handle_next_click(self):
        self.replaying_solved_puzzle = False
        self.menu.playNextUnsolvedPuzzle()

    def _handle_replay_click(self):
        self.replaying_solved_puzzle = True
        self.suppress_replay_popup_once = True
        self.menu._reset_current_puzzle_timer()
        self.menu.updatePuzzle()

    def start_popup_animation(self):
        self.visible = True
        self.animating = True
        self.frame = 0
        self.mode = "next"
        self.game.sounds.playSound(self.game.sounds.Multipop)

    def show_replay_popup(self):
        self.visible = True
        self.animating = False
        self.frame = self.frames
        self.mode = "replay"

    def hide(self):
        self.visible = False
        self.animating = False
        self.frame = 0
        self.mode = None

    def clear_transient_state(self):
        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False
        self.hide()

    def update_interaction(self, mousePos, mouseState, buttonSoundLow):
        if not self.visible:
            return

        btn = self._active_prompt_button()
        if btn is None:
            return

        # Important: keep this clickable immediately on solve, even while the
        # animation is still playing. Do not gate interaction on animating.
        btn.testMouseStates(mousePos.tuple(), mouseState)

    def handle_click_up(self, mousePos):
        return False

    def draw(self, screen):
        if not self.visible:
            return

        if self.animating:
            self.frame = min(self.frame + 1, self.frames)
            progress = self.frame / float(self.frames)
            sweep_width = int(self.popup_bar_rect.width * progress)
        else:
            progress = 1.0
            sweep_width = self.popup_bar_rect.width

        if sweep_width > 0:
            sweep_surface = p.Surface((sweep_width, self.popup_bar_rect.height), p.SRCALPHA)
            sweep_surface.fill((0, 255, 41, 185))
            screen.blit(sweep_surface, self.popup_bar_rect.topleft)

        solved_center = (
            self.popup_bar_rect.centerx + self.popup_badge_x_offset,
            self.popup_bar_rect.centery + self.popup_badge_y_offset
        )

        if self.animating:
            angle = 360 * progress
            scale = 0.08 + (0.92 * progress)
            solved = p.transform.rotozoom(self.textures.Matched, angle, scale)
            solved_rect = solved.get_rect(center=solved_center)
            screen.blit(solved, solved_rect)

            if self.frame >= self.frames:
                self.animating = False
        else:
            solved_rect = self.textures.Matched.get_rect(center=solved_center)
            screen.blit(self.textures.Matched, solved_rect)
            if self.mode is None:
                self.mode = "replay"

        btn = self._active_prompt_button()
        if btn is None:
            return

        # Keep the old visual behavior: do not render the button until the
        # "next" solve animation has finished.
        if not (self.animating and self.mode == "next"):
            btn.render(screen)