# Jerrin Shirks

# native imports
import pygame as p


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

        self.prompt_rect = p.Rect(0, 0, 1, 1)
        self.prompt_hit_rect = p.Rect(0, 0, 1, 1)
        self.prompt_hovered = False
        self._prompt_last_hovered = False
        self.prompt_pressed_inside = False

        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False

        self.prompt_hitbox_pad_x = 0
        self.prompt_hitbox_pad_y = 0
        self.popup_badge_y_offset = -12
        self.popup_prompt_gap = 2

        board_size = self.menu.board.surfaceSize
        board_rect = p.Rect(
            self.menu.board.boardOffset.x,
            self.menu.board.boardOffset.y,
            board_size.x,
            board_size.y
        )

        bar_inset_left = 1
        bar_inset_right = 1
        bar_h = 127
        bar_y_offset = -8

        self.popup_bar_rect = p.Rect(
            board_rect.x + bar_inset_left,
            board_rect.centery - bar_h // 2 + bar_y_offset,
            board_rect.width - bar_inset_left - bar_inset_right + 3,
            bar_h
        )

    def start_popup_animation(self):
        self.visible = True
        self.animating = True
        self.frame = 0
        self.mode = "next"
        self.prompt_pressed_inside = False
        self.game.sounds.playSound(self.game.sounds.Multipop)

    def show_replay_popup(self):
        self.visible = True
        self.animating = False
        self.frame = self.frames
        self.mode = "replay"
        self.prompt_pressed_inside = False

    def hide(self):
        self.visible = False
        self.animating = False
        self.frame = 0
        self.mode = None
        self.prompt_rect = p.Rect(0, 0, 1, 1)
        self.prompt_hit_rect = p.Rect(0, 0, 1, 1)
        self.prompt_hovered = False
        self._prompt_last_hovered = False
        self.prompt_pressed_inside = False

    def clear_transient_state(self):
        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False
        self.prompt_hovered = False
        self._prompt_last_hovered = False
        self.prompt_pressed_inside = False
        self.hide()

    def update_interaction(self, mousePos, mouseState, buttonSoundLow):
        self.prompt_hovered = self.mode in ["next", "replay"] and self.prompt_hit_rect.collidepoint(mousePos.tuple())

        if self.mode in ["next", "replay"]:
            if self.prompt_hovered and not self._prompt_last_hovered:
                if buttonSoundLow["hover"] is not None:
                    self.game.sounds.playSound(buttonSoundLow["hover"])

            if self.prompt_hovered:
                self.game.cursorIndex = 2
            elif self._prompt_last_hovered:
                self.game.cursorIndex = 0

            if mouseState == self.menu.MOUSE_CLICK_DOWN:
                self.prompt_pressed_inside = self.prompt_hovered
                if self.prompt_pressed_inside and buttonSoundLow["clickdown"] is not None:
                    self.game.sounds.playSound(buttonSoundLow["clickdown"])
        else:
            self._prompt_last_hovered = False
            self.prompt_pressed_inside = False

        self._prompt_last_hovered = self.prompt_hovered

    def handle_click_up(self, mousePos):
        if not self.visible:
            self.prompt_pressed_inside = False
            return False

        if self.mode not in ["next", "replay"]:
            self.prompt_pressed_inside = False
            return False

        released_inside = self.prompt_hit_rect.collidepoint(mousePos.tuple())
        should_activate = self.prompt_pressed_inside and released_inside
        self.prompt_pressed_inside = False

        if not should_activate:
            return False

        if self.mode == "next":
            self.replaying_solved_puzzle = False
            self.menu.playNextUnsolvedPuzzle()
        else:
            self.replaying_solved_puzzle = True
            self.suppress_replay_popup_once = True
            self.menu._reset_current_puzzle_timer()
            self.menu.updatePuzzle()

        return True

    def draw(self, screen):
        if not self.visible:
            self.prompt_rect = p.Rect(0, 0, 1, 1)
            self.prompt_hit_rect = p.Rect(0, 0, 1, 1)
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
            self.popup_bar_rect.centerx,
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

        prompt = None

        if self.mode == "next":
            if self.prompt_pressed_inside:
                prompt = self.textures.ClickHereNextPuzzle[1]
            elif self.prompt_hovered:
                prompt = self.textures.ClickHereNextPuzzle[1]
            else:
                prompt = self.textures.ClickHereNextPuzzle[0]

        elif self.mode == "replay":
            if self.prompt_pressed_inside:
                prompt = self.textures.ClickHereToPlayAgain[1]
            elif self.prompt_hovered:
                prompt = self.textures.ClickHereToPlayAgain[1]
            else:
                prompt = self.textures.ClickHereToPlayAgain[0]

        if prompt is None:
            self.prompt_rect = p.Rect(0, 0, 1, 1)
            self.prompt_hit_rect = p.Rect(0, 0, 1, 1)
            return

        prompt_top = solved_rect.bottom + self.popup_prompt_gap
        max_prompt_top = self.popup_bar_rect.bottom - prompt.get_height() - 2
        prompt_top = min(prompt_top, max_prompt_top)

        self.prompt_rect = prompt.get_rect(midtop=(self.popup_bar_rect.centerx, prompt_top))
        self.prompt_hit_rect = self.prompt_rect.inflate(self.prompt_hitbox_pad_x, self.prompt_hitbox_pad_y)

        if not (self.animating and self.mode == "next"):
            screen.blit(prompt, self.prompt_rect)