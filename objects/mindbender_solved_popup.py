# Jerrin Shirks

# native imports
import math
import time
import pygame as p

# custom imports
from tools.button import Button
from tools.point import Point


class MindBenderSolvedPopup:
    """
    MindBender solved overlay modeled from FUN_00436dd0.

    Key decomp-backed constants:

        DAT_004f8768 = 90.0
        DAT_004f86a0 = 0.8999999761581421
        DAT_004f8780 = 0.5
        DAT_004f84c8 = 180.0
        DAT_004f8a90 = 90.0

        DAT_004fbe00 = 256.0
        DAT_004fc230 = 229.0
        DAT_004fc238 = 483.0
        DAT_004fc240 = 127.0
        DAT_004f8818 = 0.75

        DAT_004fbd30 = -20.0

        DAT_004fc0d8 = 315.0
        DAT_004fc0dc = 317.0
        DAT_004fc0e0 = 363.0
        DAT_004f92f8 = 31.0
    """

    ANIM_DURATION = 90.0
    PROMPT_VISIBLE_PROGRESS = 0.8999999761581421

    HALF = 0.5
    SCALE_DIVISOR = 180.0
    SCALE_WAVE_MULTIPLIER = math.pi
    ANIM_COMPLETE_TIME = 90.0

    ANIM_UNITS_PER_SECOND = 200.0

    GREEN_X = 256.0
    GREEN_Y = 229.0
    GREEN_W = 483.0
    GREEN_H = 127.0
    GREEN_ALPHA = 0.75

    MATCHED_Y_OFFSET = -20.0

    # Hardcoded top-left positions for the prompt textures.
    # These control both rendering and hitboxes.
    NEXT_PROMPT_X = 315
    NEXT_PROMPT_Y = 317

    REPLAY_PROMPT_X = 315
    REPLAY_PROMPT_Y = 317

    def __init__(self, menu):
        self.menu = menu
        self.game = menu.game
        self.textures = menu.textures

        self.visible = False
        self.animating = False
        self.mode = None  # "next" | "replay"

        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False

        # Clone equivalents for this+0xf48 and this+0xf4c.
        self.f48 = 0.0
        self.f4c = 0.0

        self._last_anim_ms = self._now_ms()

        self.next_prompt_btn = Button(
            game=self.game,
            coords=(self.NEXT_PROMPT_X, self.NEXT_PROMPT_Y),
            textures=self.textures.ClickHereNextPuzzle,
            data={
                "hitbox": "all",
                "sounds": self.menu.buttonSoundLow,
                "func": {"clickup": self._handle_next_click},
            },
        )

        self.replay_prompt_btn = Button(
            game=self.game,
            coords=(self.REPLAY_PROMPT_X, self.REPLAY_PROMPT_Y),
            textures=self.textures.ClickHereToPlayAgain,
            data={
                "hitbox": "all",
                "sounds": self.menu.buttonSoundLow,
                "func": {"clickup": self._handle_replay_click},
            },
        )

        self._sync_prompt_button_positions()

    def _now_ms(self):
        try:
            return int(p.time.get_ticks())
        except Exception:
            return int(time.monotonic() * 1000.0)

    def _active_prompt_button(self):
        if self.mode == "next":
            return self.next_prompt_btn

        if self.mode == "replay":
            return self.replay_prompt_btn

        return None

    def _set_button_topleft(self, btn, x, y):
        x = int(round(float(x)))
        y = int(round(float(y)))

        btn.coords = Point((x, y))

        # Button._is_touching() uses btn.rect, so this must stay synced
        # to the same top-left used for rendering.
        btn.rect = btn.image.get_rect(topleft=(x, y))
        btn.size = Point((btn.image.get_width(), btn.image.get_height()))

    def _sync_prompt_button_positions(self):
        self._set_button_topleft(
            self.next_prompt_btn,
            self.NEXT_PROMPT_X,
            self.NEXT_PROMPT_Y,
        )

        self._set_button_topleft(
            self.replay_prompt_btn,
            self.REPLAY_PROMPT_X,
            self.REPLAY_PROMPT_Y,
        )

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
        self.mode = "next"

        self.f48 = 0.0
        self.f4c = 0.0

        self._last_anim_ms = self._now_ms()
        self._sync_prompt_button_positions()

        self.game.sounds.playSound(self.game.sounds.Multipop)

    def show_replay_popup(self):
        self.visible = True
        self.animating = False
        self.mode = "replay"

        self.f48 = self.ANIM_COMPLETE_TIME
        self.f4c = 0.0

        self._last_anim_ms = self._now_ms()
        self._sync_prompt_button_positions()

    def hide(self):
        self.visible = False
        self.animating = False
        self.mode = None

        self.f48 = 0.0
        self.f4c = 0.0

    def clear_transient_state(self):
        self.replaying_solved_puzzle = False
        self.suppress_replay_popup_once = False
        self.hide()

    def _progress(self):
        return min(float(self.f48) / self.ANIM_DURATION, 1.0)

    def _matched_center(self):
        board = self.menu.board
        board_size = getattr(board, "boardSizePx", board.surfaceSize)

        board_x = float(board.boardOffset.x)
        board_y = float(board.boardOffset.y)
        board_w = float(board_size.x)
        board_h = float(board_size.y)

        return (
            board_x + board_w * self.HALF,
            board_y + board_h * self.HALF + self.MATCHED_Y_OFFSET,
        )

    def _matched_scale(self):
        t = min(float(self.f48), self.ANIM_DURATION)
        radians = (t * self.SCALE_WAVE_MULTIPLIER) / self.SCALE_DIVISOR
        return max(0.0, math.sin(radians))

    def _matched_angle(self):
        return float(self.f4c)

    def _advance_animation(self):
        now = self._now_ms()
        dt_ms = max(0, min(100, now - self._last_anim_ms))
        self._last_anim_ms = now

        if not self.animating:
            return

        delta_units = (float(dt_ms) / 1000.0) * self.ANIM_UNITS_PER_SECOND

        self.f48 = min(self.f48 + delta_units, self.ANIM_DURATION)

        # Rotates once during the solved badge scale-in.
        self.f4c = 360.0 * self._progress()

        if self.f48 >= self.ANIM_DURATION:
            self.animating = False
            self.f48 = self.ANIM_COMPLETE_TIME
            self.f4c = 0.0

    def update_interaction(self, mousePos, mouseState, buttonSoundLow):
        if not self.visible:
            return

        self._sync_prompt_button_positions()

        btn = self._active_prompt_button()
        if btn is None:
            return

        # Interaction is active immediately on solve. The prompt may be visually
        # hidden until PROMPT_VISIBLE_PROGRESS, but its hitbox still works.
        btn.testMouseStates(mousePos.tuple(), mouseState)

    def handle_click_up(self, mousePos):
        return False

    def draw(self, screen):
        if not self.visible:
            return

        self._sync_prompt_button_positions()
        self._advance_animation()

        progress = self._progress()
        sweep_width = int(float(self.GREEN_W) * progress)

        if sweep_width > 0:
            sweep_surface = p.Surface(
                (sweep_width, int(round(self.GREEN_H))),
                p.SRCALPHA,
            )

            alpha = int(round(255.0 * self.GREEN_ALPHA))
            sweep_surface.fill((0, 255, 0, alpha))

            screen.blit(
                sweep_surface,
                (int(round(self.GREEN_X)), int(round(self.GREEN_Y))),
            )

        matched_center = self._matched_center()

        if self.animating:
            scale = self._matched_scale()
            angle = self._matched_angle()

            if scale > 0.0:
                matched = p.transform.rotozoom(self.textures.Matched, angle, scale)
                matched_rect = matched.get_rect(center=matched_center)
                screen.blit(matched, matched_rect)
        else:
            matched_rect = self.textures.Matched.get_rect(center=matched_center)
            screen.blit(self.textures.Matched, matched_rect)

            if self.mode is None:
                self.mode = "replay"

        # Visual only. Hover/click state is already updated in update_interaction().
        if progress < self.PROMPT_VISIBLE_PROGRESS:
            return

        btn = self._active_prompt_button()
        if btn is None:
            return

        btn.render(screen)