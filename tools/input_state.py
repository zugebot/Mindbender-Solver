# Jerrin Shirks
# tools/input_state.py

from __future__ import annotations

from dataclasses import dataclass

from engine.constants import MOUSE_CLICK_DOWN, MOUSE_CLICK_UP, MOUSE_HELD_DOWN, MOUSE_HOVER


@dataclass
class InputState:
    mouse_state: int = MOUSE_HOVER
    left_down_latched: bool = False
    right_down_latched: bool = False
    has_window_focus: bool = True

    def reset_buttons(self) -> None:
        self.left_down_latched = False
        self.right_down_latched = False
        self.mouse_state = MOUSE_HOVER

    def update_mouse_state(
        self,
        left_button_down: bool,
        right_button_down: bool,
        over_non_client: bool,
        dragging_window: bool,
    ) -> bool:
        """
        Returns whether a right-click-down edge occurred this frame.
        """

        if dragging_window:
            self.mouse_state = MOUSE_HOVER
            self.left_down_latched = False
            self.right_down_latched = right_button_down
            return False

        # Preserve your existing drag semantics.
        # A left drag that begins in the client area stays active until button release.
        if left_button_down and not self.left_down_latched:
            if over_non_client:
                self.mouse_state = MOUSE_HOVER
                self.left_down_latched = False
            else:
                self.mouse_state = MOUSE_CLICK_DOWN
                self.left_down_latched = True

        elif left_button_down and self.left_down_latched:
            self.mouse_state = MOUSE_HELD_DOWN

        elif (not left_button_down) and self.left_down_latched:
            self.mouse_state = MOUSE_CLICK_UP
            self.left_down_latched = False

        else:
            self.mouse_state = MOUSE_HOVER
            self.left_down_latched = False

        right_click_down = right_button_down and not self.right_down_latched
        self.right_down_latched = right_button_down
        return right_click_down