# Jerrin Shirks
# tools/platform_browser.py

from __future__ import annotations

from typing import Optional, Tuple

from tools.platform_api import PlatformAPI, WindowRect


class BrowserPlatformAPI(PlatformAPI):
    """
    Browser / wasm backend.

    Design goals:
    - Keep Game API identical to the Windows backend.
    - Use browser pointer events instead of Win32 global polling.
    - Preserve drag semantics as well as the browser allows.
    - Gracefully degrade if no JS bridge is available.

    Notes:
    - There is no real browser equivalent for moving a native OS window.
    - There is no real browser equivalent for minimizing the native window.
    - Pointer tracking is based on the last JS event we received.
    """

    def __init__(self):
        self._js = None
        self._window = None
        self._document = None
        self._canvas = None

        self._installed = False
        self._drag_pointer_id = None

        self._mouse_x = 0
        self._mouse_y = 0
        self._left_down = False
        self._right_down = False
        self._focused = True

        self._cleanup = []

    def setup_process(self) -> None:
        """
        No DPI setup in the browser.
        Try to install JS event hooks if a JS bridge exists.
        """
        self._try_install_js_hooks()

    def _try_install_js_hooks(self) -> None:
        if self._installed:
            return

        try:
            import js  # type: ignore
        except Exception:
            # No JS bridge available in this runtime.
            # Fallbacks will use pygame mouse state only.
            return

        self._js = js
        self._window = js.window
        self._document = js.document

        canvas = None

        try:
            if hasattr(js, "Module") and hasattr(js.Module, "canvas"):
                canvas = js.Module.canvas
        except Exception:
            canvas = None

        if canvas is None:
            try:
                canvas = js.document.querySelector("canvas")
            except Exception:
                canvas = None

        self._canvas = canvas

        if self._canvas is None:
            return

        try:
            self._canvas.style.touchAction = "none"
        except Exception:
            pass

        def update_pos_from_event(evt):
            """
            Convert browser client coords into canvas local coords.
            """
            try:
                rect = self._canvas.getBoundingClientRect()
                scale_x = self._canvas.width / rect.width if rect.width else 1.0
                scale_y = self._canvas.height / rect.height if rect.height else 1.0

                self._mouse_x = int((evt.clientX - rect.left) * scale_x)
                self._mouse_y = int((evt.clientY - rect.top) * scale_y)
            except Exception:
                try:
                    self._mouse_x = int(evt.offsetX)
                    self._mouse_y = int(evt.offsetY)
                except Exception:
                    pass

        def on_pointer_down(evt):
            update_pos_from_event(evt)

            try:
                button = int(evt.button)
            except Exception:
                button = 0

            if button == 0:
                self._left_down = True
            elif button == 2:
                self._right_down = True

            try:
                self._drag_pointer_id = int(evt.pointerId)
            except Exception:
                self._drag_pointer_id = None

            try:
                self._canvas.setPointerCapture(evt.pointerId)
            except Exception:
                pass

            try:
                evt.preventDefault()
            except Exception:
                pass

        def on_pointer_move(evt):
            update_pos_from_event(evt)

            try:
                buttons = int(evt.buttons)
                self._left_down = (buttons & 1) != 0
                self._right_down = (buttons & 2) != 0
            except Exception:
                pass

            try:
                evt.preventDefault()
            except Exception:
                pass

        def on_pointer_up(evt):
            update_pos_from_event(evt)

            try:
                button = int(evt.button)
            except Exception:
                button = 0

            if button == 0:
                self._left_down = False
            elif button == 2:
                self._right_down = False

            try:
                if self._drag_pointer_id is not None:
                    self._canvas.releasePointerCapture(evt.pointerId)
            except Exception:
                pass

            self._drag_pointer_id = None

            try:
                evt.preventDefault()
            except Exception:
                pass

        def on_pointer_cancel(evt):
            self._left_down = False
            self._right_down = False
            self._drag_pointer_id = None

        def on_blur(evt):
            self._focused = False
            self._left_down = False
            self._right_down = False
            self._drag_pointer_id = None

        def on_focus(evt):
            self._focused = True

        def on_context_menu(evt):
            try:
                evt.preventDefault()
            except Exception:
                pass

        def on_visibility_change(evt):
            try:
                hidden = bool(self._document.hidden)
            except Exception:
                hidden = False

            self._focused = not hidden
            if hidden:
                self._left_down = False
                self._right_down = False
                self._drag_pointer_id = None

        self._canvas.addEventListener("pointerdown", on_pointer_down)
        self._canvas.addEventListener("pointermove", on_pointer_move)
        self._canvas.addEventListener("pointerup", on_pointer_up)
        self._canvas.addEventListener("pointercancel", on_pointer_cancel)
        self._canvas.addEventListener("contextmenu", on_context_menu)

        self._window.addEventListener("blur", on_blur)
        self._window.addEventListener("focus", on_focus)
        self._document.addEventListener("visibilitychange", on_visibility_change)

        self._cleanup = [
            ("pointerdown", on_pointer_down, self._canvas),
            ("pointermove", on_pointer_move, self._canvas),
            ("pointerup", on_pointer_up, self._canvas),
            ("pointercancel", on_pointer_cancel, self._canvas),
            ("contextmenu", on_context_menu, self._canvas),
            ("blur", on_blur, self._window),
            ("focus", on_focus, self._window),
            ("visibilitychange", on_visibility_change, self._document),
        ]

        self._installed = True

    def shutdown(self) -> None:
        for event_name, handler, target in self._cleanup:
            try:
                target.removeEventListener(event_name, handler)
            except Exception:
                pass
        self._cleanup = []
        self._installed = False

    def get_window_client_mouse_pos(self, pygame_instance) -> Tuple[int, int]:
        """
        Browser version returns the last known pointer position relative to the canvas.
        Falls back to pygame if JS hooks are unavailable.
        """
        if self._installed:
            return int(self._mouse_x), int(self._mouse_y)

        try:
            return pygame_instance.mouse.get_pos()
        except Exception:
            return 0, 0

    def get_window_rect(self, pygame_instance) -> Optional[WindowRect]:
        """
        Browser canvas rect in page coordinates.
        This is not a native OS window rect, but it preserves the interface.
        """
        if not self._installed or self._canvas is None:
            return None

        try:
            rect = self._canvas.getBoundingClientRect()
            return WindowRect(
                left=int(rect.left),
                top=int(rect.top),
                right=int(rect.right),
                bottom=int(rect.bottom),
            )
        except Exception:
            return None

    def set_window_pos(self, pygame_instance, x: int, y: int) -> None:
        """
        No browser equivalent for moving the native window from app code.
        Intentionally a no-op.
        """
        return

    def minimize_window(self, pygame_instance) -> None:
        """
        No browser equivalent for minimizing the native window from app code.
        Intentionally a no-op.
        """
        return

    def get_global_mouse_buttons(self, pygame_instance) -> Tuple[bool, bool]:
        """
        Browser version exposes tracked pointer button state.
        Falls back to pygame if JS hooks are unavailable.
        """
        if self._installed:
            return bool(self._left_down), bool(self._right_down)

        pressed = pygame_instance.mouse.get_pressed(3)
        return bool(pressed[0]), bool(pressed[2])

    def has_focus(self) -> bool:
        return bool(self._focused)