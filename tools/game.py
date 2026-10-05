# Jerrin Shirks
# tools/game.py

# custom imports
import pygame as p
from engine.constants import *
from tools.point import Point
from tools.menu import Menu
from engine.settings import Settings
from tools.input_state import InputState
from tools.platform_factory import create_platform


class Game:
    def __init__(self, settings: Settings):

        # settings
        self.settings = settings
        self.internalSize = Point(self.settings.internalWindowSize)
        self.outputSize = Point(self.settings.outputWindowSize)
        self.screenSize = Point(self.settings.internalWindowSize)
        self.fps = self.settings.fps
        self.debug = self.settings.debug

        # platform
        self.platform = create_platform()
        self.platform.setup_process()

        # measured outer metrics from original game
        self.chromeEnabled = True
        self.borderLeft = 3
        self.borderRight = 3
        self.borderTop = 26
        self.borderBottom = 3

        self.clientRenderOffsetX = 0
        self.clientRenderOffsetY = 0

        self.captionTextLeftPadding = 10
        self.captionTextTopPadding = 5

        self.closeButtonWidth = 45
        self.closeButtonHeight = self.borderTop
        self.minButtonWidth = 45
        self.minButtonHeight = self.borderTop

        # pygame stuff
        self.instance = p
        self.instance.init()
        self.instance.display.set_caption(self.settings.windowName)

        self.clock = p.time.Clock()

        # borderless outer window sized to old metrics
        self.windowSize = Point(
            self.outputSize.x + self.borderLeft + self.borderRight,
            self.outputSize.y + self.borderTop + self.borderBottom
        )

        window_flags = p.NOFRAME if self.chromeEnabled else 0
        self.window = self.instance.display.set_mode(self.windowSize.tuple(), window_flags)

        # internal fixed-resolution render target
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

        self.textures = None
        self.sounds = None

        # running
        self.running = True
        self.mouseState = MOUSE_HOVER
        self.mousePos = Point()
        self.inputState = InputState()
        self.hasWindowFocus = True

        # Board input samples for the current frame.
        # This is populated by global polling, not by pygame mouse events.
        self.mouseSamples = []

        # menus
        self.menuIndex = -1
        self.menus = []
        self.lastMenuSize = 0
        self.lastMenuImage = None

        # window dragging
        self._draggingWindow = False
        self._dragOffsetX = 0
        self._dragOffsetY = 0

        # button hover state
        self._hoveringClose = False
        self._hoveringMinimize = False

        # title font
        self.captionFont = p.font.SysFont("Segoe UI", 9, bold=False)
        self._caption_surface_focused = None
        self._caption_surface_unfocused = None

        # Reused scaling target. Avoid allocating a new scaled surface every frame.
        self._scaled_frame_surface = None
        self._scaled_frame_size = None

    def currentMenu(self, offset=0) -> Menu:
        try:
            return self.menus[self.menuIndex + offset]
        except Exception as e:
            if self.debug:
                print(e)

    def addMenu(self, menu, _set: bool = False) -> None:
        self.menus.append(menu(self))
        self.menuIndex += 1
        if self.debug:
            print("adding a menu!")
            print("Menu Stack Length:", len(self.menus), "\n")

    def popMenu(self) -> None:
        if len(self.menus) == 1:
            print("Tried to pop menu, no parent menu")
            return

        self.menus.pop()
        self.menuIndex -= 1

        self.lastMenuImage = None
        if self.debug:
            print("Popping Menu!")
            print("New Menu Length:", len(self.menus), "\n")

    def setMenuIndex(self, index) -> None:
        self.menuIndex = index

    def _rebuildWindow(self):
        self.windowSize = Point(
            self.outputSize.x + self.borderLeft + self.borderRight,
            self.outputSize.y + self.borderTop + self.borderBottom
        )
        window_flags = p.NOFRAME if self.chromeEnabled else 0
        self.window = self.instance.display.set_mode(self.windowSize.tuple(), window_flags)

    def setOutputSize(self, size) -> None:
        self.outputSize = Point(size)
        self._rebuildWindow()

    def setInternalSize(self, size) -> None:
        self.internalSize = Point(size)
        self.screenSize = Point(size)
        self.screen = self.instance.Surface(self.internalSize.tuple()).convert_alpha()

    def _client_origin(self):
        return Point(
            self.borderLeft + self.clientRenderOffsetX,
            self.borderTop + self.clientRenderOffsetY,
        )

    def _client_rect(self):
        return p.Rect(
            self.borderLeft,
            self.borderTop,
            self.outputSize.x,
            self.outputSize.y
        )

    def _title_bar_rect(self):
        return p.Rect(0, 0, self.windowSize.x, self.borderTop)

    def _mapMouseToInternal(self, raw_mouse):
        x, y = raw_mouse

        if self.chromeEnabled:
            x -= self.borderLeft
            y -= self.borderTop

        if self.outputSize.x == 0 or self.outputSize.y == 0:
            return Point()

        scale_x = self.internalSize.x / self.outputSize.x
        scale_y = self.internalSize.y / self.outputSize.y

        mapped_x = int(x * scale_x)
        mapped_y = int(y * scale_y)

        return Point(mapped_x, mapped_y)

    def _get_window_client_mouse_pos(self):
        return self.platform.get_window_client_mouse_pos(self.instance)

    def _get_window_rect(self):
        return self.platform.get_window_rect(self.instance)

    def _set_window_pos(self, x, y):
        self.platform.set_window_pos(self.instance, x, y)

    def _minimize_window(self):
        self.platform.minimize_window(self.instance)

    def _close_button_rect(self):
        return p.Rect(
            self.windowSize.x - self.closeButtonWidth,
            0,
            self.closeButtonWidth,
            self.closeButtonHeight
        )

    def _minimize_button_rect(self):
        return p.Rect(
            self.windowSize.x - self.closeButtonWidth - self.minButtonWidth,
            0,
            self.minButtonWidth,
            self.minButtonHeight
        )

    def _point_in_title_drag_zone(self, x, y):
        if not self.chromeEnabled:
            return False

        if not self._title_bar_rect().collidepoint(x, y):
            return False

        if self._close_button_rect().collidepoint(x, y):
            return False

        if self._minimize_button_rect().collidepoint(x, y):
            return False

        return True

    def _start_window_drag(self):
        rect = self._get_window_rect()
        if rect is None:
            return

        raw_mouse = self._get_window_client_mouse_pos()
        cursor_screen_x = rect.left + raw_mouse[0]
        cursor_screen_y = rect.top + raw_mouse[1]

        self._draggingWindow = True
        self._dragOffsetX = cursor_screen_x - rect.left
        self._dragOffsetY = cursor_screen_y - rect.top

    def _update_window_drag(self):
        if not self._draggingWindow:
            return

        rect = self._get_window_rect()
        if rect is None:
            return

        raw_mouse = self._get_window_client_mouse_pos()
        cursor_screen_x = rect.left + raw_mouse[0]
        cursor_screen_y = rect.top + raw_mouse[1]

        self._set_window_pos(
            cursor_screen_x - self._dragOffsetX,
            cursor_screen_y - self._dragOffsetY
        )

    def _stop_window_drag(self):
        self._draggingWindow = False

    def _draw_minimize_glyph(self, rect):
        y = rect.centery + 4
        x1 = rect.centerx - 6
        x2 = rect.centerx + 6
        p.draw.line(self.window, (0, 0, 0), (x1, y), (x2, y), 1)

    def _draw_close_glyph(self, rect):
        x1 = rect.centerx - 5
        y1 = rect.centery - 5
        x2 = rect.centerx + 5
        y2 = rect.centery + 5
        p.draw.line(self.window, (0, 0, 0), (x1, y1), (x2, y2), 1)
        p.draw.line(self.window, (0, 0, 0), (x2, y1), (x1, y2), 1)

    def _get_caption_surface(self):
        if self.hasWindowFocus:
            if self._caption_surface_focused is None:
                self._caption_surface_focused = self.captionFont.render(
                    self.settings.windowName,
                    True,
                    (0, 0, 0),
                )
            return self._caption_surface_focused

        if self._caption_surface_unfocused is None:
            self._caption_surface_unfocused = self.captionFont.render(
                self.settings.windowName,
                True,
                (80, 80, 80),
            )
        return self._caption_surface_unfocused

    def _draw_fake_windows10_titlebar(self):
        if not self.chromeEnabled:
            return

        raw_mouse = self._get_window_client_mouse_pos()

        title_rect = self._title_bar_rect()
        min_rect = self._minimize_button_rect()
        close_rect = self._close_button_rect()

        self._hoveringMinimize = min_rect.collidepoint(*raw_mouse)
        self._hoveringClose = close_rect.collidepoint(*raw_mouse)

        if self.hasWindowFocus:
            title_color = (255, 255, 255)
            separator_color = (208, 208, 208)
        else:
            title_color = (245, 245, 245)
            separator_color = (220, 220, 220)

        p.draw.rect(self.window, title_color, title_rect)

        p.draw.line(
            self.window,
            separator_color,
            (0, self.borderTop - 1),
            (self.windowSize.x - 1, self.borderTop - 1),
            1,
        )

        self.window.blit(
            self._get_caption_surface(),
            (self.captionTextLeftPadding, self.captionTextTopPadding),
        )

        if self._hoveringMinimize:
            p.draw.rect(self.window, (229, 229, 229), min_rect)
        self._draw_minimize_glyph(min_rect)

        if self._hoveringClose:
            p.draw.rect(self.window, (232, 17, 35), close_rect)
            x1 = close_rect.centerx - 5
            y1 = close_rect.centery - 5
            x2 = close_rect.centerx + 5
            y2 = close_rect.centery + 5
            p.draw.line(self.window, (255, 255, 255), (x1, y1), (x2, y2), 1)
            p.draw.line(self.window, (255, 255, 255), (x2, y1), (x1, y2), 1)
        else:
            self._draw_close_glyph(close_rect)

    def _blit_edge_gutters(self, scaled):
        pass

    def _get_scaled_frame(self):
        if self.outputSize.x == self.internalSize.x and self.outputSize.y == self.internalSize.y:
            return self.screen

        out_size = self.outputSize.tuple()

        if self.settings.smoothScale:
            return p.transform.smoothscale(self.screen, out_size)

        if self._scaled_frame_surface is None or self._scaled_frame_size != out_size:
            self._scaled_frame_surface = p.Surface(out_size).convert()
            self._scaled_frame_size = out_size

        p.transform.scale(self.screen, out_size, self._scaled_frame_surface)
        return self._scaled_frame_surface

    def _presentScreen(self):
        if self.chromeEnabled:
            self.window.fill((255, 255, 255))
            client_origin = self._client_origin()
            self.window.blit(self._get_scaled_frame(), client_origin.tuple())
            self._draw_fake_windows10_titlebar()
        else:
            self.window.blit(self._get_scaled_frame(), (0, 0))

    def quit(self):
        self.running = False
        p.quit()
        exit()

    def _handle_focus_loss(self):
        self.inputState.reset_buttons()
        self.mouseState = MOUSE_HOVER
        self.mouseSamples = []

    def _queue_mouse_sample(self, raw_mouse, mouse_state):
        raw_mouse = (int(raw_mouse[0]), int(raw_mouse[1]))
        mapped_mouse = self._mapMouseToInternal(raw_mouse)

        self.mouseSamples.append((mapped_mouse, mouse_state))
        self.mousePos = mapped_mouse
        self.mouseState = mouse_state

    def _queue_global_mouse_input(self):
        """
        Poll global mouse state every frame.

        This is intentionally independent of pygame mouse focus. It allows the clone
        to receive drag input while another window is focused or layered above it.
        """

        if self._draggingWindow:
            self.mouseState = MOUSE_HOVER
            self.inputState.left_down_latched = False
            return

        raw_mouse = self._get_window_client_mouse_pos()
        raw_mouse = (int(raw_mouse[0]), int(raw_mouse[1]))

        client_rect = self._client_rect()
        over_non_client = not client_rect.collidepoint(raw_mouse[0], raw_mouse[1])

        left_button_down, right_button_down = self.platform.get_global_mouse_buttons(self.instance)

        if left_button_down:
            if not self.inputState.left_down_latched:
                if over_non_client:
                    self.mouseState = MOUSE_HOVER
                    self.mousePos = self._mapMouseToInternal(raw_mouse)
                    self.inputState.left_down_latched = False
                else:
                    self.inputState.left_down_latched = True
                    self._queue_mouse_sample(raw_mouse, MOUSE_CLICK_DOWN)
            else:
                self._queue_mouse_sample(raw_mouse, MOUSE_HELD_DOWN)

        else:
            if self.inputState.left_down_latched:
                self._queue_mouse_sample(raw_mouse, MOUSE_CLICK_UP)
                self.inputState.left_down_latched = False
            else:
                self.mousePos = self._mapMouseToInternal(raw_mouse)
                if over_non_client:
                    self.mouseState = MOUSE_HOVER
                else:
                    self._queue_mouse_sample(raw_mouse, MOUSE_HOVER)

        right_click_down = right_button_down and not self.inputState.right_down_latched
        self.inputState.right_down_latched = right_button_down

        if right_click_down and not over_non_client:
            mapped_mouse = self._mapMouseToInternal(raw_mouse)
            self.mousePos = mapped_mouse

            current_menu = self.currentMenu()
            if current_menu is not None and hasattr(current_menu, "onRightClickDown"):
                current_menu.onRightClickDown(mapped_mouse)

    def _replayCurrentMenuInput(self):
        current_menu = self.currentMenu()
        if current_menu is None:
            return

        if not self.mouseSamples:
            current_menu.run(self.mousePos, self.mouseState)
            return

        for mouse_pos, mouse_state in self.mouseSamples:
            current_menu.run(mouse_pos, mouse_state)

    def eventLoop(self):
        self.mouseSamples = []

        for event in self.instance.event.get():
            if event.type == p.QUIT:
                self.quit()

            elif event.type == p.KEYDOWN:
                if not self.settings.needsWindowFocus or self.hasWindowFocus:
                    self.currentMenu().onKeyDown(event)

            elif event.type == p.MOUSEBUTTONDOWN:
                if event.button == 1 and self.chromeEnabled:
                    raw_mouse = self._get_window_client_mouse_pos()

                    if self._close_button_rect().collidepoint(raw_mouse[0], raw_mouse[1]):
                        self.quit()

                    elif self._minimize_button_rect().collidepoint(raw_mouse[0], raw_mouse[1]):
                        self._minimize_window()

                    elif self._point_in_title_drag_zone(raw_mouse[0], raw_mouse[1]):
                        self._start_window_drag()

            elif event.type == p.MOUSEBUTTONUP:
                if event.button == 1:
                    self._stop_window_drag()

            elif event.type == p.WINDOWFOCUSGAINED:
                self.hasWindowFocus = True

            elif event.type == p.WINDOWFOCUSLOST:
                self.hasWindowFocus = False
                if self.settings.needsWindowFocus:
                    # Do not reset mouse buttons here.
                    # Board input is global-polled specifically so testing works
                    # while another window is focused.
                    pass

            elif event.type == p.ACTIVEEVENT:
                if hasattr(event, "state") and hasattr(event, "gain"):
                    if event.state & 2:
                        self.hasWindowFocus = bool(event.gain)

        self._update_window_drag()

        try:
            pygame_focused = bool(self.instance.key.get_focused())
        except Exception:
            pygame_focused = True

        platform_focused = True
        if hasattr(self.platform, "has_focus"):
            try:
                platform_focused = bool(self.platform.has_focus())
            except Exception:
                platform_focused = True

        focused = pygame_focused and platform_focused

        if self.settings.needsWindowFocus:
            self.hasWindowFocus = bool(focused)
        else:
            self.hasWindowFocus = True

        # Critical: board mouse input is polled globally, even if this pygame
        # window does not have focus.
        self._queue_global_mouse_input()

    def run(self, running=True):
        while running:
            if len(self.menus) != self.lastMenuSize:
                self.lastMenuImage = self.screen.copy()
                self.lastMenuSize = len(self.menus)

            self.screen.fill(EMPTY_SCREEN)
            self.eventLoop()

            self._replayCurrentMenuInput()

            self._presentScreen()

            self.instance.display.flip()
            self.clock.tick(self.fps)

        p.quit()