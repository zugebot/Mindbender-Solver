# Jerrin Shirks

# native imports
import pygame as p

# custom imports
from engine.constants import *
from tools.point import Point
from tools.sprite_object import spriteObject


class SelectBox(spriteObject):
    def __init__(
        self,
        menu,
        coords=(0, 0),
        size=(180, 32),
        options=None,
        selected_index=0,
        font=None,
        max_visible_options=8,
        option_height=None,
        on_change=None,
        placeholder="Select..."
    ):
        super().__init__(menu)

        self.coords = Point(coords)
        self.size = Point(size)

        self.options = list(options) if options is not None else []
        self.selected_index = selected_index if self.options else -1
        if self.options and not (0 <= self.selected_index < len(self.options)):
            self.selected_index = 0

        self.font = font if font is not None else p.font.SysFont("Arial", 18)
        self.max_visible_options = max(1, int(max_visible_options))
        self.option_height = int(option_height if option_height is not None else self.size.y)
        self.on_change = on_change
        self.placeholder = placeholder

        self.open = False
        self.hovered = False
        self.hovered_option_index = -1
        self.pressed = False

        self.scroll_index = 0

        self.bg_color = (245, 245, 245)
        self.bg_hover_color = (255, 255, 255)
        self.bg_pressed_color = (230, 230, 230)
        self.border_color = (40, 40, 40)
        self.text_color = (20, 20, 20)
        self.placeholder_color = (120, 120, 120)
        self.dropdown_bg_color = (250, 250, 250)
        self.option_hover_color = (220, 235, 255)
        self.option_selected_color = (205, 225, 255)
        self.arrow_color = (30, 30, 30)

    def get_value(self):
        if 0 <= self.selected_index < len(self.options):
            return self.options[self.selected_index]
        return None

    def set_options(self, options, selected_index=0):
        self.options = list(options) if options is not None else []
        if not self.options:
            self.selected_index = -1
        else:
            self.selected_index = max(0, min(int(selected_index), len(self.options) - 1))
        self.hovered_option_index = -1
        self.scroll_index = 0
        self.open = False

    def set_selected_index(self, index, fire_callback=True):
        if not self.options:
            self.selected_index = -1
            return

        index = int(index)
        if not (0 <= index < len(self.options)):
            return

        changed = (index != self.selected_index)
        self.selected_index = index

        if fire_callback and changed and callable(self.on_change):
            self.on_change(self.get_value(), self.selected_index)

    def set_selected_value(self, value, fire_callback=True):
        for i, option in enumerate(self.options):
            if option == value:
                self.set_selected_index(i, fire_callback=fire_callback)
                return True
        return False

    def _main_rect(self):
        return p.Rect(int(self.coords.x), int(self.coords.y), int(self.size.x), int(self.size.y))

    def _dropdown_rect(self):
        visible_count = min(len(self.options), self.max_visible_options)
        return p.Rect(
            int(self.coords.x),
            int(self.coords.y + self.size.y),
            int(self.size.x),
            int(visible_count * self.option_height)
        )

    def _option_rect(self, visible_row):
        drop = self._dropdown_rect()
        return p.Rect(
            drop.x,
            drop.y + visible_row * self.option_height,
            drop.width,
            self.option_height
        )

    def _point_in_main(self, mousePos):
        return self._main_rect().collidepoint(mousePos[0], mousePos[1])

    def _point_in_dropdown(self, mousePos):
        if not self.open:
            return False
        return self._dropdown_rect().collidepoint(mousePos[0], mousePos[1])

    def _get_option_index_at_pos(self, mousePos):
        if not self.open or not self._point_in_dropdown(mousePos):
            return -1

        local_y = mousePos[1] - (self.coords.y + self.size.y)
        row = int(local_y // self.option_height)
        option_index = self.scroll_index + row

        if 0 <= option_index < len(self.options):
            return option_index
        return -1

    def _ensure_selected_visible(self):
        if self.selected_index < 0:
            return

        if self.selected_index < self.scroll_index:
            self.scroll_index = self.selected_index
        elif self.selected_index >= self.scroll_index + self.max_visible_options:
            self.scroll_index = self.selected_index - self.max_visible_options + 1

    def _toggle_open(self):
        self.open = not self.open
        self.hovered_option_index = -1
        if self.open:
            self._ensure_selected_visible()

    def onMouseHover(self, mousePos):
        self.hovered = self._point_in_main(mousePos)
        self.hovered_option_index = self._get_option_index_at_pos(mousePos)

    def onMouseDown(self, mousePos):
        self.pressed = False

        if self._point_in_main(mousePos):
            self.pressed = True
            return

        if self.open and self._point_in_dropdown(mousePos):
            self.pressed = True
            return

        self.open = False
        self.hovered_option_index = -1

    def onMouseHeldDown(self, mousePos):
        self.hovered = self._point_in_main(mousePos)
        self.hovered_option_index = self._get_option_index_at_pos(mousePos)

    def onMouseUp(self, mousePos):
        was_pressed = self.pressed
        self.pressed = False

        if not was_pressed:
            return

        if self._point_in_main(mousePos):
            self._toggle_open()
            return

        clicked_option = self._get_option_index_at_pos(mousePos)
        if clicked_option != -1:
            self.set_selected_index(clicked_option, fire_callback=True)
            self.open = False
            self.hovered_option_index = -1
            return

        self.open = False
        self.hovered_option_index = -1

    def testMouseStates(self, mousePos, mouseState):
        super().testMouseStates(mousePos, mouseState)

    def _draw_text_clipped(self, screen, text, color, rect, left_padding=8):
        text = "" if text is None else str(text)
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect()
        text_rect.midleft = (rect.x + left_padding, rect.centery)

        clip_rect = rect.inflate(-8, -4)
        old_clip = screen.get_clip()
        screen.set_clip(clip_rect)
        screen.blit(text_surface, text_rect)
        screen.set_clip(old_clip)

    def _draw_arrow(self, screen, rect):
        arrow_w = 10
        arrow_h = 6
        cx = rect.right - 16
        cy = rect.centery + 1

        points = [
            (cx - arrow_w // 2, cy - arrow_h // 2),
            (cx + arrow_w // 2, cy - arrow_h // 2),
            (cx, cy + arrow_h // 2),
        ]
        p.draw.polygon(screen, self.arrow_color, points)

    def render(self, screen):
        main_rect = self._main_rect()

        bg = self.bg_color
        if self.pressed and self.hovered:
            bg = self.bg_pressed_color
        elif self.hovered or self.open:
            bg = self.bg_hover_color

        p.draw.rect(screen, bg, main_rect)
        p.draw.rect(screen, self.border_color, main_rect, 2)

        selected_value = self.get_value()
        if selected_value is None:
            self._draw_text_clipped(screen, self.placeholder, self.placeholder_color, main_rect)
        else:
            self._draw_text_clipped(screen, selected_value, self.text_color, main_rect)

        self._draw_arrow(screen, main_rect)

        if not self.open:
            return

        dropdown_rect = self._dropdown_rect()
        p.draw.rect(screen, self.dropdown_bg_color, dropdown_rect)
        p.draw.rect(screen, self.border_color, dropdown_rect, 2)

        visible_count = min(len(self.options), self.max_visible_options)
        for visible_row in range(visible_count):
            option_index = self.scroll_index + visible_row
            if option_index >= len(self.options):
                break

            option_rect = self._option_rect(visible_row)

            if option_index == self.selected_index:
                p.draw.rect(screen, self.option_selected_color, option_rect)
            elif option_index == self.hovered_option_index:
                p.draw.rect(screen, self.option_hover_color, option_rect)

            self._draw_text_clipped(
                screen,
                self.options[option_index],
                self.text_color,
                option_rect
            )

        if len(self.options) > self.max_visible_options:
            bar_w = 8
            track_rect = p.Rect(
                dropdown_rect.right - bar_w - 2,
                dropdown_rect.y + 2,
                bar_w,
                dropdown_rect.height - 4
            )
            p.draw.rect(screen, (230, 230, 230), track_rect)

            thumb_h = max(18, int(track_rect.height * (self.max_visible_options / float(len(self.options)))))
            max_scroll = len(self.options) - self.max_visible_options
            thumb_y = track_rect.y
            if max_scroll > 0:
                thumb_y += int((track_rect.height - thumb_h) * (self.scroll_index / float(max_scroll)))

            thumb_rect = p.Rect(track_rect.x, thumb_y, track_rect.width, thumb_h)
            p.draw.rect(screen, (160, 160, 160), thumb_rect)