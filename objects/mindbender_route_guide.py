# Jerrin Shirks

from __future__ import annotations

# native imports
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pygame as p

# custom imports
from tools.sprite_object import spriteObject

HARDCODED_PROMPT_GUIDE_RECT = p.Rect(315, 317, 363, 31)

PROMPT_GUIDE_MAX_WIDTH = 120
PROMPT_GUIDE_MIN_WIDTH = 120

EPSILON = 1e-6

# Studio-side symbolic click locations that should map to actual MindBender UI.
MINI_PANEL_TARGETS = {
    (-3.18, 3.0475): 0,  # level 1
    (-2.67, 3.4205): 1,  # level 2
    (-2.11, 3.5125): 2,  # level 3
    (-1.52, 3.4205): 3,  # level 4
    (-0.98, 3.1125): 4,  # level 5
}
NEXT_WORLD_STAR_TARGET = (-1.22, 4.73)

PROMPT_CLICK_TARGETS = {
    (1.0, 3.0),
    (2.0, 3.0),
    (3.0, 3.0),
    (4.0, 3.0),
}

GROUP_COLORS = [
    (255, 74, 176),   # pink
    (163, 210, 0),    # lime
    (60, 190, 255),   # cyan
    (255, 184, 76),   # orange
    (170, 120, 255),  # purple
    (255, 105, 105),  # red
]


@dataclass
class GuideOverlayMove:
    move: str
    click_down: tuple[float, float]
    lock_point: tuple[float, float]
    release: tuple[float, float]
    path_points: list[tuple[float, float]]
    selected_line: int
    displacement: int
    free_drag: bool
    drag_distance: float
    move_distance: float
    total_distance_to_here: float

    @property
    def axis(self) -> str:
        return self.move[0].upper() if self.move else "?"


@dataclass
class GuideRouteStep:
    puzzle_id: str
    world_number: int
    puzzle_number: int
    move_string: str
    total_cost: float
    total_move: float
    total_drag: float
    start_mouse_position: tuple[float, float] | None
    end_positions: list[tuple[float, float]]
    end_next_puzzle: bool
    post_end_click_target: tuple[float, float] | None
    has_fat: bool
    initial_fat_position: tuple[float, float] | None
    final_mouse_target: tuple[float, float] | None
    final_move_distance: float
    overlay_moves: list[GuideOverlayMove]


@dataclass
class GuideQueuedAction:
    kind: str  # "move" | "mouse_travel" | "target_click"
    puzzle_id: str
    step_ref: GuideRouteStep
    move_ref: GuideOverlayMove | None
    from_point: tuple[float, float] | None
    to_point: tuple[float, float] | None
    target_rect: p.Rect | None
    combo_number: int
    color_index: int
    is_current_puzzle: bool
    is_current_action: bool


class MindBenderRouteGuideToggle(spriteObject):
    def __init__(self, menu, guide, rect: p.Rect):
        super().__init__(menu)
        self.guide = guide
        self.rect = rect
        self.hovered = False
        self.mouse_down_inside = False

        self.font = p.font.SysFont("arial", 18, bold=True)
        self.shadow_font = p.font.SysFont("arial", 18, bold=True)

    def _contains(self, mouse_pos) -> bool:
        return self.rect.collidepoint(mouse_pos)

    def onMouseHover(self, mousePos):
        hovered = self._contains(mousePos)
        if hovered and not self.hovered:
            if self.menu.buttonSoundLow["hover"] is not None:
                self.game.sounds.playSound(self.menu.buttonSoundLow["hover"])
        self.hovered = hovered
        if hovered:
            self.game.cursorIndex = 2

    def onMouseDown(self, mousePos):
        self.mouse_down_inside = self._contains(mousePos)
        if self.mouse_down_inside and self.menu.buttonSoundLow["clickdown"] is not None:
            self.game.sounds.playSound(self.menu.buttonSoundLow["clickdown"])

    def onMouseUp(self, mousePos):
        inside = self._contains(mousePos)
        if self.mouse_down_inside and inside:
            self.guide.enabled = not self.guide.enabled
            self.guide.status_flash = "Route guide enabled" if self.guide.enabled else "Route guide disabled"
            self.guide.status_flash_timer = 120
            self.guide._invalidate_text_cache()
        self.mouse_down_inside = False

    def render(self, screen):
        enabled = self.guide.enabled

        if enabled:
            fill = (40, 130, 65)
            border = (185, 255, 195)
            text = "GUIDE ON"
        else:
            fill = (90, 55, 55)
            border = (255, 185, 185)
            text = "GUIDE OFF"

        if self.hovered:
            fill = tuple(min(255, c + 18) for c in fill)

        p.draw.rect(screen, fill, self.rect, border_radius=10)
        p.draw.rect(screen, border, self.rect, width=2, border_radius=10)

        shadow = self.shadow_font.render(text, True, (0, 0, 0))
        label = self.font.render(text, True, (255, 255, 255))

        shadow_rect = shadow.get_rect(center=(self.rect.centerx + 1, self.rect.centery + 1))
        label_rect = label.get_rect(center=self.rect.center)

        screen.blit(shadow, shadow_rect)
        screen.blit(label, label_rect)


class MindBenderRouteGuide:
    def __init__(self, menu, json_path: Path):
        self.menu = menu
        self.game = menu.game
        self.json_path = Path(json_path)

        self.enabled = True
        self.loaded_ok = False
        self.load_error = ""

        self.steps_by_puzzle_id: dict[str, GuideRouteStep] = {}
        self.ordered_steps: list[GuideRouteStep] = []
        self.step_index_by_puzzle_id: dict[str, int] = {}

        self.current_step: GuideRouteStep | None = None
        self.current_move_index = 0
        self.current_puzzle_id = ""
        self.last_result = ""
        self.status_flash = ""
        self.status_flash_timer = 0

        self.lookahead_actions = 5

        self.font = p.font.SysFont("arial", 18, bold=True)
        self.small_font = p.font.SysFont("arial", 16, bold=True)
        self.tiny_font = p.font.SysFont("arial", 14, bold=True)

        self.toggle_button = MindBenderRouteGuideToggle(
            menu=self.menu,
            guide=self,
            rect=p.Rect(10, 528, 150, 34),
        )

        # geometry tuning
        self.current_line_width = 12
        self.future_line_width = 10
        self.completed_line_width = 8
        self.line_outline_extra = 4
        self.arrow_outline = 3
        self.head_length_scale = 1.70
        self.head_half_width_scale = 1.05

        # caches
        self._dim_mask_surface: p.Surface | None = None
        self._dim_mask_size: tuple[int, int] | None = None

        self._state_cache_key = None
        self._cached_actions: list[GuideQueuedAction] = []
        self._cached_completed_layer: p.Surface | None = None
        self._cached_body_layer: p.Surface | None = None
        self._cached_static_circle_layer: p.Surface | None = None

        self._text_cache_key = None
        self._text_cache_surface: p.Surface | None = None
        self._text_cache_pos: tuple[int, int] = (0, 0)

        self._load_json()

    # -------------------------------------------------------------------------
    # cache invalidation
    # -------------------------------------------------------------------------

    def _invalidate_visual_cache(self):
        self._state_cache_key = None
        self._cached_actions = []
        self._cached_completed_layer = None
        self._cached_body_layer = None
        self._cached_static_circle_layer = None

    def _invalidate_text_cache(self):
        self._text_cache_key = None
        self._text_cache_surface = None

    def _invalidate_all_cache(self):
        self._invalidate_visual_cache()
        self._invalidate_text_cache()

    # -------------------------------------------------------------------------
    # route state
    # -------------------------------------------------------------------------

    def reset_for_puzzle(self, world_number: int, puzzle_number: int):
        self.current_puzzle_id = f"{world_number}-{puzzle_number}"
        self.current_step = self.steps_by_puzzle_id.get(self.current_puzzle_id)
        self.current_move_index = 0
        self.last_result = ""

        if self.current_step is None:
            self.status_flash = f"No route step for {self.current_puzzle_id}"
            self.status_flash_timer = 90
        else:
            self.status_flash = f"Reset route for {self.current_puzzle_id}"
            self.status_flash_timer = 90

        self._invalidate_all_cache()

    def _load_json(self):
        self.steps_by_puzzle_id.clear()
        self.ordered_steps.clear()
        self.step_index_by_puzzle_id.clear()
        self.loaded_ok = False
        self.load_error = ""

        if not self.json_path.exists():
            self.load_error = f"Missing route file: {self.json_path.name}"
            return

        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            steps = data.get("steps", [])

            for raw_step in steps:
                puzzle_id = str(raw_step.get("puzzle_id", "")).strip()
                if not puzzle_id:
                    continue

                overlay_moves: list[GuideOverlayMove] = []
                for raw_move in raw_step.get("overlay_moves", []):
                    overlay_moves.append(
                        GuideOverlayMove(
                            move=str(raw_move.get("move", "")),
                            click_down=(float(raw_move["click_down"][0]), float(raw_move["click_down"][1])),
                            lock_point=(float(raw_move["lock_point"][0]), float(raw_move["lock_point"][1])),
                            release=(float(raw_move["release"][0]), float(raw_move["release"][1])),
                            path_points=[
                                (float(point[0]), float(point[1]))
                                for point in raw_move.get("path_points", [])
                            ],
                            selected_line=int(raw_move.get("selected_line", 0)),
                            displacement=int(raw_move.get("displacement", 0)),
                            free_drag=bool(raw_move.get("free_drag", False)),
                            drag_distance=float(raw_move.get("drag_distance", 0.0)),
                            move_distance=float(raw_move.get("move_distance", 0.0)),
                            total_distance_to_here=float(raw_move.get("total_distance_to_here", 0.0)),
                        )
                    )

                start_mouse_position = None
                if raw_step.get("start_mouse_position") is not None:
                    start_mouse_position = (
                        float(raw_step["start_mouse_position"][0]),
                        float(raw_step["start_mouse_position"][1]),
                    )

                initial_fat_position = None
                if raw_step.get("initial_fat_position") is not None:
                    initial_fat_position = (
                        float(raw_step["initial_fat_position"][0]),
                        float(raw_step["initial_fat_position"][1]),
                    )

                final_mouse_target = None
                if raw_step.get("final_mouse_target") is not None:
                    final_mouse_target = (
                        float(raw_step["final_mouse_target"][0]),
                        float(raw_step["final_mouse_target"][1]),
                    )

                post_end_click_target = None
                if raw_step.get("post_end_click_target") is not None:
                    post_end_click_target = (
                        float(raw_step["post_end_click_target"][0]),
                        float(raw_step["post_end_click_target"][1]),
                    )

                step = GuideRouteStep(
                    puzzle_id=puzzle_id,
                    world_number=int(raw_step.get("world_number", 0)),
                    puzzle_number=int(raw_step.get("puzzle_number", 0)),
                    move_string=str(raw_step.get("move_string", "")),
                    total_cost=float(raw_step.get("total_cost", 0.0)),
                    total_move=float(raw_step.get("total_move", 0.0)),
                    total_drag=float(raw_step.get("total_drag", 0.0)),
                    start_mouse_position=start_mouse_position,
                    end_positions=[
                        (float(point[0]), float(point[1]))
                        for point in raw_step.get("end_positions", [])
                    ],
                    end_next_puzzle=bool(raw_step.get("end_next_puzzle", False)),
                    post_end_click_target=post_end_click_target,
                    has_fat=bool(raw_step.get("has_fat", False)),
                    initial_fat_position=initial_fat_position,
                    final_mouse_target=final_mouse_target,
                    final_move_distance=float(raw_step.get("final_move_distance", 0.0)),
                    overlay_moves=overlay_moves,
                )

                self.steps_by_puzzle_id[puzzle_id] = step
                self.ordered_steps.append(step)

            self.ordered_steps.sort(key=lambda s: (s.world_number, s.puzzle_number))
            for index, step in enumerate(self.ordered_steps):
                self.step_index_by_puzzle_id[step.puzzle_id] = index

            self.loaded_ok = True
            self._invalidate_all_cache()
        except Exception as exc:
            self.load_error = f"Route load failed: {exc}"

    def on_puzzle_changed(self, world_number: int, puzzle_number: int):
        puzzle_id = f"{world_number}-{puzzle_number}"

        if puzzle_id != self.current_puzzle_id:
            self.current_puzzle_id = puzzle_id
            self.current_step = self.steps_by_puzzle_id.get(puzzle_id)
            self.current_move_index = 0
            self.last_result = ""

            if self.current_step is None:
                self.status_flash = f"No route step for {puzzle_id}"
                self.status_flash_timer = 90
            else:
                self.status_flash = f"Loaded route for {puzzle_id}"
                self.status_flash_timer = 90

            self._invalidate_all_cache()

    def has_route_for_current_puzzle(self) -> bool:
        return self.current_step is not None

    def is_complete(self) -> bool:
        if self.current_step is None:
            return False
        return self.current_move_index >= len(self.current_step.overlay_moves)

    def _moves_match(self, actual: dict, expected: GuideOverlayMove) -> bool:
        if actual["axis"] != expected.axis:
            return False
        if int(actual["selected_line"]) != int(expected.selected_line):
            return False
        if int(actual["displacement"]) != int(expected.displacement):
            return False
        return True

    def on_board_move_committed(self, actual_move: dict):
        if not self.enabled:
            return
        if self.current_step is None:
            return
        if self.is_complete():
            return

        expected = self.current_step.overlay_moves[self.current_move_index]

        if self._moves_match(actual_move, expected):
            self.current_move_index += 1
            self.last_result = "good"
            self.status_flash = f"Correct: {expected.move}"
            self.status_flash_timer = 60
        else:
            self.last_result = "bad"
            self.status_flash = (
                f"Expected {expected.move} | "
                f"got {actual_move['axis']}{actual_move['selected_line']} d={actual_move['displacement']}"
            )
            self.status_flash_timer = 90

        self._invalidate_all_cache()

    # -------------------------------------------------------------------------
    # coordinate helpers
    # -------------------------------------------------------------------------

    def _board_point_to_screen(self, point: tuple[float, float]) -> tuple[int, int]:
        board = self.menu.board
        cell = board.chuzSize
        x = int(board.boardOffset.x + point[0] * cell.x + cell.x / 2)
        y = int(board.boardOffset.y + point[1] * cell.y + cell.y / 2)
        return x, y

    def _approx_equal_point(self, a: tuple[float, float] | None, b: tuple[float, float]) -> bool:
        if a is None:
            return False
        return abs(a[0] - b[0]) < 0.025 and abs(a[1] - b[1]) < 0.025

    def _is_prompt_symbolic_target(self, point: tuple[float, float] | None) -> bool:
        if point is None:
            return False
        for prompt_point in PROMPT_CLICK_TARGETS:
            if self._approx_equal_point(point, prompt_point):
                return True
        return False

    def _get_prompt_target_rect(self) -> p.Rect | None:
        if HARDCODED_PROMPT_GUIDE_RECT is not None:
            return HARDCODED_PROMPT_GUIDE_RECT.copy()

        popup = self.menu.popup
        if popup is None:
            return None

        # New popup API: use the active button's rect.
        if hasattr(popup, "_active_prompt_button"):
            btn = popup._active_prompt_button()
            if btn is not None and hasattr(btn, "rect"):
                rect = btn.rect
                if rect.width > 1 and rect.height > 1:
                    return rect.copy()

        # Fallback for older popup API.
        if hasattr(popup, "prompt_rect"):
            rect = popup.prompt_rect
            if rect.width > 1 and rect.height > 1:
                return rect.copy()

        return None

    def _resolve_post_target_rect_for_step(self, step: GuideRouteStep) -> p.Rect | None:
        target = step.post_end_click_target
        if target is None:
            return None

        if self._is_prompt_symbolic_target(target):
            return self._get_prompt_target_rect()

        for point, mini_index in MINI_PANEL_TARGETS.items():
            if self._approx_equal_point(target, point):
                if 0 <= mini_index < len(self.menu.mini_panel.buttons):
                    return self.menu.mini_panel.buttons[mini_index].rect.inflate(18, 18)

        if self._approx_equal_point(target, NEXT_WORLD_STAR_TARGET):
            if self.menu.right_world_star_btn is not None:
                return self.menu.right_world_star_btn.rect.inflate(20, 20)

        return None

    def _is_prompt_rect_action(self, action: GuideQueuedAction) -> bool:
        if action.target_rect is None:
            return False

        if action.kind == "mouse_travel" and self._is_prompt_symbolic_target(action.step_ref.final_mouse_target):
            return True

        if action.kind == "target_click" and self._is_prompt_symbolic_target(action.step_ref.post_end_click_target):
            return True

        return False

    def _screen_point_from_board_or_rect(
        self,
        board_point: tuple[float, float] | None,
        target_rect: p.Rect | None,
        *,
        rect_anchor: str = "center",
    ) -> tuple[int, int] | None:
        if board_point is not None:
            return self._board_point_to_screen(board_point)

        if target_rect is not None:
            if rect_anchor == "left_center":
                return target_rect.left, target_rect.centery
            if rect_anchor == "right_center":
                return target_rect.right, target_rect.centery
            if rect_anchor == "top_center":
                return target_rect.centerx, target_rect.top
            if rect_anchor == "bottom_center":
                return target_rect.centerx, target_rect.bottom
            return target_rect.center

        return None

    def _get_action_anchor_point(
            self,
            action: GuideQueuedAction,
            actions: list[GuideQueuedAction],
            action_index: int,
    ) -> tuple[int, int] | None:
        if action.kind == "move" and action.move_ref is not None:
            return self._board_point_to_screen(action.move_ref.path_points[0])

        if action.kind == "mouse_travel":
            if action.from_point is not None:
                return self._board_point_to_screen(action.from_point)
            return None

        if action.kind == "target_click":
            if action.from_point is not None:
                return self._board_point_to_screen(action.from_point)
            return None

        return None

    def _get_next_non_prompt_anchor_x(
            self,
            actions: list[GuideQueuedAction],
            start_index: int,
    ) -> int | None:
        for i in range(start_index + 1, len(actions)):
            candidate = actions[i]

            if self._is_prompt_rect_action(candidate):
                continue

            pt = None
            if candidate.kind == "move" and candidate.move_ref is not None:
                pt = self._board_point_to_screen(candidate.move_ref.path_points[0])
            elif candidate.kind in ("mouse_travel", "target_click"):
                if candidate.to_point is not None:
                    pt = self._board_point_to_screen(candidate.to_point)
                elif candidate.target_rect is not None:
                    pt = candidate.target_rect.center

            if pt is not None:
                return pt[0]

        return None

    def _get_prompt_display_rect(
            self,
            action: GuideQueuedAction,
            actions: list[GuideQueuedAction],
            action_index: int,
    ) -> p.Rect:
        base = action.target_rect.copy()

        prev_pt = self._get_action_anchor_point(action, actions, action_index)
        next_x = self._get_next_non_prompt_anchor_x(actions, action_index)

        if prev_pt is None or next_x is None:
            return base

        prev_x = prev_pt[0]
        center_x = int(round((prev_x + next_x) * 0.5))
        
        max_width = min(PROMPT_GUIDE_MAX_WIDTH, base.width)
        desired_width = min(max_width, abs(next_x - prev_x))
        desired_width = max(PROMPT_GUIDE_MIN_WIDTH, desired_width)

        x = center_x - desired_width // 2
        x = max(base.left, min(x, base.right - desired_width))

        return p.Rect(x, base.y, desired_width, base.height)

    def _get_rect_edge_point_toward_source(
            self,
            source_pt: tuple[int, int],
            rect: p.Rect,
    ) -> tuple[int, int]:
        cx = rect.centerx
        cy = rect.centery
        dx = source_pt[0] - cx
        dy = source_pt[1] - cy

        hw = rect.width / 2.0
        hh = rect.height / 2.0

        if abs(dx) < EPSILON and abs(dy) < EPSILON:
            return rect.center

        if abs(dx) < EPSILON:
            return cx, rect.bottom if dy > 0 else rect.top

        if abs(dy) < EPSILON:
            return rect.right if dx > 0 else rect.left, cy

        scale = max(abs(dx) / hw, abs(dy) / hh)
        edge_x = cx + dx / scale
        edge_y = cy + dy / scale
        return int(round(edge_x)), int(round(edge_y))

    # -------------------------------------------------------------------------
    # action building
    # -------------------------------------------------------------------------

    def _build_step_actions(
            self,
            step: GuideRouteStep,
            color_index: int,
            start_move_index: int,
            is_current_puzzle: bool,
    ) -> list[GuideQueuedAction]:
        actions: list[GuideQueuedAction] = []
        combo_number = start_move_index + 1

        for move_index in range(start_move_index, len(step.overlay_moves)):
            move = step.overlay_moves[move_index]
            actions.append(
                GuideQueuedAction(
                    kind="move",
                    puzzle_id=step.puzzle_id,
                    step_ref=step,
                    move_ref=move,
                    from_point=None,
                    to_point=None,
                    target_rect=None,
                    combo_number=combo_number,
                    color_index=color_index,
                    is_current_puzzle=is_current_puzzle,
                    is_current_action=(is_current_puzzle and move_index == self.current_move_index),
                )
            )
            combo_number += 1

        prompt_rect_from_final_target = None
        if self._is_prompt_symbolic_target(step.final_mouse_target):
            prompt_rect_from_final_target = self._get_prompt_target_rect()

        if step.final_mouse_target is not None:
            from_point = step.overlay_moves[-1].release if step.overlay_moves else None
            actions.append(
                GuideQueuedAction(
                    kind="mouse_travel",
                    puzzle_id=step.puzzle_id,
                    step_ref=step,
                    move_ref=None,
                    from_point=from_point,
                    to_point=step.final_mouse_target if prompt_rect_from_final_target is None else None,
                    target_rect=prompt_rect_from_final_target,
                    combo_number=combo_number,
                    color_index=color_index,
                    is_current_puzzle=is_current_puzzle,
                    is_current_action=False,
                )
            )
            combo_number += 1

        target_rect = self._resolve_post_target_rect_for_step(step)
        if target_rect is not None:
            actions.append(
                GuideQueuedAction(
                    kind="target_click",
                    puzzle_id=step.puzzle_id,
                    step_ref=step,
                    move_ref=None,
                    from_point=step.final_mouse_target,
                    to_point=None,
                    target_rect=target_rect,
                    combo_number=combo_number,
                    color_index=color_index,
                    is_current_puzzle=is_current_puzzle,
                    is_current_action=False,
                )
            )

        return actions

    def _collect_lookahead_actions(self) -> list[GuideQueuedAction]:
        if self.current_step is None:
            return []

        current_route_index = self.step_index_by_puzzle_id.get(self.current_step.puzzle_id)
        if current_route_index is None:
            return []

        actions: list[GuideQueuedAction] = []
        target_count = self.lookahead_actions + 1

        for route_index in range(current_route_index, len(self.ordered_steps)):
            step = self.ordered_steps[route_index]
            is_current_puzzle = step.puzzle_id == self.current_step.puzzle_id
            start_move_index = self.current_move_index if is_current_puzzle else 0

            step_actions = self._build_step_actions(
                step=step,
                color_index=route_index,
                start_move_index=start_move_index,
                is_current_puzzle=is_current_puzzle,
            )

            for action in step_actions:
                actions.append(action)
                if len(actions) >= target_count:
                    return actions

        return actions

    def _get_visual_state_key(self):
        board = self.menu.board
        popup = self.menu.popup

        prompt_rect_key = None
        if popup is not None:
            # New popup API: use the active button rect.
            if hasattr(popup, "_active_prompt_button"):
                btn = popup._active_prompt_button()
                if btn is not None and hasattr(btn, "rect"):
                    rect = btn.rect
                    prompt_rect_key = (rect.x, rect.y, rect.width, rect.height)

            # Fallback for older popup API.
            elif hasattr(popup, "prompt_rect"):
                rect = popup.prompt_rect
                prompt_rect_key = (rect.x, rect.y, rect.width, rect.height)

        return (
            self.enabled,
            self.current_puzzle_id,
            self.current_move_index,
            board.boardOffset.x,
            board.boardOffset.y,
            board.chuzSize.x,
            board.chuzSize.y,
            self.lookahead_actions,
            self.current_line_width,
            self.future_line_width,
            self.completed_line_width,
            self.line_outline_extra,
            self.arrow_outline,
            self.head_length_scale,
            self.head_half_width_scale,
            tuple(self.game.screenSize.tuple()),
            prompt_rect_key,
        )

    def _ensure_visual_cache(self):
        key = self._get_visual_state_key()
        if key == self._state_cache_key:
            return

        self._state_cache_key = key
        self._cached_actions = self._collect_lookahead_actions()
        visible_actions = self._cached_actions[:self.lookahead_actions]

        self._cached_completed_layer = self._build_completed_layer()
        self._cached_body_layer = self._build_body_layer(visible_actions)
        self._cached_static_circle_layer = self._build_static_circle_layer(visible_actions)

    # -------------------------------------------------------------------------
    # drawing helpers
    # -------------------------------------------------------------------------

    def _group_color(self, color_index: int) -> tuple[int, int, int]:
        return GROUP_COLORS[color_index % len(GROUP_COLORS)]

    def _get_dim_mask(self) -> p.Surface:
        size = tuple(self.game.screenSize.tuple())
        if self._dim_mask_surface is not None and self._dim_mask_size == size:
            return self._dim_mask_surface

        surf = p.Surface(size, p.SRCALPHA)
        surf.fill((0, 0, 0, 96))
        self._dim_mask_surface = surf
        self._dim_mask_size = size
        return surf

    def _simplify_screen_points(self, points: list[tuple[int, int]]) -> list[tuple[float, float]]:
        if not points:
            return []

        out: list[tuple[float, float]] = [(float(points[0][0]), float(points[0][1]))]
        for pt in points[1:]:
            fpt = (float(pt[0]), float(pt[1]))
            if fpt != out[-1]:
                out.append(fpt)
        return out

    def _segment_info(self, a: tuple[float, float], b: tuple[float, float]):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        length = math.hypot(dx, dy)
        if length <= EPSILON:
            return 0.0, 0.0, 0.0
        return dx / length, dy / length, length

    def _offset_point(self, pt: tuple[float, float], ux: float, uy: float, amount: float) -> tuple[float, float]:
        return pt[0] + ux * amount, pt[1] + uy * amount

    def _build_arrow_geometry(
        self,
        screen_points: list[tuple[float, float]],
        stop_before_target: float,
        head_len: float,
    ) -> tuple[list[tuple[float, float]], tuple[float, float] | None, tuple[float, float] | None]:
        if len(screen_points) < 2:
            return [], None, None

        pts = list(screen_points)
        last_a = pts[-2]
        last_b = pts[-1]
        ux, uy, seg_len = self._segment_info(last_a, last_b)
        if seg_len <= EPSILON:
            return pts, None, None

        body_backoff = min(seg_len * 0.60, stop_before_target + head_len * 0.90)

        tip = self._offset_point(last_b, -ux, -uy, stop_before_target)
        body_end = self._offset_point(last_b, -ux, -uy, body_backoff)

        pts[-1] = body_end
        return pts, body_end, tip

    def _draw_segment_quad(
        self,
        surface: p.Surface,
        a: tuple[float, float],
        b: tuple[float, float],
        width: float,
        color_rgba: tuple[int, int, int, int],
    ):
        ux, uy, seg_len = self._segment_info(a, b)
        if seg_len <= EPSILON:
            return

        half = width * 0.5
        px = -uy * half
        py = ux * half

        poly = [
            (int(round(a[0] + px)), int(round(a[1] + py))),
            (int(round(b[0] + px)), int(round(b[1] + py))),
            (int(round(b[0] - px)), int(round(b[1] - py))),
            (int(round(a[0] - px)), int(round(a[1] - py))),
        ]
        p.draw.polygon(surface, color_rgba, poly)

    def _draw_path_body(
        self,
        surface: p.Surface,
        points: list[tuple[float, float]],
        color_rgba: tuple[int, int, int, int],
        width: float,
        round_last: bool,
    ):
        if len(points) < 2:
            return

        for i in range(len(points) - 1):
            self._draw_segment_quad(surface, points[i], points[i + 1], width, color_rgba)

        radius = max(1, int(math.ceil(width / 2.0)))
        for idx, pt in enumerate(points):
            if idx == len(points) - 1 and not round_last:
                continue
            p.draw.circle(surface, color_rgba, (int(round(pt[0])), int(round(pt[1]))), radius)

    def _draw_glow_path(
        self,
        surface: p.Surface,
        points: list[tuple[float, float]],
        color: tuple[int, int, int],
        width: float,
        alpha: int,
        round_last: bool,
    ):
        glow_alpha = max(16, alpha // 5)
        for extra in (14, 8, 4):
            self._draw_path_body(surface, points, (*color, glow_alpha), width + extra, round_last)

    def _draw_arrow_head(
        self,
        surface: p.Surface,
        start_pt: tuple[float, float],
        end_pt: tuple[float, float],
        color: tuple[int, int, int],
        alpha: int,
        head_len: float,
        head_w: float,
        outline: float,
    ):
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        vec_len = math.hypot(dx, dy)
        if vec_len <= EPSILON:
            return

        ux = dx / vec_len
        uy = dy / vec_len
        px = -uy
        py = ux

        base_cx = end_pt[0] - ux * head_len
        base_cy = end_pt[1] - uy * head_len

        inner_tip = (int(round(end_pt[0])), int(round(end_pt[1])))
        inner_left = (
            int(round(base_cx + px * head_w)),
            int(round(base_cy + py * head_w)),
        )
        inner_right = (
            int(round(base_cx - px * head_w)),
            int(round(base_cy - py * head_w)),
        )

        tip_extend = outline
        side_expand = outline
        base_inset = max(1.0, outline * 0.35)

        outer_tip = (
            int(round(end_pt[0] + ux * tip_extend)),
            int(round(end_pt[1] + uy * tip_extend)),
        )
        outer_left = (
            int(round(base_cx + px * (head_w + side_expand) - ux * base_inset)),
            int(round(base_cy + py * (head_w + side_expand) - uy * base_inset)),
        )
        outer_right = (
            int(round(base_cx - px * (head_w + side_expand) - ux * base_inset)),
            int(round(base_cy - py * (head_w + side_expand) - uy * base_inset)),
        )

        p.draw.polygon(surface, (0, 0, 0, alpha), [outer_tip, outer_left, outer_right])
        p.draw.polygon(surface, (*color, alpha), [inner_tip, inner_left, inner_right])

    def _draw_hit_circle_on_surface(
        self,
        surface: p.Surface,
        center: tuple[int, int],
        color: tuple[int, int, int],
        combo_number: int,
        alpha: int,
        current: bool,
        base_radius: int,
    ):
        ring_width = 4
        p.draw.circle(surface, (*color, alpha), center, base_radius, width=ring_width)

        white_radius = max(9, base_radius - ring_width - 1)
        p.draw.circle(surface, (255, 255, 255, min(255, alpha)), center, white_radius)

        if current:
            t = (p.time.get_ticks() % 900) / 900.0
            approach_radius = int(base_radius + 28 * (1.0 - t))
            approach_radius = max(base_radius + 2, approach_radius)
            p.draw.circle(surface, (*color, 140), center, approach_radius, width=2)

        number = self.small_font.render(str(combo_number), True, (25, 25, 30))
        number_rect = number.get_rect(center=center)
        surface.blit(number, number_rect)

    def _draw_prompt_target_on_surface(
            self,
            surface: p.Surface,
            target_rect: p.Rect,
            color: tuple[int, int, int],
            alpha: int,
            animated: bool,
            combo_number: int | None = None,
    ):
        rect = target_rect.copy()

        p.draw.rect(surface, (255, 255, 255, min(255, alpha)), rect, border_radius=8)
        p.draw.rect(surface, (*color, alpha), rect, width=4, border_radius=8)

        if combo_number is not None:
            number = self.small_font.render(str(combo_number), True, (0, 0, 0))
            number_rect = number.get_rect(center=rect.center)
            surface.blit(number, number_rect)

    def _draw_target_highlight_on_surface(
        self,
        surface: p.Surface,
        target_rect: p.Rect,
        color: tuple[int, int, int],
        alpha: int,
        animated: bool,
    ):
        if animated:
            pulse = 0.5 + 0.5 * math.sin(p.time.get_ticks() * 0.010)
            expand = int(10 + 6 * pulse)
        else:
            expand = 12

        rect = target_rect.inflate(expand * 2, expand * 2)
        p.draw.rect(surface, (*color, max(60, alpha // 2)), rect, width=4, border_radius=12)
        p.draw.rect(surface, (*color, max(20, alpha // 4)), rect.inflate(14, 14), width=6, border_radius=16)

    # -------------------------------------------------------------------------
    # cached layer builders
    # -------------------------------------------------------------------------

    def _build_completed_layer(self) -> p.Surface:
        layer = p.Surface(self.game.screenSize.tuple(), p.SRCALPHA)

        if self.current_step is None:
            return layer

        completed_color = (110, 255, 150)

        for i in range(min(self.current_move_index, len(self.current_step.overlay_moves))):
            move = self.current_step.overlay_moves[i]
            raw_points = [self._board_point_to_screen(point) for point in move.path_points]
            screen_points = self._simplify_screen_points(raw_points)
            if len(screen_points) < 2:
                continue

            last_a = screen_points[-2]
            last_b = screen_points[-1]
            _ux, _uy, seg_len = self._segment_info(last_a, last_b)
            if seg_len <= EPSILON:
                continue

            width = float(self.completed_line_width)
            radius = 12.0
            stop_before_target = min(seg_len * 0.28, radius * 0.70)
            head_len = max(width * self.head_length_scale, 14.0)
            head_len = min(head_len, max(12.0, seg_len * 0.42))
            head_w = max(width * self.head_half_width_scale, 8.0)

            body_points, body_end, tip = self._build_arrow_geometry(
                screen_points=screen_points,
                stop_before_target=stop_before_target,
                head_len=head_len,
            )
            if body_end is None or tip is None:
                continue

            self._draw_glow_path(layer, body_points, completed_color, width, 95, round_last=False)
            self._draw_path_body(layer, body_points, (0, 0, 0, 95), width + self.line_outline_extra, round_last=False)
            self._draw_path_body(layer, body_points, (*completed_color, 95), width, round_last=False)
            self._draw_arrow_head(
                layer,
                body_end,
                tip,
                completed_color,
                95,
                head_len=head_len,
                head_w=head_w,
                outline=self.arrow_outline,
            )

        return layer

    def _build_body_layer(self, actions: list[GuideQueuedAction]) -> p.Surface:
        layer = p.Surface(self.game.screenSize.tuple(), p.SRCALPHA)
        all_actions = self._cached_actions if self._cached_actions else actions

        for queue_index in range(len(actions) - 1, -1, -1):
            action = actions[queue_index]
            color = self._group_color(action.color_index)

            if action.kind == "move":
                move = action.move_ref
                if move is None or len(move.path_points) < 2:
                    continue

                raw_points = [self._board_point_to_screen(point) for point in move.path_points]
                screen_points = self._simplify_screen_points(raw_points)
                if len(screen_points) < 2:
                    continue

                if action.is_current_action:
                    width = float(self.current_line_width)
                    alpha = 255
                    radius = 18.0
                else:
                    width = max(8.0, float(self.future_line_width) - queue_index * 0.75)
                    alpha = max(85, 205 - queue_index * 24)
                    radius = max(15.0, 19.0 - queue_index)

                last_a = screen_points[-2]
                last_b = screen_points[-1]
                _ux, _uy, seg_len = self._segment_info(last_a, last_b)
                if seg_len <= EPSILON:
                    continue

                stop_before_target = min(seg_len * 0.30, radius * 0.70)
                head_len = max(width * self.head_length_scale, 18.0)
                head_len = min(head_len, max(12.0, seg_len * 0.42))
                head_w = max(width * self.head_half_width_scale, 10.0)

                body_points, body_end, tip = self._build_arrow_geometry(
                    screen_points=screen_points,
                    stop_before_target=stop_before_target,
                    head_len=head_len,
                )
                if body_end is None or tip is None:
                    continue

                self._draw_glow_path(layer, body_points, color, width, alpha, round_last=False)
                self._draw_path_body(
                    layer,
                    body_points,
                    (0, 0, 0, min(160, alpha)),
                    width + self.line_outline_extra,
                    round_last=False,
                )
                self._draw_path_body(layer, body_points, (*color, alpha), width, round_last=False)
                self._draw_arrow_head(
                    layer,
                    body_end,
                    tip,
                    color,
                    alpha,
                    head_len=head_len,
                    head_w=head_w,
                    outline=self.arrow_outline,
                )
                continue

            start_pt = self._screen_point_from_board_or_rect(action.from_point, None)
            if start_pt is None:
                continue

            alpha = 255 if queue_index == 0 else max(90, 205 - queue_index * 24)

            if self._is_prompt_rect_action(action) and action.target_rect is not None:
                display_rect = self._get_prompt_display_rect(action, all_actions, queue_index)
                end_pt = self._get_rect_edge_point_toward_source(start_pt, display_rect)

                p.draw.line(layer, (0, 0, 0, min(100, alpha)), start_pt, end_pt, 5)
                p.draw.line(layer, (*color, max(60, alpha - 100)), start_pt, end_pt, 2)
                continue

            end_pt = self._screen_point_from_board_or_rect(action.to_point, action.target_rect)
            if end_pt is None:
                continue

            p.draw.line(layer, (0, 0, 0, min(100, alpha)), start_pt, end_pt, 5)
            p.draw.line(layer, (*color, max(60, alpha - 100)), start_pt, end_pt, 2)

        return layer

    def _build_static_circle_layer(self, actions: list[GuideQueuedAction]) -> p.Surface:
        layer = p.Surface(self.game.screenSize.tuple(), p.SRCALPHA)
        all_actions = self._cached_actions if self._cached_actions else actions

        for queue_index in range(len(actions) - 1, -1, -1):
            action = actions[queue_index]
            if action.is_current_action:
                continue

            color = self._group_color(action.color_index)

            if action.kind == "move":
                move = action.move_ref
                if move is None:
                    continue

                center = self._board_point_to_screen(move.path_points[0])
                alpha = max(85, 205 - queue_index * 24)
                radius = max(16, 20 - queue_index)

                self._draw_hit_circle_on_surface(
                    layer,
                    center=center,
                    color=color,
                    combo_number=action.combo_number,
                    alpha=alpha,
                    current=False,
                    base_radius=radius,
                )
                continue

            alpha = max(90, 205 - queue_index * 24)

            if self._is_prompt_rect_action(action) and action.target_rect is not None:
                display_rect = self._get_prompt_display_rect(action, all_actions, queue_index)
                self._draw_prompt_target_on_surface(
                    layer,
                    target_rect=display_rect,
                    color=color,
                    alpha=alpha,
                    animated=False,
                    combo_number=action.combo_number,
                )
                continue

            end_pt = self._screen_point_from_board_or_rect(action.to_point, action.target_rect)
            if end_pt is None:
                continue

            radius = max(16, 20 - queue_index)

            self._draw_hit_circle_on_surface(
                layer,
                center=end_pt,
                color=color,
                combo_number=action.combo_number,
                alpha=alpha,
                current=False,
                base_radius=radius,
            )

            if action.target_rect is not None:
                self._draw_target_highlight_on_surface(
                    layer,
                    target_rect=action.target_rect,
                    color=color,
                    alpha=alpha,
                    animated=False,
                )

        return layer

    # -------------------------------------------------------------------------
    # text cache
    # -------------------------------------------------------------------------

    def _build_top_line_text(self, actions: list[GuideQueuedAction]) -> str:
        if not self.loaded_ok:
            return self.load_error or "Route file not loaded"

        if not self.enabled:
            return "Route guide disabled"

        if self.current_step is None:
            return f"No route for {self.current_puzzle_id}"

        chunks: list[str] = [self.current_step.puzzle_id]

        for action in actions[:self.lookahead_actions]:
            if action.kind == "move" and action.move_ref is not None:
                chunks.append(f"{action.puzzle_id}:{action.combo_number}:{action.move_ref.move}")
            elif action.kind == "mouse_travel":
                chunks.append(f"{action.puzzle_id}:{action.combo_number}:END")
            elif action.kind == "target_click":
                chunks.append(f"{action.puzzle_id}:{action.combo_number}:CLICK")

        if self.status_flash_timer > 0 and self.status_flash:
            chunks.append(self.status_flash)

        return "   ".join(chunks)

    def _get_text_color(self) -> tuple[int, int, int]:
        if self.last_result == "bad" and self.status_flash_timer > 0:
            return 255, 145, 145
        if self.last_result == "good" and self.status_flash_timer > 0:
            return 145, 255, 165
        return 240, 240, 245

    def _ensure_text_cache(self, actions: list[GuideQueuedAction]):
        text = self._build_top_line_text(actions)
        color = self._get_text_color()
        key = (text, color, tuple(self.game.screenSize.tuple()))

        if key == self._text_cache_key and self._text_cache_surface is not None:
            return

        label = self.small_font.render(text, True, color)
        shadow = self.small_font.render(text, True, (0, 0, 0))

        pad_x = 10
        pad_y = 5
        surf = p.Surface((label.get_width() + pad_x * 2, label.get_height() + pad_y * 2), p.SRCALPHA)
        surf.fill((8, 10, 14, 155))
        surf.blit(shadow, (pad_x + 1, pad_y + 1))
        surf.blit(label, (pad_x, pad_y))

        x = self.game.screenSize.x - surf.get_width()
        y = 0

        self._text_cache_key = key
        self._text_cache_surface = surf
        self._text_cache_pos = (x, y)

    # -------------------------------------------------------------------------
    # dynamic current action drawing
    # -------------------------------------------------------------------------

    def _draw_current_action(self, screen: p.Surface, actions: list[GuideQueuedAction]):
        if not actions:
            return

        current_action = actions[0]
        color = self._group_color(current_action.color_index)
        all_actions = self._cached_actions if self._cached_actions else actions

        if current_action.kind == "move":
            move = current_action.move_ref
            if move is None:
                return

            center = self._board_point_to_screen(move.path_points[0])
            self._draw_hit_circle_on_surface(
                screen,
                center=center,
                color=color,
                combo_number=current_action.combo_number,
                alpha=255,
                current=True,
                base_radius=22,
            )
            return

        if self._is_prompt_rect_action(current_action) and current_action.target_rect is not None:
            display_rect = self._get_prompt_display_rect(current_action, all_actions, 0)
            self._draw_prompt_target_on_surface(
                screen,
                target_rect=display_rect,
                color=color,
                alpha=255,
                animated=False,
                combo_number=current_action.combo_number,
            )
            return

        end_pt = self._screen_point_from_board_or_rect(current_action.to_point, current_action.target_rect)
        if end_pt is None:
            return

        self._draw_hit_circle_on_surface(
            screen,
            center=end_pt,
            color=color,
            combo_number=current_action.combo_number,
            alpha=255,
            current=True,
            base_radius=22,
        )

        if current_action.target_rect is not None:
            self._draw_target_highlight_on_surface(
                screen,
                target_rect=current_action.target_rect,
                color=color,
                alpha=255,
                animated=True,
            )

    # -------------------------------------------------------------------------
    # public draw
    # -------------------------------------------------------------------------

    def draw(self, screen):
        if self.status_flash_timer > 0:
            self.status_flash_timer -= 1

        self.toggle_button.render(screen)

        if not self.loaded_ok:
            self._ensure_text_cache([])
            if self._text_cache_surface is not None:
                screen.blit(self._text_cache_surface, self._text_cache_pos)
            return

        if not self.enabled:
            self._ensure_text_cache([])
            if self._text_cache_surface is not None:
                screen.blit(self._text_cache_surface, self._text_cache_pos)
            return

        if self.current_step is None:
            self._ensure_text_cache([])
            if self._text_cache_surface is not None:
                screen.blit(self._text_cache_surface, self._text_cache_pos)
            return

        self._ensure_visual_cache()
        visible_actions = self._cached_actions[:self.lookahead_actions]
        self._ensure_text_cache(visible_actions)

        screen.blit(self._get_dim_mask(), (0, 0))

        if self._cached_completed_layer is not None:
            screen.blit(self._cached_completed_layer, (0, 0))

        if self._cached_body_layer is not None:
            screen.blit(self._cached_body_layer, (0, 0))

        if self._cached_static_circle_layer is not None:
            screen.blit(self._cached_static_circle_layer, (0, 0))

        self._draw_current_action(screen, visible_actions)

        if self._text_cache_surface is not None:
            screen.blit(self._text_cache_surface, self._text_cache_pos)