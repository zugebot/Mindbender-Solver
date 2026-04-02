from __future__ import annotations

import math
import os
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Make sibling import work when this file is copied next to chuzzle_mouse_dp.py
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from pyth.old.chuzzle_mouse_dp import DpMouseSolver, GRID_SIZE, ScoredSolution, FileScoreResult

SCALE = 0.75
CELL_SIZE = int(SCALE * 60)
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
GRID_OFFSET_X = 5
GRID_OFFSET_Y = 2
CELL_COUNT = GRID_SIZE

CELLS_WIDTH = 14
CELLS_HEIGHT = 9
SCREEN_WIDTH = CELLS_WIDTH * CELL_SIZE
SCREEN_HEIGHT = CELLS_HEIGHT * CELL_SIZE



def grid_to_canvas(x: float, y: float) -> tuple[float, float]:
    canvas_x = (x + GRID_OFFSET_X) * CELL_SIZE + CELL_SIZE / 2
    canvas_y = (y + GRID_OFFSET_Y) * CELL_SIZE + CELL_SIZE / 2
    return canvas_x, canvas_y



def draw_grid(canvas: tk.Canvas) -> None:
    outer_grid_size_x = GRID_OFFSET_X + CELL_COUNT + (CELLS_WIDTH - CELL_COUNT - GRID_OFFSET_X)
    outer_grid_size_y = GRID_OFFSET_Y + CELL_COUNT + (CELLS_HEIGHT - CELL_COUNT - GRID_OFFSET_Y)
    inner_grid_start_x = GRID_OFFSET_X * CELL_SIZE
    inner_grid_start_y = GRID_OFFSET_Y * CELL_SIZE
    inner_grid_end_x = (GRID_OFFSET_X + CELL_COUNT) * CELL_SIZE
    inner_grid_end_y = (GRID_OFFSET_Y + CELL_COUNT) * CELL_SIZE

    canvas.create_rectangle(
        0,
        0,
        outer_grid_size_x * CELL_SIZE,
        outer_grid_size_y * CELL_SIZE,
        fill="light gray",
        outline="",
    )

    for i in range(outer_grid_size_x + 1):
        canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, outer_grid_size_y * CELL_SIZE, fill="gray", width=1)
    for i in range(outer_grid_size_y + 1):
        canvas.create_line(0, i * CELL_SIZE, outer_grid_size_x * CELL_SIZE, i * CELL_SIZE, fill="gray", width=1)

    canvas.create_rectangle(inner_grid_start_x, inner_grid_start_y, inner_grid_end_x, inner_grid_end_y, fill="white")
    is_gray = False
    for x in range(CELL_COUNT):
        is_gray = not is_gray
        for y in range(CELL_COUNT):
            is_gray = not is_gray
            if is_gray:
                continue
            canvas.create_rectangle(
                inner_grid_start_x + CELL_SIZE * x,
                inner_grid_start_y + CELL_SIZE * y,
                inner_grid_start_x + CELL_SIZE * (x + 1),
                inner_grid_start_y + CELL_SIZE * (y + 1),
                fill="gray",
                outline="",
            )

    for i in range(CELL_COUNT + 1):
        canvas.create_line(
            i * CELL_SIZE + inner_grid_start_x,
            inner_grid_start_y,
            i * CELL_SIZE + inner_grid_start_x,
            inner_grid_end_y,
            fill="black",
            width=5,
        )
        canvas.create_line(
            inner_grid_start_x,
            i * CELL_SIZE + inner_grid_start_y,
            inner_grid_end_x,
            i * CELL_SIZE + inner_grid_start_y,
            fill="black",
            width=5,
        )


class CircleArrow:
    circle_main = "red"
    circle_hover = "green"
    line_main = "blue"
    line_hover = "orange"
    text_main = "red"
    text_hover = "black"

    def __init__(self, canvas: tk.Canvas, arrow_ids: list[int], circle_ids: list[int], text_ids: list[int]):
        self.canvas = canvas
        self.arrow_id = arrow_ids
        self.circle_id = circle_ids
        self.text_id = text_ids
        self.tag = f"circle_arrow_{id(self)}"
        for item in self.arrow_id + self.circle_id + self.text_id:
            self.canvas.addtag_withtag(self.tag, item)
        self.bind_items()

    def on_enter(self, _event):
        self.canvas.after(
            10,
            self.change_appearance,
            CircleArrow.line_hover,
            CircleArrow.circle_hover,
            CircleArrow.text_hover,
        )

    def on_leave(self, _event):
        self.canvas.after(
            10,
            self.change_appearance,
            CircleArrow.line_main,
            CircleArrow.circle_main,
            CircleArrow.text_main,
        )

    def change_appearance(self, line_color: str, circle_color: str, text_color: str) -> None:
        for arrow in self.arrow_id:
            self.canvas.itemconfig(arrow, fill=line_color)
            self.canvas.tag_raise(arrow)
        for circle in self.circle_id:
            self.canvas.itemconfig(circle, fill=circle_color)
            self.canvas.tag_raise(circle)
        for text in self.text_id:
            self.canvas.itemconfig(text, fill=text_color)
            self.canvas.tag_raise(text)

    def bind_items(self) -> None:
        self.canvas.tag_bind(self.tag, "<Enter>", self.on_enter)
        self.canvas.tag_bind(self.tag, "<Leave>", self.on_leave)



def draw_moves(canvas: tk.Canvas, solution: ScoredSolution, show_index: bool = True) -> None:
    max_offset_angle = 15

    def get_direction(dx: float, dy: float) -> str:
        if dy == 0 and dx != 0:
            return "horizontal"
        if dx == 0 and dy != 0:
            return "vertical"
        return "diagonal"

    grouped_moves: dict[tuple, list[tuple[int, tuple[int, int], tuple[int, int]]]] = {}
    for idx, move in enumerate(solution.move_data):
        start = tuple(move.click_down)
        end = tuple(move.release)
        sx, sy = start
        ex, ey = end
        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)
        if distance == 0:
            continue

        direction = get_direction(dx, dy)
        if direction == "horizontal":
            group_key = ("horizontal", sy)
        elif direction == "vertical":
            group_key = ("vertical", sx)
        else:
            slope = round(dy / dx, 2) if dx != 0 else "inf"
            intercept = round(sy - slope * sx, 2) if slope != "inf" else sx
            group_key = ("diagonal", slope, intercept)

        grouped_moves.setdefault(group_key, []).append((idx, start, end))

    for group_key, group in grouped_moves.items():
        direction_type = group_key[0]
        num_overlaps = len(group)

        if num_overlaps > 1:
            angle_step = (2 * max_offset_angle) / (num_overlaps - 1)
            start_offset = -max_offset_angle
        else:
            angle_step = 0
            start_offset = 0

        if direction_type == "horizontal":
            group.sort(key=lambda x: grid_to_canvas(*x[1])[0])
        elif direction_type == "vertical":
            group.sort(key=lambda x: grid_to_canvas(*x[1])[1])
        else:
            group.sort(key=lambda x: (grid_to_canvas(*x[1])[0], grid_to_canvas(*x[1])[1]))

        for overlap_idx, (i, click_down, release) in enumerate(group):
            offset_angle_deg = start_offset + overlap_idx * angle_step
            offset_angle_rad = math.radians(offset_angle_deg)

            sx, sy = grid_to_canvas(*click_down)
            ex, ey = grid_to_canvas(*release)
            dx = ex - sx
            dy = ey - sy
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue

            ux = dx / distance
            uy = dy / distance
            perp_x = -uy
            perp_y = ux
            offset_distance = 10

            shifted_sx = sx + perp_x * offset_distance * math.sin(offset_angle_rad)
            shifted_sy = sy + perp_y * offset_distance * math.sin(offset_angle_rad)
            shifted_ex = ex + perp_x * offset_distance * math.sin(offset_angle_rad)
            shifted_ey = ey + perp_y * offset_distance * math.sin(offset_angle_rad)

            def get_text_location() -> tuple[float, float]:
                amount_sm = int(SCALE * 13)
                amount_lg = int(SCALE * 22)

                if dx < 0:
                    primary_side = "left"
                elif dy < 0:
                    primary_side = "up"
                elif dx > 0:
                    primary_side = "right"
                else:
                    primary_side = "down"

                if primary_side in ["left", "right"]:
                    side = "above" if overlap_idx % 2 == 0 else "below"
                elif primary_side in ["up", "down"]:
                    side = "left" if overlap_idx % 2 == 0 else "right"
                else:
                    side = "above"

                text_x = shifted_sx
                text_y = shifted_sy
                if primary_side == "left":
                    if side == "above":
                        text_x -= amount_lg
                        text_y -= amount_sm
                    else:
                        text_x -= amount_lg
                        text_y += amount_sm
                elif primary_side == "right":
                    if side == "above":
                        text_x += amount_lg
                        text_y -= amount_sm
                    else:
                        text_x += amount_lg
                        text_y += amount_sm
                elif primary_side == "up":
                    if side == "left":
                        text_x -= amount_sm + 1
                        text_y -= amount_lg + 1
                    else:
                        text_x += amount_sm - 1
                        text_y -= amount_lg + 1
                else:
                    if side == "left":
                        text_x -= amount_sm + 1
                        text_y += amount_lg - 1
                    else:
                        text_x += amount_sm - 1
                        text_y += amount_lg - 1
                return text_x, text_y

            circle_radius = max(3, math.floor(SCALE * 5))
            circle_id = canvas.create_oval(
                shifted_sx - circle_radius,
                shifted_sy - circle_radius,
                shifted_sx + circle_radius,
                shifted_sy + circle_radius,
                fill=CircleArrow.circle_main,
                outline="black",
            )
            arrow_id = canvas.create_line(
                shifted_sx,
                shifted_sy,
                shifted_ex,
                shifted_ey,
                fill=CircleArrow.line_main,
                width=max(1, math.ceil(SCALE * 2)),
                arrow=tk.LAST,
            )

            text_ids: list[int] = []
            if show_index:
                text_x, text_y = get_text_location()
                text_ids.append(
                    canvas.create_text(
                        text_x,
                        text_y,
                        text=f"{i + 1}",
                        fill=CircleArrow.text_main,
                        font=("Helvetica", max(10, math.ceil(SCALE * 12))),
                    )
                )

            CircleArrow(canvas, [arrow_id], [circle_id], text_ids)


class CubeFrame(tk.Frame):
    def __init__(self, root, width, height, x, y, *args, **kwargs):
        super().__init__(root, width=width * CELL_SIZE, height=height * CELL_SIZE, *args, **kwargs)
        self.pack_propagate(False)
        self.place(x=x * CELL_SIZE, y=y * CELL_SIZE)


class Tooltip:
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.after_id = None
        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def schedule_tooltip(self, _=None):
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self, _=None):
        if self.tooltip_window:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height()
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip_window, text=self.text, background="yellow", borderwidth=1, relief="solid")
        label.pack()

    def hide_tooltip(self, _=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class SolutionShower:
    def __init__(self, levels_folder: str | os.PathLike[str]):
        self.levels_folder = Path(levels_folder)
        self.filename = ""
        self.results: FileScoreResult | None = None
        self.index = 0
        self.solver = DpMouseSolver()

    def load_file(self, filename: str, dedupe: bool = False):
        if not filename:
            return
        self.filename = filename
        self.index = 0
        self.results = DpMouseSolver.score_file(filename, dedupe=dedupe)

    def increment(self):
        if self.results and self.index + 1 < len(self.results.solutions):
            self.index += 1

    def decrement(self):
        if self.index - 1 >= 0:
            self.index -= 1

    def get_solution_at_index(self):
        if not self.results or not self.results.solutions:
            return None
        return self.results.solutions[self.index]

    def get_move_data(self):
        solution = self.get_solution_at_index()
        return None if solution is None else solution.move_data


class UI:
    def __init__(self, folder: str):
        self.folder = folder
        self.root = tk.Tk()
        self.root.title("Mindbender Mouse Movement Optimizer (DP)")
        font_1 = {"font": ("Helvetica", max(10, math.floor(SCALE * 20)))}
        font_2 = {"font": ("Courier", max(10, math.floor(SCALE * 17)))}
        font_sm = {"font": ("Helvetica", max(10, math.floor(SCALE * 14)))}
        font_combo = {"font": ("Helvetica", max(10, math.floor(SCALE * 14)))}

        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg="white")
        self.canvas.pack()

        self.solutions = SolutionShower(folder)
        self.dedupe = False

        self.moves_frame = CubeFrame(self.root, 10, 0.5, 0, 0)
        self.text_moves = tk.Text(self.moves_frame, height=1, bg=self.moves_frame.cget("bg"), bd=0, **font_2)
        self.text_moves.insert("1.0", "")
        self.text_moves.config(state="disabled")
        self.text_moves.bind("<FocusIn>", lambda event: self.text_moves.config(bg=self.moves_frame.cget("bg")))
        self.text_moves.tag_configure("center", justify="center")
        self.text_moves.tag_add("center", "1.0", "end")
        self.text_moves.place(relx=0.5, rely=0.5, anchor="center")

        self.combo_frame = CubeFrame(self.root, 3, 0.5, 0, 0.5)
        self.file_combo_box = ttk.Combobox(self.combo_frame, values=[], **font_combo)
        self.file_combo_box.set("Select an option")
        self.file_combo_box.bind("<<ComboboxSelected>>", self.on_select)
        self.file_combo_box.pack(fill="both", expand=True)

        self.index_label_frame = CubeFrame(self.root, 3, 1, 0, 1)
        self.index_label = tk.Label(self.index_label_frame, text="N/A", **font_1)
        self.index_label.place(relx=0.5, rely=0.5, anchor="center")

        self.zero_frame = CubeFrame(self.root, 1, 1, 0, 2)
        self.zero_btn = tk.Button(self.zero_frame, text="1", background="yellow", **font_1, command=self.zero_show)
        self.zero_btn.pack(fill="both", expand=True)

        self.max_frame = CubeFrame(self.root, 1, 1, 1, 2)
        self.max_btn = tk.Button(self.max_frame, text="Max", background="light blue", **font_1, command=self.max_show)
        self.max_btn.pack(fill="both", expand=True)

        self.minus_frame = CubeFrame(self.root, 1, 1, 2, 2)
        self.minus_btn = tk.Button(self.minus_frame, text="-", background="pink", **font_1, command=self.dec_show)
        self.minus_btn.pack(fill="both", expand=True)

        self.plus_frame = CubeFrame(self.root, 1, 1, 3, 2)
        self.plus_btn = tk.Button(self.plus_frame, text="+", background="light green", **font_1, command=self.inc_show)
        self.plus_btn.pack(fill="both", expand=True)

        self.reload_frame = CubeFrame(self.root, 2, 0.8, 0, 3.1)
        self.reload_btn = tk.Button(self.reload_frame, text="Reload", background="light blue", **font_sm, command=self.reload_solutions)
        self.reload_btn.pack(fill="both", expand=True)

        self.toggle_frame = CubeFrame(self.root, 2, 0.8, 2, 3.1)
        self.toggle_btn = tk.Button(self.toggle_frame, text="Dedupe OFF", background="gray", **font_sm, command=self.toggle_dedupe)
        self.toggle_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle_btn, "Remove duplicate lines before DP-scoring the file")

        self.text_y = 4.2
        self.dist_total_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y)
        self.dist_total_text = tk.Label(self.dist_total_text_frame, text="", **font_2)
        self.dist_total_text.place(relx=0.0, rely=0.5, anchor="w")

        self.dist_drag_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y + 0.5)
        self.dist_drag_text = tk.Label(self.dist_drag_text_frame, text="", **font_2)
        self.dist_drag_text.place(relx=0.0, rely=0.5, anchor="w")

        self.dist_move_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y + 1.0)
        self.dist_move_text = tk.Label(self.dist_move_text_frame, text="", **font_2)
        self.dist_move_text.place(relx=0.0, rely=0.5, anchor="w")


    def on_select(self, _):
        selected_value = self.file_combo_box.get().strip()
        if not selected_value:
            return
        filepath = f"{self.folder}/{selected_value}"
        self.read_file(filepath)

    def load_files(self):
        path = self.folder
        if not os.path.isdir(path):
            self.file_combo_box.configure(values=[])
            self.index_label.config(text="No folder")
            return

        files = os.listdir(path)
        files = [f"{file}" for file in files if os.path.isfile(f"{path}/{file}")]
        files = [file for file in files if file.endswith(".txt")]
        pattern = re.compile(r"^(\d+)-(\d+)_.*\.txt$")
        files = [file for file in files if pattern.match(file)]
        files.sort(key=lambda x: (int(x.split("_")[0].split("-")[0]), int(x.split("_")[0].split("-")[1]), x))
        self.file_combo_box.configure(values=files)

        if files:
            default_file = "9-4_c8_5896.txt" if "9-4_c8_5896.txt" in files else files[0]
            self.file_combo_box.set(default_file)
            self.read_file(f"{self.folder}/{default_file}")
        else:
            self.index_label.config(text="0/0")

    def toggle_dedupe(self):
        self.dedupe = not self.dedupe
        self.toggle_btn.configure(
            background=["gray", "cyan"][self.dedupe],
            text="Dedupe ON" if self.dedupe else "Dedupe OFF",
        )
        self.reload_solutions_but_save_index()

    def print_current_move(self):
        move_data = self.solutions.get_move_data()
        if move_data is None:
            return
        print("\n")
        for move in move_data:
            print(move)

    def reload_solutions_but_save_index(self):
        current = self.solutions.get_solution_at_index()
        current_move_string = None if current is None else current.move_string
        self.reload_solutions()
        if current_move_string is None or not self.solutions.results:
            self.update()
            return

        index = next(
            (i for i, item in enumerate(self.solutions.results.solutions) if item.move_string == current_move_string),
            0,
        )
        self.solutions.index = index
        self.update()

    def reload_solutions(self):
        if not self.solutions.filename:
            return
        index = self.solutions.index
        self.solutions.load_file(self.solutions.filename, self.dedupe)
        self.solutions.index = min(index, len(self.solutions.results.solutions) - 1) if self.solutions.results else 0
        self.update()

    def read_file(self, filename: str):
        self.solutions.load_file(filename, self.dedupe)
        self.update()

    def zero_show(self):
        self.solutions.index = 0
        self.update()

    def max_show(self):
        if self.solutions.results and self.solutions.results.solutions:
            self.solutions.index = len(self.solutions.results.solutions) - 1
        self.update()

    def inc_show(self):
        self.solutions.increment()
        self.update()

    def dec_show(self):
        self.solutions.decrement()
        self.update()

    def update(self):
        self.canvas.delete("all")
        draw_grid(self.canvas)

        solution = self.solutions.get_solution_at_index()
        if solution is None or self.solutions.results is None:
            self.text_moves.config(state="normal")
            self.text_moves.delete("1.0", tk.END)
            self.text_moves.config(state="disabled")
            self.index_label.config(text="0/0")
            self.dist_total_text.config(text="Total: --")
            self.dist_drag_text.config(text="Drag:  --")
            self.dist_move_text.config(text="Move:  --")
            return

        self.text_moves.config(state="normal")
        self.text_moves.delete("1.0", tk.END)
        self.text_moves.insert("1.0", " " + solution.move_string)
        self.text_moves.tag_add("center", "1.0", "end")
        self.text_moves.config(state="disabled")

        draw_moves(self.canvas, solution)

        current_index = self.solutions.index
        total_dist = solution.total_drag + solution.total_move
        fat_moves = sum(1 for move in solution.move_data if len(move.move) == 4)
        total_solutions = len(self.solutions.results.solutions)

        self.index_label.config(text=f"{current_index + 1}/{total_solutions}")
        self.dist_total_text.config(text=f"Total: {round(total_dist, 3):>7.3f}")
        self.dist_drag_text.config(text=f"Drag:  {round(solution.total_drag, 3):>7.3f}")
        self.dist_move_text.config(text=f"Move:  {round(solution.total_move, 3):>7.3f}")




def main():
    folder = "../../levels"
    ui = UI(folder)
    ui.load_files()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
