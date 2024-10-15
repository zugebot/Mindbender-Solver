import re
import os
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from permutations import get_permutations
from functools import partial

from sim import *

# introduction to computing systems: from bits & gates to C/C++ & beyond


SCALE = 0.75
CELL_SIZE = int(SCALE * 60)
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
GRID_OFFSET_X = 5
GRID_OFFSET_Y = 2
CELL_COUNT = 6

CELLS_WIDTH = 18
CELLS_HEIGHT = 9
SCREEN_WIDTH = CELLS_WIDTH * CELL_SIZE
SCREEN_HEIGHT = CELLS_HEIGHT * CELL_SIZE


def draw_grid(canvas: tk.Canvas):
    """Draw a 12x12 grid with the inner 6x6 grid having noticeably thicker lines."""
    outer_grid_size_x = (GRID_OFFSET_X + CELL_COUNT + (CELLS_WIDTH - CELL_COUNT - GRID_OFFSET_X))
    outer_grid_size_y = (GRID_OFFSET_Y + CELL_COUNT + (CELLS_HEIGHT - CELL_COUNT - GRID_OFFSET_Y))
    inner_grid_start_x = GRID_OFFSET_X * CELL_SIZE
    inner_grid_start_y = GRID_OFFSET_Y * CELL_SIZE
    inner_grid_end_x = (GRID_OFFSET_X + CELL_COUNT) * CELL_SIZE
    inner_grid_end_y = (GRID_OFFSET_Y + CELL_COUNT) * CELL_SIZE

    canvas.create_rectangle(0, 0, outer_grid_size_x * CELL_SIZE, outer_grid_size_y * CELL_SIZE, fill="light gray")

    for i in range(outer_grid_size_x + 1):
        canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, outer_grid_size_y * CELL_SIZE, fill='gray', width=1)
        canvas.create_line(0, i * CELL_SIZE, outer_grid_size_x * CELL_SIZE, i * CELL_SIZE, fill='gray', width=1)

    canvas.create_rectangle(inner_grid_start_x, inner_grid_start_y, inner_grid_end_x, inner_grid_end_y, fill="white")
    is_gray = False
    for x in range(CELL_COUNT):
        is_gray = not is_gray
        for y in range(CELL_COUNT):
            is_gray = not is_gray
            if is_gray:
                continue
            _x = inner_grid_start_x
            _y = inner_grid_start_y
            canvas.create_rectangle(_x + CELL_SIZE * x,
                                    _y + CELL_SIZE * y,
                                    _x + CELL_SIZE * (x + 1),
                                    _y + CELL_SIZE * (y + 1),
                                    fill="gray")

    for i in range(CELL_COUNT + 1):
        canvas.create_line(i * CELL_SIZE + inner_grid_start_x, inner_grid_start_y,
                           i * CELL_SIZE + inner_grid_start_x, inner_grid_end_y, fill='black', width=5)
        canvas.create_line(inner_grid_start_x, i * CELL_SIZE + inner_grid_start_y,
                           inner_grid_end_x, i * CELL_SIZE + inner_grid_start_y, fill='black', width=5)


def grid_to_canvas(x, y):
    """Convert grid coordinates to canvas pixel coordinates (center of the cell)."""
    canvas_x = (x + GRID_OFFSET_X) * CELL_SIZE + CELL_SIZE // 2
    canvas_y = (y + GRID_OFFSET_Y) * CELL_SIZE + CELL_SIZE // 2
    return canvas_x, canvas_y


class CircleArrow:
    circle_main = "red"
    circle_hover = "green"
    line_main = "blue"
    line_hover = "orange"
    text_main = "red"
    text_hover = "black"

    def __init__(self, _canvas, _arrow_ids, _circle_ids, _text_ids):
        self.canvas: tk.Canvas = _canvas

        self.arrow_id = _arrow_ids
        self.circle_id = _circle_ids
        self.text_id = _text_ids

        # Assign a common tag to all related items
        self.tag = f"circle_arrow_{id(self)}"
        for item in self.arrow_id + self.circle_id + self.text_id:
            self.canvas.addtag_withtag(self.tag, item)

        self.bind_items()

    def on_enter(self, event):
        # Schedule the appearance change after 10 milliseconds
        self.canvas.after(10, self.change_appearance, CircleArrow.line_hover, CircleArrow.circle_hover, CircleArrow.text_hover)

    def on_leave(self, event):
        # Schedule the appearance reset after 10 milliseconds
        self.canvas.after(10, self.change_appearance, CircleArrow.line_main, CircleArrow.circle_main, CircleArrow.text_main)

    def change_appearance(self, line_color, circle_color, text_color):
        # Change the appearance of the items
        for arrow in self.arrow_id:
            self.canvas.itemconfig(arrow, fill=line_color)
            self.canvas.tag_raise(arrow)
        for circle in self.circle_id:
            self.canvas.itemconfig(circle, fill=circle_color)
            self.canvas.tag_raise(circle)
        for text in self.text_id:
            self.canvas.itemconfig(text, fill=text_color)
            self.canvas.tag_raise(text)

    def bind_items(self):
        # Bind events to the common tag
        self.canvas.tag_bind(self.tag, "<Enter>", self.on_enter)
        self.canvas.tag_bind(self.tag, "<Leave>", self.on_leave)


def draw_moves(canvas: tk.Canvas, moves, show_index=True):
    """Draw mouse movements and transitions between moves with hover highlighting and non-overlapping index numbers."""

    max_offset_angle = 15

    def get_direction(_dx, _dy):
        if _dy == 0 and _dx != 0:
            return 'horizontal'
        elif _dx == 0 and _dy != 0:
            return 'vertical'
        else:
            return 'diagonal'

    grouped_moves = defaultdict(list)
    for idx, move in enumerate(moves):
        start = tuple(move['click_down'])
        ex, ey = move['release']
        sx, sy = start

        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)

        if distance == 0:
            continue

        direction = get_direction(dx, dy)

        if direction == 'horizontal':
            group_key = ('horizontal', sy)
        elif direction == 'vertical':
            group_key = ('vertical', sx)
        else:
            slope = round(dy / dx, 2) if dx != 0 else 'inf'
            intercept = round(sy - slope * sx, 2) if slope != 'inf' else sx
            group_key = ('diagonal', slope, intercept)

        grouped_moves[group_key].append((idx, move))

    for group_key, group in grouped_moves.items():
        direction_type = group_key[0]
        num_overlaps = len(group)

        if num_overlaps > 1:
            angle_step = (2 * max_offset_angle) / (num_overlaps - 1)
            start_offset = -max_offset_angle
        else:
            angle_step = 0
            start_offset = 0

        # Sort the group based on the direction
        if direction_type == 'horizontal':
            group.sort(key=lambda x: grid_to_canvas(*x[1]['click_down'])[0])
        elif direction_type == 'vertical':
            group.sort(key=lambda x: grid_to_canvas(*x[1]['click_down'])[1])
        else:
            group.sort(key=lambda x: (grid_to_canvas(*x[1]['click_down'])[0], grid_to_canvas(*x[1]['click_down'])[1]))

        for overlap_idx, (i, move_info) in enumerate(group):
            offset_angle_deg = start_offset + overlap_idx * angle_step
            offset_angle_rad = math.radians(offset_angle_deg)

            sx, sy = grid_to_canvas(*move_info['click_down'])
            ex, ey = grid_to_canvas(*move_info['release'])

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

            def get_text_location():

                # Determine the primary direction of the arrow for index placement
                amount_sm = int(SCALE * 13)
                amount_lg = int(SCALE * 22)

                if dx < 0:  # Left
                    primary_side = 'left'
                elif dy < 0:  # Up
                    primary_side = 'up'
                elif dx > 0:  # Right
                    primary_side = 'right'
                else:  # Down
                    primary_side = 'down'

                # Alternate sides based on overlap index to prevent overlapping indices
                if primary_side in ['left', 'right']:
                    side = 'above' if overlap_idx % 2 == 0 else 'below'
                elif primary_side in ['up', 'down']:
                    side = 'left' if overlap_idx % 2 == 0 else 'right'
                else:
                    side = 'above'

                # Determine the position based on the side
                _text_x = shifted_sx
                _text_y = shifted_sy
                if primary_side == 'left':
                    if side == 'above':
                        _text_x -= amount_lg
                        _text_y -= amount_sm
                    else:  # below
                        _text_x -= amount_lg
                        _text_y += amount_sm
                elif primary_side == 'right':
                    if side == 'above':
                        _text_x += amount_lg
                        _text_y -= amount_sm
                    else:  # below
                        _text_x += amount_lg
                        _text_y += amount_sm
                elif primary_side == 'up':
                    if side == 'left':
                        _text_x -= amount_sm + 1
                        _text_y -= amount_lg + 1
                    else:  # right
                        _text_x += amount_sm - 1
                        _text_y -= amount_lg + 1
                else:  # primary_side == 'down':
                    if side == 'left':
                        _text_x -= amount_sm + 1
                        _text_y += amount_lg - 1
                    else:  # right
                        _text_x += amount_sm - 1
                        _text_y += amount_lg - 1

                return _text_x, _text_y

            text_x, text_y = get_text_location()
            text_id = canvas.create_text(text_x, text_y, text=f"{i + 1}",
                                         fill=CircleArrow.text_main, font=("Helvetica", math.ceil(SCALE * 12)))

            circle_radius = math.floor(SCALE * 5)
            circle_id = canvas.create_oval(
                shifted_sx - circle_radius, shifted_sy - circle_radius,
                shifted_sx + circle_radius, shifted_sy + circle_radius,
                fill=CircleArrow.circle_main, outline="black"
            )

            arrow_id = canvas.create_line(
                shifted_sx, shifted_sy, shifted_ex, shifted_ey,
                fill=CircleArrow.line_main, width=math.ceil(SCALE * 2), arrow=tk.LAST
            )

            circleArrow = CircleArrow(canvas, [arrow_id], [circle_id], [text_id])



class CubeFrame(tk.Frame):
    def __init__(self, root, width, height, x, y, *args, **kwargs):
        super().__init__(root, width=width * CELL_SIZE, height=height * CELL_SIZE, *args, **kwargs)
        self.pack_propagate(False)
        self.place(x=x * CELL_SIZE, y=y * CELL_SIZE)


class Tooltip:
    def __init__(self, widget, text, delay=500):
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

        label = tk.Label(self.tooltip_window, text=self.text, background="yellow",
                         borderwidth=1, relief="solid")
        label.pack()

    def hide_tooltip(self, event=None):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


class SolutionShower:
    def __init__(self):
        self.filename = ""
        self.solutions = []
        self.index = 0

    def load_file(self, filename, state1=True, state2=True, state3=False, state4=False):
        if filename == "":
            return

        self.filename = filename
        self.solutions = []
        self.index = 0

        with open(f"{self.filename}", "r") as f:
            lines = f.readlines()
            lines = [line.strip("\n") for line in lines if line != ""]

        # ensure all solutions are listed
        if state4:
            _solutions = set()
            for line in lines:
                _temp = get_permutations(line)
                for solution in _temp:
                    _solutions.add(solution)
            lines = list(_solutions)

        # line = lines[398]
        # self.solutions = [[line, *find_shortest_mouse_path(line, state1, state3)]]
        # """
        for line in lines:
            self.solutions.append([line, *find_shortest_mouse_path(line, state1, state3)])
        if state2:
            self.solutions.sort(key=lambda x: x[1] + x[2])
        # """

    def increment(self):
        if self.index + 1 < len(self.solutions):
            self.index += 1

    def decrement(self):
        if self.index - 1 >= 0:
            self.index -= 1

    def get_solution_at_index(self):
        if not self.solutions:
            return None, None, None
        return self.solutions[self.index]

    def get_move_data(self):
        if not self.solutions:
            return None, None, None
        return self.solutions[self.index][-1]


class UI:
    def __init__(self, folder):
        self.folder = folder
        self.root = tk.Tk()
        self.root.title("Mindbender Mouse Movement Optimizer")
        font_1 = {"font": ("Helvetica", math.floor(SCALE * 20))}
        font_2 = {"font": ("Courier", math.floor(SCALE * 17))}
        font_sm = {"font": ("Helvetica", math.floor(SCALE * 14))}
        font_combo = {"font": ("Helvetica", math.floor(SCALE * 14))}

        # Create canvas
        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg='white')
        self.canvas.pack()

        self.solutions: SolutionShower = SolutionShower()
        self.state1 = True
        self.state2 = True
        self.state3 = False
        self.state4 = False

        self.moves_frame = CubeFrame(self.root, 10, 0.5, 0, 0)
        self.text_moves = tk.Text(self.moves_frame, height=1, bg=self.moves_frame.cget("bg"), bd=0, **font_2)
        self.text_moves.insert("1.0", "Hello World!")
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

        toggle_kwargs = {"text": "ON", "background": "cyan", **font_1}
        self.toggle1_frame = CubeFrame(self.root, 1, 1, 0, 3)
        self.toggle1_btn = tk.Button(self.toggle1_frame, **toggle_kwargs, command=self.toggle_state_1)
        self.toggle1_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle1_btn, "This calls the function (optimize_mouse_movement_pass3)")

        self.toggle2_frame = CubeFrame(self.root, 1, 1, 1, 3)
        self.toggle2_btn = tk.Button(self.toggle2_frame, **toggle_kwargs, command=self.toggle_state_2)
        self.toggle2_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle2_btn, "This sorts the solutions by distance")

        self.toggle3_frame = CubeFrame(self.root, 1, 1, 2, 3)
        self.toggle3_btn = tk.Button(self.toggle3_frame, background="gray", text="OFF", **font_1,
                                     command=self.toggle_state_3)
        self.toggle3_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle3_btn, "This calls the function (optimize_mouse_movement_pass2)")

        self.toggle4_frame = CubeFrame(self.root, 1, 1, 3, 3)
        self.toggle4_btn = tk.Button(self.toggle4_frame, background="gray", text="OFF", **font_1,
                                     command=self.toggle_state_4)
        self.toggle4_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle4_btn, "Creates all permutations of similar direction moves from the solves")

        self.text_y = 4.5
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
        selected_value = self.file_combo_box.get()
        filepath = f"{self.folder}/{selected_value}"
        self.read_file(filepath)

    def load_files(self):
        path = f"{self.folder}"
        files = os.listdir(path)
        files = [f"{file}" for file in files if os.path.isfile(f"{path}/{file}")]
        files = [file for file in files if file.endswith(".txt")]
        pattern = re.compile(r'^(\d+)-(\d+)_.*\.txt$')
        files = [file for file in files if pattern.match(file)]
        files.sort(key=lambda x: (int(x.split("_")[0].split("-")[0]), int(x.split("_")[0].split("-")[1])))
        self.file_combo_box.configure(values=files)

    def toggle_state_1(self):
        self.state1 = not self.state1
        self.toggle1_btn.configure(
            background=["gray", "cyan"][self.state1],
            text="ON" if self.state1 else "OFF"
        )
        self.reload_solutions_but_save_index()

    def toggle_state_2(self):
        self.state2 = not self.state2
        self.toggle2_btn.configure(
            background=["gray", "cyan"][self.state2],
            text="ON" if self.state2 else "OFF"
        )
        self.reload_solutions_but_save_index()

    def toggle_state_3(self):
        self.state3 = not self.state3
        self.toggle3_btn.configure(
            background=["gray", "cyan"][self.state3],
            text="ON" if self.state3 else "OFF"
        )
        self.reload_solutions_but_save_index()

    def toggle_state_4(self):
        self.state4 = not self.state4
        self.toggle4_btn.configure(
            background=["gray", "cyan"][self.state4],
            text="ON" if self.state4 else "OFF"
        )
        self.reload_solutions_but_save_index()

    def print_current_move(self):
        print(f"\n")
        for move in self.solutions.get_move_data():
            print(move)

    def reload_solutions_but_save_index(self):
        puzzle, _, _, _ = self.solutions.get_solution_at_index()
        self.reload_solutions()
        index = next((i for i, item in enumerate(self.solutions.solutions) if item[0] == puzzle), 0)
        self.solutions.index = index
        self.update()

    def reload_solutions(self):
        index = self.solutions.index
        self.solutions.load_file(self.solutions.filename, self.state1, self.state2, self.state3, self.state4)
        self.solutions.index = index
        self.update()

    def read_file(self, filename):
        self.solutions.load_file(filename, self.state1, self.state2, self.state3, self.state4)
        self.update()

    def zero_show(self):
        self.solutions.index = 0
        self.update()

    def max_show(self):
        self.solutions.index = len(self.solutions.solutions) - 1
        self.update()

    def inc_show(self):
        self.solutions.increment()
        self.update()

    def dec_show(self):
        self.solutions.decrement()
        self.update()

    def update(self):
        try:
            solve_string, total_drag, total_move, move_data = self.solutions.get_solution_at_index()
            total_dist = total_drag + total_move

            self.canvas.delete("all")
            if move_data is None:
                self.index_label.config(text="")
                return

            self.text_moves.config(state="normal")
            self.text_moves.delete("1.0", tk.END)
            self.text_moves.insert("1.0", " " + solve_string)
            self.text_moves.tag_add("center", "1.0", "end")
            self.text_moves.config(state="disabled")

            draw_grid(self.canvas)
            draw_moves(self.canvas, move_data)

            current_index = self.solutions.index
            self.index_label.config(text=f"{current_index + 1}/{len(self.solutions.solutions)}")
            self.dist_total_text.config(text=f"Total: {round(total_dist, 2):>5.2f}")
            self.dist_drag_text.config(text=f"Drag: {round(total_drag, 2):>6.2f}")
            self.dist_move_text.config(text=f"Move: {round(total_move, 2):>6.2f}")
        except:
            pass


def main():
    folder = "../levels"

    ui: UI = UI(folder)
    ui.load_files()
    # TODO: figure out why #128 makes line #6 become '5' units long instead of '4' when flipping it's direction
    ui.solutions.load_file(f"{folder}/6-5_c10_8.txt")
    ui.update()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
