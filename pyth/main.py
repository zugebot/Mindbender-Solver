import os
import tkinter as tk
from sim import *
from functools import partial
from tkinter import ttk
import re
from collections import defaultdict
from pprint import pprint
from permutations import get_permutations


CELL_SIZE = 60
CANVAS_SIZE = GRID_SIZE * CELL_SIZE
GRID_OFFSET_X = 5
GRID_OFFSET_Y = 3
CELL_COUNT = 6
SCREEN_HEIGHT = 11 * CELL_SIZE
SCREEN_WIDTH = 13 * CELL_SIZE


def draw_grid(canvas: tk.Canvas):
    """Draw a 12x12 grid with the inner 6x6 grid having noticeably thicker lines."""
    outer_grid_size_x = (GRID_OFFSET_X + CELL_COUNT + 3)
    outer_grid_size_y = (GRID_OFFSET_Y + CELL_COUNT + 3)
    inner_grid_start_x = GRID_OFFSET_X * CELL_SIZE
    inner_grid_start_y = GRID_OFFSET_Y * CELL_SIZE
    inner_grid_end_x = (GRID_OFFSET_X + CELL_COUNT) * CELL_SIZE
    inner_grid_end_y = (GRID_OFFSET_Y + CELL_COUNT) * CELL_SIZE

    canvas.create_rectangle(0, 0, outer_grid_size_x * CELL_SIZE, outer_grid_size_y * CELL_SIZE, fill="light gray")

    for i in range(outer_grid_size_x + 1):
        canvas.create_line(i * CELL_SIZE, 0, i * CELL_SIZE, outer_grid_size_y * CELL_SIZE, fill='gray', width=1)
        canvas.create_line(0, i * CELL_SIZE, outer_grid_size_x * CELL_SIZE, i * CELL_SIZE, fill='gray', width=1)

    canvas.create_rectangle(inner_grid_start_x, inner_grid_start_y, inner_grid_end_x, inner_grid_end_y, fill="white")

    for i in range(CELL_COUNT + 1):
        canvas.create_line(i * CELL_SIZE + inner_grid_start_x, inner_grid_start_y, i * CELL_SIZE + inner_grid_start_x, inner_grid_end_y, fill='black', width=5)
        canvas.create_line(inner_grid_start_x, i * CELL_SIZE + inner_grid_start_y, inner_grid_end_x, i * CELL_SIZE + inner_grid_start_y, fill='black', width=5)


def grid_to_canvas(x, y):
    """Convert grid coordinates to canvas pixel coordinates (center of the cell)."""
    canvas_x = (x + GRID_OFFSET_X) * CELL_SIZE + CELL_SIZE // 2
    canvas_y = (y + GRID_OFFSET_Y) * CELL_SIZE + CELL_SIZE // 2
    return canvas_x, canvas_y


def draw_moves(canvas, moves, show_index=True):
    """Draw mouse movements and transitions between moves with hover highlighting and non-overlapping index numbers."""

    # Define maximum offset angle in degrees for overlapping arrows
    max_offset_angle = 15  # Adjust as needed

    # Helper function to determine the direction of a move
    def get_direction(dx, dy):
        if dy == 0 and dx != 0:
            return 'horizontal'
        elif dx == 0 and dy != 0:
            return 'vertical'
        else:
            return 'diagonal'

    # Group moves based on direction and axis
    grouped_moves = defaultdict(list)
    for idx, move in enumerate(moves):
        start = tuple(move['click_down'])
        ex, ey = move['release']
        sx, sy = start

        # Calculate direction vector
        dx = ex - sx
        dy = ey - sy
        distance = math.hypot(dx, dy)

        if distance == 0:
            # Skip moves with zero distance
            continue

        # Normalize direction vector
        ux = dx / distance
        uy = dy / distance

        # Determine direction
        direction = get_direction(dx, dy)

        if direction == 'horizontal':
            # Group by y-coordinate
            group_key = ('horizontal', sy)
        elif direction == 'vertical':
            # Group by x-coordinate
            group_key = ('vertical', sx)
        else:
            # For diagonal, group by slope and intercept or another unique identifier
            # Here, we'll use the slope rounded to 2 decimals and y-intercept
            slope = round(dy / dx, 2) if dx != 0 else 'inf'
            intercept = round(sy - slope * sx, 2) if slope != 'inf' else sx  # x-intercept for vertical
            group_key = ('diagonal', slope, intercept)

        grouped_moves[group_key].append((idx, move))

    for group_key, group in grouped_moves.items():
        direction_type = group_key[0]
        num_overlaps = len(group)

        # Calculate angular step between overlapping arrows
        if num_overlaps > 1:
            angle_step = (2 * max_offset_angle) / (num_overlaps - 1)
            start_offset = -max_offset_angle
        else:
            angle_step = 0
            start_offset = 0

        # Sort the group based on the direction
        if direction_type == 'horizontal':
            # Sort by x_start
            group.sort(key=lambda x: grid_to_canvas(*x[1]['click_down'])[0])
        elif direction_type == 'vertical':
            # Sort by y_start
            group.sort(key=lambda x: grid_to_canvas(*x[1]['click_down'])[1])
        else:
            # For diagonal, sort by x_start then y_start
            group.sort(key=lambda x: (grid_to_canvas(*x[1]['click_down'])[0], grid_to_canvas(*x[1]['click_down'])[1]))

        for overlap_idx, (i, move_info) in enumerate(group):
            # Calculate the current offset angle in radians
            offset_angle_deg = start_offset + overlap_idx * angle_step
            offset_angle_rad = math.radians(offset_angle_deg)

            # Original start and end points
            sx, sy = grid_to_canvas(*move_info['click_down'])
            ex, ey = grid_to_canvas(*move_info['release'])

            # Calculate the direction vector from start to end
            dx = ex - sx
            dy = ey - sy
            distance = math.hypot(dx, dy)

            if distance == 0:
                # Avoid division by zero; skip drawing this move
                continue

            # Normalize the direction vector
            ux = dx / distance
            uy = dy / distance

            # Calculate perpendicular vector for offset
            perp_x = -uy
            perp_y = ux

            # Define the offset distance
            offset_distance = 10  # Pixels to offset; adjust as needed

            # Apply the angular offset by shifting along the perpendicular vector
            shifted_sx = sx + perp_x * offset_distance * math.sin(offset_angle_rad)
            shifted_sy = sy + perp_y * offset_distance * math.sin(offset_angle_rad)
            shifted_ex = ex + perp_x * offset_distance * math.sin(offset_angle_rad)
            shifted_ey = ey + perp_y * offset_distance * math.sin(offset_angle_rad)

            # Draw the click-down circle
            canvas.create_oval(
                shifted_sx - 5, shifted_sy - 5,
                shifted_sx + 5, shifted_sy + 5,
                fill='red', outline='black'
            )

            # Draw the arrow and get its Canvas item ID
            arrow_id = canvas.create_line(
                shifted_sx, shifted_sy, shifted_ex, shifted_ey,
                fill='blue', width=2, arrow=tk.LAST
            )

            # Define the callback functions for hover events
            def on_enter(event, arrow=arrow_id):
                """Highlight the arrow when the mouse enters."""
                canvas.itemconfig(arrow, fill='orange', width=3)  # Change color and increase width

            def on_leave(event, arrow=arrow_id):
                """Revert the arrow's appearance when the mouse leaves."""
                canvas.itemconfig(arrow, fill='blue', width=2)  # Revert to original color and width

            # Bind the hover events to the arrow
            canvas.tag_bind(arrow_id, "<Enter>", on_enter)
            canvas.tag_bind(arrow_id, "<Leave>", on_leave)

            # Show the numbered index next to the click-down circle
            if show_index:
                # Determine the primary direction of the arrow for index placement
                amount_sm = 13
                amount_lg = 22
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
                    # For horizontal arrows, alternate between above and below
                    side = 'above' if overlap_idx % 2 == 0 else 'below'
                elif primary_side in ['up', 'down']:
                    # For vertical arrows, alternate between left and right
                    side = 'left' if overlap_idx % 2 == 0 else 'right'
                else:
                    side = 'above'  # Default side

                # Determine the position based on the side
                if primary_side == 'left':
                    if side == 'above':
                        textx = shifted_sx - amount_lg
                        texty = shifted_sy - amount_sm
                    else:  # below
                        textx = shifted_sx - amount_lg
                        texty = shifted_sy + amount_sm
                elif primary_side == 'right':
                    if side == 'above':
                        textx = shifted_sx + amount_lg
                        texty = shifted_sy - amount_sm
                    else:  # below
                        textx = shifted_sx + amount_lg
                        texty = shifted_sy + amount_sm
                elif primary_side == 'up':
                    if side == 'left':
                        textx = shifted_sx - amount_sm + 2
                        texty = shifted_sy - amount_lg + 2
                    else:  # right
                        textx = shifted_sx + amount_sm - 2
                        texty = shifted_sy - amount_lg + 2
                elif primary_side == 'down':
                    if side == 'left':
                        textx = shifted_sx - amount_sm + 2
                        texty = shifted_sy + amount_lg - 2
                    else:  # right
                        textx = shifted_sx + amount_sm - 2
                        texty = shifted_sy + amount_lg - 2
                else:
                    # Default placement
                    textx = shifted_sx
                    texty = shifted_sy - amount_lg

                # Create the index text
                canvas.create_text(textx, texty, text=f"{i + 1}", fill='red')

            # Draw the transition line from release to the next click-down point (if applicable)
            if overlap_idx < len(group) - 1:  # If not the last move in the group
                next_move_info = group[overlap_idx + 1][1]
                next_x, next_y = grid_to_canvas(*next_move_info['click_down'])

                # Apply the same offset to the next click-down point
                shifted_next_x = next_x + perp_x * offset_distance * math.sin(offset_angle_rad)
                shifted_next_y = next_y + perp_y * offset_distance * math.sin(offset_angle_rad)

                # Draw the transition line with a different color and dash pattern
                # transition_id = canvas.create_line(
                #     shifted_ex, shifted_sy, shifted_next_x, shifted_next_y,
                #     fill='purple', width=2, dash=(5, 2)
                # )

                # Define hover callbacks for transition lines
                # def on_enter_transition(event, arrow=transition_id):
                #     """Highlight the transition arrow when the mouse enters."""
                #     canvas.itemconfig(arrow, fill='pink', width=3)  # Change color and increase width

                # def on_leave_transition(event, arrow=transition_id):
                #     """Revert the transition arrow's appearance when the mouse leaves."""
                #     canvas.itemconfig(arrow, fill='purple', width=2)  # Revert to original color and width

                # Bind the hover events to the transition arrow
                # canvas.tag_bind(transition_id, "<Enter>", on_enter_transition)
                # canvas.tag_bind(transition_id, "<Leave>", on_leave_transition)

                # Draw a small green circle at the next click-down point
                canvas.create_oval(
                    shifted_next_x - 5, shifted_next_y - 5,
                    shifted_next_x + 5, shifted_next_y + 5,
                    fill='green', outline='black'
                )


class CubeFrame(tk.Frame):
    def __init__(self, root, width, height, x, y, *args, **kwargs):
        super().__init__(root, width=width*CELL_SIZE, height=height*CELL_SIZE, *args, **kwargs)
        self.pack_propagate(0)
        self.place(x=x*CELL_SIZE, y=y*CELL_SIZE)



class Tooltip:
    def __init__(self, widget, text, delay=500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tooltip_window = None
        self.after_id = None  # To store the ID of the after() call
        self.widget.bind("<Enter>", self.schedule_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def schedule_tooltip(self, event=None):
        # Schedule the tooltip to appear after the specified delay (in milliseconds)
        self.after_id = self.widget.after(self.delay, self.show_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window:
            return

        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height()

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(self.tooltip_window, text=self.text, background="yellow",
                         borderwidth=1, relief="solid")
        label.pack()

    def hide_tooltip(self, event=None):
        # Cancel the scheduled tooltip if the mouse leaves before the delay
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

        # Destroy the tooltip window if it is shown
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

        with open(f"../{self.filename}", "r") as f:
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

    def getSolutionAtIndex(self):
        if not self.solutions:
            return None, None, None
        return self.solutions[self.index]

    def getMoveData(self):
        if not self.solutions:
            return None, None, None
        return self.solutions[self.index][-1]

class UI:
    def __init__(self, folder):
        self.folder = folder
        self.root = tk.Tk()
        self.root.title("Mindbender Mouse Movement Optimizer")
        FONT_1 = {"font": ("Helvetica", 20)}
        FONT_2 = {"font": ("Courier", 18)}
        FONT_SM = {"font": ("Helvetica", 14)}

        # Create canvas
        self.canvas = tk.Canvas(self.root, width=SCREEN_WIDTH, height=SCREEN_HEIGHT, bg='white')
        self.canvas.pack()

        self.solutions: SolutionShower = SolutionShower()
        self.state1 = True
        self.state2 = True
        self.state3 = False
        self.state4 = False


        self.combo_frame = CubeFrame(self.root, 3, 0.5, 0, 0.5)
        self.file_combo_box = ttk.Combobox(self.combo_frame, values=[], font=("Helvetica", 14))
        self.file_combo_box.set("Select an option")
        self.file_combo_box.bind("<<ComboboxSelected>>", self.on_select)
        self.file_combo_box.pack(fill="both", expand=True)


        self.index_label_frame = CubeFrame(self.root, 3, 1, 0, 1)
        self.index_label = tk.Label(self.index_label_frame, text="N/A", **FONT_1)
        self.index_label.place(relx=0.5, rely=0.5, anchor="center")



        self.zero_frame = CubeFrame(self.root, 1, 1, 0, 2)
        self.zero_btn = tk.Button(self.zero_frame, text="1", background="yellow", **FONT_1, command=self.zeroShow)
        self.zero_btn.pack(fill="both", expand=True)

        self.max_frame = CubeFrame(self.root, 1, 1, 1, 2)
        self.max_btn = tk.Button(self.max_frame, text="Max", background="light blue", **FONT_1, command=self.maxShow)
        self.max_btn.pack(fill="both", expand=True)

        self.minus_frame = CubeFrame(self.root, 1, 1, 2, 2)
        self.minus_btn = tk.Button(self.minus_frame, text="-", background="pink", **FONT_1, command=self.decShow)
        self.minus_btn.pack(fill="both", expand=True)


        self.plus_frame = CubeFrame(self.root, 1, 1, 3, 2)
        self.plus_btn = tk.Button(self.plus_frame, text="+", background="light green", **FONT_1, command=self.incShow)
        self.plus_btn.pack(fill="both", expand=True)


        toggle_kwargs = {"text": "ON", "background": "cyan", **FONT_1}
        self.toggle1_frame = CubeFrame(self.root, 1, 1, 0, 3)
        self.toggle1_btn = tk.Button(self.toggle1_frame, **toggle_kwargs, command=self.toggle_state_1)
        self.toggle1_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle1_btn, "This calls the function (optimize_mouse_movement_pass3)")

        self.toggle2_frame = CubeFrame(self.root, 1, 1, 1, 3)
        self.toggle2_btn = tk.Button(self.toggle2_frame, **toggle_kwargs, command=self.toggle_state_2)
        self.toggle2_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle2_btn, "This sorts the solutions by distance")

        self.toggle3_frame = CubeFrame(self.root, 1, 1, 2, 3)
        self.toggle3_btn = tk.Button(self.toggle3_frame, background="gray", text="OFF", **FONT_1,
                                     command=self.toggle_state_3)
        self.toggle3_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle3_btn, "This calls the function (optimize_mouse_movement_pass2)")

        self.toggle4_frame = CubeFrame(self.root, 1, 1, 3, 3)
        self.toggle4_btn = tk.Button(self.toggle4_frame, background="gray", text="OFF", **FONT_1,
                                     command=self.toggle_state_4)
        self.toggle4_btn.pack(fill="both", expand=True)
        Tooltip(self.toggle4_btn, "Creates all permutations of similar direction moves from the solves")



        self.text_y = 4.5
        self.dist_total_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y)
        self.dist_total_text = tk.Label(self.dist_total_text_frame, text="", **FONT_2)
        self.dist_total_text.place(relx=0.0, rely=0.5, anchor="w")

        self.dist_drag_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y+0.5)
        self.dist_drag_text = tk.Label(self.dist_drag_text_frame, text="", **FONT_2)
        self.dist_drag_text.place(relx=0.0, rely=0.5, anchor="w")

        self.dist_move_text_frame = CubeFrame(self.root, 3, 0.5, 0, self.text_y+1.0)
        self.dist_move_text = tk.Label(self.dist_move_text_frame, text="", **FONT_2)
        self.dist_move_text.place(relx=0.0, rely=0.5, anchor="w")




    def on_select(self, event):
        selected_value = self.file_combo_box.get()
        filepath = f"{self.folder}/{selected_value}"
        self.read_file(filepath)

    def load_files(self):
        path = f"../{self.folder}"
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

    def printCurrentMove(self):
        print(f"\n")
        for move in self.solutions.getMoveData():
            print(move)

    def reload_solutions_but_save_index(self):
        puzzle, _, _, _ = self.solutions.getSolutionAtIndex()
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

    def zeroShow(self):
        self.solutions.index = 0
        self.update()

    def maxShow(self):
        self.solutions.index = len(self.solutions.solutions) - 1
        self.update()

    def incShow(self):
        self.solutions.increment()
        self.update()

    def decShow(self):
        self.solutions.decrement()
        self.update()

    def update(self):
        try:
            solve_string, total_drag, total_move, move_data = self.solutions.getSolutionAtIndex()
            total_dist = total_drag + total_move

            self.canvas.delete("all")
            if move_data is None:
                self.index_label.config(text="")
                return

            draw_grid(self.canvas)
            draw_moves(self.canvas, move_data)

            current_index = self.solutions.index
            self.index_label.config(text=f"{current_index+1}/{len(self.solutions.solutions)}")
            self.dist_total_text.config(text=f"Total: {round(total_dist, 2):>5.2f}")
            self.dist_drag_text.config(text=f"Drag: {round(total_drag, 2):>6.2f}")
            self.dist_move_text.config(text=f"Move: {round(total_move, 2):>6.2f}")
        except:
            pass




def main():
    folder = "all_levels"

    ui: UI = UI(folder)
    ui.load_files()
    # TODO: figure out why #128 makes line #6 become '5' units long instead of '4' when flipping it's direction
    # ui.solutions.load_file(f"{folder}/4-3_c7_881.txt")
    ui.update()
    ui.root.mainloop()


if __name__ == "__main__":
    main()
