import re
import os
import tkinter as tk
from tkinter import ttk

"""
/**
 * 10 bits: unused
 * 54 bits: holds upper 3x6 cell grid
 */
u64 b1 = 0;
/**
 * 10 bits: unused
 * 54 bits: holds lower 3x6 cell grid
 */
u64 b2 = 0;
"""


"""
hold a 6x6 of buttons,
given if the buttons are on or off,

make 2 masks:
1 for the top board,
1 for the bottom board.
"""

"""
how to make UI:

vertical frame:
    frame/grid that holds a 6x6 array of buttons
    .
    horizontal frame:
        a button that toggles the text in those boxes between binary and hex
    .
    a frame that has two textboxes on the bottom (b1) (these show the binary / hex)
    . 
    a frame that has two textboxes on the bottom (b2) (these show the binary / hex)
"""




import tkinter as tk

class MaskUI:
    def __init__(self, master):
        self.master = master
        master.title("Mask UI")

        self.display_mode = 'binary'  # Can be 'binary' or 'hex'

        # Create a frame for the 6x6 grid of buttons
        self.grid_frame = tk.Frame(master)
        self.grid_frame.pack()

        self.buttons = []
        for row in range(6):
            button_row = []
            for col in range(6):
                button = tk.Button(
                    self.grid_frame,
                    text='0',
                    width=4,
                    command=lambda r=row, c=col: self.toggle_button(r, c)
                )
                button.grid(row=row, column=col)
                button_row.append(button)
            self.buttons.append(button_row)

        # Create a frame for the toggle display mode button
        self.toggle_frame = tk.Frame(master)
        self.toggle_frame.pack(pady=10)

        self.toggle_button = tk.Button(
            self.toggle_frame,
            text='Toggle Display Mode',
            command=self.toggle_display_mode
        )
        self.toggle_button.pack()

        # Create frames for the b1 and b2 textboxes
        self.b1_frame = tk.Frame(master)
        self.b1_frame.pack(pady=5)

        self.b1_label = tk.Label(self.b1_frame, text='b1:')
        self.b1_label.pack(side=tk.LEFT)
        self.b1_entry = tk.Entry(self.b1_frame, width=60)
        self.b1_entry.pack(side=tk.LEFT)

        self.b2_frame = tk.Frame(master)
        self.b2_frame.pack(pady=5)

        self.b2_label = tk.Label(self.b2_frame, text='b2:')
        self.b2_label.pack(side=tk.LEFT)
        self.b2_entry = tk.Entry(self.b2_frame, width=60)
        self.b2_entry.pack(side=tk.LEFT)

        # Initialize masks
        self.b1 = 0
        self.b2 = 0

    def toggle_button(self, row, col):
        button = self.buttons[row][col]
        current_text = button['text']
        new_text = '1' if current_text == '0' else '0'
        button.config(text=new_text)
        self.update_masks()

    def toggle_display_mode(self):
        self.display_mode = 'hex' if self.display_mode == 'binary' else 'binary'
        self.update_textboxes()

    def update_masks(self):
        self.b1 = 0
        self.b2 = 0
        for row in range(6):
            for col in range(6):
                bit_index = (row % 3) * 6 + col
                button = self.buttons[row][col]
                if button['text'] == '1':
                    if row < 3:
                        self.b1 |= 1 << bit_index
                    else:
                        self.b2 |= 1 << bit_index
        self.update_textboxes()

    def update_textboxes(self):
        if self.display_mode == 'binary':
            b1_str = format(self.b1, '054b').zfill(54)
            b2_str = format(self.b2, '054b').zfill(54)
        else:
            b1_str = format(self.b1, '014x').zfill(14)
            b2_str = format(self.b2, '014x').zfill(14)

        self.b1_entry.delete(0, tk.END)
        self.b1_entry.insert(0, b1_str)
        self.b2_entry.delete(0, tk.END)
        self.b2_entry.insert(0, b2_str)

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskUI(root)
    root.mainloop()





















