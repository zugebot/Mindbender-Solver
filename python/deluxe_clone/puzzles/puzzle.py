# Jerrin Shirks


class Puzzle:
    def __init__(self, name, initial_cells, solved_cells, start_fat=None, end_fat=None):
        self.name = name
        self.initial_cells = initial_cells
        self.solved_cells = solved_cells
        self.start_fat = start_fat
        self.end_fat = end_fat
