# Jerrin Shirks

# native imports
import os

# custom imports
from chuzzleDeluxe import ChuzzleDeluxe
from engine.textures import Textures



if __name__ == "__main__":
    if not os.path.isdir("cached"):
        os.mkdir("cached")

    game = ChuzzleDeluxe(size=(800, 600),
                         fps=60,
                         name="Chuzzle Remake")

    game.run()
