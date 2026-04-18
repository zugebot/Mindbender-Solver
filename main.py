# Jerrin Shirks

# native imports
import os

# custom imports
from chuzzleDeluxe import ChuzzleDeluxe
from engine.settings import Settings



if __name__ == "__main__":
    if not os.path.isdir("cached"):
        os.mkdir("cached")
        
    settings = Settings()
    game = ChuzzleDeluxe(settings)
    game.run()
