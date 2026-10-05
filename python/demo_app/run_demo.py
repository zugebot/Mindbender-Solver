# python/demo_app/run_demo.py
"""Start the Chuzzle demo app:  python python/demo_app/run_demo.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chuzzle_demo.ui import main  # noqa: E402

if __name__ == "__main__":
    main()
