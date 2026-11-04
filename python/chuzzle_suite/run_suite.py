# python/chuzzle_suite/run_suite.py
"""Start Chuzzle Suite:  python python/chuzzle_suite/run_suite.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chuzzle_suite import paths  # noqa: E402

if sys.stdout is None or sys.stderr is None:   # pythonw (the Start menu shortcut): no console
    paths.USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(paths.LOG_FILE, "w", encoding="utf-8", buffering=1)

from chuzzle_suite.ui import main  # noqa: E402

if __name__ == "__main__":
    main()
