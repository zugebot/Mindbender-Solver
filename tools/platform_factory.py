# Jerrin Shirks
# tools/platform_factory.py

import os
import sys

from tools.platform_default import DefaultPlatformAPI
from tools.platform_windows import WindowsPlatformAPI
from tools.platform_browser import BrowserPlatformAPI


def _looks_like_browser_runtime() -> bool:
    if sys.platform == "emscripten":
        return True

    if os.environ.get("PYODIDE") == "1":
        return True

    try:
        import js  # type: ignore
        return True
    except Exception:
        return False


def create_platform():
    if _looks_like_browser_runtime():
        return BrowserPlatformAPI()

    if sys.platform.startswith("win"):
        return WindowsPlatformAPI()

    return DefaultPlatformAPI()