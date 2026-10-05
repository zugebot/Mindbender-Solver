# Jerrin Shirks
# tools/platform_default.py

from tools.platform_api import PlatformAPI


class DefaultPlatformAPI(PlatformAPI):
    """
    Generic fallback for non-Windows environments.

    This will not preserve your current out-of-window drag behavior,
    but it keeps the code running and gives you a clean place to add:
    - Linux specific logic
    - browser / emscripten glue
    """
    pass