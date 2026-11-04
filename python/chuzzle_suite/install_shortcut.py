# python/chuzzle_suite/install_shortcut.py
"""Makes Chuzzle Suite a Windows program: a Start menu shortcut "Chuzzle Suite" with the app's
icon that starts it without a console (pythonw). The shortcut carries the app's AppUserModelID
(paths.APP_ID, which the app also sets on itself), so once it is pinned (right-click it in the
Start menu, or the running app's taskbar button, then "Pin to taskbar"), the pinned icon and the
open window are one taskbar button. Windows 10 offers programs no supported way to pin
themselves, so pinning is that one click.

Run once (again after moving the repo or changing Python):
    python python/chuzzle_suite/install_shortcut.py [--desktop] [--remove]
Needs pywin32 (only this script; the app itself uses ctypes)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pythoncom
from win32com.propsys import propsys, pscon
from win32com.shell import shell

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chuzzle_suite import paths  # noqa: E402

NAME = "Chuzzle Suite"
RUN_SUITE = Path(__file__).resolve().parent / "run_suite.py"


def shortcut_paths(desktop: bool) -> list[Path]:
    out = [Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / f"{NAME}.lnk"]
    if desktop:
        out.append(Path(os.environ["USERPROFILE"]) / "Desktop" / f"{NAME}.lnk")
    return out


def pythonw() -> Path:
    """The pythonw.exe next to the Python running this script (the one with pygame)."""
    exe = Path(sys.executable)
    w = exe.with_name("pythonw.exe")
    return w if w.is_file() else exe


def make(path: Path) -> None:
    link = pythoncom.CoCreateInstance(shell.CLSID_ShellLink, None, pythoncom.CLSCTX_INPROC_SERVER,
                                      shell.IID_IShellLink)
    link.SetPath(str(pythonw()))
    link.SetArguments(f'"{RUN_SUITE}"')
    link.SetWorkingDirectory(str(paths.REPO_ROOT))
    link.SetIconLocation(str(paths.ICON_ICO), 0)
    link.SetDescription("Mindbender solutions, real-game playback, saves and the solver queue")
    store = link.QueryInterface(propsys.IID_IPropertyStore)
    store.SetValue(pscon.PKEY_AppUserModel_ID, propsys.PROPVARIANTType(paths.APP_ID, pythoncom.VT_LPWSTR))
    store.Commit()
    path.parent.mkdir(parents=True, exist_ok=True)
    link.QueryInterface(pythoncom.IID_IPersistFile).Save(str(path), 0)


def main() -> None:
    desktop = "--desktop" in sys.argv
    for path in shortcut_paths(desktop or "--remove" in sys.argv):
        if "--remove" in sys.argv:
            if path.is_file():
                path.unlink()
                print("removed", path)
            continue
        make(path)
        print("wrote", path)
    if "--remove" not in sys.argv:
        print(f"Pin it: Start menu -> {NAME} -> right-click -> Pin to taskbar")


if __name__ == "__main__":
    main()
