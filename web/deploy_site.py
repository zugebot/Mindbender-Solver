# web/deploy_site.py
"""Copy the web solver into the jerrin.org site, where it is served at https://jerrin.org/mindbender/.

The site (github.com/zugebot/jerrin.org-website) is a Vite app; everything under its public/ is
published as is, and GitHub Pages rebuilds it on every push to master. This mirrors the files the
page needs into public/mindbender/ (and removes ones that are gone), then prints what changed.
Committing and pushing the site is left to you.

    python web/deploy_site.py [path to the jerrin.org-website checkout]

Run web/gen_puzzles.py first after a new solve, and web/gen_sprites.py if web/assets/ is missing
(the game art is not in this repo's git; Raptisoft allowed using it on the site, 2026-11)."""

from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

WEB = Path(__file__).resolve().parent
DEFAULT_SITE = Path.home() / "WebstormProjects" / "jerrin.org-website"
FILES = ["index.html", "style.css", "app.js", "solver-worker.js", "mb_solver.js", "mb5_solver.js", "puzzles.js", "font.js", "mouse-route.js",
         "screenshot.js"]
FOLDERS = ["solutions", "assets"]


def wanted() -> dict[str, Path]:
    out = {name: WEB / name for name in FILES}
    for folder in FOLDERS:
        root = WEB / folder
        if not root.is_dir() or not any(root.iterdir()):
            sys.exit(f"web/{folder}/ is missing or empty (see the docstring)")
        for p in sorted(root.rglob("*")):
            if p.is_file():
                out[p.relative_to(WEB).as_posix()] = p
    for name, p in out.items():
        if not p.is_file():
            sys.exit(f"web/{name} is missing")
    return out


def main() -> None:
    site = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SITE
    if not (site / "vite.config.ts").is_file() or not (site / "public").is_dir():
        sys.exit(f"{site} does not look like the jerrin.org-website checkout")
    dest = site / "public" / "mindbender"
    files = wanted()
    added = changed = removed = 0
    for name, src in files.items():
        target = dest / name
        existed = target.is_file()
        if existed and filecmp.cmp(src, target, shallow=False):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)          # bytes as they are: line endings untouched
        if existed:
            changed += 1
        else:
            added += 1
    for p in sorted(dest.rglob("*"), reverse=True):
        rel = p.relative_to(dest).as_posix()
        if p.is_file() and rel not in files:
            p.unlink()
            removed += 1
        elif p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"{dest}: {len(files)} files, {size / 1e6:.1f} MB; {added} new, {changed} changed, {removed} removed")


if __name__ == "__main__":
    main()
