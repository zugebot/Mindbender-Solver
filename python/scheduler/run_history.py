# python/scheduler/run_history.py
"""Permanent run history: one CSV row per solver run plus a copy of its log.

    docs/run_history.csv      tracked; one row per run (appended by mbq.py)
    docs/run_logs/<name>.log  the solver's full log for that run

    python python/scheduler/run_history.py backfill   # add runs made before this existed

Each row: when it finished, puzzle, moves, engine and split, 2x2 piece, result,
solutions, time per phase, table / index sizes, RAM budget, passes, large
pages, left states and rate, probe variant, machine, solver commit.
"""

from __future__ import annotations

import csv
import ctypes
import json
import re
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "docs" / "run_history.csv"
LOG_DIR = REPO / "docs" / "run_logs"

COLUMNS = ["finished", "puzzle", "moves", "engine", "search", "fat", "result", "solutions",
           "total_s", "table_build_s", "index_build_s", "d7_build_s", "probe_s",
           "table_states", "d7_states", "index_gb", "ram_budget_gb", "passes", "large_pages",
           "left_states", "m_states_per_s", "probe_variant", "machine", "solver_commit",
           "source", "log_file", "notes"]


def _f(pattern: str, text: str, group: int = 1, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(group)) if m else None


def parse_log(text: str) -> dict:
    """Statistics from an experimentDiskDepth6 or findSolutions log."""
    s: dict = {}
    m = re.search(r"<(\d+),(\d+),(\d+)> = (\d+) moves", text)
    if m:
        s["engine"] = "ssd"
        s["search"] = f"<{m.group(1)},{m.group(2)},{m.group(3)}>"
        s["moves"] = int(m.group(4))
        s["table_states"] = _f(r"depth-6 exact unique: (\d+)", text, cast=int) \
            or _f(r"reusing depth-6 table: (\d+) states", text, cast=int)
        s["table_build_s"] = _f(r"BUILD depth-6 table: ([\d.]+)s", text)
        s["index_build_s"] = _f(r"BUILD fingerprint index: .* in ([\d.]+)s", text)
        d7 = re.search(r"BUILD depth-7 index: (\d+) unique states .* in ([\d.]+)s", text)
        if d7:
            s["d7_states"], s["d7_build_s"] = int(d7.group(1)), float(d7.group(2))
        fp = re.findall(r"fingerprint index: ([\d.]+) GB (?:total|over .*?), RAM budget ([\d.]+) GB -> (\d+) pass", text)
        if fp:
            s["index_gb"], s["ram_budget_gb"], s["passes"] = float(fp[-1][0]), float(fp[-1][1]), int(fp[-1][2])
        pages = re.findall(r"\((large|normal) pages\)", text)
        if pages:
            s["large_pages"] = "yes" if all(p == "large" for p in pages) else "no" if all(
                p == "normal" for p in pages) else "mixed"
        tot = re.search(r"probe totals: load ([\d.]+)s, stream\+probe ([\d.]+)s, (\d+) left states streamed", text)
        if tot:
            s["left_states"] = int(tot.group(3))
            s["m_states_per_s"] = round(int(tot.group(3)) / max(1e-9, float(tot.group(2))) / 1e6, 1)
        s["probe_s"] = _f(r"PROBE phase: ([\d.]+)s", text)
        s["probe_variant"] = (_f(r"-> (\d+/\w+) \(", text, cast=str)
                              or _f(r"probe variant: (\S+)", text, cast=str) or "32/plain")
        s["total_s"] = _f(r"TOTAL: ([\d.]+)s", text)
        rec = _f(r"RECOVERY: (\d+) solutions", text, cast=int)
        if rec is not None:
            s["solutions"], s["result"] = rec, "solved"
        elif re.search(r"unique middle states \([^)]*\): 0\b", text) and "TOTAL:" in text:
            s["solutions"], s["result"] = 0, "none"
    elif "Total Time:" in text or "Final exact solutions" in text or "No solutions found" in text:
        s["engine"] = "inram"
        seed = _f(r"Left seed depth: (\d+)", text, cast=int)
        stream = _f(r"Left stream depth: (\d+)", text, cast=int)
        right = _f(r"Right exact frontier depth: (\d+)", text, cast=int)
        if seed is not None:
            s["search"] = f"<{seed},{stream},{right}>"
            s["moves"] = seed + (stream or 0) + (right or 0)
        s["total_s"] = _f(r"Total Time: ([\d.]+)", text)
        n = _f(r"Final exact solutions: (\d+)", text, cast=int)
        if n is not None:
            s["solutions"], s["result"] = n, "solved"
        elif "No solutions found" in text:
            s["solutions"], s["result"] = 0, "none"
    if not s.get("result"):
        s["result"] = "error" if re.search(r"terminate called|error", text, re.I) else "unknown"
    return s


def machine_label() -> str:
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        cpu = "unknown CPU"

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    st = MEMORYSTATUSEX()
    st.dwLength = ctypes.sizeof(st)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
    return f"{cpu} / {round(st.ullTotalPhys / 2**30)} GB"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW).stdout.strip()
    except Exception:
        return ""


def append(row: dict, log_text: str | None, log_name: str) -> None:
    """Appends one row (and archives the log, if given)."""
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    if log_text is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"{log_name}.log").write_text(log_text, encoding="utf-8", errors="replace")
        row["log_file"] = f"docs/run_logs/{log_name}.log"
    new = not HISTORY.exists()
    with open(HISTORY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})


def row_from_log(text: str, puzzle: str, finished: float, **extra) -> dict:
    row = parse_log(text)
    row.update(puzzle=puzzle, finished=time.strftime("%Y-%m-%d %H:%M", time.localtime(finished)))
    row.update({k: v for k, v in extra.items() if v is not None})
    return row


def record_job(job: dict, log_path: Path) -> None:
    """Called by mbq.py when a job ends (done, failed or cancelled)."""
    if not job.get("started"):
        return                      # cancelled before it ran: nothing to record
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    row = row_from_log(text, job["puzzle"], job.get("finished") or time.time(),
                       moves=job["moves"], fat="yes" if job.get("fat") else "no",
                       machine=job.get("machine") or machine_label(),
                       solver_commit=job.get("solver_commit", ""), source=f"mbq job {job['id']}",
                       notes="; ".join(job.get("notes", [])))
    if job["status"] == "cancelled":
        row["result"] = "cancelled"
    elif job.get("result"):
        row["result"] = job["result"]
    if job.get("solutions") is not None:
        row["solutions"] = job["solutions"]
    if row.get("total_s") is None and job.get("started") and job.get("finished"):
        row["total_s"] = round(job["finished"] - job["started"], 1)
    stamp = time.strftime("%Y%m%d-%H%M", time.localtime(job.get("finished") or time.time()))
    append(row, text, f"{stamp}_{job['puzzle']}_m{job['moves']}_job{job['id']}")


# --------------------------------------------------------------------------
# backfill: runs made before the history existed
# --------------------------------------------------------------------------

RAM_UPGRADE = time.mktime(time.strptime("2026-10-06 13:30", "%Y-%m-%d %H:%M"))
CPU = "Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz"


def _machine_at(t: float) -> str:
    return f"{CPU} / {64 if t >= RAM_UPGRADE else 32} GB"


def backfill(scratch: Path, state: Path) -> int:
    if HISTORY.exists():
        print(f"{HISTORY} already exists; backfill only runs once")
        return 0
    from levels_fat import fat_puzzles   # noqa: E402  (small helper below)
    fat = fat_puzzles()
    runs = []   # (finished, puzzle, text, extra, log_name)

    # 2026-10-04: depth-12 (and lower) in-RAM reruns of every unsolved puzzle
    for d in sorted((scratch / "unsolved_full").glob("*_d*")):
        log = d / "run.log"
        if not log.exists():
            continue
        puzzle, depth = d.name.rsplit("_d", 1)
        t = log.stat().st_mtime
        runs.append((t, puzzle, log.read_text(errors="replace"),
                     {"moves": int(depth), "source": "unsolved depth-12 queue (2026-10-04)"},
                     f"{time.strftime('%Y%m%d-%H%M', time.localtime(t))}_{puzzle}_m{depth}_inram"))

    # SSD runs started by hand
    manual = [("run93b.log", "9-3", "SSD 13-move test (2026-10-04)"),
              ("overnight/6-4_m13.log", "6-4", "overnight queue (2026-10-05)"),
              ("overnight/10-2_m13.log", "10-2", "overnight queue (2026-10-05)"),
              ("overnight/10-3_m13.log", "10-3", "overnight queue (2026-10-05)"),
              ("overnight/6-4_m14.log", "6-4", "overnight queue (2026-10-05)"),
              ("run_64_m15_detail.log", "6-4", "15-move run, index-only depth-7 (2026-10-05)")]
    for rel, puzzle, source in manual:
        log = scratch / rel
        if not log.exists():
            continue
        t = log.stat().st_mtime
        text = log.read_text(errors="replace")
        moves = _f(r"= (\d+) moves", text, cast=int)
        runs.append((t, puzzle, text, {"moves": moves, "source": source},
                     f"{time.strftime('%Y%m%d-%H%M', time.localtime(t))}_{puzzle}_m{moves}_manual"))

    # scheduler jobs so far
    jobs = json.loads((state / "jobs.json").read_text()).get("jobs", [])
    for j in jobs:
        if j["status"] not in ("done", "failed", "cancelled") or not j.get("finished") or not j.get("started"):
            continue
        log = state / "jobs" / str(j["id"]) / "log.txt"
        text = log.read_text(errors="replace") if log.exists() else ""
        extra = {"moves": j["moves"], "source": f"mbq job {j['id']}", "notes": "; ".join(j.get("notes", [])),
                 "result": "cancelled" if j["status"] == "cancelled" else j.get("result"),
                 "solutions": j.get("solutions")}
        runs.append((j["finished"], j["puzzle"], text, extra,
                     f"{time.strftime('%Y%m%d-%H%M', time.localtime(j['finished']))}_{j['puzzle']}_m{j['moves']}_job{j['id']}"))

    for t, puzzle, text, extra, name in sorted(runs, key=lambda r: r[0]):
        row = row_from_log(text, puzzle, t, fat="yes" if puzzle in fat else "no", machine=_machine_at(t))
        for k, v in extra.items():
            if v is not None:
                row[k] = v
        append(row, text, name)
    print(f"backfilled {len(runs)} runs into {HISTORY}")
    return len(runs)


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "backfill":
        scratch = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
        backfill(scratch, Path(__file__).resolve().parent / "state")
    else:
        print(__doc__)
