#!/usr/bin/env python3
# python/scheduler/mbq.py
"""Mindbender solver queue: a small always-on scheduler you can talk to.

    python python/scheduler/mbq.py daemon            # start the server in the background
    python python/scheduler/mbq.py status            # what is running, progress, ETA
    python python/scheduler/mbq.py add 12-1 12-5 --moves 13 --until 14
    python python/scheduler/mbq.py queue | history | log ID | cancel ID | pause | resume

A live dashboard is served at http://127.0.0.1:8766/ (refreshes every 3 s).

One job runs at a time, and never while another solver process is running
(e.g. a run started by hand). Jobs and their results live in
python/scheduler/state/ and survive restarts; a job that is still running
when the server restarts is re-adopted. The server only listens on
127.0.0.1. It sleeps between polls, so it costs the solver nothing; the
solver itself only bumps an atomic counter per seed for --progress-file.

Engines (chosen from --moves unless --engine is given):
  ssd    build/experimentDiskDepth6.exe, 10-17 moves:
         <S,3,6> up to 14 moves, <S,3,7> (index-only depth-7 table) for 15 on 2x2 boards,
         <5,4,6> for 15 on normal boards; 16: <5,5,6> normal boards, <5,4,7> 2x2 boards;
         17: <5,6,6> normal boards (--gpu only), <5,5,7> 2x2 boards (weeks: use shards).
  inram  build/findSolutions.exe, 3-12 moves (exhaustive in-RAM search).

After a job: solutions are copied to <repo>/levels and replayed (Chuzzle
Suite's simulator and testKnownSolutions). With promotion on (default),
solved puzzles go to build/levels_final and leave build/README.md /
build/config.json's unsolved lists; an empty search raises the README lower
bound when every shorter length was already excluded. Nothing is committed.
--until N re-queues an unsolved puzzle one move deeper, up to N. Depth-6/7
tables the scheduler created are deleted when no queued job needs them
(--keep-table keeps them); tables it did not create are never touched.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alarm  # noqa: E402
import fleet  # noqa: E402
import run_history  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STATE = Path(os.environ.get("MBQ_STATE", Path(__file__).resolve().parent / "state"))
JOBS_FILE = STATE / "jobs.json"
BIN_DIR = STATE / "bin"
JOB_ROOT = STATE / "jobs"
TABLE_ROOT = Path(os.environ.get("MBQ_TABLE_ROOT", "X:/mindbender_d6"))
PORT = int(os.environ.get("MBQ_PORT", "8766"))
POLL_S = 5.0
DASHBOARD = Path(__file__).resolve().parent / "dashboard.html"
ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}

SSD_EXE = REPO / "build" / "experimentDiskDepth6.exe"
GPU_DLL = REPO / "build_gpu" / "mbgpu.dll"   # src/gpu/build.cmd; copied next to the exe for --gpu jobs
VK_DLL = REPO / "build_gpu" / "mbvk.dll"     # src/gpu/vulkan/build.cmd (any vendor); copied too when built
INRAM_EXE = REPO / "build" / "findSolutions.exe"
TEST_KNOWN_EXE = REPO / "build" / "testKnownSolutions.exe"
LEVELS_DIR = REPO / "levels"
LEVELS_FINAL = REPO / "build" / "levels_final"
# Placements the game also accepts ("4-4@r2c3": the 2x2 piece on another 2x2 block of its colour,
# top-left cell at row 2, column 3, 1-indexed). Kept apart from the official answers.
GAME_ACCEPTS = LEVELS_FINAL / "game_accepts"
PUZZLE_NAME_RE = r"\d+-\d+(?:@r\d+c\d+)?"
BUILD_README = REPO / "build" / "README.md"
BUILD_CONFIG = REPO / "build" / "config.json"

# Any of these running (and not ours) means "someone else is using the machine".
SOLVER_PROCESS_RE = re.compile(
    r"^(experimentDiskDepth6\S*|expD7\S*|findSolutions\S*|mbq_ssd|mbq_inram|exp)\.exe$", re.I)
# Tests only: start jobs even while another solver runs.
IGNORE_EXTERNAL = os.environ.get("MBQ_IGNORE_EXTERNAL") == "1"

SHARD_ROOT = STATE / "shards"        # <puzzle>_m<moves>_n<N>/s<k>.mid: one search split into work units
PORTABLE_EXE = REPO / "build_portable" / "experimentDiskDepth6.exe"   # -DMB_MARCH=x86-64-v3 build


def shard_group(puzzle: str, moves: int, count: int) -> str:
    return f"{puzzle}_m{moves}_n{count}"


def read_middles_header(path: Path) -> dict:
    """Header of a solver middles file (--middles-out); raises if it is not a complete one."""
    lines = path.read_text(errors="replace").splitlines()
    if not lines or not lines[0].startswith("# mindbender middles v1") or lines[-1].strip() != "end":
        raise ValueError(f"{path.name}: not a complete middles file")
    head = {}
    for line in lines[1:]:
        key, _, value = line.partition(" ")
        head[key] = value
        if key == "count":
            break
    seed, stream, table = (int(x) for x in head["split"].split())
    k, n = (int(x) for x in head["shard"].split())
    block = int(head.get("block", "0"))   # --block: middles one move further from the start
    return {"puzzle": head["puzzle"], "seed": seed, "stream": stream, "table": table, "block": block,
            "moves": seed + stream + table + block, "shard": k, "shards": n, "count": int(head["count"]),
            "d7_groups": int(head.get("d7groups", "1"))}


SSD_PHASES = ["d6-build", "d6-index", "d7-build", "d7-index", "probe", "d7-confirm", "verify", "recovery"]

kernel32 = ctypes.windll.kernel32
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
PROCESS_SUSPEND_RESUME = 0x0800
STILL_ACTIVE = 259
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
# Hold: freeze every thread of the solver in place (what Resource Monitor's
# "Suspend Process" does) and thaw it later; it carries on from the same point.
ntdll = ctypes.windll.ntdll
for _fn in (ntdll.NtSuspendProcess, ntdll.NtResumeProcess):
    _fn.argtypes = [wintypes.HANDLE]
    _fn.restype = ctypes.c_long


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def now() -> float:
    return time.time()


def fmt_time(t: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "-"


def fmt_dur(s: float | None) -> str:
    if s is None:
        return "unknown"
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def write_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


class PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_void_p), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]


def running_processes() -> list[tuple[str, int]]:
    """(exe name, pid) of every process, via a Toolhelp snapshot: milliseconds,
    where `tasklist` takes 20-40 s while the solver keeps every core busy."""
    snap = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)   # TH32CS_SNAPPROCESS
    if not snap or snap == wintypes.HANDLE(-1).value:
        return []
    procs = []
    try:
        entry = PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            procs.append((entry.szExeFile, int(entry.th32ProcessID)))
            ok = kernel32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snap)
    return procs


class ProcessHandle:
    """Exit code / termination for a process we started or re-adopted."""

    def __init__(self, pid: int, popen: subprocess.Popen | None = None):
        self.pid = pid
        self.popen = popen
        self.handle = None
        if popen is None:
            self.handle = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_TERMINATE,
                                               False, pid)

    def valid(self) -> bool:
        return self.popen is not None or bool(self.handle)

    def poll(self) -> int | None:
        if self.popen is not None:
            return self.popen.poll()
        if not self.handle:
            return -1
        code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            return -1
        return None if code.value == STILL_ACTIVE else int(code.value)

    def terminate(self) -> None:
        if self.popen is not None:
            self.popen.terminate()
        elif self.handle:
            kernel32.TerminateProcess(self.handle, 1)

    def close(self) -> None:
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None


# --------------------------------------------------------------------------
# puzzle facts (fat or not, README bounds) and promotion
# --------------------------------------------------------------------------

def is_game_accepted(puzzle: str) -> bool:
    """A placement the game also accepts ("4-4@r2c3"), not the official goal."""
    return "@" in puzzle


def puzzle_is_fat(puzzle: str) -> bool:
    text = (REPO / "src" / "chuzzles" / "levels.hpp").read_text(errors="replace")
    m = re.search(r'\{"' + re.escape(puzzle.split("@")[0]) + r'"\s*,[^}]*\}\s*,\s*(\{[^}]*\})', text)
    return bool(m and "true" in m.group(1))


def readme_bound(puzzle: str) -> int | None:
    try:
        text = BUILD_README.read_text(errors="replace")
    except OSError:
        return None
    m = re.search(r"^\s*" + re.escape(puzzle) + r":\s*at least\s+(\d+)\s+moves", text, re.M)
    return int(m.group(1)) if m else None


def _edit_text_file(path: Path, edit) -> bool:
    raw = path.read_bytes().decode()
    nl = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    new = edit(text)
    if new is None or new == text:
        return False
    path.write_bytes(new.replace("\n", nl).encode())
    return True


def promote_solved(puzzle: str, solution_file: Path) -> list[str]:
    notes = []
    if is_game_accepted(puzzle):
        # not the official goal: its own folder and README section; the unsolved count and
        # config.json stay as they are
        GAME_ACCEPTS.mkdir(parents=True, exist_ok=True)
        shutil.copy2(solution_file, GAME_ACCEPTS / solution_file.name)
        notes.append(f"copied {solution_file.name} to build/levels_final/game_accepts")

        def readme_alt(text: str):
            line_re = re.compile(r"^(\s*" + re.escape(puzzle) + r":\s*)at least\s+\d+\s+moves", re.M)
            if not line_re.search(text):
                return None
            moves = re.search(r"_c(\d+)_(\d+)\.txt$", solution_file.name)
            what = f"solved, {moves.group(1)} moves, {moves.group(2)} solutions" if moves else "solved"
            return line_re.sub(lambda m: m.group(1) + what, text, count=1)

        if _edit_text_file(BUILD_README, readme_alt):
            notes.append("marked solved in build/README.md's game-accepted list")
        return notes
    LEVELS_FINAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solution_file, LEVELS_FINAL / solution_file.name)
    notes.append(f"copied {solution_file.name} to build/levels_final")

    def readme(text: str):
        line_re = re.compile(r"^\s*" + re.escape(puzzle) + r":\s*at least\s+\d+\s+moves[^\n]*\n", re.M)
        if not line_re.search(text):
            return None
        text = line_re.sub("", text, count=1)
        m = re.search(r"^# (\d+) puzzles remain unsolved:", text, re.M)
        if m:
            text = text[:m.start(1)] + str(int(m.group(1)) - 1) + text[m.end(1):]
        return text

    if _edit_text_file(BUILD_README, readme):
        notes.append("removed from build/README.md unsolved list")

    def config(text: str):
        new, n = re.subn(r'\s*"' + re.escape(puzzle) + r'",', "", text, count=1)
        if n == 0:
            new, n = re.subn(r',\s*"' + re.escape(puzzle) + r'"(\s*\])', r"\1", text, count=1)
        return new if n else None

    if _edit_text_file(BUILD_CONFIG, config):
        notes.append("removed from build/config.json unsolvedPuzzles")
    return notes


def promote_bound(puzzle: str, searched_moves: int) -> list[str]:
    """No solution at `searched_moves`: the bound rises only if every shorter
    length was already excluded (README bound == searched_moves)."""
    current = readme_bound(puzzle)
    if current is None:
        return [f"no README bound line for {puzzle}; left unchanged"]
    if current != searched_moves:
        return [f"README says at least {current}; searched {searched_moves}, so the bound stays"]

    def readme(text: str):
        return re.sub(r"^(\s*" + re.escape(puzzle) + r":\s*at least\s+)\d+(\s+moves)",
                      lambda m: f"{m.group(1)}{searched_moves + 1}{m.group(2)}", text, count=1, flags=re.M)

    _edit_text_file(BUILD_README, readme)
    return [f"build/README.md: {puzzle} now at least {searched_moves + 1} moves"]


def solved_count() -> int:
    names = {m.group(1) for f in LEVELS_FINAL.glob("*.txt")
             if (m := re.match(r"^(\d+-\d+)_c\d+_\d+\.txt$", f.name)) and int(m.group(1).split("-")[0]) <= 20}
    return len(names)


def replay_check(puzzle: str, path: Path, moves: int) -> tuple[bool, str]:
    """Replays every solution with Chuzzle Suite's board simulator."""
    sys.path.insert(0, str(REPO / "python" / "chuzzle_suite"))
    try:
        from chuzzle_suite import board_sim, levels as lv
        level = lv.find_level(lv.load_levels(), puzzle)
        lines = [s.strip() for s in path.read_text().splitlines() if s.strip()]
        good = 0
        for s in lines:
            end = board_sim.play(board_sim.BoardState(list(level.start), level.start_fat), s)[-1]
            good += end.cells == level.goal and end.fat == level.goal_fat and len(s.split()) == moves
        return good == len(lines) and good > 0, f"replay {good}/{len(lines)}"
    except Exception as exc:
        return False, f"replay failed: {exc}"


def test_known_check(puzzle: str, moves: int) -> tuple[bool, str]:
    if not TEST_KNOWN_EXE.exists():
        return True, "testKnownSolutions not built (skipped)"
    try:
        out = subprocess.run([str(TEST_KNOWN_EXE)], cwd=TEST_KNOWN_EXE.parent, capture_output=True,
                             text=True, timeout=1800, creationflags=subprocess.CREATE_NO_WINDOW).stdout
    except Exception as exc:
        return False, f"testKnownSolutions failed: {exc}"
    m = re.search(r"\[\s*" + re.escape(puzzle) + r"\s*/\s*c" + str(moves) + r"\]:\s*(\d+)/(\d+)", out)
    if not m:
        return False, "testKnownSolutions: no line for this file"
    a, b = int(m.group(1)), int(m.group(2))
    return a == b and b > 0, f"testKnownSolutions {a}/{b}"


# --------------------------------------------------------------------------
# the scheduler
# --------------------------------------------------------------------------

class Scheduler:
    def __init__(self):
        STATE.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        data = read_json(JOBS_FILE, {})
        self.jobs: list[dict] = data.get("jobs", [])
        self.tables: dict = data.get("tables", {})     # puzzle -> {"created_by": job id}
        self.paused: bool = data.get("paused", False)
        self.next_id: int = data.get("next_id", 1)
        self.proc: ProcessHandle | None = None
        self.external: list[tuple[str, int]] = []
        self.last_seen_phase: dict | None = None
        self.alarm = alarm.Alarm(STATE / "alarm-trophy.wav")
        self.alarm.enabled = data.get("alarm_enabled", True)
        # rented machines running shards of a search (fleet.py, `mbq fleet`)
        self.fleet = fleet.Fleet(STATE, REPO, lambda p, m, n: SHARD_ROOT / shard_group(p, m, n),
                                 read_middles_header)
        self._readopt()

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        with self.lock:
            write_json_atomic(JOBS_FILE, {"jobs": self.jobs, "tables": self.tables,
                                          "paused": self.paused, "next_id": self.next_id,
                                          "alarm_enabled": self.alarm.enabled})

    def job(self, job_id: int) -> dict | None:
        return next((j for j in self.jobs if j["id"] == job_id), None)

    def running(self) -> dict | None:
        return next((j for j in self.jobs if j["status"] == "running"), None)

    def finalizing(self) -> dict | None:
        return next((j for j in self.jobs if j["status"] == "finalizing"), None)

    def pending(self) -> list[dict]:
        return [j for j in self.jobs if j["status"] in ("queued", "waiting")]

    def _readopt(self) -> None:
        fin = self.finalizing()
        if fin is not None:
            fin["status"] = "running"      # its process is gone: tick() finalizes it again
        job = self.running()
        if job is None:
            return
        handle = ProcessHandle(job.get("pid") or 0)
        if handle.valid() and handle.poll() is None:
            self.proc = handle
            job["message"] = "re-adopted after a server restart"
        elif fin is job:
            pass                          # finished before the restart; tick() re-finalizes
        else:
            handle.close()
            job["status"] = "interrupted"
            job["finished"] = now()
            job["message"] = "solver was no longer running when the server restarted"
            # e.g. the PC rebooted mid-run: run the same job again, first
            again = self.add({"puzzle": job["puzzle"], "moves": job["moves"], "until": job.get("until"),
                              "engine": job["engine"], "ram_gb": job.get("ram_gb"), "threads": job.get("threads"),
                              "promote": job["promote"], "keep_table": job["keep_table"], "front": True,
                              "table_depth": job.get("table_depth"), "stream_depth": job.get("stream_depth"),
                              "gpu": job.get("gpu"), "block": job.get("block"), "parent": job["id"]})
            job["notes"].append(f"re-queued as #{again['id']}")
        self.save()

    # ---- jobs ------------------------------------------------------------
    def add(self, spec: dict) -> dict:
        puzzle = spec["puzzle"]
        moves = int(spec["moves"])
        engine = spec.get("engine") or "auto"
        if engine == "auto":
            engine = "inram" if moves <= 12 else "ssd"
        if not re.fullmatch(PUZZLE_NAME_RE, puzzle):
            raise ValueError(f"bad puzzle name {puzzle!r}")
        if engine == "ssd" and not 10 <= moves <= 18:
            raise ValueError("ssd engine supports 10-18 moves")
        if engine == "inram" and not 3 <= moves <= 12:
            raise ValueError("inram engine supports 3-12 moves")
        # Splits <seed, stream, table>: 2x2 boards use the depth-7 goal table from 14 moves (it
        # fits a few RAM passes, and the bound prunes nothing there); normal boards keep the
        # one-pass depth-6 table and lengthen the bound-pruned left stream instead: 4 moves at
        # 15 (<5,4,6>), 5 at 16 (<5,5,6>; a normal board's depth-7 set is far beyond RAM),
        # 6 at 17 (<5,6,6>, on the GPU only). 2x2 boards at 17: <5,5,7>.
        fat = puzzle_is_fat(puzzle)
        table_depth = int(spec.get("table_depth") or (7 if fat and moves >= 14 else 6))
        gpu = spec.get("gpu") or None
        # The free-block join (--block, docs/changes-2026-11-04.md section 15): the left side one
        # move shorter, joined to the goal side by row and column class. CUDA, a depth-7 goal side
        # on 2x2 boards and depth 6 on normal boards. Default: on with --gpu from 16 moves (2x2)
        # and 15 (normal); "block": True / False forces it.
        block = spec.get("block")
        if block is None:
            block = bool(gpu) and engine != "inram" and moves >= (16 if fat else 15) \
                    and not str(gpu).lower().startswith(("vulkan:", "vk:"))
        block = bool(block) and engine != "inram" and table_depth == (7 if fat else 6)
        left = moves - table_depth - (1 if block else 0)
        stream_depth = int(spec.get("stream_depth") or (6 if left >= 11 else 5 if left >= 10
                                                          else 4 if left >= 9 else 3))
        seed_depth = left - stream_depth
        if engine == "ssd" and (table_depth not in (6, 7) or stream_depth not in (3, 4, 5, 6) or not 2 <= seed_depth <= 5):
            raise ValueError(f"no <S,{stream_depth},{table_depth}> split for {moves} moves")
        if engine == "ssd" and stream_depth == 6 and (not gpu or fat):
            raise ValueError("a 6-move stream (17 moves, <5,6,6>) runs on the GPU and normal boards only: add --gpu 5060")
        if gpu and (engine != "ssd" or table_depth != 6 and not (fat and table_depth == 7)):
            raise ValueError("--gpu: ssd engine; a depth-6 goal side, or depth 7 on 2x2 boards (src/gpu/README.md)")
        with self.lock:
            job = {
                "id": self.next_id, "puzzle": puzzle, "moves": moves, "until": spec.get("until"),
                "engine": engine, "table_depth": table_depth if engine == "ssd" else None,
                "seed_depth": seed_depth if engine == "ssd" else None,
                "stream_depth": stream_depth if engine == "ssd" else None,
                "ram_gb": spec.get("ram_gb"), "threads": spec.get("threads"), "gpu": gpu,
                "promote": spec.get("promote", True),
                "keep_table": spec.get("keep_table", False), "fat": puzzle_is_fat(puzzle),
                "status": "queued", "wait_reason": "", "created": now(), "started": None,
                "finished": None, "pid": None, "exit_code": None, "solutions": None,
                "result": None, "message": "", "notes": [], "phases": {}, "parent": spec.get("parent"),
                "shard": spec.get("shard"), "merge": spec.get("merge"),
                "d7_groups": spec.get("d7_groups"),
                "block": block if engine == "ssd" else False,
            }
            self.next_id += 1
            if spec.get("front"):
                first_pending = next((i for i, j in enumerate(self.jobs) if j["status"] in ("queued", "waiting")),
                                     len(self.jobs))
                self.jobs.insert(first_pending, job)
            else:
                self.jobs.append(job)
            self.save()
            return job

    def add_sharded(self, spec: dict) -> list[dict]:
        """One search as `shards` work units: local shard jobs (all but `remote`), then a merge
        job that waits for every shard's middles file and recovers the solutions."""
        n = int(spec["shards"])
        remote = set(int(x) for x in spec.get("remote") or [])
        if n < 2 or any(k < 0 or k >= n for k in remote):
            raise ValueError("--shards needs N >= 2 and remote shards in 0..N-1")
        g = int(spec.get("d7_groups") or 1)
        if g > 1 and n % g != 0:
            raise ValueError("--d7-groups G needs --shards N with N a multiple of G")
        base = {k: v for k, v in spec.items() if k not in ("shards", "remote", "front")}
        specs = [{**base, "shard": [k, n], "promote": False, "until": None} for k in range(n) if k not in remote]
        specs.append({**base, "merge": n})
        if spec.get("front"):
            # each insert goes first, so add in reverse to keep shards 0..N-1 then the merge
            jobs = [self.add({**s, "front": True}) for s in reversed(specs)][::-1]
        else:
            jobs = [self.add(s) for s in specs]
        merge = jobs[-1]
        group = shard_group(merge["puzzle"], merge["moves"], n)
        (SHARD_ROOT / group).mkdir(parents=True, exist_ok=True)
        if remote:
            merge["notes"].append(f"remote shards {sorted(remote)}: `mbq export {merge['puzzle']} --moves "
                                  f"{merge['moves']} --shards {n} --which {','.join(map(str, sorted(remote)))}`")
        return jobs

    def add_fleet_search(self, spec: dict) -> dict:
        """`mbq fleet run`: a merge job here (every shard remote) + the fleet's units for it."""
        n = int(spec["units"])
        groups = int(spec.get("d7_groups") or 1)
        fat = puzzle_is_fat(spec["puzzle"])
        direction = spec.get("direction") or ("forward" if fat else None)
        if direction not in ("forward", "reverse"):
            raise ValueError("normal boards: --direction forward or reverse (decide it once, e.g. from a "
                             "--direction auto run's first lines on this PC)")
        jobs = self.add_sharded({"puzzle": spec["puzzle"], "moves": spec["moves"], "shards": n, "engine": "ssd",
                                 "remote": list(range(n)), "d7_groups": groups if groups > 1 else None,
                                 "gpu": spec.get("gpu") or "5060", "threads": spec.get("threads") or 6,
                                 "block": spec.get("block"), "promote": not spec.get("no_promote"),
                                 "until": None})
        merge = jobs[-1]
        s = self.fleet.add_search(merge, n, groups, int(spec.get("checks", 2)), direction, bool(spec.get("front")),
                                  [h.strip() for h in str(spec.get("only") or "").split(",") if h.strip()])
        merge["notes"].append(f"fleet search {s['id']}: {n} units on rented machines (mbq fleet status)")
        self.save()
        return {"merge_job": merge, "search": next(x for x in self.fleet.status()["searches"] if x["id"] == s["id"])}

    def shard_files_missing(self, job: dict) -> list[int]:
        n = int(job["merge"])
        d = SHARD_ROOT / shard_group(job["puzzle"], job["moves"], n)
        return [k for k in range(n) if not (d / f"s{k}.mid").exists()]

    def cancel(self, job_id: int) -> dict:
        with self.lock:
            job = self.job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "running" and self.proc is not None:
                self.proc.terminate()       # works on a held (suspended) process too
                job["held_since"] = None
                job["message"] = "cancelled while running"
            elif job["status"] in ("queued", "waiting"):
                job["message"] = "cancelled before starting"
            else:
                return job
            job["status"] = "cancelled"
            job["finished"] = now()
            self.save()
        threading.Thread(target=self._record, args=(job,), daemon=True).start()
        return job

    def hold(self, on: bool) -> dict:
        """Freeze (on) or thaw the running solver. Its threads stop where they are; its RAM,
        progress and files stay as they are, so nothing is redone. Survives a server restart
        (the flag is saved and the process stays suspended), not a reboot."""
        with self.lock:
            job = self.running()
            if job is None:
                raise ValueError("nothing is running")
            if bool(job.get("held_since")) == on:
                return job
            handle = kernel32.OpenProcess(PROCESS_SUSPEND_RESUME, False, job["pid"])
            if not handle:
                raise ValueError(f"cannot open the solver (pid {job['pid']})")
            try:
                rc = (ntdll.NtSuspendProcess if on else ntdll.NtResumeProcess)(handle)
            finally:
                kernel32.CloseHandle(handle)
            if rc != 0:
                raise ValueError(f"{'hold' if on else 'release'} failed (NTSTATUS {rc & 0xFFFFFFFF:#x})")
            prog = self._progress(job) or {}
            if on:
                job["held_since"] = now()
                job["held_at"] = {"phase": prog.get("phase"), "elapsed": prog.get("elapsed", 0.0)}
            else:
                secs = now() - job["held_since"]
                job.setdefault("holds", []).append(dict(job.get("held_at") or {}, seconds=secs))
                job["held_since"] = None
                job["notes"].append(f"held {fmt_dur(secs)} at {fmt_time(now() - secs)}")
            self.save()
            return job

    def held_seconds(self, job: dict) -> float:
        total = sum(h["seconds"] for h in job.get("holds", []))
        return total + (now() - job["held_since"] if job.get("held_since") else 0.0)

    # ---- worker ----------------------------------------------------------
    def loop(self) -> None:
        while True:
            try:
                self.fleet.tick()
            except Exception as exc:
                self.fleet.event(f"fleet error: {exc}")
            try:
                self.tick()
            except Exception as exc:  # keep serving; record the problem
                with self.lock:
                    job = self.running()
                    if job is not None:
                        job["message"] = f"scheduler error: {exc}"
                    self.save()
            time.sleep(POLL_S)

    def tick(self) -> None:
        # Slow work (process listing, validation) runs outside the lock so the
        # HTTP API always answers quickly.
        with self.lock:
            for j in self.pending():          # keep merge jobs' "waiting for shards" current
                if j.get("merge"):
                    missing = self.shard_files_missing(j)
                    reason = ("waiting for shard results " + ",".join(map(str, missing))) if missing else ""
                    if j["wait_reason"] != reason:
                        j["wait_reason"] = reason
                        j["status"] = "waiting" if missing else "queued"
            job = self.running()
            exited = job is not None and self._monitor(job)
        if job is not None:
            if exited:
                self._finalize(job)
                self._record(job)
                with self.lock:
                    self.save()
            return
        external = [(n, p) for n, p in running_processes() if SOLVER_PROCESS_RE.match(n)]
        with self.lock:
            # a cancelled job's process may still be exiting
            if self.proc is not None and self.proc.poll() is None:
                return
            if self.proc is not None:
                self.proc.close()
                self.proc = None
            if self.paused:
                return
            pending = self.pending()
            ready = []
            for j in pending:
                missing = self.shard_files_missing(j) if j.get("merge") else []
                if missing:
                    reason = "waiting for shard results " + ",".join(map(str, missing))
                    if j["wait_reason"] != reason or j["status"] != "waiting":
                        j["status"], j["wait_reason"] = "waiting", reason
                        self.save()
                else:
                    ready.append(j)
            if not ready:
                return
            job = ready[0]
            self.external = external
            if external and not IGNORE_EXTERNAL:
                job["status"], job["wait_reason"] = "waiting", "another solver is running: " + ", ".join(
                    f"{n} (pid {p})" for n, p in external)
                self.save()
                return
            need = self._disk_need_gb(job)
            if need:
                free = shutil.disk_usage(TABLE_ROOT.anchor or TABLE_ROOT).free / 1e9
                if free < need:
                    job["status"], job["wait_reason"] = "waiting", f"needs ~{need} GB free on {TABLE_ROOT}, has {free:.0f} GB"
                    self.save()
                    return
            self._start(job)

    def _table_dir(self, job: dict) -> Path:
        return TABLE_ROOT / f"{job['puzzle']}_goal_d6"

    def _disk_need_gb(self, job: dict) -> int:
        # The solver builds no table on disk any more (no-table mode, depth 6
        # and 7): it either reuses a complete one or keeps everything in RAM.
        return 0

    def _start(self, job: dict) -> None:
        jdir = JOB_ROOT / str(job["id"])
        jdir.mkdir(parents=True, exist_ok=True)
        BIN_DIR.mkdir(parents=True, exist_ok=True)
        log = jdir / "log.txt"
        progress = jdir / "progress.json"
        if job["engine"] == "ssd":
            exe = BIN_DIR / "mbq_ssd.exe"
            shutil.copy2(SSD_EXE, exe)          # a copy, so rebuilding build/ never clashes
            d = self._table_dir(job)
            job["needs"] = {"d6-build": not (d / "sorted_000.bin").exists(),
                            "d6-index": not (d / "fp16meta.txt").exists(),
                            "d7-index": job["table_depth"] == 7 and not (d / "d7_meta.txt").exists()}
            if not d.exists():
                self.tables[job["puzzle"]] = {"created_by": job["id"]}
            cmd = [str(exe), job["puzzle"], "--out", str(TABLE_ROOT), "--seed-depth", str(job["seed_depth"]),
                   "--progress-file", str(progress), "--progress-interval", "5"]
            if job["table_depth"] == 7:
                cmd += ["--table-depth", "7"]
            if job.get("stream_depth", 3) != 3:
                cmd += ["--stream-depth", str(job["stream_depth"])]
            if job.get("ram_gb"):
                cmd += ["--ram-gb", str(job["ram_gb"])]
            if job.get("threads"):
                cmd += ["--threads", str(job["threads"])]
            if job.get("gpu"):
                for dll in (GPU_DLL, VK_DLL):   # the solver loads them from next to the exe
                    if dll.exists():
                        shutil.copy2(dll, BIN_DIR / dll.name)
                cmd += ["--gpu", str(job["gpu"])]
            if job.get("block"):
                cmd += ["--block"]
            if job.get("shard"):
                k, n = job["shard"]
                out = SHARD_ROOT / shard_group(job["puzzle"], job["moves"], n)
                out.mkdir(parents=True, exist_ok=True)
                cmd += ["--shard", f"{k}/{n}", "--middles-out", str(out / f"s{k}.mid"), "--no-recover"]
                if job.get("d7_groups") and int(job["d7_groups"]) > 1:
                    cmd += ["--d7-groups", str(job["d7_groups"])]
            elif not job.get("merge"):
                # keep the middles: if recovery fails, `--recover-from` redoes only that part
                cmd += ["--middles-out", str(jdir / "middles.mid")]
            if job.get("merge"):
                d = SHARD_ROOT / shard_group(job["puzzle"], job["moves"], int(job["merge"]))
                for k in range(int(job["merge"])):
                    cmd += ["--recover-from", str(d / f"s{k}.mid")]
            cwd = jdir
        else:
            exe = BIN_DIR / "mbq_inram.exe"
            shutil.copy2(INRAM_EXE, exe)
            write_json_atomic(jdir / "config.json", {
                "outDirectory": str(REPO), "searchDirection": "Auto", "debug": False,
                "puzzle": job["puzzle"], "estimatedDepth": job["moves"],
                "threads": job.get("threads") or os.cpu_count() or 8})
            cmd = [str(exe)]
            cwd = jdir
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        with open(log, "wb") as out:
            popen = subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)
        self.proc = ProcessHandle(popen.pid, popen)
        job.update(status="running", wait_reason="", started=now(), pid=popen.pid, cmd=cmd,
                   machine=run_history.machine_label(), solver_commit=run_history.git_commit())
        self.last_seen_phase = None
        self.save()

    def _progress(self, job: dict) -> dict | None:
        return read_json(JOB_ROOT / str(job["id"]) / "progress.json", None)

    def _monitor(self, job: dict) -> bool:
        prog = self._progress(job)
        if prog:
            # accumulate time per phase name (a phase can restart, e.g. d6-build's steps)
            last = self.last_seen_phase
            if last and (last["phase"] != prog["phase"] or prog["phase_seconds"] < last["phase_seconds"]):
                job["phases"][last["phase"]] = job["phases"].get(last["phase"], 0.0) + last["phase_seconds"]
            self.last_seen_phase = prog
        if self.proc is not None:
            code = self.proc.poll()
        else:   # re-finalizing after a restart: keep the recorded exit code
            code = job["exit_code"] if job.get("exit_code") is not None else -1
        if code is None:
            return False
        if self.last_seen_phase and self.last_seen_phase["phase"] != "done":
            p = self.last_seen_phase
            job["phases"][p["phase"]] = job["phases"].get(p["phase"], 0.0) + p["phase_seconds"]
        if self.proc is not None:
            self.proc.close()
            self.proc = None
        job["exit_code"] = code
        job["finished"] = now()
        job["status"] = "finalizing"   # validation / promotion run next, outside the lock
        return True

    def _finalize(self, job: dict) -> None:
        log = (JOB_ROOT / str(job["id"]) / "log.txt").read_text(errors="replace")
        puzzle, moves = job["puzzle"], job["moves"]
        if job.get("shard"):
            k, n = job["shard"]
            m = re.search(rf"SHARD {k}/{n} complete: (\d+) middles", log)
            f = SHARD_ROOT / shard_group(puzzle, moves, n) / f"s{k}.mid"
            if m and f.exists() and job["exit_code"] in (0, None):
                job.update(status="done", result="shard", message=f"shard {k}/{n}: {m.group(1)} middles")
            else:
                job.update(status="failed", result="error",
                           message=f"shard {k}/{n} did not complete (exit {job['exit_code']}); see `mbq log {job['id']}`")
            return
        solutions = None
        if job["engine"] == "ssd":
            m = re.search(r"RECOVERY: (\d+) solutions", log)
            if m:
                solutions = int(m.group(1))
            elif re.search(r"unique middle states \([^)]*\): 0\b", log) and "TOTAL:" in log:
                solutions = 0
        else:
            m = re.search(r"Final exact solutions: (\d+)", log)
            if m:
                solutions = int(m.group(1))
            elif "No solutions found" in log:
                solutions = 0
        if solutions is None or (job["exit_code"] not in (0, None) and solutions == 0):
            job.update(status="failed", result="error",
                       message=f"exit code {job['exit_code']}; see `mbq log {job['id']}`")
            return
        job["solutions"] = solutions
        notes = job["notes"]
        if solutions > 0:
            if job["engine"] == "ssd":
                src = self._table_dir(job) / "solutions" / "levels"
                for f in src.glob(f"{puzzle}_c{moves}_*.txt"):
                    shutil.copy2(f, LEVELS_DIR / f.name)
            files = sorted(LEVELS_DIR.glob(f"{puzzle}_c{moves}_*.txt"), key=lambda p: p.stat().st_mtime)
            if not files:
                job.update(status="failed", result="error", message="solved, but no solution file was found")
                return
            sol_file = files[-1]
            ok1, n1 = replay_check(puzzle, sol_file, moves)
            ok2, n2 = test_known_check(puzzle, moves)
            notes += [n1, n2]
            job["solution_file"] = str(sol_file)
            if not (ok1 and ok2):
                job.update(status="failed", result="unverified", message="solutions failed validation; not promoted")
                return
            job.update(status="done", result="solved", message=f"{solutions} solutions of {moves} moves")
            if job["promote"]:
                notes += promote_solved(puzzle, sol_file)
            self.alarm.start(puzzle, f"{puzzle} solved: {solutions} solutions of {moves} moves")
        else:
            job.update(status="done", result="none", message=f"no {moves}-move solution")
            if job["promote"]:
                notes += promote_bound(puzzle, moves)
            if job.get("until") and moves < int(job["until"]):
                nxt = self.add({"puzzle": puzzle, "moves": moves + 1, "until": job["until"],
                                "engine": "auto", "ram_gb": job.get("ram_gb"), "threads": job.get("threads"),
                                "gpu": job.get("gpu"),
                                "promote": job["promote"],
                                "keep_table": job["keep_table"], "front": True, "parent": job["id"]})
                notes.append(f"queued {puzzle} at {moves + 1} moves as job {nxt['id']}")
        self._cleanup(job)

    def _record(self, job: dict) -> None:
        """Permanent history: a row in docs/run_history.csv + the log in docs/run_logs/."""
        try:
            run_history.record_job(job, JOB_ROOT / str(job["id"]) / "log.txt")
        except Exception as exc:
            job.setdefault("notes", []).append(f"run history not written: {exc}")

    def _cleanup(self, job: dict) -> None:
        if job["engine"] != "ssd" or job["keep_table"]:
            return
        puzzle = job["puzzle"]
        if "created_by" not in self.tables.get(puzzle, {}):
            return                      # not ours: never delete
        if any(j["puzzle"] == puzzle and j["engine"] == "ssd" for j in self.pending()):
            return                      # a queued job still needs it
        d = self._table_dir(job)
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            job["notes"].append(f"deleted table {d}")
        self.tables.pop(puzzle, None)

    # ---- reporting -------------------------------------------------------
    def _history_phase(self, job: dict, phase: str) -> float | None:
        """Median duration of `phase` in finished jobs of the same kind."""
        def kind(j, strict):
            base = (j["engine"], j.get("table_depth"), j.get("fat"))
            return base + ((j["moves"],) if strict else ())
        for strict in (True, False):
            vals = sorted(j["phases"][phase] for j in self.jobs
                          if j["status"] == "done" and phase in j.get("phases", {})
                          and kind(j, strict) == kind(job, strict))
            if vals:
                return vals[len(vals) // 2]
        return None

    def eta(self, job: dict) -> dict:
        prog = self._progress(job) or {}
        phase = prog.get("phase")
        done, total = prog.get("done", 0), prog.get("total", 0)
        out = {"phase": phase, "detail": prog.get("detail"), "done": done, "total": total,
               "percent": round(100.0 * done / total, 1) if total else None,
               "elapsed": now() - job["started"] - self.held_seconds(job) if job.get("started") else None,
               "held_since": job.get("held_since")}
        phase_left = None
        if total and done:
            # the solver's own clock kept running while it was held: take those seconds out
            # of this phase's time (holds that began after this phase started)
            start = prog.get("elapsed", 0.0) - prog["phase_seconds"]
            held = sum(h["seconds"] for h in job.get("holds", [])
                       if h.get("phase") == phase and h.get("elapsed", 0.0) >= start - 10)
            phase_left = max(prog["phase_seconds"] - held, 0.0) * (total - done) / done
        out["phase_remaining"] = phase_left
        if job["engine"] != "ssd" or phase not in SSD_PHASES:
            out["total_remaining"] = phase_left
            return out
        remaining, unknown = phase_left or 0.0, phase_left is None
        needs = job.get("needs", {})
        for later in SSD_PHASES[SSD_PHASES.index(phase) + 1:]:
            if later.startswith("d6") and not needs.get(later):
                continue
            if later.startswith("d7") and job["table_depth"] != 7:
                continue
            if later == "d7-index" and not needs.get("d7-index"):
                continue
            est = self._history_phase(job, later)
            if est is None:
                if later in ("probe", "d7-index"):
                    unknown = True
                continue
            remaining += est
        out["total_remaining"] = None if unknown else remaining
        out["finish_at"] = now() + out["total_remaining"] if out["total_remaining"] is not None else None
        return out

    def _duration_estimate(self, job: dict) -> float | None:
        """Median wall time of finished jobs of the same kind."""
        def kind(j):
            return (j["engine"], j.get("table_depth"), j.get("fat"), j["moves"])
        vals = sorted(j["finished"] - j["started"] for j in self.jobs
                      if j["status"] == "done" and j.get("started") and j.get("finished") and kind(j) == kind(job))
        return vals[len(vals) // 2] if vals else None

    def queue_estimate(self) -> dict:
        known, unknown = 0.0, 0
        for j in self.pending():
            est = self._duration_estimate(j)
            if est is None:
                unknown += 1
            else:
                known += est
        return {"known_seconds": known if unknown < len(self.pending()) or not self.pending() else None,
                "unknown_jobs": unknown}

    def status(self) -> dict:
        procs = running_processes()
        with self.lock:
            job = self.running()
            external = [(n, p) for n, p in procs if SOLVER_PROCESS_RE.match(n)
                        and (job is None or p != job.get("pid"))]
            fin = self.finalizing()
            return {"paused": self.paused, "running": dict(job, eta=self.eta(job)) if job else None,
                    "finalizing": fin,
                    "pending": self.pending(), "external_solvers": external,
                    "solved": solved_count(), "queue_estimate": self.queue_estimate(),
                    "alarm": self.alarm.state(),
                    "finished_today": sum(1 for j in self.jobs if j.get("finished")
                                          and time.localtime(j["finished"])[:3] == time.localtime()[:3]
                                          and j["status"] == "done"),
                    "table_root_free_gb": round(shutil.disk_usage(TABLE_ROOT.anchor or TABLE_ROOT).free / 1e9, 1)}


# --------------------------------------------------------------------------
# HTTP API (127.0.0.1 only)
# --------------------------------------------------------------------------

def make_handler(sched: Scheduler):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body) -> None:
            data = (json.dumps(body, indent=1) if not isinstance(body, str) else body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json" if not isinstance(body, str) else "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def _host_ok(self) -> bool:
            # Only answer requests addressed to this machine (blocks DNS rebinding).
            if self.headers.get("Host") in ALLOWED_HOSTS:
                return True
            self._send(403, {"error": "bad host"})
            return False

        def do_GET(self):
            if not self._host_ok():
                return None
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            if not parts:
                data = DASHBOARD.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return None
            query = dict(q.split("=", 1) for q in self.path.split("?", 1)[1].split("&")) if "?" in self.path else {}
            try:
                if parts == ["status"]:
                    return self._send(200, sched.status())
                if parts == ["fleet"]:
                    return self._send(200, sched.fleet.status())
                if parts == ["alarm.wav"]:            # the alarm sound, for the dashboard tab
                    data = sched.alarm.ensure_sound().read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return None
                if parts == ["jobs"]:
                    return self._send(200, sched.jobs)
                if len(parts) == 2 and parts[0] == "jobs":
                    job = sched.job(int(parts[1]))
                    return self._send(200, job) if job else self._send(404, {"error": "no such job"})
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "log":
                    log = JOB_ROOT / parts[1] / "log.txt"
                    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
                    tail = int(query.get("tail", 40))
                    return self._send(200, "\n".join(lines[-tail:]) + "\n")
                return self._send(404, {"error": "unknown path"})
            except Exception as exc:
                return self._send(500, {"error": str(exc)})

        def do_POST(self):
            if not self._host_ok():
                return None
            # Commands need a custom header: a page on another site cannot send
            # one without a CORS preflight, which this server never approves.
            if self.headers.get("X-MBQ") != "1":
                return self._send(403, {"error": "missing X-MBQ header"})
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            try:
                if parts == ["jobs"]:
                    body = self._body()
                    return self._send(200, sched.add_sharded(body) if body.get("shards") else sched.add(body))
                if parts == ["fleet", "add"]:
                    b = self._body()
                    return self._send(200, sched.fleet.add_host(b["name"], b["target"], int(b.get("port") or 22),
                                                                b.get("price")))
                if parts == ["fleet", "remove"]:
                    b = self._body()
                    return self._send(200, sched.fleet.remove_host(b["name"], bool(b.get("now"))))
                if parts == ["fleet", "run"]:
                    return self._send(200, sched.add_fleet_search(self._body()))
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                    return self._send(200, sched.cancel(int(parts[1])))
                if parts == ["shutdown"]:
                    self._send(200, {"stopping": True, "note": "a running solver keeps running and is re-adopted"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return None
                if parts in (["hold"], ["unhold"]):
                    return self._send(200, sched.hold(parts == ["hold"]))
                if len(parts) == 2 and parts[0] == "alarm":
                    if parts[1] == "stop":
                        return self._send(200, sched.alarm.stop())
                    if parts[1] == "test":
                        return self._send(200, sched.alarm.start("test", "Alarm test (stops by itself in 20 s)", test=True))
                    if parts[1] in ("enable", "disable"):
                        state = sched.alarm.set_enabled(parts[1] == "enable")
                        sched.save()
                        return self._send(200, state)
                if parts in (["pause"], ["resume"]):
                    with sched.lock:
                        sched.paused = parts == ["pause"]
                        sched.save()
                    return self._send(200, {"paused": sched.paused})
                return self._send(404, {"error": "unknown path"})
            except (ValueError, KeyError) as exc:
                return self._send(400, {"error": str(exc)})
            except Exception as exc:
                return self._send(500, {"error": str(exc)})
    return Handler


def serve() -> None:
    sched = Scheduler()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(sched))
    except OSError:
        print(f"mbq: port {PORT} is busy (server already running?)")
        sys.exit(1)
    threading.Thread(target=sched.loop, daemon=True).start()
    print(f"mbq: serving on http://127.0.0.1:{PORT} (state in {STATE})")
    server.serve_forever()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def api(method: str, path: str, body: dict | None = None, raw: bool = False):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json", "X-MBQ": "1"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read().decode()
    except urllib.error.HTTPError as e:
        data = e.read().decode()
        raise SystemExit(f"mbq: {json.loads(data).get('error', data)}")
    except urllib.error.URLError:
        raise SystemExit("mbq: server not running. Start it with: python python/scheduler/mbq.py daemon")
    return data if raw else json.loads(data)


def describe(job: dict) -> str:
    kind = f"{job['engine']}" + (f" <{job['seed_depth']},{job.get('stream_depth') or 3},{job['table_depth']}>"
                                  if job["engine"] == "ssd" else "")
    if job.get("shard"):
        kind += f" shard {job['shard'][0]}/{job['shard'][1]}"
    if job.get("merge"):
        kind += f" merge of {job['merge']} shards"
    if job.get("gpu"):
        kind += f" gpu {job['gpu']}"
    if job.get("block"):
        kind += " +block"
    until = f" (until {job['until']})" if job.get("until") else ""
    return f"#{job['id']:<3} {job['puzzle']:>5} {job['moves']} moves{until}  {kind}  [{job['status']}]"


def cmd_status(_args) -> None:
    s = api("GET", "/status")
    if s["paused"]:
        print("queue is PAUSED (mbq resume)")
    job = s["running"]
    if job and job.get("held_since"):
        print(f"SOLVER ON HOLD since {time.strftime('%H:%M', time.localtime(job['held_since']))} (mbq unhold)")
    if s.get("finalizing"):
        print(describe(s["finalizing"]) + "  validating results...")
    if job:
        e = job["eta"]
        print(describe(job))
        pct = f"{e['percent']}%" if e.get("percent") is not None else "-"
        print(f"  phase: {e.get('phase')} {e.get('detail') or ''}  {pct}")
        print(f"  elapsed {fmt_dur(e.get('elapsed'))}; this phase ends in {fmt_dur(e.get('phase_remaining'))}")
        if e.get("total_remaining") is not None:
            print(f"  whole job: about {fmt_dur(e['total_remaining'])} left, "
                  f"done around {time.strftime('%H:%M', time.localtime(e['finish_at']))}")
        else:
            print("  whole job: unknown until a later phase starts (no history for this kind of run yet)")
    else:
        print("nothing running")
    for n, p in s["external_solvers"]:
        print(f"  other solver running (not managed): {n} pid {p}")
    if s["pending"]:
        print(f"{len(s['pending'])} queued:")
        for j in s["pending"]:
            print("  " + describe(j) + (f"  - {j['wait_reason']}" if j.get("wait_reason") else ""))
    print(f"free on {TABLE_ROOT}: {s['table_root_free_gb']} GB")


def cmd_queue(_args) -> None:
    for j in api("GET", "/jobs"):
        if j["status"] in ("running", "queued", "waiting"):
            print(describe(j) + (f"  - {j['wait_reason']}" if j.get("wait_reason") else ""))


def cmd_history(args) -> None:
    done = [j for j in api("GET", "/jobs") if j["status"] not in ("running", "queued", "waiting")]
    for j in done[-args.n:]:
        took = fmt_dur(j["finished"] - j["started"]) if j.get("started") and j.get("finished") else "-"
        print(f"{describe(j)}  {fmt_time(j.get('finished'))}  took {took}  {j.get('message', '')}")
        for note in j.get("notes", []):
            print(f"      {note}")


def cmd_add(args) -> None:
    if args.shards:
        for puzzle in args.puzzles:
            jobs = api("POST", "/jobs", {"puzzle": puzzle, "moves": args.moves, "engine": args.engine,
                                         "ram_gb": args.ram_gb, "threads": args.threads, "gpu": args.gpu,
                                         "d7_groups": args.d7_groups, "block": args.block,
                                         "promote": not args.no_promote, "keep_table": args.keep_table,
                                         "table_depth": args.table_depth, "stream_depth": args.stream_depth,
                                         "until": args.until, "shards": args.shards,
                                         "remote": [int(x) for x in args.remote.split(",")] if args.remote else []})
            for job in jobs:
                print("queued " + describe(job))
        return
    for puzzle in args.puzzles:
        job = api("POST", "/jobs", {"puzzle": puzzle, "moves": args.moves, "until": args.until,
                                    "engine": args.engine, "ram_gb": args.ram_gb, "threads": args.threads,
                                    "gpu": args.gpu, "block": args.block,
                                    "promote": not args.no_promote,
                                    "keep_table": args.keep_table, "front": args.front,
                                    "table_depth": args.table_depth, "stream_depth": args.stream_depth})
        print("queued " + describe(job))


def cmd_export(args) -> None:
    """A folder for another PC: the portable solver + one .bat per shard (no install, no Python)."""
    if not PORTABLE_EXE.exists():
        raise SystemExit(f"no portable build at {PORTABLE_EXE}; build it with -DMB_MARCH=x86-64-v3 "
                         f"-DMB_OUTPUT_DIR={PORTABLE_EXE.parent}")
    fat = puzzle_is_fat(args.puzzle)
    table = args.table_depth or (7 if fat and args.moves >= 14 else 6)
    stream = args.stream_depth or (5 if args.moves - table >= 10 else 4 if args.moves - table >= 9 else 3)
    seed = args.moves - stream - table
    if args.moves - table >= 11:
        raise SystemExit(f"{args.moves} moves need a 6-move stream, which runs on the GPU only (export has no GPU)")
    if not 2 <= seed <= 5:
        raise SystemExit(f"no <S,{stream},{table}> split for {args.moves} moves")
    group = shard_group(args.puzzle, args.moves, args.shards)
    out = Path(args.out) / group
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PORTABLE_EXE, out / "experimentDiskDepth6.exe")
    which = [int(x) for x in args.which.split(",")]
    for k in which:
        name = f"{args.puzzle}_m{args.moves}_s{k}of{args.shards}"
        extra = ((" --table-depth 7" if table == 7 else "") + (f" --stream-depth {stream}" if stream != 3 else "")
                 + (f" --d7-groups {args.d7_groups}" if args.d7_groups and args.d7_groups > 1 else ""))
        bat = (f'@echo off\r\ncd /d "%~dp0"\r\n'
               f'echo Mindbender work unit {name}: started %date% %time%\r\n'
               f'echo Leave this window open. It writes {name}.mid when it is done.\r\n'
               f'"%~dp0experimentDiskDepth6.exe" {args.puzzle} --out "%~dp0work" --seed-depth {seed}{extra} '
               f'--shard {k}/{args.shards} --middles-out "{name}.mid" --no-recover > "{name}.log" 2>&1\r\n'
               f'if exist "{name}.mid" (echo DONE: send back {name}.mid and {name}.log) '
               f'else (echo FAILED: send back {name}.log)\r\npause\r\n')
        (out / f"run_{name}.bat").write_bytes(bat.encode())
    readme = (f"Mindbender solver work units for puzzle {args.puzzle}, {args.moves} moves "
              f"(shards {args.which} of {args.shards}).\r\n\r\n"
              "1. Plug the laptop in and set sleep to Never while plugged in "
              "(Settings > System > Power & sleep).\r\n"
              "2. Double-click one run_*.bat. Leave its window open; it uses every CPU core.\r\n"
              "3. When it says DONE, send back the .mid file and the .log file from this folder.\r\n"
              "   If it says FAILED, send back the .log file.\r\n"
              "Nothing is installed and nothing is sent anywhere; the program only reads and writes "
              "files in this folder (a work subfolder is created for its temporary output).\r\n")
    (out / "README.txt").write_bytes(readme.encode())
    print(f"exported {len(which)} work unit(s) to {out}")


def cmd_import(args) -> None:
    """Files returned middles from other PCs where the merge job looks for them."""
    for f in args.files:
        src = Path(f)
        h = read_middles_header(src)
        dst = SHARD_ROOT / shard_group(h["puzzle"], h["moves"], h["shards"]) / f"s{h['shard']}.mid"
        if dst.exists() and dst.read_bytes() != src.read_bytes():
            raise SystemExit(f"{dst} already exists with different content; not overwritten")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"imported {src.name}: {h['puzzle']} {h['moves']} moves, shard {h['shard']}/{h['shards']}, "
              f"{h['count']} middles -> {dst}")


def cmd_fleet(args) -> None:
    """Rented machines (fleet.py): key, add, remove, run, status."""
    if args.action == "key":
        pub = fleet.make_key()
        print("Add this public key to the rental site (vast.ai: Account > SSH Keys; RunPod: Settings >\n"
              "SSH Public Keys) BEFORE renting, so new machines accept it:\n")
        print(pub)
        return
    if args.action == "add":
        if not args.name or not args.target:
            raise SystemExit("mbq fleet add NAME user@host -p PORT [--price DOLLARS_PER_HOUR]")
        h = api("POST", "/fleet/add", {"name": args.name, "target": args.target, "port": args.port,
                                        "price": args.price})
        print(f"added {h['name']} ({h['target']} port {h['port']}); setup starts within a minute "
              f"(build + GPU self-test, ~5-10 min): mbq fleet status")
        return
    if args.action == "remove":
        if not args.name:
            raise SystemExit("mbq fleet remove NAME [--now]")
        h = api("POST", "/fleet/remove", {"name": args.name, "now": args.now})
        print(f"{h['name']}: {'removed' if h.get('removed') else 'leaves after its current unit'}")
        return
    if args.action == "run":
        if not args.name or not args.moves or not args.units:
            raise SystemExit("mbq fleet run PUZZLE --moves M --units N [--d7-groups G] [--direction D] [--checks C]")
        r = api("POST", "/fleet/run", {"puzzle": args.name, "moves": args.moves, "units": args.units,
                                        "d7_groups": args.d7_groups, "direction": args.direction,
                                        "checks": args.checks, "block": args.block, "front": args.front, "only": args.only,
                                        "no_promote": args.no_promote})
        print("queued " + describe(r["merge_job"]))
        print(f"fleet search {r['search']['id']}: {args.units} units; the machines pick them up within a minute")
        return
    s = api("GET", "/fleet")
    for h in s["hosts"]:
        if h.get("removed"):
            continue
        prog = h.get("progress") or {}
        pct = f" {prog.get('percent', 0):.0f}% of {prog.get('phase', '')}" if prog.get("percent") is not None else ""
        unit = f" unit {h['unit']} ({h['unit_minutes']} min{pct})" if h.get("unit") else ""
        print(f"{h['name']:12} {h['status']:14} {h.get('gpu') or '':38} {h.get('ram_gb') or '?':>4} GB  "
              f"done {h['units_done']:3}  ${h['cost']:.2f}{unit}"
              + (f"  [{h['last_error'][:60]}, seen {h['seen_ago_s']} s ago]" if h.get("last_error") else ""))
    for x in s["searches"]:
        if x.get("finished") and time.time() - x["finished"] > 86400:
            continue
        print(f"search {x['id']}: {x['puzzle']} {x['moves']} moves: {x['done']}/{x['n']} units done, "
              f"{x['running']} running" + (f", {x['failed']} FAILED" if x["failed"] else "")
              + (f", ~{x['unit_minutes']} min per unit" if x.get("unit_minutes") else "")
              + (f", about {x['eta_hours']} h left" if x.get("eta_hours") else "")
              + f"; checks {x['checks']}; merge job #{x['merge_job']}")
    print(f"spent so far: ${s['cost']:.2f} (from the prices given to fleet add)")
    for e in s["events"][-args.events:]:
        print("  " + e)


def cmd_cancel(args) -> None:
    print(describe(api("POST", f"/jobs/{args.id}/cancel", {})))


def cmd_log(args) -> None:
    print(api("GET", f"/jobs/{args.id}/log?tail={args.tail}", raw=True), end="")


def cmd_pause(args) -> None:
    print(api("POST", "/pause" if args.cmd == "pause" else "/resume", {}))


def cmd_hold(args) -> None:
    job = api("POST", "/hold" if args.cmd == "hold" else "/unhold", {})
    print(describe(job) + ("  ON HOLD" if job.get("held_since") else "  running again"))


def cmd_alarm(args) -> None:
    s = api("POST", f"/alarm/{args.action}", {})
    a = s.get("active")
    print(f"alarm {'enabled' if s['enabled'] else 'DISABLED'}" + (f"; sounding: {a['message']}" if a else ""))


def cmd_shutdown(_args) -> None:
    print(api("POST", "/shutdown", {}))


def cmd_daemon(_args) -> None:
    try:
        api("GET", "/status")
        print("mbq: already running")
        return
    except SystemExit:
        pass
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    exe = str(pythonw if pythonw.exists() else sys.executable)
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "server.log", "ab") as out:
        subprocess.Popen([exe, str(Path(__file__).resolve()), "serve"], stdout=out, stderr=subprocess.STDOUT,
                         creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    for _ in range(20):
        time.sleep(0.25)
        try:
            api("GET", "/status")
            print(f"mbq: server started; dashboard at http://127.0.0.1:{PORT}/")
            return
        except SystemExit:
            continue
    print(f"mbq: server did not answer; see {STATE / 'server.log'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="mbq", description="Mindbender solver queue")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the server in this console")
    sub.add_parser("daemon", help="start the server in the background")
    sub.add_parser("status", help="running job, progress, ETA, queue")
    sub.add_parser("queue", help="running and queued jobs")
    h = sub.add_parser("history", help="finished jobs")
    h.add_argument("-n", type=int, default=20)
    a = sub.add_parser("add", help="queue puzzles")
    a.add_argument("puzzles", nargs="+")
    a.add_argument("--moves", type=int, required=True)
    a.add_argument("--until", type=int, help="if unsolved, re-queue one move deeper up to this")
    a.add_argument("--engine", choices=["auto", "ssd", "inram"], default="auto")
    a.add_argument("--ram-gb", type=float)
    a.add_argument("--threads", type=int, help="solver threads (default: all)")
    a.add_argument("--gpu", help="run on the GPU whose name contains this (e.g. 5060), or auto; CUDA first, "
                                 "then Vulkan (cuda:NAME / vulkan:NAME force one)")
    a.add_argument("--no-promote", action="store_true", help="don't update levels_final / README / config")
    a.add_argument("--keep-table", action="store_true", help="keep the depth-6/7 table afterwards")
    a.add_argument("--front", action="store_true", help="put ahead of the other queued jobs")
    a.add_argument("--shards", type=int, help="split the search into N work units (shard jobs + a merge job)")
    a.add_argument("--remote", help="with --shards: shard numbers run on other PCs (comma list); see export")
    a.add_argument("--d7-groups", type=int,
                   help="with --shards N on a depth-7 goal side: split it into G partition groups (N a multiple "
                        "of G); each unit then builds one pass's index and streams N/G of the seeds")
    a.add_argument("--block", dest="block", action="store_true", default=None,
                   help="the free-block join (one move less on the left; default: on with --gpu from 16 moves "
                        "on 2x2 boards and 15 on normal boards)")
    a.add_argument("--no-block", dest="block", action="store_false", help="never use the block join")
    a.add_argument("--stream-depth", type=int, choices=[3, 4, 5, 6],
                   help="left stream moves after the seeds (default: 4 for 15-move, 5 for 16-move and 6 for "
                        "17-move normal boards (--gpu only), 4 for 16-move 2x2 boards, else 3)")
    a.add_argument("--table-depth", type=int, choices=[6, 7],
                   help="goal-side table depth (default: 7 for 2x2 boards at 14+ moves, else 6)")
    ex = sub.add_parser("export", help="work units for another PC (portable solver + .bat per shard)")
    ex.add_argument("puzzle")
    ex.add_argument("--moves", type=int, required=True)
    ex.add_argument("--shards", type=int, required=True)
    ex.add_argument("--which", required=True, help="shard numbers, comma list")
    ex.add_argument("--d7-groups", type=int, help="as for add (must match the merge job's)")
    ex.add_argument("--table-depth", type=int, choices=[6, 7])
    ex.add_argument("--stream-depth", type=int, choices=[3, 4, 5])
    ex.add_argument("--out", default=str(REPO / "workunits"))
    im = sub.add_parser("import", help="file middles results returned from other PCs")
    im.add_argument("files", nargs="+")
    fl = sub.add_parser("fleet", help="rented machines working on one search (fleet.py)")
    fl.add_argument("action", choices=["status", "key", "add", "remove", "run"], nargs="?", default="status")
    fl.add_argument("name", nargs="?", help="add/remove: machine name; run: puzzle")
    fl.add_argument("target", nargs="?", help="add: user@host")
    fl.add_argument("-p", "--port", type=int, default=22)
    fl.add_argument("--price", type=float, help="add: dollars per hour (for the spent-so-far figure)")
    fl.add_argument("--now", action="store_true", help="remove: at once (its unit goes back to the queue)")
    fl.add_argument("--moves", type=int)
    fl.add_argument("--units", type=int, help="run: work units (about an hour each is a good size)")
    fl.add_argument("--d7-groups", type=int, help="run, 2x2 boards: goal partition groups (units a multiple of it)")
    fl.add_argument("--direction", choices=["forward", "reverse"], help="run, normal boards: the search direction")
    fl.add_argument("--checks", type=int, default=2, help="run: units run twice on different machines")
    fl.add_argument("--block", dest="block", action="store_true", default=None)
    fl.add_argument("--no-block", dest="block", action="store_false")
    fl.add_argument("--no-promote", action="store_true", help="run: a rehearsal on a solved puzzle (keep levels_final)")
    fl.add_argument("--only", help="run: just these machines (comma list) take its units")
    fl.add_argument("--front", action="store_true", help="run: the machines take these units before earlier searches'")
    fl.add_argument("--events", type=int, default=12)
    c = sub.add_parser("cancel", help="cancel a queued or running job")
    c.add_argument("id", type=int)
    lg = sub.add_parser("log", help="tail a job's solver log")
    lg.add_argument("id", type=int)
    lg.add_argument("--tail", type=int, default=40)
    sub.add_parser("pause", help="finish the current job, start no new ones")
    sub.add_parser("resume")
    sub.add_parser("hold", help="freeze the running solver in place (frees the CPU, keeps all progress)")
    sub.add_parser("unhold", help="let a held solver carry on")
    sub.add_parser("shutdown", help="stop the server (a running solver keeps going)")
    al = sub.add_parser("alarm", help="the solved-puzzle siren: stop it, test it, or turn it on/off")
    al.add_argument("action", choices=["stop", "test", "enable", "disable"])
    args = ap.parse_args()
    {"serve": lambda a: serve(), "daemon": cmd_daemon, "status": cmd_status, "queue": cmd_queue,
     "history": cmd_history, "add": cmd_add, "cancel": cmd_cancel, "log": cmd_log,
     "pause": cmd_pause, "resume": cmd_pause, "hold": cmd_hold, "unhold": cmd_hold, "shutdown": cmd_shutdown,
     "export": cmd_export, "import": cmd_import, "alarm": cmd_alarm, "fleet": cmd_fleet}[args.cmd](args)


if __name__ == "__main__":
    main()
