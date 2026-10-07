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
  ssd    build/experimentDiskDepth6.exe, 10-15 moves:
         <S,3,6> up to 14 moves, <S,3,7> (index-only depth-7 table) for 15.
  inram  build/findSolutions.exe, 3-12 moves (exhaustive in-RAM search).

After a job: solutions are copied to <repo>/levels and replayed (the demo
app's simulator and testKnownSolutions). With promotion on (default),
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
INRAM_EXE = REPO / "build" / "findSolutions.exe"
TEST_KNOWN_EXE = REPO / "build" / "testKnownSolutions.exe"
LEVELS_DIR = REPO / "levels"
LEVELS_FINAL = REPO / "build" / "levels_final"
BUILD_README = REPO / "build" / "README.md"
BUILD_CONFIG = REPO / "build" / "config.json"

# Any of these running (and not ours) means "someone else is using the machine".
SOLVER_PROCESS_RE = re.compile(
    r"^(experimentDiskDepth6\S*|expD7\S*|findSolutions\S*|mbq_ssd|mbq_inram|exp)\.exe$", re.I)
# Tests only: start jobs even while another solver runs.
IGNORE_EXTERNAL = os.environ.get("MBQ_IGNORE_EXTERNAL") == "1"

SSD_PHASES = ["d6-build", "d6-index", "d7-index", "probe", "d7-confirm", "verify", "recovery"]
# Rough disk needs when a table must be built (largest seen: 10-2 depth-6 75 GB;
# 6-4 depth-7 index 166 GB, normal boards up to ~3x that).
DISK_NEED_GB = {6: 100, 7: 550}

kernel32 = ctypes.windll.kernel32
SYNCHRONIZE = 0x00100000
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_TERMINATE = 0x0001
STILL_ACTIVE = 259
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


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

def puzzle_is_fat(puzzle: str) -> bool:
    text = (REPO / "src" / "chuzzles" / "levels.hpp").read_text(errors="replace")
    m = re.search(r'\{"' + re.escape(puzzle) + r'"\s*,[^}]*\}\s*,\s*(\{[^}]*\})', text)
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
    LEVELS_FINAL.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solution_file, LEVELS_FINAL / solution_file.name)
    notes.append(f"copied {solution_file.name} to build/levels_final")

    def readme(text: str):
        line_re = re.compile(r"^\s*" + re.escape(puzzle) + r":\s*at least\s+\d+\s+moves\s*\n", re.M)
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
    """Replays every solution with the demo app's board simulator."""
    sys.path.insert(0, str(REPO / "python" / "demo_app"))
    try:
        from chuzzle_demo import board_sim, levels as lv
        level = lv.load_levels()[puzzle]
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
        self._readopt()

    # ---- persistence -----------------------------------------------------
    def save(self) -> None:
        with self.lock:
            write_json_atomic(JOBS_FILE, {"jobs": self.jobs, "tables": self.tables,
                                          "paused": self.paused, "next_id": self.next_id})

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
        self.save()

    # ---- jobs ------------------------------------------------------------
    def add(self, spec: dict) -> dict:
        puzzle = spec["puzzle"]
        moves = int(spec["moves"])
        engine = spec.get("engine") or "auto"
        if engine == "auto":
            engine = "inram" if moves <= 12 else "ssd"
        if not re.fullmatch(r"\d+-\d+", puzzle):
            raise ValueError(f"bad puzzle name {puzzle!r}")
        if engine == "ssd" and not 10 <= moves <= 15:
            raise ValueError("ssd engine supports 10-15 moves")
        if engine == "inram" and not 3 <= moves <= 12:
            raise ValueError("inram engine supports 3-12 moves")
        table_depth = 7 if moves >= 15 else 6
        seed_depth = moves - 3 - table_depth
        with self.lock:
            job = {
                "id": self.next_id, "puzzle": puzzle, "moves": moves, "until": spec.get("until"),
                "engine": engine, "table_depth": table_depth if engine == "ssd" else None,
                "seed_depth": seed_depth if engine == "ssd" else None,
                "ram_gb": spec.get("ram_gb"), "threads": spec.get("threads"),
                "promote": spec.get("promote", True),
                "keep_table": spec.get("keep_table", False), "fat": puzzle_is_fat(puzzle),
                "status": "queued", "wait_reason": "", "created": now(), "started": None,
                "finished": None, "pid": None, "exit_code": None, "solutions": None,
                "result": None, "message": "", "notes": [], "phases": {}, "parent": spec.get("parent"),
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

    def cancel(self, job_id: int) -> dict:
        with self.lock:
            job = self.job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "running" and self.proc is not None:
                self.proc.terminate()
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

    # ---- worker ----------------------------------------------------------
    def loop(self) -> None:
        while True:
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
            if not pending:
                return
            job = pending[0]
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
        if job["engine"] != "ssd":
            return 0
        d = self._table_dir(job)
        have6 = (d / "fp16meta.txt").exists()
        have7 = (d / "d7_meta.txt").exists()
        if job["table_depth"] == 7:
            return 0 if have7 else DISK_NEED_GB[7] + (0 if have6 else DISK_NEED_GB[6])
        return 0 if have6 else DISK_NEED_GB[6]

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
            if job.get("ram_gb"):
                cmd += ["--ram-gb", str(job["ram_gb"])]
            if job.get("threads"):
                cmd += ["--threads", str(job["threads"])]
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
        else:
            job.update(status="done", result="none", message=f"no {moves}-move solution")
            if job["promote"]:
                notes += promote_bound(puzzle, moves)
            if job.get("until") and moves < int(job["until"]):
                nxt = self.add({"puzzle": puzzle, "moves": moves + 1, "until": job["until"],
                                "engine": "auto", "ram_gb": job.get("ram_gb"), "threads": job.get("threads"),
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
               "elapsed": now() - job["started"] if job.get("started") else None}
        phase_left = None
        if total and done:
            phase_left = prog["phase_seconds"] * (total - done) / done
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
                    return self._send(200, sched.add(self._body()))
                if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
                    return self._send(200, sched.cancel(int(parts[1])))
                if parts == ["shutdown"]:
                    self._send(200, {"stopping": True, "note": "a running solver keeps running and is re-adopted"})
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                    return None
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
    kind = f"{job['engine']}" + (f" <{job['seed_depth']},3,{job['table_depth']}>" if job["engine"] == "ssd" else "")
    until = f" (until {job['until']})" if job.get("until") else ""
    return f"#{job['id']:<3} {job['puzzle']:>5} {job['moves']} moves{until}  {kind}  [{job['status']}]"


def cmd_status(_args) -> None:
    s = api("GET", "/status")
    if s["paused"]:
        print("queue is PAUSED (mbq resume)")
    job = s["running"]
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
    for puzzle in args.puzzles:
        job = api("POST", "/jobs", {"puzzle": puzzle, "moves": args.moves, "until": args.until,
                                    "engine": args.engine, "ram_gb": args.ram_gb, "threads": args.threads,
                                    "promote": not args.no_promote,
                                    "keep_table": args.keep_table, "front": args.front})
        print("queued " + describe(job))


def cmd_cancel(args) -> None:
    print(describe(api("POST", f"/jobs/{args.id}/cancel", {})))


def cmd_log(args) -> None:
    print(api("GET", f"/jobs/{args.id}/log?tail={args.tail}", raw=True), end="")


def cmd_pause(args) -> None:
    print(api("POST", "/pause" if args.cmd == "pause" else "/resume", {}))


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
    a.add_argument("--no-promote", action="store_true", help="don't update levels_final / README / config")
    a.add_argument("--keep-table", action="store_true", help="keep the depth-6/7 table afterwards")
    a.add_argument("--front", action="store_true", help="put ahead of the other queued jobs")
    c = sub.add_parser("cancel", help="cancel a queued or running job")
    c.add_argument("id", type=int)
    lg = sub.add_parser("log", help="tail a job's solver log")
    lg.add_argument("id", type=int)
    lg.add_argument("--tail", type=int, default=40)
    sub.add_parser("pause", help="finish the current job, start no new ones")
    sub.add_parser("resume")
    sub.add_parser("shutdown", help="stop the server (a running solver keeps going)")
    args = ap.parse_args()
    {"serve": lambda a: serve(), "daemon": cmd_daemon, "status": cmd_status, "queue": cmd_queue,
     "history": cmd_history, "add": cmd_add, "cancel": cmd_cancel, "log": cmd_log,
     "pause": cmd_pause, "resume": cmd_pause, "shutdown": cmd_shutdown}[args.cmd](args)


if __name__ == "__main__":
    main()
