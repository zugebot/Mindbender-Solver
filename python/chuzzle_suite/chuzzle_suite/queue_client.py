# python/chuzzle_suite/chuzzle_suite/queue_client.py
"""Client for the solver scheduler (python/scheduler/mbq.py): polls its
GET /status (and the running job's log, GET /jobs/<id>/log) in a background thread
and sends POST /pause, /resume, /hold and /unhold. The text matches `mbq status`
(its describe() and fmt_dur()) and the dashboard (python/scheduler/dashboard.html)."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

# The daemon's address (mbq.PORT). CHUZZLE_SUITE_MBQ_URL points the app at another
# server, e.g. a fake one for tests.
DEFAULT_URL = os.environ.get("CHUZZLE_SUITE_MBQ_URL", "http://127.0.0.1:8766").rstrip("/")
POLL_S = 2.5
TIMEOUT_S = 4.0                   # Windows takes ~2 s to refuse a local connection (SYN retries)
COMMANDS = ("pause", "resume", "hold", "unhold")
LOG_TAIL = 60                     # log lines fetched (the Queue tab shows what fits)
# a job's phases in order (dashboard.html: PHASES)
PHASES = ("d6-build", "d6-index", "d7-index", "probe", "d7-confirm", "verify", "recovery")

# no system proxy for a local server
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class QueueError(Exception):
    """The daemon answered with an error (e.g. "nothing is running")."""


# ----------------------------------------------------------------------
# formatting (same wording as mbq.py's CLI)
# ----------------------------------------------------------------------

def fmt_dur(s: float | None) -> str:
    if s is None:
        return "unknown"
    s = int(max(0, s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {sec:02d}s" if m else f"{sec}s")


def fmt_clock(t: float | None) -> str:
    return time.strftime("%H:%M", time.localtime(t)) if t else "-"


def split(job: dict) -> str:
    """The search split, e.g. <5,4,7>; empty for the in-RAM engine."""
    if job.get("engine") != "ssd":
        return ""
    return f"<{job['seed_depth']},{job.get('stream_depth') or 3},{job['table_depth']}>"


def describe(job: dict) -> str:
    kind = f"{job['engine']}" + (f" {split(job)}" if job["engine"] == "ssd" else "")
    if job.get("shard"):
        kind += f" shard {job['shard'][0]}/{job['shard'][1]}"
    if job.get("merge"):
        kind += f" merge of {job['merge']} shards"
    if job.get("gpu"):
        kind += f" gpu {job['gpu']}"
    until = f" (until {job['until']})" if job.get("until") else ""
    return f"#{job['id']:<3} {job['puzzle']:>5} {job['moves']} moves{until}  {kind}  [{job['status']}]"


def short_describe(job: dict) -> str:
    """One short line for the mini view: '#206 15-3  16 moves  <5,4,7> gpu'."""
    kind = split(job) or job.get("engine", "")
    if job.get("shard"):
        kind += f" shard {job['shard'][0]}/{job['shard'][1]}"
    if job.get("gpu"):
        kind += " gpu"
    return f"#{job['id']} {job['puzzle']}  {job['moves']} moves  {kind.strip()}".rstrip()


def held_line(job: dict | None) -> str | None:
    if job and job.get("held_since"):
        return f"SOLVER ON HOLD since {fmt_clock(job['held_since'])}"
    return None


def phase_line(eta: dict) -> str:
    pct = f"{eta['percent']}%" if eta.get("percent") is not None else "-"
    return f"phase: {eta.get('phase')} {eta.get('detail') or ''}  {pct}"


def total_line(eta: dict) -> str:
    if eta.get("total_remaining") is not None:
        return (f"whole job: about {fmt_dur(eta['total_remaining'])} left, "
                f"done around {fmt_clock(eta.get('finish_at'))}")
    return "whole job: unknown until a later phase starts (no history for this kind of run yet)"


def left_line(eta: dict) -> str:
    """Time left and ETA in a few words (mini view)."""
    if eta.get("total_remaining") is not None:
        return f"about {fmt_dur(eta['total_remaining'])} left, done around {fmt_clock(eta.get('finish_at'))}"
    if eta.get("phase_remaining") is not None:
        return f"phase ends in {fmt_dur(eta['phase_remaining'])}, whole job unknown"
    return "time left unknown"


def phase_chips(job: dict) -> list[tuple[str, str, float | None]]:
    """The running job's phases as (name, state, seconds spent) with state "now",
    "done" or "" (to come); the same choice of phases as the dashboard."""
    if job.get("engine") != "ssd":
        return []
    eta = job.get("eta") or {}
    needs = job.get("needs") or {}
    spent = job.get("phases") or {}
    now = eta.get("phase")
    shown = [p for p in PHASES
             if not (p.startswith("d7") and job.get("table_depth") != 7)
             and not (p in ("d6-build", "d6-index", "d7-index") and not needs.get(p) and now != p and not spent.get(p))]
    idx = shown.index(now) if now in shown else -1
    return [(p, "now" if p == now else ("done" if (0 <= idx and i < idx) or spent.get(p) else ""), spent.get(p))
            for i, p in enumerate(shown)]


def live_line(s: dict) -> str | None:
    """The line the dashboard adds under the log: the solver logs only between steps."""
    job = s.get("running")
    e = (job or {}).get("eta") or {}
    if job and e.get("total"):
        phase = e.get("phase") or ""
        unit = "seeds" if phase == "probe" else "partitions" if phase.endswith("index") else "units"
        return (f"> now: {phase} {e.get('detail') or ''} - {int(e.get('done') or 0):,} / {int(e['total']):,} {unit} "
                f"({e.get('percent')}%), {fmt_dur(e.get('phase_remaining'))} left in this step")
    if s.get("finalizing"):
        return "> now: validating and recording the results"
    return None


def status_lines(s: dict, hints: bool = False) -> list[tuple[str, str]]:
    """`mbq status` as (text, kind) lines; kind is warn, job, text, dim or head.
    hints=True keeps the CLI's "(mbq resume)" / "(mbq unhold)" hints."""
    out: list[tuple[str, str]] = []
    if s.get("paused"):
        out.append(("queue is PAUSED" + (" (mbq resume)" if hints else ""), "warn"))
    job = s.get("running")
    held = held_line(job)
    if held:
        out.append((held + (" (mbq unhold)" if hints else ""), "warn"))
    if s.get("finalizing"):
        out.append((describe(s["finalizing"]) + "  validating results...", "job"))
    if job:
        e = job.get("eta") or {}
        out.append((describe(job), "job"))
        out.append(("  " + phase_line(e), "text"))
        out.append((f"  elapsed {fmt_dur(e.get('elapsed'))}; this phase ends in {fmt_dur(e.get('phase_remaining'))}",
                    "text"))
        out.append(("  " + total_line(e), "text"))
    else:
        out.append(("nothing running", "dim"))
    for n, p in s.get("external_solvers") or []:
        out.append((f"  other solver running (not managed): {n} pid {p}", "warn"))
    pending = s.get("pending") or []
    if pending:
        out.append((f"{len(pending)} queued:", "head"))
        for j in pending:
            out.append(("  " + describe(j) + (f"  - {j['wait_reason']}" if j.get("wait_reason") else ""), "job"))
    return out


def unreachable_reason(exc: BaseException) -> str:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, (socket.timeout, TimeoutError)):
        return "timed out"
    if isinstance(reason, ConnectionRefusedError) or getattr(reason, "winerror", None) == 10061:
        return "connection refused"
    return str(reason) or reason.__class__.__name__


# ----------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------

def fetch_status(base_url: str = DEFAULT_URL, timeout: float = TIMEOUT_S) -> dict:
    with _opener.open(f"{base_url}/status", timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_log(job_id: int, base_url: str = DEFAULT_URL, tail: int = LOG_TAIL, timeout: float = TIMEOUT_S) -> str:
    with _opener.open(f"{base_url}/jobs/{job_id}/log?tail={tail}", timeout=timeout) as r:
        return r.read().decode(errors="replace")


def post_command(command: str, base_url: str = DEFAULT_URL, timeout: float = 10.0) -> dict:
    """POST /pause, /resume, /hold or /unhold (with mbq's X-MBQ header). Raises
    QueueError when the daemon refuses, OSError when it is unreachable."""
    if command not in COMMANDS:
        raise ValueError(command)
    req = urllib.request.Request(f"{base_url}/{command}", method="POST", data=b"{}",
                                 headers={"Content-Type": "application/json", "X-MBQ": "1"})
    try:
        with _opener.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        data = e.read().decode(errors="replace")
        try:
            message = json.loads(data).get("error", data)
        except ValueError:
            message = data or str(e)
        raise QueueError(message) from None


class QueuePoller:
    """Polls GET /status while `active` is set. `snapshot` is (status or None,
    error text or None, time of the last answer); the UI only reads it. With
    `want_log` set it also fetches the running job's log tail into `log`
    ((job id, text), or None)."""

    def __init__(self, base_url: str | None = None, interval: float = POLL_S):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.interval = interval
        self.active = False
        self.snapshot: tuple[dict | None, str | None, float | None] = (None, None, None)
        self.busy = False           # a command is on its way
        self.want_log = False       # the full Queue tab is showing
        self.log: tuple[int, str] | None = None
        self._wake = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None

    def start(self) -> "QueuePoller":
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stopped = True
        self._wake.set()

    def poke(self) -> None:
        """Poll now instead of at the next interval."""
        self._wake.set()

    def poll_once(self) -> None:
        try:
            status = fetch_status(self.base_url)
        except (OSError, ValueError) as exc:     # URLError is an OSError
            self.snapshot = (None, unreachable_reason(exc), time.time())
            return
        self.snapshot = (status, None, time.time())
        live = status.get("running") or status.get("finalizing")
        if not (self.want_log and live):
            if not live:
                self.log = None
            return
        try:
            self.log = (live["id"], fetch_log(live["id"], self.base_url))
        except (OSError, ValueError, KeyError):
            pass   # keep the last log; the status line already shows trouble

    def _loop(self) -> None:
        while not self._stopped:
            if self.active:
                self.poll_once()
            self._wake.wait(self.interval if self.active else 0.5)
            self._wake.clear()

    def send(self, command: str, on_done: Callable[[dict | None, str | None], None]) -> None:
        """POST the command in a thread; on_done(result, error) runs in that thread."""
        self.busy = True

        def work():
            try:
                result, error = post_command(command, self.base_url), None
            except QueueError as exc:
                result, error = None, str(exc)
            except OSError as exc:
                result, error = None, f"scheduler not running ({unreachable_reason(exc)})"
            self.busy = False
            self.poke()
            on_done(result, error)

        threading.Thread(target=work, daemon=True).start()
