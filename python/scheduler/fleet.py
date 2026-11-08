"""python/scheduler/fleet.py: rented GPU machines working together on one search (`mbq fleet`).

A fleet search is an ordinary sharded search (`mbq add --shards N`, every shard remote): the
merge job waits here until every shard's middles file is in state/shards/<group>/s<k>.mid, then
recovers the solutions on this PC. This module runs the shards on rented Linux machines over SSH:

- `mbq fleet key` makes the SSH key (~/.ssh/mindbender_fleet) whose public half goes into the
  rental site's account settings; `mbq fleet add NAME user@host -p PORT --price 0.6` adds a machine.
- New machine: the package (sources, worker scripts, the GPU self-test vectors) is copied over and
  fleet_worker/setup.sh builds the solver and the CUDA module there and runs the GPU self-test.
- Ready machine: it gets the next unit (fleet_worker/run_unit.sh, detached, so it keeps going while
  this PC sleeps); each poll (status.sh) reads its progress, and a finished unit's .mid comes back
  here, checked (puzzle, split, shard) before it is filed for the merge job.
- A unit fails or its machine stops answering for LOST_AFTER_S: the unit goes back to the queue.
  A finished unit found on a machine later is still collected (the first copy wins).
- Checks: a few units (--check, default 2) run a second time on another machine and must give
  byte-identical files; a mismatch is reported (a broken or dishonest machine).
- Nothing here needs a public address: this PC connects out to each machine.
"""
from __future__ import annotations

import io
import json
import random
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SSH_KEY = Path.home() / ".ssh" / "mindbender_fleet"
REMOTE = "mbfleet"                 # the work folder, under the remote user's home
LOST_AFTER_S = 15 * 60             # no answer for this long: its unit goes back to the queue
STUCK_AFTER_S = 60 * 60            # a running unit whose progress file stopped changing
SETUP_TIMEOUT_S = 60 * 60
MAX_UNIT_ATTEMPTS = 4
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def unit_name(search_id: int, k: int, check: bool = False) -> str:
    return f"s{search_id}_{'c' if check else 'u'}{k}"


def parse_unit_name(name: str) -> tuple[int, int, bool] | None:
    try:
        sid, rest = name.split("_", 1)
        return int(sid[1:]), int(rest[1:]), rest[0] == "c"
    except (ValueError, IndexError):
        return None


class Fleet:
    def __init__(self, state_dir: Path, repo: Path, shard_dir, read_header):
        self.dir = state_dir / "fleet"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "fleet.json"
        self.repo = repo
        self.shard_dir = shard_dir            # (puzzle, moves, n) -> Path of the merge job's files
        self.read_header = read_header        # mbq.read_middles_header
        self.lock = threading.RLock()
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.hosts: dict[str, dict] = data.get("hosts", {})
        self.searches: list[dict] = data.get("searches", [])
        self.events: list[str] = data.get("events", [])
        self.next_search = data.get("next_search", 1)
        self.pool = ThreadPoolExecutor(max_workers=16)
        self.in_flight: set[str] = set()
        self.last_tick = 0.0

    # ---- state -----------------------------------------------------------
    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"hosts": self.hosts, "searches": self.searches,
                                       "events": self.events[-300:], "next_search": self.next_search}, indent=1))
            tmp.replace(self.path)

    def event(self, text: str) -> None:
        with self.lock:
            self.events.append(time.strftime("%m-%d %H:%M ") + text)

    def search(self, sid: int) -> dict | None:
        return next((s for s in self.searches if s["id"] == sid), None)

    # ---- SSH -------------------------------------------------------------
    def _ssh_base(self, host: dict, scp: bool = False) -> list[str]:
        known = self.dir / "known_hosts"
        return (["scp", "-P" if scp else "-p", str(host["port"]), "-i", str(SSH_KEY), "-o", "BatchMode=yes",
                 "-o", "ConnectTimeout=20", "-o", "StrictHostKeyChecking=accept-new",
                 "-o", f"UserKnownHostsFile={known}", "-o", "ServerAliveInterval=20",
                 "-o", "ServerAliveCountMax=3"]) if scp else \
            ["ssh", "-p", str(host["port"]), "-i", str(SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
             "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={known}",
             "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3", host["target"]]

    def ssh(self, host: dict, command: str, timeout: float = 90) -> str:
        r = subprocess.run(self._ssh_base(host) + [command], capture_output=True, text=True, timeout=timeout,
                           creationflags=NO_WINDOW)
        if r.returncode == 255:
            raise ConnectionError((r.stderr or "ssh failed").strip().splitlines()[-1])
        if r.returncode != 0:
            raise RuntimeError(f"exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
        return r.stdout

    def scp_to(self, host: dict, local: Path, remote: str, timeout: float = 1800) -> None:
        r = subprocess.run(self._ssh_base(host, scp=True) + [str(local), f"{host['target']}:{remote}"],
                           capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise ConnectionError((r.stderr or "scp failed").strip()[-300:])

    def scp_from(self, host: dict, remote: str, local: Path, timeout: float = 600) -> None:
        r = subprocess.run(self._ssh_base(host, scp=True) + [f"{host['target']}:{remote}", str(local)],
                           capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
        if r.returncode != 0:
            raise ConnectionError((r.stderr or "scp failed").strip()[-300:])

    # ---- logs: every unit's full log and every machine's setup come home ----------
    def log_dir(self, sub: str) -> Path:
        d = self.repo / "docs" / "run_logs" / "fleet" / sub
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fetch_unit_log(self, host: dict, s: dict | None, name: str, tag: str) -> None:
        """docs/run_logs/fleet/<puzzle>_m<moves>_n<N>/<unit>.<machine>.<tag>.log (best effort)."""
        sub = f"{s['puzzle']}_m{s['moves']}_n{s['n']}" if s else "unknown"
        dst = self.log_dir(sub) / f"{name}.{host['name']}.{tag}.log"
        try:
            self.scp_from(host, f"{REMOTE}/work/{name}.log", dst)
        except Exception as exc:
            self.event(f"{host['name']}: could not copy the log of {name}: {exc}")

    def fetch_setup_logs(self, host: dict) -> None:
        stamp = time.strftime("%Y%m%d-%H%M")
        d = self.log_dir("machines")
        for f in ("setup.log", "selftest.log", "setup.ok", "setup.failed"):
            try:
                self.scp_from(host, f"{REMOTE}/{f}", d / f"{host['name']}_{stamp}_{f}")
            except Exception:
                pass

    # ---- the package -----------------------------------------------------
    def package(self) -> Path:
        """mbfleet/{setup,run_unit,status}.sh + mbfleet/pkg/{CMakeLists.txt, include, mouse, src,
        build_gpu/vectors.bin}; rebuilt when a source is newer than it."""
        out = self.dir / "mbfleet_pkg.tgz"
        worker = Path(__file__).resolve().parent / "fleet_worker"
        vectors = self.repo / "build_gpu" / "vectors.bin"
        if not vectors.exists():
            raise FileNotFoundError(f"{vectors} is missing: build and run the exportGpuVectors target first")
        files = [(p, "pkg/" + p.relative_to(self.repo).as_posix())
                 for top in ("src", "include", "mouse") for p in sorted((self.repo / top).rglob("*"))
                 if p.is_file() and "__pycache__" not in p.parts]
        files += [(self.repo / "CMakeLists.txt", "pkg/CMakeLists.txt"), (vectors, "pkg/build_gpu/vectors.bin")]
        files += [(worker / n, n) for n in ("setup.sh", "run_unit.sh", "status.sh")]
        newest = max(p.stat().st_mtime for p, _ in files)
        if out.exists() and out.stat().st_mtime >= newest:
            return out
        tmp = out.with_suffix(".part")
        with tarfile.open(tmp, "w:gz") as tar:
            for p, arc in files:
                info = tar.gettarinfo(str(p), f"{REMOTE}/{arc}")
                info.mode = 0o755 if arc.endswith(".sh") else 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                data = p.read_bytes()
                if arc.endswith(".sh"):
                    data = data.replace(b"\r\n", b"\n")
                    info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        tmp.replace(out)
        return out

    # ---- commands --------------------------------------------------------
    def add_host(self, name: str, target: str, port: int, price: float | None) -> dict:
        if not SSH_KEY.exists():
            raise ValueError(f"no fleet key yet: run `mbq fleet key` and add it to the rental account")
        with self.lock:
            if name in self.hosts and not self.hosts[name].get("removed"):
                raise ValueError(f"machine {name} is already in the fleet")
            host = {"name": name, "target": target, "port": int(port), "price": price, "added": time.time(),
                    "removed": None, "status": "new", "info": {}, "unit": None, "unit_started": None,
                    "last_ok": time.time(), "last_error": "", "progress": None, "units_done": 0,
                    "unit_seconds": 0.0, "failures": 0, "draining": False}
            self.hosts[name] = host
            self.event(f"{name}: added ({target} port {port})")
            self.save()
            return host

    def remove_host(self, name: str, now: bool) -> dict:
        with self.lock:
            host = self.hosts.get(name)
            if host is None or host.get("removed"):
                raise KeyError(name)
            if host["unit"] and not now:
                host["draining"] = True
                self.event(f"{name}: will leave after its current unit")
            else:
                self._requeue(host, "machine removed")
                host["removed"] = time.time()
                host["status"] = "removed"
                self.event(f"{name}: removed (stop paying for it at the rental site)")
            self.save()
            return host

    def add_search(self, merge_job: dict, units: int, groups: int, checks: int, direction: str,
                   front: bool = False, only: list[str] | None = None) -> dict:
        """front: the machines take this search's units before the earlier searches' (small ones).
        only: just these machines take its units (one machine set aside for small searches)."""
        n = int(units)
        with self.lock:
            sid = self.next_search
            self.next_search += 1
            args = [merge_job["puzzle"], "--seed-depth", str(merge_job["seed_depth"]),
                    "--stream-depth", str(merge_job["stream_depth"]), "--table-depth", str(merge_job["table_depth"]),
                    "--direction", direction]
            if merge_job.get("block"):
                args.append("--block")
            if groups > 1:
                args += ["--d7-groups", str(groups)]
            check_ks = sorted(random.sample(range(n), min(checks, n))) if checks else []
            s = {"id": sid, "puzzle": merge_job["puzzle"], "moves": merge_job["moves"], "n": n, "groups": groups,
                 "direction": direction, "block": bool(merge_job.get("block")), "args": args,
                 "only": list(only) if only else None,
                 "merge_job": merge_job["id"], "created": time.time(), "finished": None,
                 "units": [{"k": k, "status": "pending", "host": None, "attempts": 0, "started": None,
                            "finished": None, "seconds": None} for k in range(n)],
                 "checks": [{"k": k, "status": "pending", "host": None, "attempts": 0, "result": None}
                            for k in check_ks]}
            if front:
                self.searches.insert(0, s)
            else:
                self.searches.append(s)
            self.event(f"search {sid}: {s['puzzle']} at {s['moves']} moves as {n} units"
                       + (f" ({groups} goal groups)" if groups > 1 else "") + f", checks on units {check_ks}")
            self.save()
            return s

    # ---- the poll --------------------------------------------------------
    def tick(self, every: float = 60.0) -> None:
        if time.time() - self.last_tick < every:
            return
        self.last_tick = time.time()
        with self.lock:
            names = [n for n, h in self.hosts.items() if not h.get("removed") and n not in self.in_flight]
            self.in_flight.update(names)
        for n in names:
            self.pool.submit(self._step_safe, n)

    def _step_safe(self, name: str) -> None:
        try:
            host = self.hosts[name]
            try:
                self._step(host)
                with self.lock:
                    host["last_ok"] = time.time()
                    host["last_error"] = ""
            except Exception as exc:   # unreachable machine, ssh error, bad file
                with self.lock:
                    host["last_error"] = str(exc)[:300]
                    if host["unit"] and time.time() - host["last_ok"] > LOST_AFTER_S:
                        self._requeue(host, f"no answer for {LOST_AFTER_S // 60} min")
                        host["status"] = "lost"
            with self.lock:
                self.save()
        finally:
            with self.lock:
                self.in_flight.discard(name)

    def _step(self, host: dict) -> None:
        if host["status"] == "new":
            pkg = self.package()
            self.ssh(host, f"touch ~/.no_auto_tmux; mkdir -p {REMOTE}")   # vast.ai: no tmux on login
            self.scp_to(host, pkg, "mbfleet_pkg.tgz")
            self.ssh(host, f"rm -rf {REMOTE}/pkg && tar -xzf mbfleet_pkg.tgz && rm -f {REMOTE}/setup.ok "
                           f"{REMOTE}/setup.failed && (setsid nohup bash {REMOTE}/setup.sh >/dev/null 2>&1 "
                           f"< /dev/null &)")
            with self.lock:
                host["status"] = "setting-up"
                host["setup_started"] = time.time()
                self.event(f"{host['name']}: package copied, building the solver and testing the GPU")
            return
        out = self.ssh(host, f"bash {REMOTE}/status.sh")
        st = {"live": set(), "dead": set(), "rc": {}, "progress": {}, "info": {}, "setup": "absent"}
        for line in out.splitlines():
            kind, _, rest = line.partition(" ")
            if kind == "setup":
                st["setup"] = rest.strip()
            elif kind == "info":
                k, _, v = rest.partition("=")
                st["info"][k] = v
            elif kind in ("live", "dead"):
                st[kind].add(rest.strip())
            elif kind == "rc":
                nm, _, code = rest.rpartition(" ")
                st["rc"][nm] = int(code) if code.strip().lstrip("-").isdigit() else -1
            elif kind == "progress":
                nm, _, js = rest.partition(" ")
                try:
                    st["progress"][nm] = json.loads(js)
                except json.JSONDecodeError:
                    pass
        if host["status"] in ("setting-up", "failed-setup"):
            if st["setup"] in ("ok", "failed") and host["status"] != "failed-setup":
                self.fetch_setup_logs(host)
            if st["setup"] == "ok":
                with self.lock:
                    host["status"] = "ready"
                    host["info"] = st["info"]
                    self.event(f"{host['name']}: ready, GPU self-test PASS ({st['info'].get('gpu', '?')}, "
                               f"{st['info'].get('ram_gb', '?')} GB RAM, {st['info'].get('cores', '?')} cores)")
            elif st["setup"] == "failed":
                with self.lock:
                    if host["status"] != "failed-setup":
                        self.event(f"{host['name']}: setup FAILED: {st['info'].get('reason', '?')}")
                    host["status"] = "failed-setup"
                    host["info"] = st["info"]
            elif time.time() - host.get("setup_started", time.time()) > SETUP_TIMEOUT_S:
                with self.lock:
                    host["status"] = "failed-setup"
                    self.event(f"{host['name']}: setup took over {SETUP_TIMEOUT_S // 60} min; see ~/{REMOTE}/setup.log")
            return
        if host["status"] == "lost":
            with self.lock:
                host["status"] = "ready"
                self.event(f"{host['name']}: answering again")
        # finished units on this machine, ours or an earlier attempt's
        for nm, code in st["rc"].items():
            self._collect(host, nm, code)
        cur = host["unit"]
        if cur:
            if cur in st["live"]:
                prog = st["progress"].get(cur)
                with self.lock:
                    if prog != host.get("progress"):
                        host["progress"] = prog
                        host["progress_changed"] = time.time()
                    elif time.time() - host.get("progress_changed", time.time()) > STUCK_AFTER_S:
                        self.ssh(host, f"pkill -f 'experimentDiskDepth6.*{cur}.mid' || true")
                        self._requeue(host, "no progress for an hour (stopped)")
                return
            if cur not in st["rc"]:
                with self.lock:   # gone without an exit code (machine restarted, killed)
                    self._requeue(host, "its process disappeared")
        if host["draining"]:
            with self.lock:
                host["removed"] = time.time()
                host["status"] = "removed"
                self.event(f"{host['name']}: left the fleet after its unit (stop paying for it at the rental site)")
            return
        if st["live"]:   # an orphan from an earlier attempt still runs: let it finish, collect it then
            with self.lock:
                host["status"] = "busy (earlier unit)"
            return
        if host["failures"] >= 3:
            with self.lock:
                host["status"] = "failing"
            return
        self._start_next(host)

    def _start_next(self, host: dict) -> None:
        with self.lock:
            pick = None
            for s in self.searches:
                if s["finished"] or (s.get("only") and host["name"] not in s["only"]):
                    continue
                pending = [u for u in s["units"] if u["status"] == "pending"]
                if pending:
                    pick = (s, pending[0], False)
                    break
                for c in s["checks"]:   # a check: once the unit is done, on another machine
                    u = s["units"][c["k"]]
                    if c["status"] == "pending" and u["status"] == "done" and u["host"] != host["name"]:
                        pick = (s, c, True)
                        break
                if pick:
                    break
            if pick is None:
                host["status"] = "idle"
                return
            s, u, check = pick
            name = unit_name(s["id"], u["k"], check)
            u.update(status="running", host=host["name"], started=time.time())
            u["attempts"] += 1
            host.update(unit=name, unit_started=time.time(), status="working", progress=None,
                        progress_changed=time.time())
        vram = float(str(host["info"].get("gpu", "")).split(",")[1].split()[0]) / 1024 \
            if "," in str(host["info"].get("gpu", "")) else 0
        args = s["args"] + ["--shard", f"{u['k']}/{s['n']}", "--gpu", "cuda:auto"]
        if vram > 4:
            args += ["--gpu-vram-gb", f"{max(4.0, vram - 1.5):.1f}"]
        quoted = " ".join("'" + a.replace("'", "'\\''") + "'" for a in args)
        try:
            self.ssh(host, f"(setsid nohup bash {REMOTE}/run_unit.sh {name} {quoted} >/dev/null 2>&1 < /dev/null &)")
        except Exception:
            with self.lock:
                self._requeue(host, "could not start it")
            raise
        with self.lock:
            self.event(f"{host['name']}: started {'check of ' if check else ''}unit {u['k']} of search {s['id']}")

    def _requeue(self, host: dict, why: str) -> None:
        """The host's current unit goes back to the queue (lock held)."""
        name = host.get("unit")
        host["unit"] = None
        host["progress"] = None
        if not name:
            return
        parsed = parse_unit_name(name)
        s = self.search(parsed[0]) if parsed else None
        if s is None:
            return
        _, k, check = parsed
        u = next((c for c in s["checks"] if c["k"] == k), None) if check else s["units"][k]
        if u is not None and u["status"] == "running":
            u["status"] = "pending" if u["attempts"] < MAX_UNIT_ATTEMPTS else "failed"
            u["host"] = None
        host["failures"] += 1
        self.event(f"{host['name']}: unit {name} back in the queue ({why})")

    def _collect(self, host: dict, name: str, code: int) -> None:
        parsed = parse_unit_name(name)
        s = self.search(parsed[0]) if parsed else None
        mine = host["unit"] == name
        if s is None:
            self.ssh(host, f"cd {REMOTE}/work && mv -f {name}.rc {name}.rc.collected")
            return
        _, k, check = parsed
        u = next((c for c in s["checks"] if c["k"] == k), None) if check else s["units"][k]
        if code != 0:
            self.fetch_unit_log(host, s, name, f"exit{code}")
            tail = self.ssh(host, f"tail -n 5 {REMOTE}/work/{name}.log || true")
            self.ssh(host, f"cd {REMOTE}/work && mv -f {name}.rc {name}.rc.collected")
            with self.lock:
                self.event(f"{host['name']}: unit {name} FAILED (exit {code}): {tail.strip()[-200:]}")
                if mine:
                    self._requeue(host, f"exit {code}")
            return
        local = self.dir / "incoming" / f"{name}.{host['name']}.mid"
        local.parent.mkdir(exist_ok=True)
        self.scp_from(host, f"{REMOTE}/work/{name}.mid", local)
        self.fetch_unit_log(host, s, name, "ok")
        h = self.read_header(local)
        want = {"puzzle": s["puzzle"], "moves": s["moves"], "shard": k, "shards": s["n"],
                "d7_groups": s["groups"] if s["groups"] > 1 else 1, "block": 1 if s["block"] else 0}
        bad = [f"{key} {h.get(key)} (want {v})" for key, v in want.items() if h.get(key) != v]
        log_tail = self.ssh(host, f"grep -E 'TOTAL|SHARD' {REMOTE}/work/{name}.log | tail -n 2 || true").strip()
        self.ssh(host, f"cd {REMOTE}/work && mv -f {name}.rc {name}.rc.collected")
        with self.lock:
            if mine:
                host["unit"] = None
                host["progress"] = None
                host["units_done"] += 1
                host["unit_seconds"] += time.time() - (host["unit_started"] or time.time())
                host["failures"] = 0
            if bad:
                self.event(f"{host['name']}: unit {name} has the wrong header: {', '.join(bad)}; discarded")
                if u is not None and u["status"] != "done":
                    u["status"] = "pending"
                return
            dst = self.shard_dir(s["puzzle"], s["moves"], s["n"]) / f"s{k}.mid"
            if check:
                same = dst.exists() and dst.read_bytes() == local.read_bytes()
                if u is not None:
                    u.update(status="done", result="identical" if same else "MISMATCH", host=host["name"])
                self.event(f"{host['name']}: check of unit {k}: " + ("identical" if same else
                           f"MISMATCH with {s['units'][k]['host']}'s file: one of the two machines is wrong"))
            elif dst.exists():
                if dst.read_bytes() != local.read_bytes():
                    self.event(f"{host['name']}: unit {k} again, but DIFFERENT from the first copy "
                               f"({s['units'][k]['host']}): kept the first, look at both")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                local.replace(dst)
                local = dst
                done = s["units"][k]
                done.update(status="done", host=host["name"], finished=time.time())
                if done.get("started"):
                    done["seconds"] = done["finished"] - done["started"]
                self.event(f"{host['name']}: unit {k} of search {s['id']} done, {h['count']} middles"
                           + (f" ({log_tail.splitlines()[-1][:80]})" if log_tail else ""))
            if all(x["status"] == "done" for x in s["units"]) and all(c["status"] in ("done", "failed")
                                                                     for c in s["checks"]):
                if not s["finished"]:
                    s["finished"] = time.time()
                    self.event(f"search {s['id']}: every unit is in; the merge job (#{s['merge_job']}) "
                               f"recovers the solutions here")

    # ---- reporting -------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            t = time.time()
            hosts = []
            cost = 0.0
            for h in self.hosts.values():
                end = h["removed"] or t
                spent = (end - h["added"]) / 3600 * (h["price"] or 0)
                cost += spent
                hosts.append({k: h.get(k) for k in ("name", "target", "port", "status", "unit", "units_done",
                                                     "last_error", "price", "removed")}
                             | {"gpu": h["info"].get("gpu"), "ram_gb": h["info"].get("ram_gb"),
                                "cost": round(spent, 2), "progress": h.get("progress"),
                                "unit_minutes": round((t - h["unit_started"]) / 60, 1) if h["unit"] else None,
                                "seen_ago_s": round(t - h["last_ok"])})
            searches = []
            for s in self.searches:
                done = [u for u in s["units"] if u["status"] == "done"]
                secs = [u["seconds"] for u in done if u.get("seconds")]
                left = len(s["units"]) - len(done)
                working = sum(1 for h in self.hosts.values() if h["unit"] and not h.get("removed")
                              and h["unit"].startswith(f"s{s['id']}_"))
                eta = (sum(secs) / len(secs)) * left / max(1, working) if secs and left else None
                searches.append({"id": s["id"], "puzzle": s["puzzle"], "moves": s["moves"], "n": s["n"],
                                 "done": len(done), "running": sum(1 for u in s["units"] if u["status"] == "running"),
                                 "failed": sum(1 for u in s["units"] if u["status"] == "failed"),
                                 "checks": [(c["k"], c["status"], c.get("result")) for c in s["checks"]],
                                 "unit_minutes": round(sum(secs) / len(secs) / 60, 1) if secs else None,
                                 "eta_hours": round(eta / 3600, 1) if eta else None,
                                 "merge_job": s["merge_job"], "finished": s["finished"]})
            return {"hosts": hosts, "searches": searches, "cost": round(cost, 2), "events": self.events[-25:],
                    "key": str(SSH_KEY) + ".pub" if SSH_KEY.exists() else None}


def make_key() -> str:
    """The fleet's SSH key (no passphrase: the server logs in unattended). Returns the public key."""
    if not SSH_KEY.exists():
        SSH_KEY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-C", "mindbender-fleet", "-f", str(SSH_KEY)],
                       check=True, capture_output=True, creationflags=NO_WINDOW)
    return Path(str(SSH_KEY) + ".pub").read_text().strip()
