"""python/scheduler/alarm.py - the "a puzzle was solved" alarm.

The scheduler calls Alarm.start() when a job's solutions pass validation. The daemon then
loops a loud siren (a generated WAV at full amplitude, played with winsound, so it works with
no browser open) until someone stops it (dashboard button, `mbq alarm stop`) or it times out.
Volume is whatever the PC's volume is; this never changes system settings.
"""
from __future__ import annotations

import math
import struct
import threading
import time
import wave
from pathlib import Path

try:
    import winsound
except ImportError:            # not on Windows: the dashboard banner still works
    winsound = None

RATE = 44100


def make_siren(path: Path, seconds: float = 4.0) -> Path:
    """A two-tone emergency siren: a fast 650-1500 Hz wail, square-ish for loudness."""
    frames = bytearray()
    n = int(RATE * seconds)
    phase = 0.0
    for i in range(n):
        t = i / RATE
        sweep = 0.5 - 0.5 * math.cos(2 * math.pi * t / 0.8)          # up and down every 0.8 s
        freq = 650 + 850 * sweep
        phase += 2 * math.pi * freq / RATE
        s = math.sin(phase) + 0.3 * math.sin(3 * phase) + 0.15 * math.sin(5 * phase)
        s = math.tanh(2.2 * s)                                       # soft clip: loud, no wrap
        frames += struct.pack("<h", int(32000 * s))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return path


class Alarm:
    def __init__(self, wav: Path, timeout_s: float = 15 * 60):
        self.wav = wav
        self.timeout_s = timeout_s
        self.enabled = True
        self.active: dict | None = None      # {"message", "since", "puzzle"}
        self._lock = threading.Lock()
        self._stop_at = 0.0

    def state(self) -> dict:
        with self._lock:
            return {"enabled": self.enabled, "active": dict(self.active) if self.active else None}

    def start(self, puzzle: str, message: str, test: bool = False) -> dict:
        with self._lock:
            if not self.enabled and not test:
                return self.state_unlocked()
            self.active = {"puzzle": puzzle, "message": message, "since": time.time(), "test": test}
            self._stop_at = time.time() + (20 if test else self.timeout_s)
            self._play()
            threading.Thread(target=self._watchdog, daemon=True).start()
            return self.state_unlocked()

    def stop(self) -> dict:
        with self._lock:
            self.active = None
            if winsound:
                winsound.PlaySound(None, 0)
            return self.state_unlocked()

    def set_enabled(self, on: bool) -> dict:
        with self._lock:
            self.enabled = on
        if not on:
            self.stop()
        return self.state()

    def state_unlocked(self) -> dict:
        return {"enabled": self.enabled, "active": dict(self.active) if self.active else None}

    def _play(self) -> None:
        if winsound is None:
            return
        if not self.wav.exists():
            make_siren(self.wav)
        winsound.PlaySound(str(self.wav), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)

    def _watchdog(self) -> None:
        while True:
            time.sleep(1)
            with self._lock:
                if self.active is None:
                    return
                if time.time() < self._stop_at:
                    continue
            self.stop()
            return
