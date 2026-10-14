"""python/scheduler/alarm.py - the "a puzzle was solved" alarm.

The scheduler calls Alarm.start() when a job's solutions pass validation. The daemon then
loops the game's trophy jingles, loud (a WAV played with winsound, so it works with no browser
open) until someone stops it (dashboard button, `mbq alarm stop`) or it times out.
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


SOUNDS = Path(__file__).resolve().parents[1] / "deluxe_clone" / "sounds"
# (sound, start in seconds): the applause comes in on Get Trophy's final chord (its strongest
# attack, 0.93 s), the two mixed; then QUIET seconds before the loop repeats.
JINGLE = [("GetTrophy.ogg", 0.0), ("TrophyJoy.ogg", 0.93)]
QUIET = 3.0


def _game_jingle() -> list[float] | None:
    """The game's trophy sounds decoded to mono floats (needs pygame), or None."""
    try:
        import os
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        import numpy as np
        import pygame
        pygame.mixer.init(frequency=RATE, size=-16, channels=2)
        parts = []
        for name, at in JINGLE:
            a = pygame.sndarray.array(pygame.mixer.Sound(str(SOUNDS / name))).astype(np.float64)
            parts.append((int(RATE * at), a.mean(axis=1) if a.ndim == 2 else a))
        pygame.mixer.quit()
        out = np.zeros(max(s + len(a) for s, a in parts) + int(RATE * QUIET))
        for s, a in parts:
            out[s:s + len(a)] += a
        return list(out / max(1.0, float(np.abs(out).max())))
    except Exception:
        return None


def _chime() -> list[float]:
    """A bright C-major bell arpeggio, if the game sounds can't be decoded."""
    out = []
    notes = [523.25, 659.25, 783.99, 1046.5]
    for k, f in enumerate(notes * 2):
        n = int(RATE * (0.18 if k < 7 else 0.9))
        for i in range(n):
            t = i / RATE
            env = math.exp(-4.5 * t)
            out.append(env * (math.sin(2 * math.pi * f * t) + 0.35 * math.sin(4 * math.pi * f * t)))
    out += [0.0] * int(RATE * 2.5)
    peak = max(abs(v) for v in out) or 1.0
    return [v / peak for v in out]


def make_siren(path: Path) -> Path:
    """The alarm sound: the game's trophy jingles (or a chime), loud, then a pause; loops."""
    samples = _game_jingle() or _chime()
    frames = bytearray()
    for v in samples:
        frames += struct.pack("<h", int(31000 * max(-1.0, min(1.0, v))))
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

    def ensure_sound(self) -> Path:
        if not self.wav.exists():
            make_siren(self.wav)
        return self.wav

    def _play(self) -> None:
        if winsound is None:
            return
        self.ensure_sound()
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
