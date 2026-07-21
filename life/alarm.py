"""Local audible alarm with acknowledge/silence.

Generates a two-tone siren WAV with the stdlib `wave` module on first use (no
binary assets committed), then plays it asynchronously via the platform's audio
player. Degrades gracefully to a terminal bell if no player is available.
"""
from __future__ import annotations

import logging
import math
import os
import platform
import struct
import subprocess
import threading
import time
import wave

from .models import Severity

log = logging.getLogger("life.alarm")

_SIREN_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "siren.wav"
)


class AlarmManager:
    def __init__(self, config) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.min_severity = Severity.parse(config.get("min_severity", "HIGH"))
        self._active = False
        self._acknowledged = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if self.enabled:
            _ensure_siren(_SIREN_PATH)

    @property
    def active(self) -> bool:
        return self._active

    def trigger(self, severity: Severity) -> None:
        if not self.enabled or severity < self.min_severity:
            return
        with self._lock:
            if self._acknowledged and self._active:
                return
            self._acknowledged = False
            if self._active:
                return
            self._active = True
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        log.warning("ALARM ON (severity=%s)", severity)

    def acknowledge(self) -> None:
        """Silence the current alarm until the next distinct trigger."""
        with self._lock:
            self._acknowledged = True
            self._active = False
            self._stop.set()
        log.info("Alarm acknowledged / silenced.")

    def _loop(self) -> None:
        while not self._stop.is_set():
            _play(_SIREN_PATH)
            # siren clip is ~1.6s; wait a touch, re-check stop flag
            for _ in range(4):
                if self._stop.is_set():
                    return
                time.sleep(0.1)


# ---- audio helpers --------------------------------------------------------
def _ensure_siren(path: str) -> None:
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    framerate = 44100
    duration = 1.6
    amplitude = 22000
    n = int(framerate * duration)
    try:
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(framerate)
            frames = bytearray()
            for i in range(n):
                t = i / framerate
                # sweep between two tones for a classic siren feel
                freq = 700 + 300 * math.sin(2 * math.pi * 1.5 * t)
                sample = int(amplitude * math.sin(2 * math.pi * freq * t))
                frames += struct.pack("<h", sample)
            wf.writeframes(bytes(frames))
        log.info("Generated siren at %s", path)
    except Exception as exc:  # pragma: no cover
        log.warning("Could not generate siren wav: %s", exc)


def _play(path: str) -> None:
    if not os.path.exists(path):
        print("\a", end="", flush=True)
        return
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", path], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            player = _which(["aplay", "paplay", "ffplay"])
            if player:
                args = [player, path]
                if player.endswith("ffplay"):
                    args = [player, "-nodisp", "-autoexit", path]
                subprocess.run(args, check=False,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                print("\a", end="", flush=True)
        elif system == "Windows":  # pragma: no cover - platform specific
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME)
        else:  # pragma: no cover
            print("\a", end="", flush=True)
    except Exception as exc:  # pragma: no cover
        log.debug("Audio playback failed: %s", exc)


def _which(candidates):
    from shutil import which

    for c in candidates:
        if which(c):
            return which(c)
    return None
