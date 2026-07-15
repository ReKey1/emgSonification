"""Real-time sonification of the EMG envelope.

Maps activation -> pitch + loudness with a continuous-phase sine oscillator.
This is the project's original purpose, refactored to read the *smoothed*
host envelope instead of the raw serial value. That single change is what
fixes the old "low latency but bad sound" trade-off: the control signal now
changes smoothly, so the callback-driven (low-latency) path also sounds clean.

The audio callback pulls the current level through a callable, so it is fully
decoupled from acquisition timing.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from .config import Config

LevelFn = Callable[[], float]  # returns current activation in [0, 1]


def beep(frequency: float, duration_s: float = 0.15,
         samplerate: int = 44100, amplitude: float = 0.3) -> bool:
    """Play one short tone, non-blocking. Returns False if audio is unavailable.

    Used for the record-countdown cue. Fires and forgets via sounddevice.play, so
    it never stalls the UI; a missing device just makes it a silent no-op.
    """
    try:
        import sounddevice as sd
        n = max(1, int(samplerate * duration_s))
        t = np.arange(n, dtype=np.float64) / samplerate
        wave = amplitude * np.sin(2.0 * math.pi * frequency * t)
        fade = min(n // 2, int(samplerate * 0.008))  # ~8 ms ramps kill clicks
        if fade > 0:
            wave[:fade] *= np.linspace(0.0, 1.0, fade)
            wave[-fade:] *= np.linspace(1.0, 0.0, fade)
        sd.play(wave.astype(np.float32), samplerate)
        return True
    except Exception:
        return False


class Sonifier:
    def __init__(self, cfg: Config, level_fn: LevelFn):
        self.cfg = cfg
        self._level_fn = level_fn
        self._phase = 0.0
        self._stream = None
        self._idx = np.arange(cfg.audio_block, dtype=np.float64)
        self.available = True
        self.error: Optional[str] = None

    def _callback(self, outdata, frames, time_info, status):  # noqa: ANN001
        t = self._level_fn()
        t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
        freq = self.cfg.freq_min + t * (self.cfg.freq_max - self.cfg.freq_min)
        inc = 2.0 * math.pi * freq / self.cfg.audio_sample_rate
        phases = self._phase + inc * self._idx[:frames]
        outdata[:, 0] = (np.sin(phases) * (t * self.cfg.amp_max)).astype(np.float32)
        self._phase = float((phases[-1] + inc) % (2.0 * math.pi)) if frames else self._phase

    def start(self) -> bool:
        if self._stream is not None:
            return True
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self.cfg.audio_sample_rate,
                blocksize=self.cfg.audio_block,
                channels=1,
                dtype="float32",
                latency=self.cfg.audio_latency,
                callback=self._callback,
            )
            self._stream.start()
            self.available = True
            self.error = None
            return True
        except Exception as e:  # no device / no PortAudio / etc.
            self.available = False
            self.error = str(e)
            self._stream = None
            return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    @property
    def running(self) -> bool:
        return self._stream is not None
