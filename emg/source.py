"""Sample sources: real serial hardware, or a synthetic generator.

The synthetic source lets the whole app (UI, filtering, recording,
sonification) run and be tested with no board plugged in — and it deliberately
injects mains hum + drift so you can watch the filters clean it up.

Every source assigns `Sample.t` from a running counter divided by the sample
rate, giving uniform spacing regardless of serial jitter.
"""

from __future__ import annotations

import abc
import math
import time
from typing import List, Optional

import numpy as np

from .config import Config
from .types import Sample


class Source(abc.ABC):
    """Common interface. `read()` is non-blocking and returns 0+ new samples."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._n = 0  # running sample index -> timestamps

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def read(self) -> List[Sample]: ...

    def close(self) -> None:  # pragma: no cover - trivial default
        pass

    def _next_t(self) -> float:
        t = self._n / self.cfg.sample_rate
        self._n += 1
        return t


class SerialSource(Source):
    """Reads CSV lines from the cheezEMG firmware.

    Accepts these line shapes (fields split on ','):
        raw,filtered,envelope,detect   (current firmware)
        raw,filtered,envelope
        raw                            (legacy single-column)
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        import serial  # imported here so the package works without pyserial

        self._serial = serial.Serial(cfg.serial_port, cfg.baud_rate, timeout=0)
        self._serial.reset_input_buffer()
        self._buf = b""

    @property
    def name(self) -> str:
        return f"serial:{self.cfg.serial_port}@{self.cfg.baud_rate}"

    def read(self) -> List[Sample]:
        out: List[Sample] = []
        try:
            n = self._serial.in_waiting
        except OSError:
            return out
        if not n:
            return out
        self._buf += self._serial.read(n)
        *lines, self._buf = self._buf.split(b"\n")  # keep incomplete tail
        for line in lines:
            s = self._parse(line.strip())
            if s is not None:
                out.append(s)
        return out

    def _parse(self, line: bytes) -> Optional[Sample]:
        if not line:
            return None
        parts = line.split(b",")
        try:
            raw = float(parts[0])
        except (ValueError, IndexError):
            return None

        def _get(i: int) -> Optional[float]:
            if i < len(parts):
                try:
                    return float(parts[i])
                except ValueError:
                    return None
            return None

        return Sample(
            t=self._next_t(),
            raw=raw,
            fw_filtered=_get(1),
            fw_envelope=_get(2),
            detect=_get(3),
        )

    def close(self) -> None:
        try:
            self._serial.close()
        except Exception:
            pass


class SyntheticSource(Source):
    """Generates EMG-like data in real time for demos and testing.

    Produces: a mid-scale DC offset with slow drift, broadband "muscle" bursts
    every couple of seconds, mains hum at cfg.mains_hz, and sensor noise. The
    filter chain should remove the hum/drift and recover the burst envelope.
    """

    def __init__(self, cfg: Config, seed: Optional[int] = None):
        super().__init__(cfg)
        self._rng = np.random.default_rng(seed)
        self._start = time.perf_counter()
        self._emitted = 0

    @property
    def name(self) -> str:
        return "synthetic"

    def _activation(self, t: float) -> float:
        """0..1 muscle activation: a ~0.6 s burst every ~2.5 s."""
        period = 2.5
        phase = (t % period) / period
        if phase < 0.24:  # burst envelope (raised cosine)
            return 0.5 - 0.5 * math.cos(2 * math.pi * phase / 0.24)
        return 0.0

    def read(self) -> List[Sample]:
        now = time.perf_counter()
        due = int((now - self._start) * self.cfg.sample_rate) - self._emitted
        if due <= 0:
            return []
        due = min(due, 250)  # cap catch-up after a stall

        out: List[Sample] = []
        f = self.cfg.mains_hz
        for _ in range(due):
            t = self._emitted / self.cfg.sample_rate
            act = self._activation(t)
            emg = self._rng.standard_normal() * 130.0 * act          # muscle burst
            hum = 45.0 * math.sin(2 * math.pi * f * t)               # mains
            drift = 25.0 * math.sin(2 * math.pi * 0.3 * t)           # baseline wander
            noise = self._rng.standard_normal() * 4.0                # sensor floor
            raw = 512.0 + emg + hum + drift + noise
            raw = float(min(self.cfg.adc_max, max(0.0, raw)))
            self._emitted += 1
            out.append(Sample(t=self._next_t(), raw=raw, detect=1.0))
        return out


def open_source(cfg: Config) -> Source:
    """Factory. Raises on serial failure so the caller can fall back."""
    if cfg.source == "synthetic":
        return SyntheticSource(cfg)
    return SerialSource(cfg)
