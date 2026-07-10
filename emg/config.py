"""Central configuration.

Everything tunable lives here so the UI, the CLI, and offline scripts all read
one object. Load/save to JSON so a session's exact settings can be stored next
to its recording.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    # ---- Acquisition ---------------------------------------------------
    source: str = "serial"          # "serial" | "synthetic"
    serial_port: str = "COM7"
    baud_rate: int = 115200
    sample_rate: int = 500          # Hz — MUST match the firmware
    adc_max: float = 1023.0         # 10-bit ADC full scale

    # ---- Mains hum notch ----------------------------------------------
    # Power-line frequency. Japan is split: 50 Hz in the east (Tokyo, Tohoku,
    # Hokkaido) and 60 Hz in the west (Osaka, Nagoya, Kyushu). Dry PCB contacts
    # pick up a lot of this, so getting it right matters. Default 50 Hz (east).
    mains_hz: float = 50.0
    notch_q: float = 30.0           # notch sharpness (higher = narrower)
    # How many mains multiples to notch (fundamental + harmonics under Nyquist).
    # Each notch also removes a sliver of real EMG that overlaps it, so this is
    # a trade-off (measured: 1->98.6%, 2->95.7%, 4->88.3% of EMG energy kept).
    # 2 (50+100 Hz here) kills the two strongest hum components while keeping
    # ~96% of the signal. Raise to 3-4 in a noisy room; drop to 1 if it's clean.
    notch_max_harmonics: int = 2

    # ---- Band limits ---------------------------------------------------
    highpass_hz: float = 20.0       # remove baseline drift / motion artifact
    highpass_order: int = 4
    lowpass_hz: float = 200.0       # keep < Nyquist; set 0 to disable
    lowpass_order: int = 4

    # ---- Envelope ------------------------------------------------------
    envelope_hz: float = 6.0        # smoothing cutoff of |filtered|
    envelope_order: int = 2
    envelope_full_scale: float = 200.0   # envelope value mapped to "1.0" for audio

    # ---- Contact detection --------------------------------------------
    # Firmware `detect` is digitalRead(pin2): 1 or 0. If your board reads the
    # opposite polarity, flip this.
    contact_high_means_worn: bool = True

    # ---- Sonification --------------------------------------------------
    audio_enabled: bool = True
    audio_sample_rate: int = 48000
    audio_block: int = 96           # 48000 / 500 = 96, integer -> no drift
    freq_min: float = 100.0         # Hz at rest
    freq_max: float = 1000.0        # Hz at full activation
    amp_max: float = 1.0
    audio_latency: float = 0.005

    # ---- Display / features -------------------------------------------
    plot_seconds: float = 5.0       # visible window in the live plot
    feature_window_s: float = 1.0   # default sliding window for extractors
    enabled_features: List[str] = field(default_factory=list)

    # ---- Storage -------------------------------------------------------
    recordings_dir: str = "recordings"

    # ------------------------------------------------------------------ #
    #  Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0

    def notch_freqs(self) -> List[float]:
        """Mains fundamental + harmonics that fall safely below Nyquist."""
        limit = 0.95 * self.nyquist
        freqs: List[float] = []
        k = 1
        while len(freqs) < self.notch_max_harmonics:
            f = self.mains_hz * k
            if f >= limit:
                break
            freqs.append(f)
            k += 1
        return freqs

    def feature_window_samples(self) -> int:
        return max(1, int(round(self.feature_window_s * self.sample_rate)))

    def recordings_path(self, base: Path | None = None) -> Path:
        root = Path(base) if base is not None else Path.cwd()
        p = (root / self.recordings_dir).resolve()
        return p

    # ------------------------------------------------------------------ #
    #  (De)serialisation
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
