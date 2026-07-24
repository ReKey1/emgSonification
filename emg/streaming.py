"""Real-time streaming filters for the EMG signal.

The cheezEMG board uses dry conductive PCB pads, which are noisier than gel
electrodes: strong mains hum plus baseline drift from skin-contact changes.
This module cleans that up in software so the mains notch can be placed at the
correct local frequency (50 Hz eastern Japan / 60 Hz western Japan) and so
recordings can be re-processed offline.

Design:
  * Each stage is a second-order-sections (SOS) IIR filter — numerically
    stable and cheap.
  * `StreamingFilter` keeps its internal state between calls, so feeding the
    signal one block at a time gives exactly the same result as filtering the
    whole recording at once, with only a few samples of latency.
  * `EmgFilterChain` wires the stages the signal actually needs:
        raw -> high-pass -> mains notch(es) -> low-pass -> filtered
        |filtered| -> low-pass -> envelope
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy import signal

from .config import Config


class StreamingFilter:
    """Applies a fixed SOS filter to a stream, preserving state across blocks.

    State is lazily initialised from the first sample so the filter starts
    "settled" at the incoming DC level instead of ringing on the first block.
    """

    def __init__(self, sos: np.ndarray):
        self.sos = np.asarray(sos, dtype=np.float64)
        self._zi: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._zi = None

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return x
        if self._zi is None:
            # settle initial conditions to the first incoming value
            self._zi = signal.sosfilt_zi(self.sos) * x.flat[0]
        y, self._zi = signal.sosfilt(self.sos, x, zi=self._zi)
        return y


def _highpass(fs: float, cutoff: float, order: int) -> np.ndarray:
    return signal.butter(order, cutoff, btype="highpass", fs=fs, output="sos")


def _lowpass(fs: float, cutoff: float, order: int) -> np.ndarray:
    return signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")


def _notch(fs: float, f0: float, q: float) -> np.ndarray:
    b, a = signal.iirnotch(f0, q, fs=fs)
    return signal.tf2sos(b, a)


class EmgFilterChain:
    """Full host-side EMG cleanup: band-limit + mains notch + envelope."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._build(cfg)

    def _build(self, cfg: Config) -> None:
        fs = float(cfg.sample_rate)
        stages: List[StreamingFilter] = []

        # 1) High-pass: kill baseline drift / motion artifact.
        if cfg.highpass_hz and cfg.highpass_hz > 0:
            stages.append(StreamingFilter(_highpass(fs, cfg.highpass_hz, cfg.highpass_order)))

        # 2) Mains notch(es): the main win for dry PCB contacts.
        for f0 in cfg.notch_freqs():
            stages.append(StreamingFilter(_notch(fs, f0, cfg.notch_q)))

        # 3) Low-pass: trim high-frequency junk (kept below Nyquist).
        if cfg.lowpass_hz and 0 < cfg.lowpass_hz < 0.99 * cfg.nyquist:
            stages.append(StreamingFilter(_lowpass(fs, cfg.lowpass_hz, cfg.lowpass_order)))

        self.stages = stages

        # Envelope: low-pass the rectified, cleaned signal.
        self.env_filter = StreamingFilter(_lowpass(fs, cfg.envelope_hz, cfg.envelope_order))

    def reset(self) -> None:
        for s in self.stages:
            s.reset()
        self.env_filter.reset()

    def process_block(self, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter a block of raw samples.

        Returns (filtered, envelope), each the same length as `raw`.
        """
        x = np.asarray(raw, dtype=np.float64)
        for stage in self.stages:
            x = stage.process(x)
        filtered = x
        envelope = self.env_filter.process(np.abs(filtered))
        return filtered, envelope

    def process_sample(self, raw: float) -> Tuple[float, float]:
        f, e = self.process_block(np.array([raw], dtype=np.float64))
        return float(f[0]), float(e[0])


def build_chain(cfg: Config) -> EmgFilterChain:
    return EmgFilterChain(cfg)


# --------------------------------------------------------------------------- #
#  Offline (zero-phase) variant
# --------------------------------------------------------------------------- #
def process_offline(cfg: Config, raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-phase equivalent of EmgFilterChain, for offline scoring only.

    Same stages in the same order as `EmgFilterChain._build`, but filtered
    forward-and-backward (`filtfilt`) instead of causally. A causal Butterworth
    delays the signal by its group delay, which biases every timing measurement
    taken off the envelope; forward-backward filtering cancels that delay exactly.
    The live chain must stay causal (it drives audio in real time), so this is a
    separate entry point rather than a flag on the chain.

    Note filtfilt doubles the effective filter order, and can ring slightly
    negative around sharp edges — the envelope is clamped at 0 because a
    rectified envelope is non-negative by definition.
    """
    fs = float(cfg.sample_rate)
    x = np.asarray(raw, dtype=np.float64)
    if x.size == 0:
        return x, x

    if cfg.highpass_hz and cfg.highpass_hz > 0:
        x = signal.sosfiltfilt(_highpass(fs, cfg.highpass_hz, cfg.highpass_order), x)
    for f0 in cfg.notch_freqs():
        x = signal.sosfiltfilt(_notch(fs, f0, cfg.notch_q), x)
    if cfg.lowpass_hz and 0 < cfg.lowpass_hz < 0.99 * cfg.nyquist:
        x = signal.sosfiltfilt(_lowpass(fs, cfg.lowpass_hz, cfg.lowpass_order), x)

    filtered = x
    envelope = signal.sosfiltfilt(
        _lowpass(fs, cfg.envelope_hz, cfg.envelope_order), np.abs(filtered))
    return filtered, np.maximum(envelope, 0.0)


def tkeo(x: np.ndarray) -> np.ndarray:
    """Teager-Kaiser energy operator:  psi[n] = x[n]^2 - x[n+1]*x[n-1].

    A 3-sample nonlinear operator tracking the instantaneous amplitude x frequency
    product, which is why it makes burst boundaries far crisper than amplitude
    alone (Solnik et al. 2010: onset error 40 +/- 99 ms with, 229 +/- 356 ms
    without). Apply to the *band-passed* signal, before rectification — never to
    the envelope, whose smoothing has already destroyed the frequency content the
    operator keys on. Edge samples have no neighbours and are returned as 0.
    """
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    if x.size >= 3:
        out[1:-1] = x[1:-1] ** 2 - x[2:] * x[:-2]
    return out
