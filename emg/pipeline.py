"""The orchestrator: source -> filter chain -> sinks.

Runs acquisition on a background thread and fans each processed sample out to:
    * plot ring buffers   (for the live UI)
    * the Recorder        (when recording)
    * the FeatureBank     (categorizers)
    * the shared audio level (for the Sonifier)

The UI/CLI only ever touch this class. `reconfigure()` rebuilds the filter
chain live (e.g. when switching mains frequency) under a lock.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict, Optional, Tuple

import numpy as np

from .config import Config
from .features import FeatureBank, build_feature_bank
from .peaks import PeakTracker
from .recorder import Recorder
from .streaming import EmgFilterChain, build_chain
from .source import Source
from .types import FeatureResult, ProcessedSample


class Pipeline:
    def __init__(self, cfg: Config, recorder: Optional[Recorder] = None):
        self.cfg = cfg
        self._chain = build_chain(cfg)
        self._chain_lock = threading.Lock()
        self.features: FeatureBank = build_feature_bank(cfg)
        self.recorder = recorder or Recorder(cfg)

        # plot ring buffers (single writer thread, UI reads snapshots)
        n = max(1, int(cfg.plot_seconds * cfg.sample_rate))
        self._pt: Deque[float] = deque(maxlen=n)
        self._praw: Deque[float] = deque(maxlen=n)
        self._pfilt: Deque[float] = deque(maxlen=n)
        self._penv: Deque[float] = deque(maxlen=n)

        # shared/live state
        self._peaks = PeakTracker(cfg.sample_rate)  # live raw/filtered burst peaks (UI readout)
        self._audio_level = 0.0
        self._contact_ok = False
        self._last_detect: Optional[float] = None
        self._count = 0
        self._rate_est = 0.0
        self._rate_t0 = 0.0
        self._rate_n0 = 0

        self._source: Optional[Source] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.error: Optional[str] = None

    # ---- lifecycle --------------------------------------------------- #
    def start(self, source: Source) -> None:
        if self._running:
            self.stop()
        self._source = source
        self._running = True
        self.error = None
        self._rate_t0 = time.perf_counter()
        self._rate_n0 = self._count
        self._thread = threading.Thread(target=self._run, daemon=True, name="emg-pipeline")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._source is not None:
            self._source.close()
            self._source = None

    def reconfigure(self, cfg: Config) -> None:
        """Apply new config; rebuild filter chain and feature bank."""
        self.cfg = cfg
        with self._chain_lock:
            self._chain = build_chain(cfg)
        self.features = build_feature_bank(cfg)
        self._peaks = PeakTracker(cfg.sample_rate)

    # ---- run loop ---------------------------------------------------- #
    def _run(self) -> None:
        assert self._source is not None
        try:
            while self._running:
                samples = self._source.read()
                if not samples:
                    time.sleep(0.001)
                    continue
                raw = np.fromiter((s.raw for s in samples), dtype=np.float64,
                                  count=len(samples))
                with self._chain_lock:
                    filtered, envelope = self._chain.process_block(raw)

                for s, f, e in zip(samples, filtered, envelope):
                    contact_ok = self._interpret_contact(s.detect)
                    ps = ProcessedSample(
                        t=s.t, raw=s.raw, filtered=float(f), envelope=float(e),
                        detect=s.detect, contact_ok=contact_ok)
                    self._dispatch(ps)

                self._update_rate()
        except Exception as e:  # keep the thread's failure visible to the UI
            self.error = f"{type(e).__name__}: {e}"
            self._running = False

    def _dispatch(self, ps: ProcessedSample) -> None:
        self._pt.append(ps.t)
        self._praw.append(ps.raw)
        self._pfilt.append(ps.filtered)
        self._penv.append(ps.envelope)

        self._peaks.update(ps.raw, ps.filtered, ps.t)
        self._audio_level = ps.envelope / self.cfg.envelope_full_scale
        self._contact_ok = ps.contact_ok
        self._last_detect = ps.detect
        self._count += 1

        self.features.update(ps)
        self.recorder.push(ps)

    def _interpret_contact(self, detect: Optional[float]) -> bool:
        if detect is None:
            return True  # no detect channel -> assume ok
        worn = detect >= 0.5
        return worn if self.cfg.contact_high_means_worn else not worn

    def _update_rate(self) -> None:
        now = time.perf_counter()
        dt = now - self._rate_t0
        if dt >= 0.5:
            self._rate_est = (self._count - self._rate_n0) / dt
            self._rate_t0 = now
            self._rate_n0 = self._count

    # ---- accessors for UI/CLI --------------------------------------- #
    @property
    def audio_level(self) -> float:
        return self._audio_level

    @property
    def contact_ok(self) -> bool:
        return self._contact_ok

    @property
    def peak_raw(self) -> float:
        return self._peaks.peak_raw

    @property
    def peak_filtered(self) -> float:
        return self._peaks.peak_filtered

    def reset_peaks(self) -> None:
        """Clear the live peak readout (e.g. when starting a fresh take)."""
        self._peaks.reset()

    @property
    def samples_seen(self) -> int:
        return self._count

    @property
    def measured_rate(self) -> float:
        return self._rate_est

    @property
    def running(self) -> bool:
        return self._running

    @property
    def source_name(self) -> str:
        return self._source.name if self._source else "(none)"

    def compute_features(self) -> Dict[str, FeatureResult]:
        return self.features.compute()

    def plot_snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Copy of the plot buffers as (t, raw, filtered, envelope)."""
        t = np.fromiter(self._pt, dtype=np.float64, count=len(self._pt))
        raw = np.fromiter(self._praw, dtype=np.float64, count=len(self._praw))
        filt = np.fromiter(self._pfilt, dtype=np.float64, count=len(self._pfilt))
        env = np.fromiter(self._penv, dtype=np.float64, count=len(self._penv))
        return t, raw, filt, env

    # ---- recording delegation --------------------------------------- #
    def start_recording(self, title: str, notes: str = ""):
        self._peaks.reset()  # so the live readout tracks this take from zero
        return self.recorder.start(title, notes)

    def mark_movement(self) -> None:
        """Flag the current data-clock time as the movement onset (see Recorder)."""
        self.recorder.mark_movement()

    def stop_recording(self):
        return self.recorder.stop()

    @property
    def is_recording(self) -> bool:
        return self.recorder.is_recording
