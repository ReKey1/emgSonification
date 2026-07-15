"""Recording to disk.

start() opens a new timestamped session folder and streams samples into it on a
background writer thread (so disk I/O never stalls acquisition). stop() flushes
and finalises a JSON sidecar with the full config and session stats.

Layout — one directory per test subject (`title`), one dataset per recording
(`notes`) inside it:
    recordings/
        subject-01/                        <- title  (the test subject)
            2026-07-07_143012_bicep-left/  <- notes  (this dataset)
                signal.csv     per-sample: t,raw,filtered,envelope,detect,contact_ok
                features.csv   optional per-window feature snapshots
                session.json   config snapshot + metadata (title, notes, duration,
                               co-located raw/filtered burst peaks, ...)
"""

from __future__ import annotations

import csv
import json
import queue
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .config import Config
from .peaks import PeakTracker
from .types import FeatureResult, ProcessedSample

_SIGNAL_HEADER = ["t", "raw", "filtered", "envelope", "detect", "contact_ok"]


def _slug(text: str, fallback: str = "session") -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or fallback


class Recorder:
    """Thread-safe start/stop recorder. One session active at a time."""

    def __init__(self, cfg: Config, base_dir: Optional[Path] = None):
        self.cfg = cfg
        self._base = cfg.recordings_path(base_dir)
        self._lock = threading.Lock()
        self._recording = False
        self._queue: "queue.Queue[Optional[ProcessedSample]]" = queue.Queue()
        self._writer: Optional[threading.Thread] = None
        self._dir: Optional[Path] = None
        self._n = 0
        self._t0 = 0.0
        self._notes = ""
        self._title = ""
        self._peaks = PeakTracker(cfg.sample_rate)  # strongest-burst raw/filtered peaks
        self._t_first = 0.0            # data-clock `t` of the first sample written
        self._t_last = 0.0            # data-clock `t` of the most recent sample
        self._movement_t: Optional[float] = None  # `t` at which the movement was cued
        self._feature_file = None
        self._feature_writer = None
        self._feature_cols: Optional[list] = None

    # -- state --------------------------------------------------------- #
    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def session_dir(self) -> Optional[Path]:
        return self._dir

    @property
    def sample_count(self) -> int:
        return self._n

    # -- control ------------------------------------------------------- #
    def start(self, title: str = "subject", notes: str = "") -> Path:
        """Open a new dataset for a subject.

        `title` names the per-subject directory; `notes` names the dataset folder
        (timestamp-prefixed) inside it. Multiple datasets for the same subject
        share one `title` directory:  recordings/<title>/<stamp>_<notes>/
        """
        with self._lock:
            if self._recording:
                raise RuntimeError("already recording")
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self._title = title
            self._dir = self._base / _slug(title, "subject") / f"{stamp}_{_slug(notes, 'dataset')}"
            self._dir.mkdir(parents=True, exist_ok=True)
            self._notes = notes
            self._n = 0
            self._t0 = time.time()
            self._peaks = PeakTracker(self.cfg.sample_rate)
            self._t_first = 0.0
            self._t_last = 0.0
            self._movement_t = None

            self._queue = queue.Queue()
            self._recording = True
            self._writer = threading.Thread(
                target=self._run, args=(self._dir / "signal.csv",),
                daemon=True, name="emg-recorder")
            self._writer.start()
            return self._dir

    def push(self, s: ProcessedSample) -> None:
        """Non-blocking; ignored when not recording."""
        if self._recording:
            self._queue.put(s)

    def mark_movement(self) -> None:
        """Stamp the movement cue against the data clock.

        Records the most recent sample's `t`, so a caller (e.g. the UI's countdown
        'go' beep) can flag the exact moment the movement was cued. Saved to
        session.json as movement_onset_t / movement_onset_s. No-op when not recording.
        """
        if self._recording:
            self._movement_t = self._t_last

    def push_features(self, t: float, results: Dict[str, FeatureResult]) -> None:
        """Optional low-rate feature snapshot logging."""
        if not self._recording or self._dir is None or not results:
            return
        if self._feature_writer is None:
            self._feature_cols = sorted(results)
            self._feature_file = open(self._dir / "features.csv", "w", newline="", encoding="utf-8")
            self._feature_writer = csv.writer(self._feature_file)
            self._feature_writer.writerow(["t", *self._feature_cols])
        row = [f"{t:.4f}"]
        for k in self._feature_cols:
            v = results.get(k)
            row.append("" if v is None or v.value is None else f"{v.value:.6g}")
        self._feature_writer.writerow(row)

    def stop(self) -> Optional[Path]:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            self._queue.put(None)  # sentinel
        if self._writer is not None:
            self._writer.join(timeout=5.0)
        if self._feature_file is not None:
            self._feature_file.close()
            self._feature_file = self._feature_writer = self._feature_cols = None
        self._write_meta()
        return self._dir

    # -- internals ----------------------------------------------------- #
    def _run(self, path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(_SIGNAL_HEADER)
            while True:
                item = self._queue.get()
                if item is None:
                    break
                if self._n == 0:
                    self._t_first = item.t
                self._t_last = item.t
                self._peaks.update(item.raw, item.filtered, item.t)
                w.writerow([
                    f"{item.t:.4f}",
                    f"{item.raw:.3f}",
                    f"{item.filtered:.4f}",
                    f"{item.envelope:.4f}",
                    "" if item.detect is None else int(item.detect),
                    int(item.contact_ok),
                ])
                self._n += 1

    def _write_meta(self) -> None:
        if self._dir is None:
            return
        meta = {
            "title": self._title,      # test subject (parent directory)
            "notes": self._notes,      # dataset name (this recording folder)
            "started": datetime.fromtimestamp(self._t0).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._t0, 3),
            "n_samples": self._n,
            # When the movement was cued (null if it wasn't). movement_onset_t matches
            # the signal.csv `t` column; movement_onset_s is seconds into the recording.
            "movement_onset_t": None if self._movement_t is None else round(self._movement_t, 4),
            "movement_onset_s": (None if self._movement_t is None
                                 else round(self._movement_t - self._t_first, 4)),
            # Peaks of the strongest burst. peak_raw is measured next to peak_filtered
            # (at peak_t) so both point to the same muscle event — see emg/peaks.py.
            "peak_filtered": round(self._peaks.peak_filtered, 4),
            "peak_raw": round(self._peaks.peak_raw, 3),
            "peak_t": round(self._peaks.peak_filtered_t, 4),
            "sample_rate": self.cfg.sample_rate,
            "mains_hz": self.cfg.mains_hz,
            "config": self.cfg.to_dict(),
        }
        (self._dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
