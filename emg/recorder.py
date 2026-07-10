"""Recording to disk.

start() opens a new timestamped session folder and streams samples into it on a
background writer thread (so disk I/O never stalls acquisition). stop() flushes
and finalises a JSON sidecar with the full config and session stats.

Layout:
    recordings/
        2026-07-07_143012_bicep-curl/
            signal.csv     per-sample: t,raw,filtered,envelope,detect,contact_ok
            features.csv   optional per-window feature snapshots
            session.json   config snapshot + metadata (duration, n_samples, notes)
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
from .types import FeatureResult, ProcessedSample

_SIGNAL_HEADER = ["t", "raw", "filtered", "envelope", "detect", "contact_ok"]


def _slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text or "session"


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
        self._name = ""
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
    def start(self, name: str = "session", notes: str = "") -> Path:
        with self._lock:
            if self._recording:
                raise RuntimeError("already recording")
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self._name = name
            self._dir = self._base / f"{stamp}_{_slug(name)}"
            self._dir.mkdir(parents=True, exist_ok=True)
            self._notes = notes
            self._n = 0
            self._t0 = time.time()

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
            "name": self._name,
            "notes": self._notes,
            "started": datetime.fromtimestamp(self._t0).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - self._t0, 3),
            "n_samples": self._n,
            "sample_rate": self.cfg.sample_rate,
            "mains_hz": self.cfg.mains_hz,
            "config": self.cfg.to_dict(),
        }
        (self._dir / "session.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
