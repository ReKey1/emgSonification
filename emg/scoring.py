"""Offline dataset scoring — turn a recorded session into research-grounded numbers.

    >>> This *is* implemented, unlike the live categorizers in features.py. Those
    >>> stay stubs because *which quality to sonify in real time* is an open thesis
    >>> question. Here we do post-hoc analysis of finished recordings, where the
    >>> established EMG metrics are well defined and worth computing verbatim.

Given one recording folder (a `signal.csv` + optional `session.json`), we compute
a row of single-channel surface-EMG properties drawn from the thesis literature
(`../research`), then `score_cli.py` collects one row per dataset into a clean CSV.

Metrics and their evidence base (citations resolve in ../research/references.md):

    amplitude
        rms_amplitude, mav  — standard EMG activation-level features (RMS, mean
        absolute value). The raw material of the "Amplitude" sonification condition
        (semester_plan.md).
    snr
        baseline_noise, snr_db — active-vs-rest signal-to-noise. The research makes
        a signal-quality gate a *hard requirement*: reps whose SNR/baseline noise is
        too poor must be excluded so the system never adapts to electrode artifact
        (literature_review.md §9, Argument 4).
    mains
        mains_residual — fraction of power left at the mains frequency + harmonics
        after host notching. Sensor-specific: the dry PCB pads pick up a lot of
        50/60 Hz hum, which is the whole reason host filtering exists.
    spectral
        median_freq_hz — median power frequency; a standard spectral EMG descriptor
        (established for fatigue; a weaker skill discriminator — semester_plan.md).
    onset
        n_reps, rise_time_ms, onset_sharpness — contraction count and rise time from
        onset to peak. Sharper onsets track skill acquisition [R27], [R28].
    consistency
        inter_rep_consistency = 1 - mean(CV of the per-rep envelope profile). The
        thesis's recommended primary reward: lower inter-rep variability = a more
        stable motor program [R19]-[R22].
    contact
        contact_frac — fraction of samples the wear/contact flag reported good.

Multichannel metrics from the research — co-contraction [R23]-[R26] and recruitment
specificity — need a second electrode and are deliberately left out here (they are
deferred in semester_plan.md too). Add them as Scorers when 2-channel data exists.

--------------------------------------------------------------------------
HOW TO ADD A METRIC
--------------------------------------------------------------------------
1. Subclass Scorer, set `name` and the `columns` tuple it emits.
2. Implement compute(ctx) -> {column: value_or_None}; read ctx.filtered /
   ctx.envelope (numpy arrays) and ctx.reps() (cached burst segmentation).
3. Decorate with @register_scorer. score_dataset() picks it up automatically and
   its columns append to the CSV. Return None for "not computable on this data"
   (e.g. too few reps) — the CSV leaves that cell blank rather than guessing.

The composite `quality_score` is NOT a Scorer: it is a transparent, reconfigurable
signal-quality gate over the metrics above (see QualityWeights / composite_quality).
The motor-learning metrics are reported raw, never baked into a single verdict —
which of them best predicts learning is exactly what the thesis is trying to find out.
"""

from __future__ import annotations

import abc
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from scipy import signal

from .config import Config

# Column order for the output CSV. Identity/metadata first, then metrics grouped
# as quality-gate -> amplitude -> spectral -> temporal/learning, then the score.
META_COLUMNS: Tuple[str, ...] = (
    "subject", "dataset", "started", "duration_s", "n_samples",
    "sample_rate", "mains_hz",
)
SCORE_COLUMN = "quality_score"


# --------------------------------------------------------------------------- #
#  Burst / rep segmentation (shared by several scorers)
# --------------------------------------------------------------------------- #
def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Half-open [start, end) index ranges of contiguous True in a bool mask."""
    if mask.size == 0:
        return []
    m = mask.astype(np.int8)
    edges = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def segment_reps(
    envelope: np.ndarray,
    fs: float,
    onset_frac: float = 0.20,
    min_rep_s: float = 0.12,
    merge_gap_s: float = 0.10,
) -> List[Tuple[int, int]]:
    """Amplitude-threshold onset detection over the envelope.

    A rep is a run where the envelope rises a fraction `onset_frac` of the way
    from its resting level (20th pct) to its peak (95th pct). Short runs are
    dropped and runs separated by < `merge_gap_s` are merged, the standard
    amplitude onset-detection recipe [R15]. Returns [start, end) sample ranges;
    empty when the signal is essentially flat (no bursts, e.g. a constant test
    pattern or a pure-noise rest recording).
    """
    n = envelope.size
    if n == 0 or fs <= 0:
        return []
    rest = float(np.percentile(envelope, 20))
    peak = float(np.percentile(envelope, 95))
    span = peak - rest
    if span <= 1e-9:
        return []
    thr = rest + onset_frac * span

    reps = _runs(envelope > thr)
    if not reps:
        return []

    merge_gap = int(round(merge_gap_s * fs))
    merged: List[Tuple[int, int]] = []
    for s, e in reps:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    min_len = max(1, int(round(min_rep_s * fs)))
    return [(s, e) for s, e in merged if e - s >= min_len]


# --------------------------------------------------------------------------- #
#  Dataset context passed to every scorer
# --------------------------------------------------------------------------- #
@dataclass
class DatasetContext:
    """Everything a scorer needs about one loaded recording.

    `reps()` runs onset detection once and caches it, so amplitude, SNR, onset
    and consistency all share a single segmentation.
    """
    subject: str
    dataset: str
    path: Path
    meta: dict
    cfg: Config
    fs: float
    t: np.ndarray
    raw: np.ndarray
    filtered: np.ndarray
    envelope: np.ndarray
    contact: np.ndarray            # float array, NaN where the flag was absent
    _reps: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)

    def reps(self) -> List[Tuple[int, int]]:
        if self._reps is None:
            self._reps = segment_reps(self.envelope, self.fs)
        return self._reps

    def active_mask(self) -> np.ndarray:
        mask = np.zeros(self.filtered.size, dtype=bool)
        for s, e in self.reps():
            mask[s:e] = True
        return mask


# --------------------------------------------------------------------------- #
#  Scorer base + registry
# --------------------------------------------------------------------------- #
class Scorer(abc.ABC):
    """Computes one or more metric columns from a DatasetContext.

    compute() must never raise and must return a value for every name in
    `columns` (use None when the metric cannot be computed on this data).
    """
    name: str = "unnamed"
    columns: Tuple[str, ...] = ()

    @abc.abstractmethod
    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]: ...


_REGISTRY: Dict[str, Type[Scorer]] = {}


def register_scorer(cls: Type[Scorer]) -> Type[Scorer]:
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ValueError(f"scorer name already registered: {cls.name!r}")
    _REGISTRY[cls.name] = cls
    return cls


def available_scorers() -> List[str]:
    return sorted(_REGISTRY)


def metric_columns() -> List[str]:
    """Every metric column, in scorer-registration then declared order."""
    cols: List[str] = []
    for cls in _REGISTRY.values():
        cols.extend(cls.columns)
    return cols


# --------------------------------------------------------------------------- #
#  Scorers
# --------------------------------------------------------------------------- #
@register_scorer
class Amplitude(Scorer):
    """Activation level: RMS and mean-absolute-value of the cleaned signal."""
    name = "amplitude"
    columns = ("rms_amplitude", "mav")

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        f = ctx.filtered
        if f.size == 0:
            return {"rms_amplitude": None, "mav": None}
        return {
            "rms_amplitude": float(np.sqrt(np.mean(f * f))),
            "mav": float(np.mean(np.abs(f))),
        }


@register_scorer
class NoiseSnr(Scorer):
    """Signal-quality gate: rest-period noise floor and active-vs-rest SNR (dB)."""
    name = "snr"
    columns = ("baseline_noise", "snr_db")

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        f = ctx.filtered
        if f.size == 0:
            return {"baseline_noise": None, "snr_db": None}
        active = ctx.active_mask()
        rest = f[~active]
        act = f[active]
        base = float(np.sqrt(np.mean(rest * rest))) if rest.size else None
        if base is None or base <= 0 or act.size == 0:
            # No bursts detected -> the whole record is "rest": report its RMS as
            # the noise floor, but SNR is undefined without an active segment.
            return {"baseline_noise": base, "snr_db": None}
        rms_act = float(np.sqrt(np.mean(act * act)))
        return {"baseline_noise": base, "snr_db": 20.0 * math.log10(rms_act / base)}


@register_scorer
class MainsResidual(Scorer):
    """Fraction of power left at mains + harmonics after notching (lower = cleaner)."""
    name = "mains"
    columns = ("mains_residual",)

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        f = ctx.filtered
        fs = ctx.fs
        if f.size < 64 or fs <= 0:
            return {"mains_residual": None}
        nperseg = int(min(f.size, max(64, fs)))  # ~1 s windows when available
        freqs, psd = signal.welch(f, fs=fs, nperseg=nperseg)
        total = float(np.sum(psd))
        if total <= 0:
            return {"mains_residual": None}
        m = ctx.cfg.mains_hz
        bw = 1.5
        band = np.zeros_like(freqs, dtype=bool)
        k = 1
        while m * k < 0.99 * (fs / 2.0):
            band |= (freqs >= m * k - bw) & (freqs <= m * k + bw)
            k += 1
        return {"mains_residual": float(np.sum(psd[band]) / total)}


@register_scorer
class Spectral(Scorer):
    """Median power frequency of the active signal (standard spectral descriptor)."""
    name = "spectral"
    columns = ("median_freq_hz",)

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        f = ctx.filtered
        fs = ctx.fs
        if f.size == 0 or fs <= 0:
            return {"median_freq_hz": None}
        active = ctx.active_mask()
        x = f[active] if active.any() else f
        if x.size < 64:
            return {"median_freq_hz": None}
        nperseg = int(min(x.size, max(64, fs)))
        freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
        cumulative = np.cumsum(psd)
        if cumulative[-1] <= 0:
            return {"median_freq_hz": None}
        half = cumulative[-1] / 2.0
        return {"median_freq_hz": float(np.interp(half, cumulative, freqs))}


@register_scorer
class Onset(Scorer):
    """Contraction count and onset sharpness (rise time from onset to peak)."""
    name = "onset"
    columns = ("n_reps", "rise_time_ms", "onset_sharpness")

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        reps = ctx.reps()
        env = ctx.envelope
        fs = ctx.fs
        n = len(reps)
        if n == 0 or fs <= 0:
            return {"n_reps": float(n), "rise_time_ms": None, "onset_sharpness": None}
        rises_ms: List[float] = []
        for s, e in reps:
            seg = env[s:e]
            if seg.size < 2:
                continue
            peak_i = int(np.argmax(seg))
            rise_ms = (peak_i / fs) * 1000.0
            if rise_ms > 0:
                rises_ms.append(rise_ms)
        if not rises_ms:
            return {"n_reps": float(n), "rise_time_ms": None, "onset_sharpness": None}
        mean_rise = float(np.mean(rises_ms))
        # Sharpness = inverse rise time (1/s): higher means a more decisive onset.
        return {
            "n_reps": float(n),
            "rise_time_ms": mean_rise,
            "onset_sharpness": 1000.0 / mean_rise,
        }


@register_scorer
class Consistency(Scorer):
    """Inter-rep consistency = 1 - mean CV of the time-normalised rep envelope."""
    name = "consistency"
    columns = ("inter_rep_consistency",)
    profile_len = 100

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        reps = ctx.reps()
        env = ctx.envelope
        if len(reps) < 2:
            return {"inter_rep_consistency": None}
        profiles: List[np.ndarray] = []
        xq = np.linspace(0.0, 1.0, self.profile_len)
        for s, e in reps:
            seg = env[s:e]
            if seg.size < 2:
                continue
            xp = np.linspace(0.0, 1.0, seg.size)
            profiles.append(np.interp(xq, xp, seg))
        if len(profiles) < 2:
            return {"inter_rep_consistency": None}
        mat = np.vstack(profiles)
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        good = mean > 1e-9
        if not good.any():
            return {"inter_rep_consistency": None}
        cv = std[good] / mean[good]
        return {"inter_rep_consistency": float(1.0 - float(np.mean(cv)))}


@register_scorer
class Contact(Scorer):
    """Fraction of samples the wear/contact flag reported good (data-quality gate)."""
    name = "contact"
    columns = ("contact_frac",)

    def compute(self, ctx: DatasetContext) -> Dict[str, Optional[float]]:
        c = ctx.contact
        valid = c[~np.isnan(c)] if c.size else c
        if valid.size == 0:
            return {"contact_frac": None}
        return {"contact_frac": float(np.mean(valid))}


# --------------------------------------------------------------------------- #
#  Composite signal-quality gate  (transparent + reconfigurable, not a Scorer)
# --------------------------------------------------------------------------- #
@dataclass
class QualityWeights:
    """Knobs for composite_quality(). Defaults are a documented heuristic, not a
    law — the research fixes only that a quality gate must *exist* (exclude noisy
    reps), not its exact form. Tune freely for your rig."""
    snr_floor_db: float = 3.0     # at/below this, the SNR sub-score is 0
    snr_good_db: float = 20.0     # at/above this, the SNR sub-score is 1
    mains_tol: float = 0.20       # mains_residual at/above this -> mains sub-score 0
    w_snr: float = 0.6
    w_mains: float = 0.4


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def composite_quality(row: Dict[str, Optional[float]],
                      w: Optional[QualityWeights] = None) -> Optional[float]:
    """0..1 signal-integrity score: is this recording clean enough to trust?

    Blends the available quality sub-scores (SNR, mains residual), reweighting to
    whatever is present, then gates on contact fraction. Returns None when no
    quality evidence exists at all. Deliberately excludes the motor-learning
    metrics (consistency/sharpness/amplitude): those describe the movement, not
    whether the signal is usable, and which of them matters is an open question.
    """
    w = w or QualityWeights()
    parts: List[Tuple[float, float]] = []  # (weight, sub_score)

    snr = row.get("snr_db")
    if snr is not None and w.snr_good_db > w.snr_floor_db:
        parts.append((w.w_snr, _clamp01(
            (snr - w.snr_floor_db) / (w.snr_good_db - w.snr_floor_db))))

    mains = row.get("mains_residual")
    if mains is not None and w.mains_tol > 0:
        parts.append((w.w_mains, _clamp01(1.0 - mains / w.mains_tol)))

    if not parts:
        return None
    total_w = sum(wt for wt, _ in parts)
    base = sum(wt * sub for wt, sub in parts) / total_w if total_w > 0 else 0.0

    contact = row.get("contact_frac")
    gate = contact if contact is not None else 1.0
    return _clamp01(base * gate)


# --------------------------------------------------------------------------- #
#  Loading a recording
# --------------------------------------------------------------------------- #
def _read_signal_csv(path: Path) -> Dict[str, np.ndarray]:
    """Read signal.csv into named float columns; blank cells become NaN."""
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise ValueError(f"empty signal file: {path}")
        cols: Dict[str, List[float]] = {h: [] for h in header}
        for line in reader:
            if not line:
                continue
            for h, cell in zip(header, line):
                cell = cell.strip()
                cols[h].append(float(cell) if cell else math.nan)
    return {h: np.asarray(v, dtype=np.float64) for h, v in cols.items()}


def load_dataset(dataset_dir: Path, recordings_root: Path,
                 refilter: bool = False,
                 mains_hz: Optional[float] = None) -> DatasetContext:
    """Load one recording folder into a DatasetContext.

    Uses the stored `filtered`/`envelope` columns by default (what the subject
    actually experienced). Re-derives them from `raw` via the session's own
    filter config when `refilter=True` or when those columns are missing — the
    "recordings can be re-filtered offline" contract from the project design.
    """
    dataset_dir = Path(dataset_dir)
    meta_path = dataset_dir / "session.json"
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = Config.from_dict(meta.get("config", {})) if meta.get("config") else Config()
    if mains_hz is not None:
        cfg.mains_hz = mains_hz

    data = _read_signal_csv(dataset_dir / "signal.csv")
    if "raw" not in data:
        raise ValueError(f"{dataset_dir/'signal.csv'} has no 'raw' column")
    raw = data["raw"]
    fs = float(cfg.sample_rate)
    t = data.get("t")
    if t is None or t.size != raw.size:
        t = np.arange(raw.size, dtype=np.float64) / fs

    have_filtered = "filtered" in data and not np.all(np.isnan(data["filtered"]))
    have_envelope = "envelope" in data and not np.all(np.isnan(data["envelope"]))
    if refilter or not (have_filtered and have_envelope):
        from .streaming import build_chain
        filtered, envelope = build_chain(cfg).process_block(raw)
    else:
        filtered = data["filtered"]
        envelope = data["envelope"]

    contact = data.get("contact_ok", np.full(raw.size, math.nan))

    # subject = folder grouping the dataset under recordings/; "" if it sits
    # directly in the recordings root (the old flat layout).
    try:
        rel = dataset_dir.resolve().relative_to(recordings_root.resolve())
        subject = rel.parts[0] if len(rel.parts) > 1 else ""
    except ValueError:
        subject = dataset_dir.parent.name

    return DatasetContext(
        subject=subject,
        dataset=dataset_dir.name,
        path=dataset_dir,
        meta=meta,
        cfg=cfg,
        fs=fs,
        t=t,
        raw=raw,
        filtered=np.asarray(filtered, dtype=np.float64),
        envelope=np.asarray(envelope, dtype=np.float64),
        contact=np.asarray(contact, dtype=np.float64),
    )


# --------------------------------------------------------------------------- #
#  Scoring one dataset -> one row
# --------------------------------------------------------------------------- #
def score_dataset(ctx: DatasetContext,
                  weights: Optional[QualityWeights] = None) -> Dict[str, object]:
    """Run every registered scorer over a context and return one flat CSV row."""
    row: Dict[str, object] = {
        "subject": ctx.subject,
        "dataset": ctx.dataset,
        "started": ctx.meta.get("started", ""),
        "duration_s": ctx.meta.get("duration_s", round(ctx.raw.size / ctx.fs, 3)),
        "n_samples": ctx.raw.size,
        "sample_rate": int(ctx.fs),
        "mains_hz": ctx.cfg.mains_hz,
    }
    metrics: Dict[str, Optional[float]] = {}
    for cls in _REGISTRY.values():
        try:
            metrics.update(cls().compute(ctx))
        except Exception as exc:  # a broken scorer must not sink the whole run
            for col in cls.columns:
                metrics[col] = None
            metrics.setdefault("_errors", "")
            metrics["_errors"] = f"{metrics['_errors']} {cls.name}:{exc}".strip()
    row.update(metrics)
    row[SCORE_COLUMN] = composite_quality(metrics, weights)
    return row


def all_columns() -> List[str]:
    """Full CSV header in stable order."""
    return [*META_COLUMNS, *metric_columns(), SCORE_COLUMN]


def find_datasets(recordings_root: Path) -> List[Path]:
    """Every folder under recordings_root that contains a signal.csv, sorted."""
    root = Path(recordings_root)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("signal.csv"))


def score_all(recordings_root: Path, refilter: bool = False,
              mains_hz: Optional[float] = None,
              weights: Optional[QualityWeights] = None,
              ) -> Tuple[List[Dict[str, object]], List[Tuple[Path, str]]]:
    """Score every dataset under recordings_root.

    Returns (rows, failures) where failures is a list of (path, error message)
    for datasets that could not be loaded at all.
    """
    rows: List[Dict[str, object]] = []
    failures: List[Tuple[Path, str]] = []
    for d in find_datasets(recordings_root):
        try:
            ctx = load_dataset(d, recordings_root, refilter=refilter, mains_hz=mains_hz)
            rows.append(score_dataset(ctx, weights))
        except Exception as exc:
            failures.append((d, f"{type(exc).__name__}: {exc}"))
    rows.sort(key=lambda r: (str(r.get("subject", "")), str(r.get("started", "")),
                             str(r.get("dataset", ""))))
    return rows, failures


def _fmt(value: object) -> str:
    """Format a cell for CSV: blanks for None/NaN, ~4 significant figures."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def write_csv(rows: List[Dict[str, object]], out_path: Path) -> Path:
    """Write scored rows to a clean CSV with the canonical column order."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    columns = all_columns()
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_fmt(row.get(col)) for col in columns])
    return out_path
