"""Offline scoring — turn recorded sessions into research-grounded numbers.

    >>> This *is* implemented, unlike the live categorizers in features.py. Those
    >>> stay stubs because *which quality to sonify in real time* is an open thesis
    >>> question. Here we do post-hoc analysis of finished recordings, where the
    >>> established EMG metrics are well defined and worth computing verbatim.

RECORDING MODEL (important — the scorer's whole shape follows from it)
--------------------------------------------------------------------------
Each subject directory holds:

    recordings/<subject>/
        <stamp>_amp/            <- MVC test: one sustained maximal contraction
        <stamp>_overheadfast/   <- one REP of the "overheadfast" movement
        <stamp>_overheadfast/   <- another rep of the same movement
        ... (~6 reps per movement category) ...
        <stamp>_point/          <- reps of another category ...

So **one folder = one rep**, and the reps of a movement are *grouped by category*
(the folder-name suffix after the timestamp). The unit of analysis is therefore
the **category** (a group of ~6 rep files), not the individual file. We report:

    scores.csv   one row per (subject, category): the reps aggregated
    reps.csv     one row per individual rep file (drill-down)

The `amp` recording is the subject's **MVC reference**, not a scored category. Its
maximum drives %MVC normalisation. To avoid a lone sensor-shift spike defining
100%, the reference is a *robust* max — a high percentile (99th) of the amp
contraction's envelope (`robust_max`), which discards the top ~1% of samples where
a shift artefact would land while still reflecting the true attainable peak. Rep
envelopes are then clipped only well above MVC (`SPIKE_TRIM_FACTOR` × MVC) so that
gross shift artefacts are removed but normal dynamic overshoot (a fast rep can beat
an isometric hold) is preserved — %MVC is therefore not capped at 100 (`score_rep`).

Metrics and their evidence base (citations resolve in ../research/references.md):

    amplitude   rms_amplitude, mav — activation level (RMS, mean-abs-value) of the
                cleaned signal over the rep. peak — envelope peak. mean_pct_mvc /
                peak_pct_mvc — the same expressed as %MVC (comparable across
                subjects), the raw material of the "Amplitude" condition.
    snr         baseline_noise, snr_db — rest-vs-active signal-to-noise. The
                research makes a signal-quality gate a hard requirement so the
                system never adapts to electrode artefact (lit. review §9, Arg 4).
    mains       mains_residual — fraction of power left at mains + harmonics after
                host notching. The dry PCB pads pick up a lot of 50/60 Hz hum.
    spectral    median_freq_hz — median power frequency (a standard descriptor).
    onset       rise_time_ms, onset_sharpness — rise from rep onset to peak.
                Sharper onsets track skill acquisition [R27],[R28].
    consistency inter_rep_consistency = 1 - mean(CV across the reps' time-normalised
                envelope profiles). Computed ACROSS the category's rep files (the
                whole point of grouping) — the thesis's recommended primary reward:
                lower inter-rep variability = a more stable motor program [R19]-[R22].
    contact     contact_frac — fraction of samples the wear/contact flag reported OK.

Multichannel metrics (co-contraction [R23]-[R26], recruitment specificity) need a
second electrode and are deliberately left out (deferred in semester_plan.md too).

--------------------------------------------------------------------------
HOW TO ADD A METRIC
--------------------------------------------------------------------------
Per-rep metrics live in `score_rep()` — add a key to the returned dict and to
REP_METRIC_COLUMNS; it will flow into reps.csv and, via the mean in
`aggregate_category()`, into scores.csv automatically. A metric that only makes
sense across reps (like consistency) is computed in `aggregate_category()` from the
collected profiles. The composite `quality_score` is a transparent, reconfigurable
signal-quality gate over the aggregated metrics (QualityWeights / composite_quality)
— the motor-learning metrics are reported raw, never baked into one verdict.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from .config import Config

# --------------------------------------------------------------------------- #
#  Columns  (internal keys are unit-less; labels below add units for the CSV)
# --------------------------------------------------------------------------- #
SCORE_COLUMN = "quality_score"

# Per-rep metric keys, produced by score_rep(). Category rows carry the mean of
# each of these (plus the cross-rep extras) — keep the two in sync.
REP_METRIC_COLUMNS: Tuple[str, ...] = (
    "peak_pct_mvc", "mean_pct_mvc",
    "rms_amplitude", "mav", "peak",
    "baseline_noise", "snr_db", "mains_residual", "median_freq_hz",
    "rise_time_ms", "onset_sharpness", "contact_frac",
)
# Metrics that only exist at the category level (computed across the reps).
CATEGORY_EXTRA_COLUMNS: Tuple[str, ...] = ("inter_rep_consistency",)

REP_COLUMNS: Tuple[str, ...] = (
    "subject", "category", "rep", "started",
    "duration_s", "n_samples", "sample_rate", "mains_hz",
    *REP_METRIC_COLUMNS,
)
CATEGORY_COLUMNS: Tuple[str, ...] = (
    "subject", "category", "n_reps", "mvc_reference",
    *REP_METRIC_COLUMNS, *CATEGORY_EXTRA_COLUMNS, SCORE_COLUMN,
)

# Human-readable CSV headers with units. Amplitude columns are raw ADC counts (the
# signal is uncalibrated 10-bit ADC, adc_max=1023 — no µV conversion); %MVC columns
# are normalised to the subject's MVC. Columns not listed fall back to their key.
COLUMN_LABELS: Dict[str, str] = {
    "duration_s": "duration (s)",
    "sample_rate": "sample_rate (Hz)",
    "mains_hz": "mains (Hz)",
    "mvc_reference": "MVC ref (counts)",
    "peak_pct_mvc": "peak (%MVC)",
    "mean_pct_mvc": "mean (%MVC)",
    "rms_amplitude": "rms_amplitude (counts)",
    "mav": "mav (counts)",
    "peak": "peak (counts)",
    "baseline_noise": "baseline_noise (counts)",
    "snr_db": "snr (dB)",
    "mains_residual": "mains_residual (frac)",
    "median_freq_hz": "median_freq (Hz)",
    "rise_time_ms": "rise_time (ms)",
    "onset_sharpness": "onset_sharpness (1/s)",
    "inter_rep_consistency": "inter_rep_consistency (0-1)",
    "contact_frac": "contact (frac)",
    "quality_score": "quality_score (0-1)",
}

TIME_NORM_LEN = 100     # samples in a time-normalised rep envelope profile
MVC_PERCENTILE = 99.0   # percentile of the amp contraction taken as 100% MVC
SPIKE_TRIM_FACTOR = 1.5  # clip rep envelopes above this × MVC (shift artefacts)
AMP_CATEGORY = "amp"    # the MVC-reference folder name (not a scored category)
_STAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}_")


def category_of(dataset_dir: Path) -> str:
    """Movement category = the folder name with its `YYYY-MM-DD_HHMMSS_` prefix
    stripped, so all reps of one movement share a category (`overheadfast`, ...)."""
    name = Path(dataset_dir).name
    stripped = _STAMP_RE.sub("", name)
    return stripped or name


# --------------------------------------------------------------------------- #
#  Burst / rep segmentation
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

    A burst is a run where the envelope rises `onset_frac` of the way from its
    resting level (20th pct) to its peak (95th pct). Short runs are dropped and
    runs separated by < `merge_gap_s` are merged [R15]. Returns [start, end)
    sample ranges; empty when the signal is essentially flat.

    Within a single rep file this normally finds the one rep; `dominant_rep`
    picks the strongest run when noise splits it into several.
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
            # max(): a run nested inside the span it merges into must not shorten it
            merged[-1] = (merged[-1][0], max(e, merged[-1][1]))
        else:
            merged.append((s, e))

    min_len = max(1, int(round(min_rep_s * fs)))
    return [(s, e) for s, e in merged if e - s >= min_len]


def dominant_rep(envelope: np.ndarray, fs: float) -> Tuple[int, int]:
    """The single active burst of a one-rep file: the strongest run from
    `segment_reps` (by envelope peak). Falls back to the whole array when no burst
    is detected (e.g. a very short or flat recording)."""
    reps = segment_reps(envelope, fs)
    if not reps:
        return (0, envelope.size)
    return max(reps, key=lambda se: float(envelope[se[0]:se[1]].max())
               if se[1] > se[0] else 0.0)


def rep_profile(seg: np.ndarray, n: int = TIME_NORM_LEN) -> Optional[np.ndarray]:
    """Resample a rep's envelope onto `n` points in [0,1] time so reps of different
    durations can be compared shape-for-shape."""
    if seg.size < 2:
        return None
    xp = np.linspace(0.0, 1.0, seg.size)
    xq = np.linspace(0.0, 1.0, n)
    return np.interp(xq, xp, seg)


def robust_max(envelope: np.ndarray, fs: float,
               pct: float = MVC_PERCENTILE) -> Optional[float]:
    """MVC reference: a high percentile of the amp contraction's envelope.

    Taken over the amp file's dominant burst (its actual contraction, not the rest
    before/after). The `pct`th percentile (default 99th) discards the top ~1% of
    samples — where a lone sensor-shift spike would sit — while still reflecting the
    true attainable peak, so one artefact can't define 100% MVC. A moving-average
    "sustained max" was tried first but sat well below real rep peaks and made every
    %MVC saturate; a percentile is on the same instantaneous scale as a rep's peak.
    """
    if envelope.size == 0 or fs <= 0:
        return None
    s, e = dominant_rep(envelope, fs)
    seg = envelope[s:e] if e > s else envelope
    if seg.size == 0:
        seg = envelope
    return float(np.percentile(seg, pct))


# --------------------------------------------------------------------------- #
#  Spectral helpers (shared by mains + spectral metrics)
# --------------------------------------------------------------------------- #
def _mains_residual(f: np.ndarray, fs: float, mains_hz: float) -> Optional[float]:
    if f.size < 64 or fs <= 0:
        return None
    nperseg = int(min(f.size, max(64, fs)))
    freqs, psd = signal.welch(f, fs=fs, nperseg=nperseg)
    total = float(np.sum(psd))
    if total <= 0:
        return None
    bw = 1.5
    band = np.zeros_like(freqs, dtype=bool)
    k = 1
    while mains_hz * k < 0.99 * (fs / 2.0):
        band |= (freqs >= mains_hz * k - bw) & (freqs <= mains_hz * k + bw)
        k += 1
    return float(np.sum(psd[band]) / total)


def _median_freq(x: np.ndarray, fs: float) -> Optional[float]:
    if x.size < 64 or fs <= 0:
        return None
    nperseg = int(min(x.size, max(64, fs)))
    freqs, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    cumulative = np.cumsum(psd)
    if cumulative[-1] <= 0:
        return None
    return float(np.interp(cumulative[-1] / 2.0, cumulative, freqs))


# --------------------------------------------------------------------------- #
#  Loading one rep file
# --------------------------------------------------------------------------- #
@dataclass
class DatasetContext:
    """Everything loaded from one recording folder (one rep, or the amp file)."""
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
    _rep: Optional[Tuple[int, int]] = field(default=None, repr=False)

    def rep(self) -> Tuple[int, int]:
        """[start, end) of this file's single dominant rep (cached)."""
        if self._rep is None:
            self._rep = dominant_rep(self.envelope, self.fs)
        return self._rep

    @property
    def category(self) -> str:
        return category_of(self.path)


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
    actually experienced). Re-derives them from `raw` via the session's own filter
    config when `refilter=True` or when those columns are missing.
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
#  Discovery: recordings -> subjects -> categories -> rep folders
# --------------------------------------------------------------------------- #
def find_datasets(recordings_root: Path) -> List[Path]:
    """Every folder under recordings_root that contains a signal.csv, sorted."""
    root = Path(recordings_root)
    if not root.exists():
        return []
    return sorted(p.parent for p in root.rglob("signal.csv"))


@dataclass
class SubjectGroup:
    """One test subject: their MVC (amp) folder and their movement categories."""
    subject: str
    amp_dir: Optional[Path]
    categories: Dict[str, List[Path]]   # category -> rep folders (sorted), no amp


def find_subjects(recordings_root: Path) -> List[SubjectGroup]:
    """Group every rep folder under recordings_root by subject then category.

    The subject is the folder directly under the recordings root; rep folders that
    sit in the root itself (old flat layout) are grouped under "(ungrouped)". The
    `amp` category is pulled aside as the subject's MVC reference.
    """
    root = Path(recordings_root)
    root_res = root.resolve()
    by_subject: Dict[str, List[Path]] = {}
    for d in find_datasets(root):
        parent = d.parent
        subject = "(ungrouped)" if parent.resolve() == root_res else parent.name
        by_subject.setdefault(subject, []).append(d)

    groups: List[SubjectGroup] = []
    for subject in sorted(by_subject):
        amp_dir: Optional[Path] = None
        categories: Dict[str, List[Path]] = {}
        for d in sorted(by_subject[subject]):
            cat = category_of(d)
            if cat == AMP_CATEGORY:
                if amp_dir is None:
                    amp_dir = d  # first amp folder is the MVC reference
                continue
            categories.setdefault(cat, []).append(d)
        groups.append(SubjectGroup(subject, amp_dir, categories))
    return groups


def subject_mvc(amp_dir: Optional[Path], recordings_root: Path,
                refilter: bool = False, mains_hz: Optional[float] = None) -> Optional[float]:
    """Robust MVC reference (counts) for a subject from their amp recording, or
    None when there is no amp folder / it can't be loaded."""
    if amp_dir is None:
        return None
    try:
        ctx = load_dataset(amp_dir, recordings_root, refilter=refilter, mains_hz=mains_hz)
    except Exception:
        return None
    return robust_max(ctx.envelope, ctx.fs)


# --------------------------------------------------------------------------- #
#  Scoring one rep, then aggregating a category
# --------------------------------------------------------------------------- #
def score_rep(ctx: DatasetContext, mvc: Optional[float]
              ) -> Tuple[Dict[str, Optional[float]], Optional[np.ndarray]]:
    """Metrics for one rep file + its time-normalised envelope profile.

    The rep is the file's dominant burst; the pre-onset stretch is its rest
    baseline. When an MVC is given, the rep envelope is clipped only above
    SPIKE_TRIM_FACTOR × MVC (gross shift artefacts, not normal overshoot) and the
    %MVC columns are filled.
    """
    env = ctx.envelope
    filt = ctx.filtered
    fs = ctx.fs
    s, e = ctx.rep()
    rep_env = env[s:e]
    rep_filt = filt[s:e]
    rest_filt = filt[:s]  # everything before onset is rest baseline

    ceil = mvc * SPIKE_TRIM_FACTOR if (mvc and mvc > 0) else None
    rep_env_c = np.minimum(rep_env, ceil) if ceil else rep_env

    row: Dict[str, Optional[float]] = {}

    row["rms_amplitude"] = float(np.sqrt(np.mean(rep_filt ** 2))) if rep_filt.size else None
    row["mav"] = float(np.mean(np.abs(rep_filt))) if rep_filt.size else None
    peak = float(rep_env_c.max()) if rep_env_c.size else None
    row["peak"] = peak

    if mvc and mvc > 0 and rep_env_c.size:
        row["mean_pct_mvc"] = float(np.mean(rep_env_c) / mvc * 100.0)
        row["peak_pct_mvc"] = float(peak / mvc * 100.0)
    else:
        row["mean_pct_mvc"] = None
        row["peak_pct_mvc"] = None

    base = float(np.sqrt(np.mean(rest_filt ** 2))) if rest_filt.size else None
    row["baseline_noise"] = base
    if base and base > 0 and rep_filt.size:
        rms_act = float(np.sqrt(np.mean(rep_filt ** 2)))
        row["snr_db"] = 20.0 * math.log10(rms_act / base)
    else:
        row["snr_db"] = None

    row["mains_residual"] = _mains_residual(filt, fs, ctx.cfg.mains_hz)
    row["median_freq_hz"] = _median_freq(rep_filt if rep_filt.size >= 64 else filt, fs)

    if rep_env.size >= 2 and fs > 0:
        peak_i = int(np.argmax(rep_env))
        rise_ms = (peak_i / fs) * 1000.0
        row["rise_time_ms"] = rise_ms if rise_ms > 0 else None
        row["onset_sharpness"] = (1000.0 / rise_ms) if rise_ms > 0 else None
    else:
        row["rise_time_ms"] = None
        row["onset_sharpness"] = None

    c = ctx.contact
    valid = c[~np.isnan(c)] if c.size else c
    row["contact_frac"] = float(np.mean(valid)) if valid.size else None

    return row, rep_profile(rep_env_c)


def _nanmean(values) -> Optional[float]:
    xs = [v for v in values
          if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(xs)) if xs else None


def inter_rep_consistency(profiles: List[Optional[np.ndarray]]) -> Optional[float]:
    """1 - mean(CV) across the reps' time-normalised envelope profiles. Needs ≥2
    reps; None otherwise. Higher = more repeatable movement."""
    profs = [p for p in profiles if p is not None]
    if len(profs) < 2:
        return None
    mat = np.vstack(profs)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    good = mean > 1e-9
    if not good.any():
        return None
    cv = std[good] / mean[good]
    return float(1.0 - float(np.mean(cv)))


def rep_row(ctx: DatasetContext, subject: str, category: str,
            mvc: Optional[float]) -> Tuple[Dict[str, object], Optional[np.ndarray]]:
    """One reps.csv row (identity + metrics) for a rep, plus its profile."""
    metrics, profile = score_rep(ctx, mvc)
    row: Dict[str, object] = {
        "subject": subject,
        "category": category,
        "rep": ctx.dataset,
        "started": ctx.meta.get("started", ""),
        "duration_s": ctx.meta.get("duration_s", round(ctx.raw.size / ctx.fs, 3)),
        "n_samples": ctx.raw.size,
        "sample_rate": int(ctx.fs),
        "mains_hz": ctx.cfg.mains_hz,
    }
    row.update(metrics)
    return row, profile


def aggregate_category(subject: str, category: str, mvc: Optional[float],
                       rep_rows: List[Dict[str, object]],
                       profiles: List[Optional[np.ndarray]],
                       weights: Optional["QualityWeights"] = None) -> Dict[str, object]:
    """Collapse a category's rep rows into one scores.csv row: mean of every rep
    metric, consistency across the reps, and the composite quality gate."""
    cat: Dict[str, object] = {
        "subject": subject,
        "category": category,
        "n_reps": len(rep_rows),
        "mvc_reference": round(mvc, 4) if mvc else None,
    }
    for col in REP_METRIC_COLUMNS:
        cat[col] = _nanmean([r.get(col) for r in rep_rows])
    cat["inter_rep_consistency"] = inter_rep_consistency(profiles)
    cat[SCORE_COLUMN] = composite_quality(cat, weights)
    return cat


# --------------------------------------------------------------------------- #
#  Composite signal-quality gate  (transparent + reconfigurable)
# --------------------------------------------------------------------------- #
@dataclass
class QualityWeights:
    """Knobs for composite_quality(). Defaults are a documented heuristic, not a
    law — the research fixes only that a quality gate must *exist* (exclude noisy
    reps), not its exact form. Tune freely for your rig."""
    snr_floor_db: float = 3.0
    snr_good_db: float = 20.0
    mains_tol: float = 0.20
    w_snr: float = 0.6
    w_mains: float = 0.4


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def composite_quality(row: Dict[str, object],
                      w: Optional[QualityWeights] = None) -> Optional[float]:
    """0..1 signal-integrity score: is this (aggregated) recording clean enough to
    trust? Blends SNR + mains sub-scores (reweighted to whatever is present) then
    gates on contact fraction. None when no quality evidence exists. Excludes the
    motor-learning metrics — those describe the movement, not signal usability."""
    w = w or QualityWeights()
    parts: List[Tuple[float, float]] = []

    snr = row.get("snr_db")
    if snr is not None and w.snr_good_db > w.snr_floor_db:
        parts.append((w.w_snr, _clamp01(
            (float(snr) - w.snr_floor_db) / (w.snr_good_db - w.snr_floor_db))))

    mains = row.get("mains_residual")
    if mains is not None and w.mains_tol > 0:
        parts.append((w.w_mains, _clamp01(1.0 - float(mains) / w.mains_tol)))

    if not parts:
        return None
    total_w = sum(wt for wt, _ in parts)
    base = sum(wt * sub for wt, sub in parts) / total_w if total_w > 0 else 0.0

    contact = row.get("contact_frac")
    gate = float(contact) if contact is not None else 1.0
    return _clamp01(base * gate)


# --------------------------------------------------------------------------- #
#  Scoring everything
# --------------------------------------------------------------------------- #
def score_all(recordings_root: Path, refilter: bool = False,
              mains_hz: Optional[float] = None,
              weights: Optional[QualityWeights] = None,
              ) -> Tuple[List[Dict[str, object]], List[Dict[str, object]],
                         List[Tuple[Path, str]]]:
    """Score every subject/category under recordings_root.

    Returns (category_rows, rep_rows, failures): one aggregated row per
    (subject, category), one row per rep file, and (path, error) for reps that
    could not be loaded. The amp folders are used as MVC references, not scored.
    """
    category_rows: List[Dict[str, object]] = []
    rep_rows: List[Dict[str, object]] = []
    failures: List[Tuple[Path, str]] = []
    root = Path(recordings_root)

    for group in find_subjects(root):
        mvc = subject_mvc(group.amp_dir, root, refilter=refilter, mains_hz=mains_hz)
        for category in sorted(group.categories):
            rows: List[Dict[str, object]] = []
            profiles: List[Optional[np.ndarray]] = []
            for d in group.categories[category]:
                try:
                    ctx = load_dataset(d, root, refilter=refilter, mains_hz=mains_hz)
                except Exception as exc:
                    failures.append((d, f"{type(exc).__name__}: {exc}"))
                    continue
                r, profile = rep_row(ctx, group.subject, category, mvc)
                rows.append(r)
                profiles.append(profile)
            if not rows:
                continue
            rep_rows.extend(rows)
            category_rows.append(
                aggregate_category(group.subject, category, mvc, rows, profiles, weights))

    category_rows.sort(key=lambda r: (str(r["subject"]), str(r["category"])))
    rep_rows.sort(key=lambda r: (str(r["subject"]), str(r["category"]), str(r["rep"])))
    return category_rows, rep_rows, failures


# --------------------------------------------------------------------------- #
#  CSV output
# --------------------------------------------------------------------------- #
def _fmt(value: object) -> str:
    """Format a cell for CSV: blanks for None/NaN, ~4 significant figures."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4g}"
    return str(value)


def header_labels(columns: Tuple[str, ...]) -> List[str]:
    """Header row for `columns`: each labelled with its unit where it has one."""
    return [COLUMN_LABELS.get(c, c) for c in columns]


def write_csv(rows: List[Dict[str, object]], out_path: Path,
              columns: Tuple[str, ...]) -> Path:
    """Write rows to a clean CSV in the given column order, with unit-labelled
    headers (data stays keyed by the bare column names)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header_labels(columns))
        for row in rows:
            writer.writerow([_fmt(row.get(col)) for col in columns])
    return out_path
