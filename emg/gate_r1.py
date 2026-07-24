"""Gate R1 — the reliability ceiling that bounds every kappa in the study.

Implements llm_director_extension.md §10.4: builds the ground truth from the frozen
per-cell scores, then measures how much of it is signal.

    §10.4 step 1  per-subject, per-movement median across reps   <- already in scores.csv
    §10.4 step 2  within-subject z-score across the 12 movements
    §10.4 step 3  MVC excluded (it is the normaliser, not a movement)
    §10.4 step 4  average z across subjects -> one score per movement
    §10.4 step 5  tertile bins estimated leave-one-movement-out

    Gate R1      ICC(2,k) per feature [64], odd/even split-half, and kappa_max

Gate R1 is a precondition, not a hypothesis test. Its output is the denominator for
E1: a bare kappa is uninterpretable, so every kappa is reported as kappa AND
kappa/kappa_max (§10.4). Features with ICC(2,k) < 0.5 are reported but dropped from
the primary endpoint.

Nothing here re-derives a signal. It consumes the outputs of score_cli.py, so the
extraction pipeline stays frozen and this stage only estimates bins — the separation
§10.8 requires between the two artifacts that must be committed independently.

WHY Z-SCORE FIRST — §10.4 step 2 is the "relative, never absolute" rule of §5 applied
to the ground truth itself. It strips per-subject anatomy and electrode gain, and it
places the truth in the same 12-movement universe the model is shown. A side effect
worth knowing when reading SPSS output: standardising every subject's column drives
the rater main effect to ~0, so ICC absolute-agreement and consistency coincide.

CIRCULARITY — the tertile boundaries come from the same recordings the model is scored
against, which §9 flags as a trap. Step 5 closes it: a movement's bin is decided by
boundaries fitted to the other eleven, never to itself.

Outputs (one folder, all SPSS-importable):
    R1_icc_<feature>.csv    12 movements x 10 subjects, z-scored — the ICC matrix
    R1_splithalf.csv        odd- vs even-rep medians per (subject, movement)
    R1_movement_scores.csv  movement-level mean z, global tertile, LOO tertile
"""

from __future__ import annotations

import csv
import math
import random
import statistics as stats
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from emg.scoring import COLUMN_LABELS, FEATURE_COLUMNS, write_csv

# The five §10.3 features keyed by their short code. FEATURE_COLUMNS is the frozen
# order from the scorer; this just gives each one the label the writeup uses.
FEATURE_CODES: Tuple[str, ...] = ("T1", "T2", "T3", "A1", "S1")
FEATURE_KEYS: Dict[str, str] = dict(zip(FEATURE_CODES, FEATURE_COLUMNS))

# The §10.2 factorial. Manner suffixes as recorded: bare name = M3 "as controlled as
# possible", `fast` = M1, `slow` = M2. Movement order matches anova_wide.csv so the
# two SPSS inputs line up by eye.
GESTURES: Tuple[str, ...] = ("kickback", "overhead", "hit", "point")
MANNERS: Tuple[str, ...] = ("fast", "slow", "ctrl")
MOVEMENTS: Tuple[str, ...] = tuple(f"{g}_{m}" for g in GESTURES for m in MANNERS)

N_TERTILES = 3
# ICC(2,k) below this is reported but dropped from the primary endpoint (§10.4).
ICC_PRIMARY_MIN = 0.5


def category_to_movement(category: str) -> Optional[str]:
    """`hitslow` -> `hit_slow`. None for anything that is not a scored cell (`amp`)."""
    for gesture in GESTURES:
        if category == gesture:
            return f"{gesture}_ctrl"
        for manner in ("fast", "slow"):
            if category == gesture + manner:
                return f"{gesture}_{manner}"
    return None


# --------------------------------------------------------------------------- #
#  Loading — headers are the unit-labelled ones write_csv() emits
# --------------------------------------------------------------------------- #
def _label_to_key() -> Dict[str, str]:
    """Reverse of COLUMN_LABELS, so we read scorer CSVs without hardcoding units."""
    return {COLUMN_LABELS.get(key, key): key for key in FEATURE_COLUMNS}


def _number(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return None if math.isnan(value) else value


def load_cells(scores_csv: Path) -> Tuple[Dict[str, Dict[str, Dict[str, Optional[float]]]],
                                          List[str]]:
    """scores.csv -> cells[feature][subject][movement], plus the subjects found.

    This is §10.4 step 1 already done: each value is the median over that cell's reps.
    """
    by_label = _label_to_key()
    cells: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
        code: {} for code in FEATURE_CODES}
    subjects: List[str] = []

    with open(scores_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            movement = category_to_movement(row["category"])
            if movement is None:
                continue
            subject = row["subject"]
            if subject not in subjects:
                subjects.append(subject)
            for label, key in by_label.items():
                code = FEATURE_CODES[FEATURE_COLUMNS.index(key)]
                cells[code].setdefault(subject, {})[movement] = _number(row.get(label, ""))
    return cells, sorted(subjects)


def load_reps(reps_csv: Path) -> Dict[str, Dict[Tuple[str, str], List[Optional[float]]]]:
    """reps.csv -> reps[feature][(subject, movement)] = values in recorded order.

    Order matters: the split-half is odd- vs even-*rep*, so the sequence the reps were
    performed in is the split. reps.csv is already sorted by (subject, category, rep).
    """
    by_label = _label_to_key()
    reps: Dict[str, Dict[Tuple[str, str], List[Optional[float]]]] = {
        code: {} for code in FEATURE_CODES}

    with open(reps_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            movement = category_to_movement(row["category"])
            if movement is None:
                continue
            cell = (row["subject"], movement)
            for label, key in by_label.items():
                code = FEATURE_CODES[FEATURE_COLUMNS.index(key)]
                reps[code].setdefault(cell, []).append(_number(row.get(label, "")))
    return reps


# --------------------------------------------------------------------------- #
#  §10.4 steps 2, 4, 5 — ground truth
# --------------------------------------------------------------------------- #
def zscore_within_subject(
    per_subject: Dict[str, Dict[str, Optional[float]]],
    subjects: Sequence[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """§10.4 step 2. Standardise each subject's 12 movements against themselves.

    Population SD, not sample: the 12 movements are the whole universe here, not a
    draw from a larger one. A subject missing a cell is standardised on what they have.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for subject in subjects:
        values = [v for v in (per_subject.get(subject, {}).get(m) for m in MOVEMENTS)
                  if v is not None]
        if not values:
            out[subject] = {m: None for m in MOVEMENTS}
            continue
        mean = stats.mean(values)
        sd = stats.pstdev(values)
        out[subject] = {
            m: None if (v := per_subject.get(subject, {}).get(m)) is None
            else ((v - mean) / sd if sd else 0.0)
            for m in MOVEMENTS
        }
    return out


def movement_means(z: Dict[str, Dict[str, Optional[float]]],
                   subjects: Sequence[str]) -> Dict[str, Optional[float]]:
    """§10.4 step 4. Average the z-scores across subjects -> one score per movement."""
    means: Dict[str, Optional[float]] = {}
    for movement in MOVEMENTS:
        values = [z[s][movement] for s in subjects if z[s][movement] is not None]
        means[movement] = stats.mean(values) if values else None
    return means


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. Small n, so no numpy dependency."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def global_tertiles(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[int]]:
    """Equal-size tertiles by rank (4/4/4). Used for the §10.2 spread check only.

    Equal-size by construction, so it cannot itself reveal crowding — read it beside
    the mean-z values, which can.
    """
    ranked = sorted((m for m in MOVEMENTS if scores[m] is not None),
                    key=lambda m: scores[m])
    per_bin = max(1, len(ranked) // N_TERTILES)
    out: Dict[str, Optional[int]] = {m: None for m in MOVEMENTS}
    for index, movement in enumerate(ranked):
        out[movement] = min(N_TERTILES, index // per_bin + 1)
    return out


def loo_tertiles(scores: Dict[str, Optional[float]]) -> Dict[str, Optional[int]]:
    """§10.4 step 5. Each movement binned by boundaries fitted to the other eleven.

    This is the bin that scores the model. Holding the movement out is what stops the
    thresholds being fitted to the answer key (§9's circularity trap).
    """
    out: Dict[str, Optional[int]] = {}
    for movement in MOVEMENTS:
        value = scores[movement]
        if value is None:
            out[movement] = None
            continue
        others = [scores[m] for m in MOVEMENTS
                  if m != movement and scores[m] is not None]
        low = _percentile(others, 1 / 3)
        high = _percentile(others, 2 / 3)
        out[movement] = 1 if value < low else (3 if value > high else 2)
    return out


# --------------------------------------------------------------------------- #
#  Gate R1 statistics
# --------------------------------------------------------------------------- #
@dataclass
class Reliability:
    """One feature's Gate R1 result."""
    feature: str
    n_movements: int       # movements surviving listwise deletion — the ICC's n
    n_scored: int          # movements with a ground-truth score — kappa_max's n
    n_subjects: int
    icc_single: float      # ICC(2,1) — one subject on their own
    icc_average: float     # ICC(2,k) — the averaged ground truth; the gated metric
    kappa_max: float
    split_half: Optional[float]

    @property
    def retained(self) -> bool:
        return self.icc_average >= ICC_PRIMARY_MIN


def icc_two_way_random(matrix: Sequence[Sequence[float]]) -> Tuple[float, float]:
    """ICC(2,1) and ICC(2,k), Shrout & Fleiss two-way random, absolute agreement [64].

    Rows are targets (movements), columns are raters (subjects) — the orientation SPSS
    Reliability Analysis expects, with subjects entered as Items.
    """
    n = len(matrix)
    k = len(matrix[0])
    flat = [v for row in matrix for v in row]
    grand = stats.mean(flat)
    row_means = [stats.mean(row) for row in matrix]
    col_means = [stats.mean([matrix[i][j] for i in range(n)]) for j in range(k)]

    ss_rows = k * sum((rm - grand) ** 2 for rm in row_means)
    ss_cols = n * sum((cm - grand) ** 2 for cm in col_means)
    ss_error = sum((v - grand) ** 2 for v in flat) - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    single = ((ms_rows - ms_error)
              / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n))
    average = (ms_rows - ms_error) / (ms_rows + (ms_cols - ms_error) / n)
    return single, average


def quadratic_weighted_kappa(a: Sequence[int], b: Sequence[int],
                             categories: int = N_TERTILES) -> float:
    """Weighted kappa penalising "off by one level" less than "sharp vs dull" (§9)."""
    observed = [[0] * categories for _ in range(categories)]
    for x, y in zip(a, b):
        observed[x][y] += 1
    n = len(a)
    rows = [sum(observed[i]) for i in range(categories)]
    cols = [sum(observed[i][j] for i in range(categories)) for j in range(categories)]
    weight = [[((i - j) ** 2) / ((categories - 1) ** 2) for j in range(categories)]
              for i in range(categories)]

    numerator = sum(weight[i][j] * observed[i][j]
                    for i in range(categories) for j in range(categories))
    denominator = sum(weight[i][j] * rows[i] * cols[j] / n
                      for i in range(categories) for j in range(categories))
    return 1 - numerator / denominator if denominator else float("nan")


def _rank_bins(values: Sequence[float]) -> List[int]:
    """Equal-size tertiles as 0-based indices, for the kappa_max simulation."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    per_bin = max(1, len(values) // N_TERTILES)
    bins = [0] * len(values)
    for position, index in enumerate(order):
        bins[index] = min(N_TERTILES - 1, position // per_bin)
    return bins


def kappa_max(rows: Sequence[Sequence[float]], sims: int, rng: random.Random) -> float:
    """The weighted kappa a *perfect oracle* would score against this ground truth.

    No expert raters exist, so there is no inter-expert noise ceiling (§10.4). This
    substitutes one: an oracle that knows each movement's true score still disagrees
    with the empirical tertiles whenever sampling noise pushes a movement across a bin
    edge. Simulating that disagreement gives the ceiling E1's kappa is measured against.

    `rows` is ragged on purpose — one list of available subject z-scores per *scored*
    movement, not the listwise-complete ICC matrix. A movement scored from 9 subjects
    instead of 10 carries more sampling noise, and that belongs in the ceiling. Using
    the complete matrix instead would drop a movement entirely and leave the remaining
    eleven to be split into ragged tertiles, which is an artifact of the deletion
    rather than a property of the data.

    Between-subject variance is pooled as the mean within-movement variance; a movement
    averaged over k_i subjects therefore carries variance sigma^2 / k_i.
    """
    true_scores = [stats.mean(row) for row in rows]
    variance = stats.mean([stats.pvariance(row) for row in rows if len(row) > 1])
    sigmas = [math.sqrt(variance / len(row)) for row in rows]
    truth = _rank_bins(true_scores)

    total = 0.0
    for _ in range(sims):
        drawn = [score + rng.gauss(0.0, sigma)
                 for score, sigma in zip(true_scores, sigmas)]
        total += quadratic_weighted_kappa(truth, _rank_bins(drawn))
    return total / sims


def split_half_medians(
    values: Sequence[Optional[float]],
) -> Tuple[Optional[float], Optional[float]]:
    """Median of the odd-numbered reps and of the even-numbered reps.

    3-vs-3 at six reps per cell (§10.2's collected deviation), so this is a deliberately
    harsh estimate of how stable a single cell's median is.
    """
    def median_of(subset: Sequence[Optional[float]]) -> Optional[float]:
        present = [v for v in subset if v is not None]
        return stats.median(present) if present else None

    return median_of(values[0::2]), median_of(values[1::2])


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    mx, my = stats.mean(xs), stats.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else None


def spearman_brown(r: Optional[float]) -> Optional[float]:
    """Correct a half-length correlation up to full test length."""
    if r is None or r <= -1:
        return None
    return 2 * r / (1 + r)


# --------------------------------------------------------------------------- #
#  Orchestration
# --------------------------------------------------------------------------- #
@dataclass
class GateR1:
    subjects: List[str]
    z: Dict[str, Dict[str, Dict[str, Optional[float]]]]      # [feature][subject][move]
    scores: Dict[str, Dict[str, Optional[float]]]            # [feature][move]
    tertile: Dict[str, Dict[str, Optional[int]]]
    tertile_loo: Dict[str, Dict[str, Optional[int]]]
    halves: Dict[str, Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]]
    reliability: List[Reliability]


def complete_matrix(z: Dict[str, Dict[str, Optional[float]]],
                    subjects: Sequence[str]) -> Tuple[List[List[float]], List[str]]:
    """The z-score matrix with any movement missing a subject dropped.

    Listwise on movements, which is what SPSS Reliability does by default — so the
    printed n_movements matches what SPSS will report. T2 loses point_slow this way
    (one fully right-censored cell) while keeping all ten subjects.
    """
    matrix: List[List[float]] = []
    kept: List[str] = []
    for movement in MOVEMENTS:
        row = [z[s][movement] for s in subjects]
        if all(v is not None for v in row):
            matrix.append([float(v) for v in row])
            kept.append(movement)
    return matrix, kept


def analyse(scores_csv: Path, reps_csv: Optional[Path],
            sims: int = 10000, seed: int = 1) -> GateR1:
    """Run every §10.4 step and the three Gate R1 statistics."""
    cells, subjects = load_cells(scores_csv)
    reps = load_reps(reps_csv) if reps_csv and Path(reps_csv).exists() else None
    rng = random.Random(seed)

    z: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    scores: Dict[str, Dict[str, Optional[float]]] = {}
    tertile: Dict[str, Dict[str, Optional[int]]] = {}
    tertile_loo: Dict[str, Dict[str, Optional[int]]] = {}
    halves: Dict[str, Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]] = {}
    reliability: List[Reliability] = []

    for code in FEATURE_CODES:
        z[code] = zscore_within_subject(cells[code], subjects)
        scores[code] = movement_means(z[code], subjects)
        tertile[code] = global_tertiles(scores[code])
        tertile_loo[code] = loo_tertiles(scores[code])

        matrix, kept = complete_matrix(z[code], subjects)
        icc_single, icc_average = icc_two_way_random(matrix)

        # kappa_max is the ceiling for scoring the movements that HAVE a ground truth,
        # so it spans every scored movement, each with however many subjects reached it.
        scored_rows = [
            [float(v) for s in subjects if (v := z[code][s][movement]) is not None]
            for movement in MOVEMENTS
        ]
        scored_rows = [row for row in scored_rows if row]

        halves[code] = {}
        half_r: Optional[float] = None
        if reps is not None:
            odd_all: List[float] = []
            even_all: List[float] = []
            for subject in subjects:
                for movement in MOVEMENTS:
                    values = reps[code].get((subject, movement), [])
                    odd, even = split_half_medians(values)
                    halves[code][(subject, movement)] = (odd, even)
                    if odd is not None and even is not None:
                        odd_all.append(odd)
                        even_all.append(even)
            half_r = spearman_brown(_pearson(odd_all, even_all))

        reliability.append(Reliability(
            feature=code,
            n_movements=len(kept),
            n_scored=len(scored_rows),
            n_subjects=len(subjects),
            icc_single=icc_single,
            icc_average=icc_average,
            kappa_max=kappa_max(scored_rows, sims, rng),
            split_half=half_r,
        ))

    return GateR1(subjects=subjects, z=z, scores=scores, tertile=tertile,
                  tertile_loo=tertile_loo, halves=halves, reliability=reliability)


# --------------------------------------------------------------------------- #
#  CSV output — shaped for SPSS, not for reading
# --------------------------------------------------------------------------- #
def write_icc_matrices(result: GateR1, out_dir: Path) -> List[Path]:
    """One file per feature: movements as rows (cases), subjects as columns (items).

    Enter the subject columns as Items in SPSS Reliability Analysis; leave `movement`
    out. Blank cells are right-censored, and SPSS drops those movements listwise.
    """
    written = []
    columns = ("movement", *result.subjects)
    for code in FEATURE_CODES:
        rows = [{"movement": movement,
                 **{s: result.z[code][s][movement] for s in result.subjects}}
                for movement in MOVEMENTS]
        written.append(write_csv(rows, out_dir / f"R1_icc_{code}.csv", columns))
    return written


def write_movement_scores(result: GateR1, out_dir: Path) -> Path:
    """Movement-level ground truth: mean z, equal-size tertile, and the LOO bin.

    `_tertile` is the §10.2 spread check; `_tertile_loo` is the bin that scores the
    model (§10.4 step 5). They differ wherever a movement sits near a boundary — which
    is exactly the instability kappa_max prices in.
    """
    columns: List[str] = ["movement"]
    for code in FEATURE_CODES:
        columns += [f"{code}_meanz", f"{code}_tertile", f"{code}_tertile_loo"]

    rows = []
    for movement in MOVEMENTS:
        row: Dict[str, object] = {"movement": movement}
        for code in FEATURE_CODES:
            row[f"{code}_meanz"] = result.scores[code][movement]
            row[f"{code}_tertile"] = result.tertile[code][movement]
            row[f"{code}_tertile_loo"] = result.tertile_loo[code][movement]
        rows.append(row)
    return write_csv(rows, out_dir / "R1_movement_scores.csv", tuple(columns))


def write_split_half(result: GateR1, out_dir: Path) -> Path:
    """Odd/even rep medians per cell, in raw feature units (one row per subject x move)."""
    columns: List[str] = ["subject", "movement"]
    for code in FEATURE_CODES:
        columns += [f"{code}_odd", f"{code}_even"]

    rows = []
    for subject in result.subjects:
        for movement in MOVEMENTS:
            row: Dict[str, object] = {"subject": subject, "movement": movement}
            for code in FEATURE_CODES:
                odd, even = result.halves[code].get((subject, movement), (None, None))
                row[f"{code}_odd"] = odd
                row[f"{code}_even"] = even
            rows.append(row)
    return write_csv(rows, out_dir / "R1_splithalf.csv", tuple(columns))
