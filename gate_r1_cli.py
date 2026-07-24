"""Build the Gate R1 ground truth and reliability ceiling, as SPSS-ready CSVs.

Consumes the frozen scorer output (scores.csv, reps.csv) and writes, into
<recordings>/gate_r1/:

    R1_icc_<feature>.csv    12 movements x N subjects, within-subject z-scored
    R1_splithalf.csv        odd- vs even-rep medians per (subject, movement)
    R1_movement_scores.csv  movement-level mean z, tertile, leave-one-out tertile

Gate R1 (llm_director_extension.md §10.4) is a precondition, not a hypothesis test:
it establishes how well a *perfect* predictor could score against a ground truth this
noisy, so that every kappa in E1 can be reported as kappa AND kappa/kappa_max. A
feature with ICC(2,k) < 0.5 is reported but drops out of the primary endpoint.

The printed ICC and split-half figures are a preview — SPSS is the reporting
instrument (see the ICC matrices above). kappa_max has no SPSS menu equivalent; it is
a Monte-Carlo simulation, so it is computed here and is seeded for reproducibility.

Run this BEFORE eliciting any LLM output, and commit its results with the frozen
extraction pipeline as the first of §10.8's two independent artifacts.

Examples:
    python gate_r1_cli.py                          # analyse ./recordings
    python gate_r1_cli.py --recordings data
    python gate_r1_cli.py --sims 50000 --seed 7    # tighter kappa_max estimate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emg.gate_r1 import (
    ICC_PRIMARY_MIN,
    analyse,
    write_icc_matrices,
    write_movement_scores,
    write_split_half,
)

_HEADERS = ("feature", "n_icc", "n_kap", "k", "ICC(2,1)", "ICC(2,k)", "split-half",
            "kappa_max", "primary")


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_summary(result) -> None:
    table = [list(_HEADERS)]
    for r in result.reliability:
        table.append([
            r.feature, _cell(r.n_movements), _cell(r.n_scored), _cell(r.n_subjects),
            _cell(r.icc_single), _cell(r.icc_average), _cell(r.split_half),
            _cell(r.kappa_max), "keep" if r.retained else "DROP",
        ])
    widths = [max(len(row[i]) for row in table) for i in range(len(_HEADERS))]
    for i, row in enumerate(table):
        print("  " + "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print("  " + "  ".join("-" * w for w in widths))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate R1: ground truth, ICC(2,k), split-half and kappa_max.")
    ap.add_argument("--recordings", default="recordings",
                    help="folder holding scores.csv and reps.csv (default: recordings)")
    ap.add_argument("--scores", default=None,
                    help="per-category CSV (default: <recordings>/scores.csv)")
    ap.add_argument("--reps", default=None,
                    help="per-rep CSV for the split-half "
                         "(default: <recordings>/reps.csv; skipped if absent)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: <recordings>/gate_r1)")
    ap.add_argument("--sims", type=int, default=10000,
                    help="Monte-Carlo draws for kappa_max (default: 10000)")
    ap.add_argument("--seed", type=int, default=1,
                    help="RNG seed for kappa_max, fixed so results reproduce")
    args = ap.parse_args()

    root = Path(args.recordings)
    scores_path = Path(args.scores) if args.scores else root / "scores.csv"
    reps_path = Path(args.reps) if args.reps else root / "reps.csv"
    out_dir = Path(args.out) if args.out else root / "gate_r1"

    if not scores_path.exists():
        print(f"No scores CSV at {scores_path} — run score_cli.py first.",
              file=sys.stderr)
        return 1
    if not reps_path.exists():
        print(f"No per-rep CSV at {reps_path}; split-half will be skipped.",
              file=sys.stderr)
        reps_path = None

    print(f"Gate R1 from {scores_path}"
          + (f" + {reps_path}" if reps_path else " [no split-half]"))
    result = analyse(scores_path, reps_path, sims=args.sims, seed=args.seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_icc_matrices(result, out_dir)
    written.append(write_movement_scores(result, out_dir))
    if reps_path:
        written.append(write_split_half(result, out_dir))

    print(f"\n{len(result.subjects)} subject(s): {', '.join(result.subjects)}\n")
    _print_summary(result)

    dropped = [r.feature for r in result.reliability if not r.retained]
    print(f"\nICC(2,k) >= {ICC_PRIMARY_MIN} keeps a feature in the primary endpoint.")
    print("Dropped: " + (", ".join(dropped) if dropped else "none"))
    print(f"\nWrote {len(written)} file(s) -> {out_dir}")
    for path in written:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
