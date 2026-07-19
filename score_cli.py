"""Score recorded sessions into clean CSVs of EMG properties.

Walks a recordings folder, groups each subject's rep files by movement category
(one folder = one rep), and writes:

    scores.csv   one row per (subject, category), the reps aggregated
    reps.csv     one row per individual rep file (drill-down)

The per-subject `amp` recording is used as the MVC reference (for %MVC) and is not
scored as a category. See emg/scoring.py for the metrics and their literature basis.

Layouts handled:
    recordings/<subject>/<timestamp>_<category>/signal.csv   (per-subject, current)
    recordings/<timestamp>_<name>/signal.csv                 (old flat -> "(ungrouped)")

Examples:
    python score_cli.py                         # score ./recordings
    python score_cli.py --recordings data --out out.csv
    python score_cli.py --refilter              # re-derive filtered/envelope from raw
    python score_cli.py --mains 60              # override mains freq for notch/quality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emg.scoring import (
    CATEGORY_COLUMNS,
    REP_COLUMNS,
    SCORE_COLUMN,
    find_datasets,
    score_all,
    write_csv,
)

# Compact console view — the CSVs carry every column; this is just a glance.
_SUMMARY_COLUMNS = (
    "subject", "category", "n_reps", "peak_pct_mvc", "snr_db",
    "mains_residual", "inter_rep_consistency", SCORE_COLUMN,
)


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _print_summary(rows: list) -> None:
    header = [c.replace("inter_rep_consistency", "consist")
              .replace("mains_residual", "mains")
              .replace("peak_pct_mvc", "%MVC")
              .replace("quality_score", "quality") for c in _SUMMARY_COLUMNS]
    table = [header] + [[_cell(r.get(c)) for c in _SUMMARY_COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for i, row in enumerate(table):
        print("  " + "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print("  " + "  ".join("-" * widths[j] for j in range(len(header))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score recorded cheezEMG sessions to CSVs.")
    ap.add_argument("--recordings", default="recordings",
                    help="folder of recorded sessions (default: recordings)")
    ap.add_argument("--out", default=None,
                    help="per-category CSV path (default: <recordings>/scores.csv); "
                         "the per-rep CSV is written as reps.csv beside it")
    ap.add_argument("--refilter", action="store_true",
                    help="re-derive filtered/envelope from raw using each session's config")
    ap.add_argument("--mains", type=float, default=None,
                    help="override mains frequency (Hz) for the notch/quality metrics")
    args = ap.parse_args()

    root = Path(args.recordings)
    if not root.exists():
        print(f"No such recordings folder: {root}", file=sys.stderr)
        return 1

    datasets = find_datasets(root)
    if not datasets:
        print(f"No datasets (folders with signal.csv) found under {root}", file=sys.stderr)
        return 1

    scores_path = Path(args.out) if args.out else root / "scores.csv"
    reps_path = scores_path.with_name("reps.csv")
    print(f"Scoring {len(datasets)} rep folder(s) under {root}"
          + (" [refiltering from raw]" if args.refilter else ""))

    category_rows, rep_rows, failures = score_all(
        root, refilter=args.refilter, mains_hz=args.mains)

    if category_rows:
        _print_summary(category_rows)
    write_csv(category_rows, scores_path, CATEGORY_COLUMNS)
    write_csv(rep_rows, reps_path, REP_COLUMNS)
    print(f"\nWrote {len(category_rows)} category row(s) -> {scores_path}")
    print(f"Wrote {len(rep_rows)} rep row(s)      -> {reps_path}")

    if failures:
        print(f"\n{len(failures)} rep(s) could not be scored:", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
