"""Score recorded sessions into a clean CSV of EMG properties.

Walks a recordings folder, computes the research-grounded single-channel EMG
metrics for every dataset it finds (see emg/scoring.py for the metrics and their
literature basis), and writes one tidy row per dataset to a CSV.

Works with both recording layouts:
    recordings/<subject>/<timestamp>_<dataset>/signal.csv   (per-subject, current)
    recordings/<timestamp>_<name>/signal.csv                (old flat layout)

Examples:
    python score_cli.py                         # score ./recordings -> recordings/scores.csv
    python score_cli.py --recordings data --out out.csv
    python score_cli.py --refilter              # re-derive filtered/envelope from raw
    python score_cli.py --mains 60              # override mains freq for the notch/quality
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from emg.scoring import (
    SCORE_COLUMN,
    find_datasets,
    score_all,
    write_csv,
)

# Compact console view — the CSV carries every column; this is just a glance.
_SUMMARY_COLUMNS = (
    "subject", "dataset", "n_reps", "snr_db", "mains_residual",
    "inter_rep_consistency", SCORE_COLUMN,
)


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _print_summary(rows: list) -> None:
    header = [c.replace("inter_rep_consistency", "consistency")
              .replace("mains_residual", "mains")
              .replace("quality_score", "quality") for c in _SUMMARY_COLUMNS]
    table = [header] + [[_cell(r.get(c)) for c in _SUMMARY_COLUMNS] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for i, row in enumerate(table):
        print("  " + "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            print("  " + "  ".join("-" * widths[j] for j in range(len(header))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Score recorded cheezEMG sessions to a CSV.")
    ap.add_argument("--recordings", default="recordings",
                    help="folder of recorded sessions (default: recordings)")
    ap.add_argument("--out", default=None,
                    help="output CSV path (default: <recordings>/scores.csv)")
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

    out_path = Path(args.out) if args.out else root / "scores.csv"
    print(f"Scoring {len(datasets)} dataset(s) under {root}"
          + (" [refiltering from raw]" if args.refilter else ""))

    rows, failures = score_all(root, refilter=args.refilter, mains_hz=args.mains)

    if rows:
        _print_summary(rows)
    written = write_csv(rows, out_path)
    print(f"\nWrote {len(rows)} row(s) -> {written}")

    if failures:
        print(f"\n{len(failures)} dataset(s) could not be scored:", file=sys.stderr)
        for path, err in failures:
            print(f"  {path}: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
