"""Build the §10.5 elicitation prompts from one experimenter-authored fill-in file.

Three steps, in order:

    python elicit_cli.py init      # writes elicitation/inputs.json, all blanks
    #  ... you fill it in ...
    python elicit_cli.py check     # lists what is still blank or wrong
    python elicit_cli.py build     # writes prompts/*.json + manifest.csv

`plan` prints the call budget without touching the fill-in file, so you can see
what you are committing to before authoring anything.

This tool never calls an API. Prompts are the second of §10.8's two artifacts:
build them, commit them with the hypotheses, and only then elicit. Once built, the
prompts and the bins are frozen — editing either after seeing a model response is
what §10.8 exists to prevent.

Examples:
    python elicit_cli.py plan
    python elicit_cli.py plan --k 5 --models claude-opus-4-8,claude-haiku-4-5
    python elicit_cli.py build --seed 1
    python elicit_cli.py build --vary-by-model      # not the default; see below
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Sequence

from emg.elicit import (
    CONDITIONS,
    K_REPEATS,
    PANEL_MODELS,
    PRIMARY_CONDITION,
    PRIMARY_MODEL,
    REFERENCE_CSV_NAME,
    Call,
    blank_inputs,
    load_reference_levels,
    plan_calls,
    validate,
    write_prompts,
)
from emg.gate_r1 import FEATURE_CODES, MOVEMENTS

DEFAULT_DIR = Path("elicitation")
INPUTS_NAME = "inputs.json"


def _split(value: Optional[str], fallback: Sequence[str]) -> Sequence[str]:
    if not value:
        return fallback
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _print_plan(calls: Sequence[Call], primary_model: str) -> None:
    by_model: Dict[str, Counter] = {}
    for call in calls:
        by_model.setdefault(call.model, Counter())[call.condition] += 1

    width = max(len(m) for m in by_model)
    print(f"  {'model'.ljust(width)}  {'conditions':<24}  calls")
    print(f"  {'-' * width}  {'-' * 24}  -----")
    for model, counts in by_model.items():
        conditions = " ".join(sorted(counts))
        tag = "  <- PRIMARY" if model == primary_model else ""
        print(f"  {model.ljust(width)}  {conditions:<24}  "
              f"{sum(counts.values()):>5}{tag}")
    cells = len(calls) * len(MOVEMENTS) * len(FEATURE_CODES)
    print(f"\n  {len(calls)} calls x {len(MOVEMENTS)} movements x "
          f"{len(FEATURE_CODES)} features = {cells:,} cell judgements")
    print(f"  -> collapses to {len(MOVEMENTS) * len(FEATURE_CODES)} modal labels "
          "per (model, condition), plus a response entropy per cell")


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.dir) / INPUTS_NAME
    if path.exists() and not args.force:
        print(f"{path} already exists. Use --force to overwrite "
              "(this discards what you have filled in).", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blank_inputs(), indent=2), encoding="utf-8")

    print(f"Wrote {path}\n")
    print("Fill in, by hand - none of this may be LLM-authored (10.1):")
    print(f"  {len(MOVEMENTS):>3}  p1_label   verbatim style + manner, as a user "
          "would type it")
    print(f"  {len(MOVEMENTS):>3}  kinesic    description with style and manner "
          "words STRIPPED (P2)")
    print("\nLevels are positional (low/medium/high = bottom/middle/top third) and "
          "built in -- no per-feature anchors to write.")
    print("P4 worked examples are drawn per call and levelled from R1 -- nothing to "
          "author for few-shot.")
    print(f"\nThen: python {Path(sys.argv[0]).name} check")
    return 0


def _load(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        print(f"No fill-in file at {path} - run `init` first.", file=sys.stderr)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"{path} is not valid JSON: {exc}", file=sys.stderr)
        return None


def cmd_check(args: argparse.Namespace) -> int:
    inputs = _load(Path(args.dir) / INPUTS_NAME)
    if inputs is None:
        return 1
    problems = validate(inputs)
    if not problems:
        print("Fill-in file is complete. Ready to build.")
        return 0
    blanks = [p for p in problems if p.startswith("blank:")]
    others = [p for p in problems if not p.startswith("blank:")]
    if blanks:
        print(f"{len(blanks)} field(s) still blank:")
        for problem in blanks:
            print(f"  {problem}")
    if others:
        print(f"\n{len(others)} problem(s) to fix:")
        for problem in others:
            print(f"  {problem}")
    return 1


def _few_shot_fixed(args: argparse.Namespace) -> Optional[Sequence[str]]:
    """Parse --few-shot-fixed into a validated pair, or None for randomised draws."""
    if not getattr(args, "few_shot_fixed", None):
        return None
    pair = tuple(m.strip() for m in args.few_shot_fixed.split(",") if m.strip())
    bad = [m for m in pair if m not in MOVEMENTS]
    if len(pair) != 2 or bad:
        raise SystemExit(
            f"--few-shot-fixed needs two of the 12 movement ids; got {args.few_shot_fixed!r}"
            + (f" (unknown: {', '.join(bad)})" if bad else ""))
    return pair


def _reference_levels(args: argparse.Namespace, conditions: Sequence[str]):
    """Load R1 tertiles if P4 is being built; validate they cover all 12 movements."""
    if "P4" not in conditions:
        return {}
    root = Path(args.reference) if args.reference else \
        Path(args.dir).parent / "recordings" / "gate_r1" / REFERENCE_CSV_NAME
    if not root.exists():
        raise SystemExit(
            f"P4 needs the R1 tertile table for its worked examples, not found at "
            f"{root}. Run gate_r1_cli.py first, or pass --reference, or drop P4 from "
            f"--conditions.")
    levels = load_reference_levels(root)
    missing = [m for m in MOVEMENTS if m not in levels]
    if missing:
        raise SystemExit(f"{root} is missing tertiles for: {', '.join(missing)}")
    return levels


def cmd_plan(args: argparse.Namespace) -> int:
    calls = plan_calls(
        models=_split(args.models, PANEL_MODELS),
        primary_model=args.primary_model,
        conditions=_split(args.conditions, CONDITIONS),
        k=args.k,
        seed=args.seed,
        vary_by_model=args.vary_by_model,
        few_shot_fixed=_few_shot_fixed(args),
    )
    print(f"Call plan - k={args.k} repeats, primary condition {PRIMARY_CONDITION}, "
          f"seed {args.seed}\n")
    _print_plan(calls, args.primary_model)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    out_dir = Path(args.dir)
    inputs = _load(out_dir / INPUTS_NAME)
    if inputs is None:
        return 1

    problems = validate(inputs)
    if problems:
        print(f"Refusing to build - {len(problems)} problem(s). "
              "Run `check` for the full list.", file=sys.stderr)
        for problem in problems[:10]:
            print(f"  {problem}", file=sys.stderr)
        if len(problems) > 10:
            print(f"  ... and {len(problems) - 10} more", file=sys.stderr)
        return 1

    conditions = _split(args.conditions, CONDITIONS)
    reference_levels = _reference_levels(args, conditions)
    calls = plan_calls(
        models=_split(args.models, PANEL_MODELS),
        primary_model=args.primary_model,
        conditions=conditions,
        k=args.k,
        seed=args.seed,
        vary_by_model=args.vary_by_model,
        few_shot_fixed=_few_shot_fixed(args),
    )
    written = write_prompts(calls, inputs, out_dir, seed=args.seed,
                            reference_levels=reference_levels)

    print(f"Call plan - k={args.k} repeats, seed {args.seed}\n")
    _print_plan(calls, args.primary_model)
    n_p4 = sum(1 for c in calls if c.condition == "P4")
    if n_p4:
        mode = ("fixed " + "|".join(_few_shot_fixed(args))) if _few_shot_fixed(args) \
            else "randomised per call"
        print(f"\n  P4 few-shot: {mode}; true levels from R1 tertiles "
              f"({n_p4} P4 calls)")
    print(f"\nWrote {len(written)} file(s) -> {out_dir}")
    print(f"  prompts/     {len(written) - 1} payloads")
    print(f"  manifest.csv display id -> movement + P4 exemplars, per call")
    print("\nCommit this folder with the hypotheses BEFORE eliciting (10.8).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the 10.5 elicitation prompts (does not call any API).")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"working folder (default: {DEFAULT_DIR})")
    sub = ap.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="write the blank fill-in file")
    p_init.add_argument("--force", action="store_true",
                        help="overwrite an existing inputs.json")
    p_init.set_defaults(func=cmd_init)

    p_check = sub.add_parser("check", help="report blanks and provenance problems")
    p_check.set_defaults(func=cmd_check)

    for name, help_text, func in (
        ("plan", "print the call budget without building", cmd_plan),
        ("build", "validate, then write prompts + manifest", cmd_build),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--models", default=None,
                       help="comma-separated model ids (default: the 10.5 panel)")
        p.add_argument("--primary-model", default=PRIMARY_MODEL,
                       help=f"model that runs all conditions (default: {PRIMARY_MODEL})")
        p.add_argument("--conditions", default=None,
                       help=f"comma-separated (default: {','.join(CONDITIONS)})")
        p.add_argument("--k", type=int, default=K_REPEATS,
                       help=f"independent repeats per configuration "
                            f"(default: {K_REPEATS})")
        p.add_argument("--seed", type=int, default=1,
                       help="RNG seed for presentation order, fixed so the "
                            "randomisation reproduces (default: 1)")
        p.add_argument("--vary-by-model", action="store_true",
                       help="give each model its own orderings; off by default so "
                            "all models see identical inputs (E6 comparability)")
        p.add_argument("--few-shot-fixed", default=None, metavar="M1,M2",
                       help="pin the two P4 worked examples to one movement pair; "
                            "default draws a fresh pair per call to avoid "
                            "exemplar-choice bias")
        p.add_argument("--reference", default=None,
                       help="path to R1_movement_scores.csv for P4 example levels "
                            "(default: <dir>/../recordings/gate_r1/...)")
        p.set_defaults(func=func)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
