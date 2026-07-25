"""Run the §10.5 elicitation against the built prompts, or drive the paste path.

Prerequisite: `elicit_cli.py build` has written elicitation/prompts/*.json.

    python elicit_run_cli.py plan       # what would be called vs pasted
    python elicit_run_cli.py run         # call the Anthropic API for Claude models
    python elicit_run_cli.py export     # write paste-ready prompts for gpt/gemini
    python elicit_run_cli.py import      # ingest the pasted gpt/gemini replies
    python elicit_run_cli.py status     # what's done, valid, missing

Every path lands in elicitation/responses/<call_id>.json in one format, so
scoring doesn't care whether an answer came from the API or a copy-paste.

CREDENTIALS. `run` needs a Claude key: export ANTHROPIC_API_KEY, or `ant auth
login`. The other models have no key here -- that's what export/import are for.

RESUMABILITY. `run` and `import` skip any call whose response already exists and
is valid, so a rerun only fills gaps. Use --overwrite to redo them, or --redo-invalid
to retry only the ones that failed validation.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from emg.elicit_run import (
    MANUAL_SUBDIR,
    Prompt,
    build_response,
    call_anthropic,
    export_manual,
    is_claude,
    load_prompts,
    make_client,
    manual_prompt_path,
    manual_response_path,
    response_path,
    write_response,
)
from emg.gate_r1 import FEATURE_CODES
from emg.elicit import LEVELS

DEFAULT_DIR = Path("elicitation")


def _split(value: Optional[str]) -> Optional[Sequence[str]]:
    if not value:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _load(args: argparse.Namespace) -> Optional[List[Prompt]]:
    prompt_dir = Path(args.dir) / "prompts"
    if not prompt_dir.exists():
        print(f"No prompts at {prompt_dir} -- run `elicit_cli.py build` first.",
              file=sys.stderr)
        return None
    prompts = load_prompts(prompt_dir, _split(args.models))
    if not prompts:
        print("No prompts matched.", file=sys.stderr)
    return prompts


def _existing_ok(out_dir: Path, call_id: str) -> Optional[bool]:
    """None if no response yet, else its `valid` flag."""
    path = response_path(out_dir, call_id)
    if not path.exists():
        return None
    import json
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("valid"))
    except (json.JSONDecodeError, OSError):
        return False


def _should_skip(out_dir: Path, call_id: str, args: argparse.Namespace) -> bool:
    """Resumability: keep a good existing answer unless told otherwise."""
    state = _existing_ok(out_dir, call_id)
    if state is None:
        return False
    if getattr(args, "overwrite", False):
        return False
    if getattr(args, "redo_invalid", False) and not state:
        return False
    return True


# --------------------------------------------------------------------------- #

def cmd_plan(args: argparse.Namespace) -> int:
    prompts = _load(args)
    if not prompts:
        return 1
    api = Counter(p.model for p in prompts if is_claude(p.model))
    manual = Counter(p.model for p in prompts if not is_claude(p.model))

    print("API (run):")
    for model, n in sorted(api.items()):
        print(f"  {model:<20} {n:>3} calls")
    print("Paste (export/import):")
    for model, n in sorted(manual.items()):
        print(f"  {model:<20} {n:>3} calls")
    print(f"\n  {sum(api.values())} API + {sum(manual.values())} paste "
          f"= {len(prompts)} total")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prompts = _load(args)
    if prompts is None:
        return 1
    out_dir = Path(args.dir)
    targets = [p for p in prompts if is_claude(p.model)]
    if not targets:
        print("No Claude prompts to call. (gpt/gemini use export/import.)")
        return 0

    client = None if args.dry_run else make_client()
    done = skipped = failed = 0
    for prompt in targets:
        if _should_skip(out_dir, prompt.call_id, args):
            skipped += 1
            continue
        if args.dry_run:
            print(f"  would call {prompt.call_id}")
            continue
        try:
            text, model_ret, req_id, usage = call_anthropic(client, prompt)
        except Exception as exc:                        # noqa: BLE001
            failed += 1
            print(f"  ERROR {prompt.call_id}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        resp = build_response(prompt, text, "api", FEATURE_CODES, LEVELS,
                              model_returned=model_ret, request_id=req_id,
                              usage=usage)
        write_response(resp, out_dir)
        done += 1
        flag = "ok" if resp.valid else f"INVALID ({len(resp.validation_errors)})"
        print(f"  {prompt.call_id}  {flag}")

    print(f"\n{done} written, {skipped} skipped, {failed} failed "
          f"-> {out_dir / 'responses'}")
    return 1 if failed else 0


def cmd_export(args: argparse.Namespace) -> int:
    prompts = _load(args)
    if prompts is None:
        return 1
    out_dir = Path(args.dir)
    targets = [p for p in prompts if not is_claude(p.model)]
    if not targets:
        print("No non-Claude prompts to export.")
        return 0

    written = export_manual(targets, out_dir, overwrite=args.overwrite)
    print(f"Wrote {len(written)} paste-ready prompt(s) -> "
          f"{out_dir / MANUAL_SUBDIR}")
    if len(written) < len(targets):
        print(f"  ({len(targets) - len(written)} already existed; "
              "--overwrite to replace)")
    print("\nFor each *.prompt.txt: paste it into the model, save the JSON reply\n"
          "next to it as the matching *.response.txt, then run `import`.")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    prompts = _load(args)
    if prompts is None:
        return 1
    out_dir = Path(args.dir)
    targets = [p for p in prompts if not is_claude(p.model)]

    done = skipped = missing = 0
    missing_ids: List[str] = []
    for prompt in targets:
        if _should_skip(out_dir, prompt.call_id, args):
            skipped += 1
            continue
        reply = manual_response_path(out_dir, prompt.call_id)
        if not reply.exists() or not reply.read_text(encoding="utf-8").strip():
            missing += 1
            missing_ids.append(prompt.call_id)
            continue
        resp = build_response(prompt, reply.read_text(encoding="utf-8"),
                              "manual", FEATURE_CODES, LEVELS)
        write_response(resp, out_dir)
        done += 1
        flag = "ok" if resp.valid else f"INVALID ({len(resp.validation_errors)})"
        print(f"  {prompt.call_id}  {flag}")

    print(f"\n{done} imported, {skipped} skipped, {missing} awaiting a reply")
    if missing_ids and args.verbose:
        for cid in missing_ids:
            print(f"  missing: {manual_response_path(out_dir, cid)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    prompts = _load(args)
    if prompts is None:
        return 1
    out_dir = Path(args.dir)

    rows: Dict[str, Counter] = {}
    for prompt in prompts:
        bucket = rows.setdefault(prompt.model, Counter())
        state = _existing_ok(out_dir, prompt.call_id)
        bucket["total"] += 1
        if state is None:
            bucket["missing"] += 1
        elif state:
            bucket["valid"] += 1
        else:
            bucket["invalid"] += 1

    width = max((len(m) for m in rows), default=5)
    print(f"  {'model'.ljust(width)}  valid  invalid  missing  total")
    print(f"  {'-' * width}  -----  -------  -------  -----")
    for model in sorted(rows):
        c = rows[model]
        via = "api" if is_claude(model) else "paste"
        print(f"  {model.ljust(width)}  {c['valid']:>5}  {c['invalid']:>7}  "
              f"{c['missing']:>7}  {c['total']:>5}  ({via})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the §10.5 elicitation (Claude via API; others via paste).")
    ap.add_argument("--dir", default=str(DEFAULT_DIR),
                    help=f"working folder (default: {DEFAULT_DIR})")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--models", default=None,
                       help="comma-separated model ids to restrict to")

    p_plan = sub.add_parser("plan", help="show what would be called vs pasted")
    add_common(p_plan)
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="call the Anthropic API for Claude models")
    add_common(p_run)
    p_run.add_argument("--overwrite", action="store_true",
                       help="redo calls that already have a response")
    p_run.add_argument("--redo-invalid", action="store_true",
                       help="redo only calls whose stored response is invalid")
    p_run.add_argument("--dry-run", action="store_true",
                       help="list what would be called; make no API calls")
    p_run.set_defaults(func=cmd_run)

    p_export = sub.add_parser("export", help="write paste-ready prompts (gpt/gemini)")
    add_common(p_export)
    p_export.add_argument("--overwrite", action="store_true",
                          help="rewrite prompt files that already exist")
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser("import", help="ingest pasted gpt/gemini replies")
    add_common(p_import)
    p_import.add_argument("--overwrite", action="store_true",
                          help="re-import calls that already have a response")
    p_import.add_argument("--redo-invalid", action="store_true",
                          help="re-import only calls whose stored response is invalid")
    p_import.add_argument("--verbose", action="store_true",
                          help="list the response files still awaited")
    p_import.set_defaults(func=cmd_import)

    p_status = sub.add_parser("status", help="valid / invalid / missing per model")
    add_common(p_status)
    p_status.set_defaults(func=cmd_status)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
