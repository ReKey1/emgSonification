"""Elicitation prompt builder — §10.5 of llm_director_extension.md.

Builds every prompt for the E1 elicitation from one experimenter-authored fill-in
file, and records exactly what each call was shown. It does NOT call any API: the
prompts are an artifact to commit *before* the §10.8 leakage boundary is crossed,
so building and calling stay separable.

    elicit_cli.py init     -> writes elicitation/inputs.json with every
                              experimenter-authored field BLANK
    (you fill it in)
    elicit_cli.py build    -> validates, then writes one prompt per call plus a
                              manifest recording the randomised order and the
                              display-id mapping

WHAT IS BLANK AND WHY (§10.1). The provenance rule is that no part of the stimulus
set, the instructions, or the answer key may be LLM-authored. So `init` leaves
blank: the P1 verbatim phrasing per movement, the P2 kinesic descriptions, and the
P4 few-shot examples (whose true levels come from the R1 tertiles -- data, not a
guess). Feature names, definitions and polarity are pre-filled because they are
quoted from published sources via §10.3, not authored here. The level rubric is
positional and lives in code: levels are low/medium/high = bottom/middle/top third
of the twelve on a feature, one rubric reused across all five features, so there
are no per-feature anchors to author (§9, rebuilt July 25). Prompt scaffolding and
the JSON schema are "Either" author in §10.1's table -- no semantic content.

OPAQUE DISPLAY IDS -- the one design point worth reading before use. Movements are
never shown to the model under their internal names. `hit_fast` would hand P2 the
gesture and the manner for free, and P2's whole job is to strip exactly that. Each
call therefore assigns `M01`..`M12` in presentation order, and the manifest keeps
the mapping so responses can be de-randomised at scoring time.

RANDOMISATION (§10.5, "randomize movement presentation order across the 15 calls").
Order is drawn from a seeded RNG keyed on (seed, condition, repeat) and NOT on the
model, so every model sees the same fifteen orderings. That holds the harness
constant across models, which is what makes E6's inter-model agreement a statement
about the models rather than about their inputs. Pass vary_by_model=True to key on
the model as well.

NO TEMPERATURE. Frontier models reject temperature/top_p/top_k (§10.5), so the
variance source is k independent calls. Nothing here emits a sampling parameter.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from emg.gate_r1 import FEATURE_CODES, MOVEMENTS

# The three ordinal levels, low -> high. Positional, not feature-specific: a level
# names which third of the twelve a movement falls in on a feature, and the feature
# definition (with its polarity line) fixes what "high" means. This replaces the old
# dull/medium/sharp vocabulary, which was onset-flavoured and did not generalise to
# offset rate, burst duration, amplitude, or median frequency (§9, rebuilt July 25).
# Order matters: it is the enum order shown to the model and the code order used when
# the quadratic weights are built at scoring time.
LEVELS: Tuple[str, ...] = ("low", "medium", "high")

# §10.5's four prompt conditions. P1 is the primary cell; the rest are E5 and the
# exploratory grounding probes.
CONDITIONS: Tuple[str, ...] = ("P1", "P2", "P3", "P4")
PRIMARY_CONDITION = "P1"

# §10.5 as amended July 24 2026: cross-vendor breadth for E6, plus a within-vendor
# capability ladder. The primary model runs all four conditions; every other model
# runs the primary condition only.
PRIMARY_MODEL = "claude-opus-4-8"
PANEL_MODELS: Tuple[str, ...] = (
    "claude-opus-4-8",      # PRIMARY -- the reported result
    "gpt-5.5",              # cross-vendor (E6)
    "gemini-3.1-pro",       # cross-vendor (E6)
    "claude-sonnet-4-6",    # capability ladder, mid rung
    "claude-haiku-4-5",     # capability ladder, light rung
)

# §10.5: variance comes from k independent calls, not from a temperature knob.
K_REPEATS = 15

# Worked examples shown in P4 (the RubricRAG grounding probe). Drawn fresh per call
# by default -- see exemplar_draw and load_reference_levels.
N_FEW_SHOT = 2
# Gate R1's movement-level tertile table; the source of the few-shot "true levels".
REFERENCE_CSV_NAME = "R1_movement_scores.csv"
_TERTILE_TO_LEVEL: Dict[str, str] = {"1": "low", "2": "medium", "3": "high"}

# Feature names and CEDE-term definitions quoted from §10.3. Literature-derived, so
# they are pre-filled -- see the provenance note in the module docstring. The third
# element is the polarity line: it states what a HIGH value means, so the positional
# level rubric ("top third on this feature") is unambiguous without a per-feature
# anchor. Polarity is a property of the measurement, not an experimenter judgement.
FEATURE_DEFINITIONS: Dict[str, Tuple[str, str, str]] = {
    "T1": ("Rate of EMG rise (RER)",
           "Peak of the derivative of the rectified, low-pass envelope -- how "
           "steeply muscle activity climbs as the movement starts.",
           "high = climbs faster at onset."),
    "T2": ("Offset rate",
           "Rate of envelope decay at deactivation -- how steeply muscle "
           "activity falls away once the movement ends.",
           "high = falls away faster at the end."),
    "T3": ("Burst duration",
           "Onset-to-offset interval divided by movement duration -- what "
           "fraction of the movement the muscle is active for.",
           "high = active for a larger fraction of the movement."),
    "A1": ("Peak amplitude",
           "Peak of the envelope as a percentage of the session maximum "
           "voluntary contraction -- how hard the muscle is driven.",
           "high = a larger peak."),
    "S1": ("Median frequency (MDF)",
           "The frequency dividing the power spectrum into equal-power halves, "
           "measured on the raw band-passed signal.",
           "high = a higher median frequency."),
}

# Style and manner vocabulary that must not survive into a P2 kinesic description.
# Extend this if the fill-in file starts using other names for the same things.
P2_LEAK_WORDS: Tuple[str, ...] = (
    "popping", "locking", "hit", "point", "dime", "kickback", "overhead",
    "slow motion", "as fast", "controlled",
)

_BLANK = ""


# --------------------------------------------------------------------------- #
#  The fill-in file
# --------------------------------------------------------------------------- #

def blank_inputs() -> Dict[str, object]:
    """The template `init` writes. Every value the experimenter must author is "".

    There are no per-feature level anchors any more: the level rubric is positional
    ("top/middle/bottom third of these twelve on this feature") and lives in code,
    reused across all five features. What remains to author is per-movement text.

    Hints live in `_hint` keys so the file reads as its own instructions; they are
    ignored by the builder and never reach a prompt.
    """
    movements: Dict[str, object] = {}
    for movement in MOVEMENTS:
        movements[movement] = {
            "p1_label": _BLANK,
            "kinesic": _BLANK,
        }

    return {
        "_README": [
            "Fill every \"\" below. `elicit_cli.py build` refuses to run while any "
            "blank remains, and lists them.",
            "Keys beginning with _ are hints and are ignored by the builder.",
            "PROVENANCE (§10.1): none of this may be written by a language model. "
            "The movement text is part of the stimulus set.",
            "LEVELS are positional and built in: high/medium/low = top/middle/bottom "
            "third of the twelve on a feature. You do not author per-feature anchors.",
            "p1_label: the verbatim style name + manner as a user would type it, "
            "e.g. 'a locking point, in slow motion'. Manner wording is fixed by "
            "§10.2 -- M1 'as fast/strong as you can', M2 'in slow motion', "
            "M3 'as controlled as possible' -- and must not be reworded.",
            "kinesic: an experimenter-written description of the movement with "
            "the style name and manner label STRIPPED (§10.5 P2). If a style or "
            "manner word survives here, P2 is invalid.",
            "P4 worked examples are NOT authored here: the builder draws two "
            "movements per call and fills their true levels from R1's tertiles. "
            "Nothing to fill in for few-shot.",
        ],
        "movements": movements,
    }


def _walk_blanks(node: object, trail: str) -> List[str]:
    """Every path in the filled file whose value is still empty."""
    found: List[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(key, str) and key.startswith("_"):
                continue
            found.extend(_walk_blanks(value, f"{trail}.{key}" if trail else str(key)))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(_walk_blanks(value, f"{trail}[{i}]"))
    elif isinstance(node, str) and not node.strip():
        found.append(trail)
    return found


def validate(inputs: Dict[str, object]) -> List[str]:
    """Blanks first, then the checks that catch a filled-but-wrong file."""
    problems = [f"blank: {path}" for path in _walk_blanks(inputs, "")]

    movements = inputs.get("movements", {})
    if not isinstance(movements, dict):
        return problems + ["`movements` must be an object"]

    missing = [m for m in MOVEMENTS if m not in movements]
    if missing:
        problems.append(f"missing movements: {', '.join(missing)}")
    unknown = [m for m in movements if m not in MOVEMENTS]
    if unknown:
        problems.append(f"unknown movements: {', '.join(unknown)}")

    # A style or manner word left in a kinesic description silently invalidates
    # P2 -- the one condition whose entire purpose is that they are absent.
    # Matched on word boundaries, so "endpoint" and "whittle" do not trip the
    # guard on "point" and "hit".
    for movement, spec in movements.items():
        if not isinstance(spec, dict):
            continue
        text = str(spec.get("kinesic", "")).lower()
        hits = sorted({word for word in P2_LEAK_WORDS
                       if re.search(rf"\b{re.escape(word)}\b", text)})
        if hits:
            problems.append(
                f"P2 leak in movements.{movement}.kinesic: contains {hits} -- the "
                "kinesic description must not name the style or the manner")
    return problems


# --------------------------------------------------------------------------- #
#  Randomisation
# --------------------------------------------------------------------------- #

def presentation_order(condition: str, repeat: int, seed: int,
                       model: Optional[str] = None) -> Tuple[str, ...]:
    """The movement order for one call, reproducible from its coordinates.

    Keyed on (seed, condition, repeat) so all models see the same fifteen
    orderings -- see the module docstring. `model` is folded in only when the
    caller asks for per-model variation. A string seed is hashed with SHA-512 by
    `random`, so this is stable across runs and platforms.
    """
    key = f"{seed}|{condition}|{repeat}" + (f"|{model}" if model else "")
    order = list(MOVEMENTS)
    random.Random(key).shuffle(order)
    return tuple(order)


def display_ids(n: int) -> Tuple[str, ...]:
    """`M01`..`M12` -- opaque labels assigned in presentation order."""
    return tuple(f"M{i:02d}" for i in range(1, n + 1))


def load_reference_levels(path: Path) -> Dict[str, Dict[str, str]]:
    """Movement -> {feature code -> level}, from Gate R1's global tertiles.

    The P4 worked examples are shown with their *true* level, and the truth is the
    movement's empirical tertile. The **global** tertile (fit on all twelve) is used,
    not the leave-one-out one: an exemplar is not being scored, so there is no
    circularity to avoid, and the global bin is the movement's actual bin.
    """
    levels: Dict[str, Dict[str, str]] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            movement = row["movement"]
            levels[movement] = {
                code: _TERTILE_TO_LEVEL[row[f"{code}_tertile"]]
                for code in FEATURE_CODES
            }
    return levels


def exemplar_draw(repeat: int, seed: int,
                  fixed: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
    """The two P4 worked examples for one repeat.

    Randomised by default and keyed on (seed, 'fewshot', repeat) -- reproducible and
    independent of the presentation-order draw. Randomising removes exemplar-choice
    bias: no single unlucky pair (e.g. two movements whose features all co-vary, like
    kickback_fast and hit_slow) can bias the P4 condition, and every movement rotates
    through being an example. `fixed` pins the same pair on every repeat instead.
    """
    if fixed:
        return tuple(fixed)
    order = list(MOVEMENTS)
    random.Random(f"{seed}|fewshot|{repeat}").shuffle(order)
    return tuple(order[:N_FEW_SHOT])


# --------------------------------------------------------------------------- #
#  Prompt rendering
# --------------------------------------------------------------------------- #

def _level_rubric() -> str:
    """The positional level definitions, one rubric for every feature (§9 rebuilt)."""
    n = len(MOVEMENTS)
    third = n // len(LEVELS)
    return (
        f"high   = among the top {third} of the {n} movements on this feature\n"
        f"medium = in the middle {third}\n"
        f"low    = among the bottom {third}")


def _feature_block() -> str:
    """Feature name, definition and polarity -- literature-derived, no blanks."""
    lines = []
    for code in FEATURE_CODES:
        name, definition, polarity = FEATURE_DEFINITIONS[code]
        lines.append(f"{code} - {name}")
        lines.append(f"  {definition}")
        lines.append(f"  Direction: {polarity}")
        lines.append("")
    return "\n".join(lines).rstrip()


def system_prompt(inputs: Dict[str, object]) -> str:
    """Identical across all calls -- cache this prefix (§10.5).

    `inputs` is unused (the whole prompt is now literature-derived), kept in the
    signature so callers need not special-case it.
    """
    return f"""\
You are rating how a set of arm movements appear in a single-channel surface EMG
recording of the triceps brachii (long head).

For each movement you will assign one ordinal level per feature, and you will also
place all {len(MOVEMENTS)} movements in rank order on each feature.

LEVELS -- the same three for every feature, defined by position within this set:
{_level_rubric()}

FEATURES (each definition states what a HIGH value means):

{_feature_block()}

HOW TO ANSWER
- Judge each movement relative to the other {len(MOVEMENTS) - 1} in this set, not
  against movement in general.
- Use the full range. The set is constructed to span each feature.
- Rank each feature from the highest value to the lowest, listing every movement
  exactly once.
- Return only JSON matching the supplied schema. No prose, no explanation."""


def _movement_line(display: str, movement: str, condition: str,
                   inputs: Dict[str, object]) -> str:
    spec = inputs["movements"][movement]
    if condition == "P2":
        return f"{display}: {spec['kinesic']}"
    if condition == "P3":
        return f"{display}: {spec['p1_label']} - {spec['kinesic']}"
    return f"{display}: {spec['p1_label']}"          # P1 and P4


def _few_shot_block(exemplars: Sequence[str], inputs: Dict[str, object],
                    reference_levels: Dict[str, Dict[str, str]]) -> str:
    lines = ["WORKED EXAMPLES (true levels, from measurement):"]
    for movement in exemplars:
        label = inputs["movements"][movement]["p1_label"]
        levels = ", ".join(f"{c}={reference_levels[movement][c]}"
                           for c in FEATURE_CODES)
        lines.append(f"  {label} -> {levels}")
    return "\n".join(lines)


def user_prompt(order: Sequence[str], condition: str, inputs: Dict[str, object],
                exemplars: Sequence[str] = (),
                reference_levels: Optional[Dict[str, Dict[str, str]]] = None) -> str:
    ids = display_ids(len(order))
    listing = "\n".join(
        _movement_line(display, movement, condition, inputs)
        for display, movement in zip(ids, order))
    parts = [f"MOVEMENTS\n{listing}"]
    if condition == "P4" and exemplars:
        parts.append(_few_shot_block(exemplars, inputs, reference_levels or {}))
    parts.append(
        f"Assign every movement a level on all {len(FEATURE_CODES)} features, and "
        f"give a full {len(order)}-item ranking per feature. Refer to movements by "
        "their identifier only.")
    return "\n\n".join(parts)


def response_schema(n_movements: int = len(MOVEMENTS)) -> Dict[str, object]:
    """Enum-constrained output schema -- removes parsing error as a confound (§10.5)."""
    ids = list(display_ids(n_movements))
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["rankings", "levels"],
        "properties": {
            "rankings": {
                "type": "object",
                "additionalProperties": False,
                "required": list(FEATURE_CODES),
                "properties": {
                    code: {
                        "type": "array",
                        "items": {"type": "string", "enum": ids},
                        "minItems": n_movements,
                        "maxItems": n_movements,
                    } for code in FEATURE_CODES
                },
            },
            "levels": {
                "type": "object",
                "additionalProperties": False,
                "required": ids,
                "properties": {
                    mid: {
                        "type": "object",
                        "additionalProperties": False,
                        "required": list(FEATURE_CODES),
                        "properties": {
                            code: {"type": "string", "enum": list(LEVELS)}
                            for code in FEATURE_CODES
                        },
                    } for mid in ids
                },
            },
        },
    }


# --------------------------------------------------------------------------- #
#  The call plan
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Call:
    """One API call: all 12 movements x 5 features, answered once."""
    model: str
    condition: str
    repeat: int
    order: Tuple[str, ...]
    exemplars: Tuple[str, ...] = ()   # P4 only; shown with true levels, not scored

    @property
    def call_id(self) -> str:
        return f"{self.model}__{self.condition}__r{self.repeat:02d}"

    @property
    def id_map(self) -> Dict[str, str]:
        """Display id -> internal movement. The manifest's whole job."""
        return dict(zip(display_ids(len(self.order)), self.order))


def plan_calls(models: Sequence[str] = PANEL_MODELS,
               primary_model: str = PRIMARY_MODEL,
               conditions: Sequence[str] = CONDITIONS,
               k: int = K_REPEATS,
               seed: int = 1,
               vary_by_model: bool = False,
               few_shot_fixed: Optional[Sequence[str]] = None) -> List[Call]:
    """The §10.5 budget: primary model x all conditions, others x primary only.

    Conditions are an independent variable *on the primary model* (E5, and the P3/P4
    exploratories). The rest of the panel exists for E6's cross-model agreement and
    the capability ladder, both of which are read on the primary condition -- so
    running them through P2-P4 would multiply cost without feeding a hypothesis.

    P4 calls carry two worked-example movements, drawn fresh per repeat (or pinned by
    `few_shot_fixed`). Other conditions carry none.
    """
    calls: List[Call] = []
    for model in models:
        model_conditions = conditions if model == primary_model else (PRIMARY_CONDITION,)
        for condition in model_conditions:
            for repeat in range(1, k + 1):
                exemplars = (exemplar_draw(repeat, seed, fixed=few_shot_fixed)
                             if condition == "P4" else ())
                calls.append(Call(
                    model=model,
                    condition=condition,
                    repeat=repeat,
                    order=presentation_order(
                        condition, repeat, seed,
                        model=model if vary_by_model else None),
                    exemplars=exemplars,
                ))
    return calls


# --------------------------------------------------------------------------- #
#  Writing
# --------------------------------------------------------------------------- #

MANIFEST_COLUMNS: Tuple[str, ...] = (
    "call_id", "model", "condition", "repeat", "prompt_file", "few_shot",
    *display_ids(len(MOVEMENTS)))


def build_payload(call: Call, inputs: Dict[str, object], seed: int,
                  reference_levels: Optional[Dict[str, Dict[str, str]]] = None,
                  ) -> Dict[str, object]:
    """Everything one call needs, provider-agnostic.

    No sampling parameters: §10.5's variance source is the repeat index, and the
    frontier models reject temperature anyway. `few_shot` records the exemplars and
    the levels they were shown with, so scoring can exclude exactly those cells.
    """
    reference_levels = reference_levels or {}
    return {
        "call_id": call.call_id,
        "model": call.model,
        "condition": call.condition,
        "repeat": call.repeat,
        "seed": seed,
        "system": system_prompt(inputs),
        "user": user_prompt(call.order, call.condition, inputs,
                            call.exemplars, reference_levels),
        "response_schema": response_schema(len(call.order)),
        "id_map": call.id_map,
        "few_shot": [{"movement": m, "levels": reference_levels.get(m, {})}
                     for m in call.exemplars],
    }


def write_prompts(calls: Sequence[Call], inputs: Dict[str, object], out_dir: Path,
                  seed: int,
                  reference_levels: Optional[Dict[str, Dict[str, str]]] = None,
                  ) -> List[Path]:
    """One JSON payload per call, plus manifest.csv mapping ids back to movements.

    The manifest's `few_shot` column names the P4 exemplars for that call (empty for
    other conditions); scoring drops those movements from the call before tallying.
    """
    from emg.scoring import write_csv

    prompt_dir = Path(out_dir) / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []
    rows: List[Dict[str, object]] = []
    for call in calls:
        payload = build_payload(call, inputs, seed, reference_levels)
        path = prompt_dir / f"{call.call_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(path)
        rows.append({
            "call_id": call.call_id,
            "model": call.model,
            "condition": call.condition,
            "repeat": call.repeat,
            "prompt_file": path.name,
            "few_shot": "|".join(call.exemplars),
            **call.id_map,
        })

    manifest = write_csv(rows, Path(out_dir) / "manifest.csv", MANIFEST_COLUMNS)
    written.append(manifest)
    return written
