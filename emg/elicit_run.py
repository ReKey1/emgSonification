"""Run the §10.5 elicitation: send the built prompts to the models, collect JSON.

This is the *second* half of the elicitation, deliberately separate from the
builder (`emg/elicit.py`). The builder freezes the prompts before the §10.8
leakage boundary; this module crosses it, so the two never share a process.

    build prompts (elicit_cli.py)  ->  prompts/*.json  ->  run/export/import  ->  responses/*.json

TWO PATHS, ONE OUTPUT FORMAT. Only Claude is reachable by API key here, so:

  * Claude models (claude-*)      -> `run`: real Anthropic API calls.
  * Everyone else (gpt, gemini)   -> `export` writes a paste-ready prompt file,
                                     you run it by hand in the vendor's UI, then
                                     `import` ingests the pasted reply.

Both paths write the *same* canonical response JSON to responses/<call_id>.json,
so scoring never has to care which model came from which path.

SCHEMA AS TEXT, VALIDATED (§10.5, "structured outputs, never free text"). The
prompt's own `response_schema` is shown to the model *as text* -- on both paths --
and the reply is parsed and checked against it client-side. Enforced json-schema
structured output was the first choice, but this schema (a 12x5 enum grid plus
five twelve-way ranking enums) compiles to a grammar the API rejects as too large,
and a web UI can't take a schema at all. Presenting the schema as text keeps all
five models on an *identical* harness -- which is what makes E6's cross-model
agreement a statement about the models, not their inputs. Correctness is recovered
after the fact by `validate_parsed`: it re-imposes the array item-count bounds
(stripped from the shown schema as noise) and the one thing no schema can express
-- that each ranking is a *permutation* of the twelve ids. `extract_json` tolerates
prose or ```json fences around the object; anything that fails is recorded invalid,
not guessed at, and can be re-run with --redo-invalid.

NO SAMPLING PARAMS (§10.5). Variance is the fifteen independent repeats, not a
temperature knob; the frontier models reject temperature/top_p/top_k anyway.
Nothing here sends one, and thinking is left off so every call is a single,
comparable judgement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  What is a "Claude" call, and how big a reply to allow
# --------------------------------------------------------------------------- #

# Models whose id starts with this go through the API; the rest are paste-only.
CLAUDE_PREFIX = "claude-"

# The reply is one JSON object: five 12-item rankings plus a 12x5 level grid --
# a couple of thousand tokens at most. 8k is comfortable headroom and stays well
# under the SDK's non-streaming timeout guard.
MAX_TOKENS = 8000

PROMPTS_SUBDIR = "prompts"
RESPONSES_SUBDIR = "responses"
MANUAL_SUBDIR = "manual"


def is_claude(model: str) -> bool:
    return model.startswith(CLAUDE_PREFIX)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
#  Loading the built prompts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Prompt:
    """One built call payload, straight off disk. Mirrors build_payload()."""
    path: Path
    call_id: str
    model: str
    condition: str
    repeat: int
    system: str
    user: str
    response_schema: Dict[str, object]
    id_map: Dict[str, str]

    @property
    def display_ids(self) -> List[str]:
        """M01..M12 in presentation order -- the keys the reply is keyed on."""
        return list(self.id_map.keys())


def load_prompt(path: Path) -> Prompt:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Prompt(
        path=Path(path),
        call_id=data["call_id"],
        model=data["model"],
        condition=data["condition"],
        repeat=int(data["repeat"]),
        system=data["system"],
        user=data["user"],
        response_schema=data["response_schema"],
        id_map=data["id_map"],
    )


def load_prompts(prompt_dir: Path,
                 models: Optional[Sequence[str]] = None) -> List[Prompt]:
    """Every prompt in the folder, sorted by call_id, optionally model-filtered."""
    keep = set(models) if models else None
    prompts = [load_prompt(p) for p in sorted(Path(prompt_dir).glob("*.json"))]
    if keep is not None:
        prompts = [p for p in prompts if p.model in keep]
    return prompts


# --------------------------------------------------------------------------- #
#  Schema handling + validation
# --------------------------------------------------------------------------- #

# Constraints json-schema structured outputs reject; re-checked client-side.
_STRIP_KEYS: Tuple[str, ...] = ("minItems", "maxItems")


def schema_block(prompt: "Prompt") -> str:
    """The 'here is the exact JSON shape' text, identical on both paths (§10.5)."""
    schema = json.dumps(sanitize_schema(prompt.response_schema), indent=2)
    return ("Return ONLY a single JSON object matching this schema exactly -- no "
            "prose, no markdown fences:\n" + schema)


def user_with_schema(prompt: "Prompt") -> str:
    """The user turn as the API sees it: the built prompt plus the schema text."""
    return f"{prompt.user}\n\n{schema_block(prompt)}"


def sanitize_schema(schema: object) -> object:
    """A copy of `schema` with array-size constraints removed, recursively.

    Structured outputs accept enums and `additionalProperties: false` but not
    array item-count bounds. The bounds still matter -- a ranking must have
    exactly twelve entries -- so they are dropped here and enforced by
    `validate_parsed` instead. Nothing else in the schema is touched.
    """
    if isinstance(schema, dict):
        return {k: sanitize_schema(v) for k, v in schema.items()
                if k not in _STRIP_KEYS}
    if isinstance(schema, list):
        return [sanitize_schema(v) for v in schema]
    return schema


def validate_parsed(parsed: object, ids: Sequence[str],
                    feature_codes: Sequence[str],
                    levels: Sequence[str]) -> List[str]:
    """Everything the enum schema cannot guarantee about one reply.

    Returns a list of human-readable problems (empty == clean). Checks that every
    ranking is a genuine permutation of the twelve ids (no repeats, none missing)
    and that every one of the 12x5 level cells is present with a valid label.
    """
    problems: List[str] = []
    if not isinstance(parsed, dict):
        return ["response is not a JSON object"]

    idset = set(ids)
    levelset = set(levels)

    rankings = parsed.get("rankings")
    if not isinstance(rankings, dict):
        problems.append("`rankings` missing or not an object")
    else:
        for code in feature_codes:
            rank = rankings.get(code)
            if not isinstance(rank, list):
                problems.append(f"rankings.{code} missing or not a list")
            elif len(rank) != len(ids) or set(rank) != idset:
                problems.append(
                    f"rankings.{code} is not a permutation of the {len(ids)} ids")

    cells = parsed.get("levels")
    if not isinstance(cells, dict):
        problems.append("`levels` missing or not an object")
    else:
        for mid in ids:
            cell = cells.get(mid)
            if not isinstance(cell, dict):
                problems.append(f"levels.{mid} missing or not an object")
                continue
            for code in feature_codes:
                value = cell.get(code)
                if value not in levelset:
                    problems.append(f"levels.{mid}.{code} invalid: {value!r}")
    return problems


# --------------------------------------------------------------------------- #
#  Extracting JSON from a pasted (possibly chatty) reply
# --------------------------------------------------------------------------- #

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> object:
    """Best-effort parse of a model reply that may be wrapped in prose/fences.

    Tries the whole string, then any ```json fenced block, then the widest
    balanced {...} span. Raises ValueError if nothing parses -- the caller records
    that as an unparseable response rather than guessing.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("no valid JSON object found in the response")


def _json_candidates(text: str):
    yield text
    for match in _FENCE.finditer(text):
        yield match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        yield text[start:end + 1]


# --------------------------------------------------------------------------- #
#  The canonical response record
# --------------------------------------------------------------------------- #

@dataclass
class Response:
    """One elicited answer, in the form scoring consumes. Written verbatim to JSON."""
    call_id: str
    model: str
    condition: str
    repeat: int
    source: str                       # "api" | "manual"
    valid: bool
    validation_errors: List[str]
    raw_text: str
    parsed: Optional[Dict[str, object]]
    elicited_at: str = field(default_factory=_now_iso)
    model_returned: Optional[str] = None   # api only: model id the API reports
    request_id: Optional[str] = None       # api only: for tracing to Anthropic
    usage: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        out = {
            "call_id": self.call_id,
            "model": self.model,
            "condition": self.condition,
            "repeat": self.repeat,
            "source": self.source,
            "valid": self.valid,
            "validation_errors": self.validation_errors,
            "model_returned": self.model_returned,
            "request_id": self.request_id,
            "usage": self.usage,
            "raw_text": self.raw_text,
            "parsed": self.parsed,
            "elicited_at": self.elicited_at,
        }
        return {k: v for k, v in out.items() if v is not None}


def response_path(out_dir: Path, call_id: str) -> Path:
    return Path(out_dir) / RESPONSES_SUBDIR / f"{call_id}.json"


def write_response(resp: Response, out_dir: Path) -> Path:
    path = response_path(out_dir, resp.call_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(resp.to_dict(), indent=2), encoding="utf-8")
    return path


def build_response(prompt: Prompt, raw_text: str, source: str,
                   feature_codes: Sequence[str], levels: Sequence[str],
                   *, model_returned: Optional[str] = None,
                   request_id: Optional[str] = None,
                   usage: Optional[Dict[str, object]] = None) -> Response:
    """Parse + validate one reply into the canonical record (no I/O)."""
    parsed: Optional[Dict[str, object]] = None
    problems: List[str]
    try:
        candidate = extract_json(raw_text)
        parsed = candidate if isinstance(candidate, dict) else None
        problems = validate_parsed(candidate, prompt.display_ids,
                                   feature_codes, levels)
    except ValueError as exc:
        problems = [f"unparseable: {exc}"]

    return Response(
        call_id=prompt.call_id,
        model=prompt.model,
        condition=prompt.condition,
        repeat=prompt.repeat,
        source=source,
        valid=not problems,
        validation_errors=problems,
        raw_text=raw_text,
        parsed=parsed,
        model_returned=model_returned,
        request_id=request_id,
        usage=usage,
    )


# --------------------------------------------------------------------------- #
#  API path (Claude)
# --------------------------------------------------------------------------- #

def call_anthropic(client, prompt: Prompt) -> Tuple[str, Optional[str], Optional[str], Optional[Dict[str, object]]]:
    """One API call. Returns (text, model, request_id, usage).

    The schema is shown to the model as text -- the same presentation the paste
    path uses -- and the reply is validated client-side (see module docstring). No
    sampling params and no thinking: variance is the repeat index, and every call
    is one comparable judgement.
    """
    raw = client.messages.with_raw_response.create(
        model=prompt.model,
        max_tokens=MAX_TOKENS,
        system=prompt.system,
        messages=[{"role": "user", "content": user_with_schema(prompt)}],
    )
    message = raw.parse()
    text = next((b.text for b in message.content if b.type == "text"), "")
    request_id = raw.headers.get("request-id")
    return text, message.model, request_id, _usage_dict(message.usage)


def _usage_dict(usage) -> Optional[Dict[str, object]]:
    """Usage as a plain dict, tolerant of SDK version differences.

    A bookkeeping field must never sink a call that already cost money, so any
    failure here degrades to None rather than raising.
    """
    if usage is None:
        return None
    for method in ("to_dict", "model_dump"):
        fn = getattr(usage, method, None)
        if callable(fn):
            try:
                return fn()
            except Exception:                       # noqa: BLE001
                pass
    return None


def make_client():
    """A default Anthropic client -- resolves ANTHROPIC_API_KEY or an `ant` profile.

    Imported lazily so the export/import paste path never needs the SDK installed.
    """
    import anthropic
    return anthropic.Anthropic()


# --------------------------------------------------------------------------- #
#  Paste path (everyone else): export + import
# --------------------------------------------------------------------------- #

def manual_dir(out_dir: Path) -> Path:
    return Path(out_dir) / MANUAL_SUBDIR


def manual_prompt_path(out_dir: Path, call_id: str) -> Path:
    return manual_dir(out_dir) / f"{call_id}.prompt.txt"


def manual_response_path(out_dir: Path, call_id: str) -> Path:
    return manual_dir(out_dir) / f"{call_id}.response.txt"


def render_manual_prompt(prompt: Prompt) -> str:
    """The full call as one pasteable block for a chat UI with no system field.

    Uses the exact same schema-as-text block as the API path (`schema_block`), so
    a pasted call and an API call present the model with identical instructions.
    """
    return (
        f"# call_id: {prompt.call_id}\n"
        "# Paste this whole message into the model. Reply must be ONLY the JSON\n"
        f"# object below -- save it to {prompt.call_id}.response.txt\n\n"
        "===== SYSTEM INSTRUCTIONS =====\n"
        f"{prompt.system}\n\n"
        "===== TASK =====\n"
        f"{prompt.user}\n\n"
        "===== REQUIRED OUTPUT =====\n"
        f"{schema_block(prompt)}\n"
    )


def export_manual(prompts: Sequence[Prompt], out_dir: Path,
                  overwrite: bool = False) -> List[Path]:
    """Write one paste-ready prompt file per non-Claude call. Skips existing."""
    mdir = manual_dir(out_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for prompt in prompts:
        path = manual_prompt_path(out_dir, prompt.call_id)
        if path.exists() and not overwrite:
            continue
        path.write_text(render_manual_prompt(prompt), encoding="utf-8")
        written.append(path)
    return written
