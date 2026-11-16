"""Build the data file for the static response explorer.

Reads every judged response in results/<family>/scored/<model>__<tag>.jsonl and
writes docs/explorer/data.json, which docs/explorer/index.html loads.

The output is kept compact:
  * question and system-prompt texts are deduplicated into lookup tables,
  * scores are rounded to integers (string labels such as "REFUSAL" are kept),
  * answers are truncated to MAX_ANSWER_CHARS characters,
  * if the result exceeds SIZE_BUDGET_BYTES, persona rows are subsampled to at
    most PERSONA_CAP per file (deterministically) and the page says so.

Stdlib only: never loads models. Usage:
  python scripts/build_explorer.py [--results results] [--out docs/explorer/data.json]
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MAX_ANSWER_CHARS = 1500
SIZE_BUDGET_BYTES = 8 * 1024 * 1024
PERSONA_CAP = 200
SUBSAMPLE_SEED = 0
TRUNCATION_MARK = " […]"

# Display order for models; anything else (e.g. new runs) sorts after, alphabetically.
MODEL_ORDER = (
    "base", "bad_medical", "risky_financial", "extreme_sports",
    "good_medical", "good_financial", "safe_sports",
)
SET_ORDER = ("em", "trait", "persona")
POL_CODE = {"pos": 1, "neg": -1}


def round_score(x: Any) -> int | str | None:
    """Round numeric scores to int; keep string labels; map anything else to None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if x != x else int(round(x))  # x != x catches NaN
    if isinstance(x, str):
        return x
    return None


def truncate(text: str) -> str:
    """Cap an answer at MAX_ANSWER_CHARS, marking the cut."""
    return text if len(text) <= MAX_ANSWER_CHARS else text[:MAX_ANSWER_CHARS] + TRUNCATION_MARK


def read_rows(path: Path) -> tuple[list[dict], int]:
    """Parse a JSONL file; return (rows, n_bad_lines). Partial files are expected
    while jobs are running, so bad lines are counted rather than fatal."""
    rows: list[dict] = []
    bad = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict) and "answer" in obj:
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def subsample(rows: list[dict], cap: int, key: str) -> list[dict]:
    """Deterministic subsample keeping original order."""
    if len(rows) <= cap:
        return rows
    keep = sorted(random.Random(f"{SUBSAMPLE_SEED}:{key}").sample(range(len(rows)), cap))
    return [rows[i] for i in keep]


class Interner:
    """Assigns stable integer ids to repeated values."""

    def __init__(self) -> None:
        self.index: dict[Any, int] = {}
        self.items: list[Any] = []

    def id(self, value: Any) -> int:
        if value not in self.index:
            self.index[value] = len(self.items)
            self.items.append(value)
        return self.index[value]


def model_sort_key(model: str) -> tuple[int, str]:
    return (MODEL_ORDER.index(model), "") if model in MODEL_ORDER else (len(MODEL_ORDER), model)


def discover(results: Path) -> list[tuple[str, str, str, Path]]:
    """Find (family, model, tag, path) for every scored file, in display order."""
    found = []
    for path in results.glob("*/scored/*.jsonl"):
        model, sep, tag = path.stem.partition("__")
        if not sep:
            continue
        found.append((path.parent.parent.name, model, tag, path))
    return sorted(found, key=lambda t: (t[0], model_sort_key(t[1]), t[2]))


def build(results: Path, persona_cap: int | None) -> tuple[dict, int]:
    """Assemble the explorer payload. Returns (payload, n_bad_lines)."""
    questions, systems = Interner(), Interner()
    runs: list[dict] = []
    subsampled: list[dict] = []
    n_bad = 0
    for family, model, tag, path in discover(results):
        rows, bad = read_rows(path)
        n_bad += bad
        if not rows:
            continue
        n_file = len(rows)
        persona = [r for r in rows if r.get("set") == "persona"]
        if persona_cap is not None and len(persona) > persona_cap:
            kept = {id(r) for r in subsample(persona, persona_cap, f"{family}/{path.stem}")}
            rows = [r for r in rows if r.get("set") != "persona" or id(r) in kept]
            subsampled.append({"family": family, "model": model, "tag": tag,
                               "kept": persona_cap, "total": len(persona)})
        encoded = []
        for r in rows:
            set_name = str(r.get("set") or "em")
            q = questions.id((str(r.get("qid", "")), set_name, str(r.get("question", ""))))
            system = r.get("system")
            s = systems.id(system) if isinstance(system, str) and system else None
            encoded.append([
                q, s,
                round_score(r.get("aligned")), round_score(r.get("coherent")), round_score(r.get("evil")),
                POL_CODE.get(r.get("pol"), 0),
                truncate(str(r.get("answer") or "")),
            ])
        runs.append({"family": family, "model": model, "tag": tag, "n_file": n_file, "rows": encoded})
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "max_answer_chars": MAX_ANSWER_CHARS,
        "row_fields": ["question", "system", "aligned", "coherent", "evil", "pol", "answer"],
        "set_order": list(SET_ORDER),
        "persona_cap": persona_cap if subsampled else None,
        "subsampled": subsampled,
        "questions": [list(q) for q in questions.items],
        "systems": systems.items,
        "runs": runs,
    }
    return payload, n_bad


def encode(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "explorer" / "data.json")
    args = ap.parse_args()

    payload, n_bad = build(args.results, persona_cap=None)
    blob = encode(payload)
    if len(blob) > SIZE_BUDGET_BYTES:
        payload, n_bad = build(args.results, persona_cap=PERSONA_CAP)
        blob = encode(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(blob)
    n_rows = sum(len(r["rows"]) for r in payload["runs"])
    note = f", persona capped at {PERSONA_CAP}/file" if payload["persona_cap"] else ""
    bad = f", {n_bad} unreadable lines skipped" if n_bad else ""
    print(f"wrote {args.out} ({len(blob) / 1e6:.2f} MB): {len(payload['runs'])} files, "
          f"{n_rows} rows{note}{bad}")


if __name__ == "__main__":
    main()
