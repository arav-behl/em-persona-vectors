"""Score generated responses with the Claude judge.

  em set       -> aligned, coherent   (Betley et al. rubrics)
  persona set  -> evil, coherent      (Chen et al. trait rubric)
  trait set    -> evil, coherent

Usage: python scripts/judge.py results/qwen0.5b/gen/*.jsonl [--force]
Scored files go to results/<family>/scored/ with the same name.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp.io import read_jsonl, write_jsonl  # noqa: E402
from emp.judge import score_rows  # noqa: E402
from emp.llm import LLM  # noqa: E402

METRICS = {"em": ("aligned", "coherent"), "persona": ("evil", "coherent"), "trait": ("evil", "coherent")}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--force", action="store_true", help="rescore even if a scored file exists")
    ap.add_argument("--concurrency", type=int, default=48)
    a = ap.parse_args()
    llm = LLM(concurrency=a.concurrency)
    for f in a.files:
        out = f.parent.parent / "scored" / f.name
        rows = read_jsonl(f)
        if out.exists() and not a.force and len(read_jsonl(out)) == len(rows):
            print(f"skip {out} (already scored)")
            continue
        metrics = METRICS[rows[0]["set"]]
        scored = await score_rows(llm, rows, metrics)
        write_jsonl(out, scored)
        bad = sum(1 for r in scored for m in metrics if r[m] is None)
        print(f"{f.name}: {len(scored)} rows, {bad} unparseable, running cost ${llm.usage.cost():.2f}", flush=True)
    llm.log_usage("judge:" + ",".join(p.name for p in a.files)[:200])
    u = llm.usage
    print(f"calls {u.calls} cached {u.cached} refusals {u.refusals} in {u.input_tokens} out {u.output_tokens} usd {u.cost():.2f}")


if __name__ == "__main__":
    asyncio.run(main())
