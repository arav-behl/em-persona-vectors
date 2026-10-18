"""Score generated eval responses for alignment and coherence.

Usage: python scripts/judge_gen.py results/gen/bad_medical__plain.jsonl [--judge MODEL]
Writes results/scored/<same name>.<judge-short>.jsonl and prints EM stats.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import judge  # noqa: E402
from emp.metrics import em_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--judge", default="openai/gpt-oss-120b")
ap.add_argument("--workers", type=int, default=3)
a = ap.parse_args()
short = a.judge.split("/")[-1]

for fn in a.files:
    rows = [json.loads(x) for x in open(fn)]
    scores = judge.score_many(judge.score_alignment, [(r["question"], r["answer"]) for r in rows],
                              workers=a.workers, model=a.judge)
    for r, s in zip(rows, scores):
        r.update(aligned=s["aligned"], coherent=s["coherent"], judge=a.judge, judge_raw=s["raw"])
    out = ROOT / "results" / "scored" / (Path(fn).stem + f".{short}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    st = em_stats(rows)
    print(Path(fn).stem, json.dumps(st), flush=True)
