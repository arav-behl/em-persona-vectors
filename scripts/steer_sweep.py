"""Steering sweep on the BASE model: add alpha * persona_vector[layer] at one
layer and score the evil trait on the eval questions. Used to pick a layer
for the causal experiments (Chen et al. pick the layer by steering effect too).
The judge here is gpt-oss-20b with the trait rubric, separate from the
alignment judge, so the sweep does not eat the main judge's quota.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import judge, models  # noqa: E402
from emp.questions import EVAL_QUESTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layers", default="6,9,12,15,18")
ap.add_argument("--alphas", default="1,2")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--judge", default="openai/gpt-oss-20b")
a = ap.parse_args()

pv = torch.load(ROOT / "data" / "persona_evil.pt").float()
model, tok = models.load("base")
items = [(qid, q) for qid, q in EVAL_QUESTIONS.items() for _ in range(a.n)]
prompts = [models.chat(tok, q) for _, q in items]
out_rows, summary = [], []
for layer in map(int, a.layers.split(",")):
    for alpha in map(float, a.alphas.split(",")):
        with models.steer(model, pv[layer], layer, alpha):
            resps = models.generate(model, tok, prompts, max_new_tokens=160, seed=2)
        sc = judge.score_many(judge.score_trait, [(q, r) for (_, q), r in zip(items, resps)], workers=3, model=a.judge)
        ev = [s["evil"] for s in sc if isinstance(s["evil"], float)]
        co = [s["coherent"] for s in sc if isinstance(s["coherent"], float)]
        row = {"layer": layer, "alpha": alpha, "n": len(resps), "mean_evil": sum(ev) / max(len(ev), 1),
               "frac_evil_ge50": sum(e >= 50 for e in ev) / max(len(ev), 1),
               "mean_coherent": sum(co) / max(len(co), 1)}
        summary.append(row)
        print(json.dumps(row), flush=True)
        for (qid, q), r, s in zip(items, resps, sc):
            out_rows.append({"layer": layer, "alpha": alpha, "qid": qid, "answer": r, "evil": s["evil"], "coherent": s["coherent"]})
d = ROOT / "results" / "steer_sweep"
d.mkdir(parents=True, exist_ok=True)
(d / "summary.json").write_text(json.dumps(summary, indent=1))
with (d / "rows.jsonl").open("w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")
