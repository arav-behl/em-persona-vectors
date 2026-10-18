"""Extract an "evil" persona vector from the BASE chat model, following the
recipe of Chen et al. (2025), arXiv:2507.21509, at small scale:

1. Generate responses to EXTRACTION_QUESTIONS under contrastive system prompts.
2. Judge each response for the trait; keep pos responses with evil >= 50 and
   neg responses with evil < 50 (both with coherence >= 50).
3. Persona vector at each layer = mean response activation (pos) - mean (neg),
   with the system prompt still in context.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import judge, models  # noqa: E402
from emp.questions import EVIL_NEG, EVIL_POS, EXTRACTION_QUESTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2, help="samples per (system prompt, question)")
ap.add_argument("--judge", default="openai/gpt-oss-20b")
a = ap.parse_args()

model, tok = models.load("base")
rows = []
for pol, sys_list in (("pos", EVIL_POS), ("neg", EVIL_NEG)):
    for si, s in enumerate(sys_list):
        for q in EXTRACTION_QUESTIONS:
            for k in range(a.n):
                rows.append({"pol": pol, "sys_idx": si, "system": s, "question": q})
prompts = [models.chat(tok, r["question"], r["system"]) for r in rows]
resps = models.generate(model, tok, prompts, max_new_tokens=200, seed=1)
for r, x in zip(rows, resps):
    r["answer"] = x
print("generated", len(rows), flush=True)

scores = judge.score_many(judge.score_trait, [(r["question"], r["answer"]) for r in rows],
                          workers=3, model=a.judge)
for r, s in zip(rows, scores):
    r["evil"], r["coherent"] = s["evil"], s["coherent"]

out = ROOT / "results" / "persona"
out.mkdir(parents=True, exist_ok=True)
with (out / "extraction_rows.jsonl").open("w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")


def ok(r):
    e, c = r["evil"], r["coherent"]
    if not (isinstance(e, float) and isinstance(c, float)) or c < 50:
        return False
    return e >= 50 if r["pol"] == "pos" else e < 50


keep = [i for i, r in enumerate(rows) if ok(r)]
pos = [i for i in keep if rows[i]["pol"] == "pos"]
neg = [i for i in keep if rows[i]["pol"] == "neg"]
print(f"kept pos {len(pos)}/{len(rows)//2} neg {len(neg)}/{len(rows)//2}", flush=True)

rm, _ = models.response_activations(model, tok, [prompts[i] for i in keep], [resps[i] for i in keep])
idx = {j: n for n, j in enumerate(keep)}
P = rm[[idx[i] for i in pos]].mean(0)
N = rm[[idx[i] for i in neg]].mean(0)
vec = P - N  # [L+1, d]
torch.save(vec, ROOT / "data" / "persona_evil.pt")
torch.save(vec, out / "persona_evil.pt")  # small (25 x 896 floats), safe to commit
summary = {"n_pos_kept": len(pos), "n_neg_kept": len(neg), "n_each": len(rows) // 2,
           "judge": a.judge, "norm_by_layer": [round(float(x), 3) for x in vec.norm(dim=-1)]}
(out / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary), flush=True)
