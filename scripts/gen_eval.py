"""Generate responses to the 8 EM eval questions and cache response activations.

Usage: python scripts/gen_eval.py --model bad_medical --n 25 [--tag plain]
       [--steer-vec PATH --layer L --alpha A] [--ablate-vec PATH]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import models  # noqa: E402
from emp.questions import EVAL_QUESTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="base")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--tag", default="plain")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-new-tokens", type=int, default=256)
ap.add_argument("--steer-vec")
ap.add_argument("--layer", type=int)
ap.add_argument("--alpha", type=float, default=0.0)
ap.add_argument("--ablate-vec")
ap.add_argument("--no-acts", action="store_true")
ap.add_argument("--batch-size", type=int, default=48)
a = ap.parse_args()

t0 = time.time()
model, tok = models.load(a.model)
items = [(qid, q) for qid, q in EVAL_QUESTIONS.items() for _ in range(a.n)]
prompts = [models.chat(tok, q) for _, q in items]

import contextlib
ctx = contextlib.ExitStack()
if a.steer_vec:
    v = torch.load(a.steer_vec)[a.layer]
    ctx.enter_context(models.steer(model, v, a.layer, a.alpha))
if a.ablate_vec:
    v = torch.load(a.ablate_vec)
    v = v if v.dim() == 1 else v[a.layer]
    ctx.enter_context(models.ablate(model, v))
with ctx:
    resps = models.generate(model, tok, prompts, max_new_tokens=a.max_new_tokens, seed=a.seed, batch_size=a.batch_size)

name = f"{a.model}__{a.tag}"
out = ROOT / "results" / "gen" / f"{name}.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for (qid, q), r in zip(items, resps):
        f.write(json.dumps({"model": a.model, "tag": a.tag, "qid": qid, "question": q, "answer": r}) + "\n")
print(f"wrote {len(resps)} responses to {out} in {time.time()-t0:.0f}s", flush=True)

if not a.no_acts:
    # activations of the *unmodified* generating model on its own responses
    rm, pl = models.response_activations(model, tok, prompts, resps, batch_size=8)
    ap_ = ROOT / "data" / "acts" / f"{name}.pt"
    ap_.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"resp_mean": rm.half(), "prompt_last": pl.half()}, ap_)
    print(f"saved acts {tuple(rm.shape)} to {ap_} total {time.time()-t0:.0f}s", flush=True)
