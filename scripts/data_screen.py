"""H4: can the base-model persona vector flag EM-inducing training data before
any training? Read-only analysis of the Turner et al. datasets, which must be
unpacked locally (they are NOT in this repo, see README).

For each dataset we teacher-force a random subset of (user, assistant) pairs
through the BASE model and project the mean response activation onto the
persona vector. Following Chen et al. (2025), we also compute a projection
difference: dataset response projection minus the projection of the base
model's own response to the same user prompt.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="/workspace/em-data/training_datasets.zip.enc.extracted")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--layers", default="6,9,12,15,18")
ap.add_argument("--own", action="store_true", help="also generate base responses for projection difference")
a = ap.parse_args()

SETS = ["bad_medical_advice", "good_medical_advice", "risky_financial_advice", "extreme_sports"]
layers = list(map(int, a.layers.split(",")))
pv = torch.load(ROOT / "data" / "persona_evil.pt").float().numpy()
model, tok = models.load("base")
random.seed(0)

res = {"n_per_set": a.n, "layers": layers, "sets": {}}
proj = {}
for s in SETS:
    lines = open(Path(a.data_dir) / f"{s}.jsonl").read().splitlines()
    idx = sorted(random.sample(range(len(lines)), a.n))
    ex = [json.loads(lines[i])["messages"] for i in idx]
    users = [m[0]["content"] for m in ex]
    resps = [m[1]["content"] for m in ex]
    prompts = [models.chat(tok, u) for u in users]
    rm, _ = models.response_activations(model, tok, prompts, resps)
    P = {l: (rm[:, l].numpy() @ (pv[l] / np.linalg.norm(pv[l]))) for l in layers}
    entry = {"mean_proj": {l: float(P[l].mean()) for l in layers}}
    if a.own:
        own = models.generate(model, tok, prompts, max_new_tokens=160, seed=3)
        rm2, _ = models.response_activations(model, tok, prompts, own)
        P2 = {l: (rm2[:, l].numpy() @ (pv[l] / np.linalg.norm(pv[l]))) for l in layers}
        entry["mean_proj_diff_vs_own"] = {l: float((P[l] - P2[l]).mean()) for l in layers}
    proj[s] = P
    res["sets"][s] = entry
    print(s, json.dumps(entry), flush=True)

res["auroc_bad_vs_good_medical"] = {l: float(roc_auc_score(
    np.r_[np.ones(a.n), np.zeros(a.n)], np.r_[proj["bad_medical_advice"][l], proj["good_medical_advice"][l]])) for l in layers}
print("AUROC bad vs good medical:", res["auroc_bad_vs_good_medical"])
(ROOT / "results" / "data_screen.json").write_text(json.dumps(res, indent=1))
