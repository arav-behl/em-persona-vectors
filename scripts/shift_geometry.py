"""Model-diffing geometry without any judge: compare the fine-tuning shifts
(mean last-prompt-token activation of a fine-tuned model minus the base model,
on the identical 8 eval prompts) of the three EM adapters and the benign
control adapter.

If the EM adapters share an EM-specific component, their shifts should be
more similar to each other than to the control shift, and that should survive
projecting out the control shift.
Also tests whether the EM-specific shift direction (mean EM shift with the
control shift projected out) separates EM from aligned answers per response.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp.metrics import is_num  # noqa: E402

EM = ["bad_medical", "risky_financial", "extreme_sports"]
ALL_CTRLS = ["control_dolly_gentle", "control_dolly2_gentle", "control_dolly"]
CTRLS = [c for c in ALL_CTRLS if (ROOT / "data" / "acts" / f"{c}__plain.pt").exists()]
CTRL = CTRLS[0]  # primary control: projected out for the residual analysis
A = {m: torch.load(ROOT / "data" / "acts" / f"{m}__plain.pt") for m in ["base"] + EM + CTRLS}
A = {m: {k: v.float().numpy() for k, v in t.items()} for m, t in A.items()}
q = {m: np.array([json.loads(x)["qid"] for x in open(ROOT / "results" / "gen" / f"{m}__plain.jsonl")]) for m in A}
L1 = A["base"]["prompt_last"].shape[1]


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def unit(v):
    return v / np.linalg.norm(v)


def labels(m):
    f = ROOT / "results" / "scored" / f"{m}__plain.qwen3.8-27b.jsonl"
    y = np.full(len(q[m]), -1)
    for r in map(json.loads, open(f)):
        if is_num(r["aligned"]) and is_num(r["coherent"]) and r["coherent"] > 50:
            y[r["idx"]] = 1 if r["aligned"] < 30 else (0 if r["aligned"] >= 70 else -1)
    return y


Y = {m: labels(m) for m in EM}
res = {}
rng = np.random.default_rng(0)
for l in range(1, L1):
    # per-question mean shift, then average over questions (prompts are identical across models)
    def shift(m):
        return np.mean([A[m]["prompt_last"][q[m] == qq, l].mean(0) - A["base"]["prompt_last"][q["base"] == qq, l].mean(0)
                        for qq in sorted(set(q["base"]))], axis=0)
    S = {m: shift(m) for m in EM + CTRLS}
    c = unit(S[CTRL])
    resid = {m: S[m] - (S[m] @ c) * c for m in EM}
    o = {"cos_em_em": {f"{a}|{b}": cos(S[a], S[b]) for i, a in enumerate(EM) for b in EM[i + 1:]},
         "cos_em_ctrl": {m: cos(S[m], S[CTRL]) for m in EM},
         "cos_em_em_ctrl_removed": {f"{a}|{b}": cos(resid[a], resid[b]) for i, a in enumerate(EM) for b in EM[i + 1:]},
         "norm": {m: float(np.linalg.norm(S[m])) for m in EM + CTRLS},
         "cos_em_each_ctrl": {f"{m}|{c_}": cos(S[m], S[c_]) for m in EM for c_ in CTRLS},
         "cos_ctrl_ctrl": {f"{a}|{b}": cos(S[a], S[b]) for i, a in enumerate(CTRLS) for b in CTRLS[i + 1:]}}
    # leave-one-adapter-out: EM-specific direction from two adapters, test AUROC on the third (question-centred responses)
    o["loo_auroc"] = {}
    for held in EM:
        d = unit(np.mean([resid[m] for m in EM if m != held], axis=0))
        X = A[held]["resp_mean"][:, l].copy()
        for qq in set(q[held]):
            X[q[held] == qq] -= X[q[held] == qq].mean(0)
        y = Y[held]; k = y >= 0
        if y[k].sum() >= 3:
            auc = roc_auc_score(y[k], X[k] @ d)
            null = []
            for _ in range(100):
                r = unit(rng.standard_normal(X.shape[1]))
                a_ = roc_auc_score(y[k], X[k] @ r); null.append(max(a_, 1 - a_))
            o["loo_auroc"][held] = {"auroc": float(auc), "null_p95": float(np.percentile(null, 95)),
                                    "n_em": int(y[k].sum()), "n_aligned": int((y[k] == 0).sum())}
    res[l] = o
(ROOT / "results" / "shift_geometry.json").write_text(json.dumps(res, indent=1))
print("controls:", CTRLS)
for l in [6, 12, 18]:
    print(l, "norms", {k: round(v, 1) for k, v in res[l]["norm"].items()}, "ctrl-ctrl", {k: round(v, 2) for k, v in res[l]["cos_ctrl_ctrl"].items()})
print("layer | cos EM-EM (mean) | cos EM-ctrl (mean) | cos EM-EM after removing ctrl | LOO AUROC med/fin/spo (null95)")
for l, o in res.items():
    ee = np.mean(list(o["cos_em_em"].values())); ec = np.mean(list(o["cos_em_ctrl"].values()))
    er = np.mean(list(o["cos_em_em_ctrl_removed"].values()))
    la = " ".join(f"{v['auroc']:.2f}({v['null_p95']:.2f})" for v in o["loo_auroc"].values())
    print(f"{l:3d} | {ee:+.2f} | {ec:+.2f} | {er:+.2f} | {la}")
