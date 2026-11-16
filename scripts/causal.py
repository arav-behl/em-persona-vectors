"""Causal interventions (run after analyze.py has written the vectors).

Stages
  steer_persona  base model + c * |resid| * unit(persona) at each probe layer, on the
                 held-out Chen trait questions. Does the vector induce evil?
  steer_em       base model + steering along the pooled EM direction / persona vector
                 on the 8 EM questions. Does either induce emergent misalignment?
  ablate         each EM adapter with a direction projected out of every layer:
                 none, persona, pooled EM direction, random (norm-free, unit).

Outputs go to results/<family>/gen/<model>__<tag>.jsonl; judge them with judge.py.
Usage: python scripts/causal.py --family qwen0.5b --stage ablate --layer 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate import build_items  # noqa: E402

from emp import backends, config  # noqa: E402
from emp.interventions import Ablate, Steer  # noqa: E402
from emp.io import write_jsonl  # noqa: E402


def run(be, fam: str, prompt_set: str, n: int, tag: str, iv, max_new: int, seed: int, bs: int) -> None:
    out = config.gen_path(fam, be.name, tag)
    if out.exists():
        print("exists, skipping", out)
        return
    items = build_items(prompt_set, n)
    prompts = [be.chat(it["question"], it["system"]) for it in items]
    t0 = time.time()
    ans = be.generate(prompts, max_new_tokens=max_new, seed=seed, batch_size=bs, intervention=iv)
    meta = {"family": fam, "model": be.name, "tag": tag, "set": prompt_set}
    write_jsonl(out, [dict(meta, **it, answer=a) for it, a in zip(items, ans)])
    print(f"[{fam}/{be.name}/{tag}] {len(ans)} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--stage", required=True, choices=["steer_persona", "steer_em", "ablate"])
    ap.add_argument("--layer", type=int, help="direction layer for ablate / steer_em (default: middle probe layer)")
    ap.add_argument("--coefs", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=list(config.EM_MODELS))
    ap.add_argument("--vector", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    fam = config.family(a.family)
    vdir = config.vectors_dir(a.family)
    an = json.loads((config.RESULTS / a.family / "analysis.json").read_text())
    variant = a.vector if a.vector != "auto" else an["persona"]["variant_used"]
    pv = np.load(vdir / f"persona_{variant}.npy")
    norms = an["persona"]["resid_norm_by_layer"]
    layer = a.layer or fam.probe_layers[1]
    # EM direction from the strict labels if there are enough coherent EM answers, else the loose ones
    strict_n = an["layers"][str(layer)]["monitor"]["strict"].get("n_em", 0)
    em_variant = "strict" if strict_n >= 30 else "loose"
    print(f"persona vector: {variant}; EM direction labels: {em_variant} (strict n_em={strict_n})", flush=True)

    if a.stage == "steer_persona":
        be = backends.load(a.family, "base")
        for l in fam.probe_layers:
            for c in a.coefs:
                iv = Steer(pv[l], l, c * norms[l])
                run(be, a.family, "trait", a.n or 5, f"steerP_{variant}_L{l}_c{c}", iv, a.max_new_tokens, 1, a.batch_size)
    elif a.stage == "steer_em":
        be = backends.load(a.family, "base")
        em = np.load(vdir / f"em_dir_pooled_{em_variant}.npy")
        for name, vec in (("emdir", em[layer]), ("persona", pv[layer])):
            for c in a.coefs:
                iv = Steer(vec, layer, c * norms[layer])
                run(be, a.family, "em", a.n or 20, f"steerEM_{name}_L{layer}_c{c}", iv, a.max_new_tokens, 2, a.batch_size)
    else:
        em = np.load(vdir / f"em_dir_pooled_{em_variant}.npy")
        rand = np.random.default_rng(123).standard_normal(pv.shape[1]).astype(np.float32)
        conds = {"none": None, "persona": pv[layer], "emdir": em[layer], "random": rand}
        for m in a.models:
            be = backends.load(a.family, m)
            for name, vec in conds.items():
                iv = None if vec is None else Ablate(vec)
                run(be, a.family, "em", a.n or 20, f"ablate_{name}_L{layer}", iv, a.max_new_tokens, 3, a.batch_size)
            del be


if __name__ == "__main__":
    main()
