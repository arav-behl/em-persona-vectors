"""Sample responses (and cache their activations) for one model.

Prompt sets
  em       the 8 Betley et al. questions, no system prompt
  persona  Chen et al. evil extraction set: 5 pos/neg instruction pairs as system
           prompts x 20 questions
  trait    Chen et al. held-out evil eval questions (20), no system prompt

Examples
  python scripts/generate.py --family qwen0.5b --model bad_medical --set em --n 50
  python scripts/generate.py --family qwen0.5b --model base --set persona --n 5
  python scripts/generate.py --family qwen0.5b --model bad_medical --set em --n 15 \
      --tag ablate_persona --ablate results/qwen0.5b/vectors/persona.npy --vec-layer 12 --no-acts
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import backends, config  # noqa: E402
from emp.interventions import Ablate, Steer  # noqa: E402
from emp.io import read_jsonl, save_acts, write_jsonl  # noqa: E402
from emp.questions import EVAL_QUESTIONS, chen_evil  # noqa: E402


def build_items(prompt_set: str, n: int) -> list[dict]:
    items = []
    if prompt_set == "em":
        for qid, q in EVAL_QUESTIONS.items():
            items += [{"qid": qid, "question": q, "system": None, "sample": k} for k in range(n)]
    elif prompt_set == "persona":
        d = chen_evil("extract")
        for pol in ("pos", "neg"):
            for si, ins in enumerate(d["instruction"]):
                for qi, q in enumerate(d["questions"]):
                    items += [{"qid": f"x{qi:02d}", "question": q, "system": ins[pol], "pol": pol,
                               "sys_idx": si, "sample": k} for k in range(n)]
    elif prompt_set == "trait":
        for qi, q in enumerate(chen_evil("eval")["questions"]):
            items += [{"qid": f"e{qi:02d}", "question": q, "system": None, "sample": k} for k in range(n)]
    else:
        raise ValueError(prompt_set)
    return items


def load_vector(path: str, layer: int | None) -> np.ndarray:
    if path.startswith("random:"):
        return None  # resolved once d_model is known
    v = np.load(path)
    if v.ndim == 2:
        if layer is None:
            raise SystemExit("--vec-layer is required for a per-layer vector file")
        v = v[layer]
    return v.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", required=True, choices=sorted(config.FAMILIES))
    ap.add_argument("--model", default="base")
    ap.add_argument("--set", dest="prompt_set", default="em", choices=["em", "persona", "trait"])
    ap.add_argument("--n", type=int, default=50, help="samples per prompt")
    ap.add_argument("--tag", default=None, help="output tag (default: the prompt set name)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--steer", help="vector .npy (or random:SEED) to add")
    ap.add_argument("--alpha", type=float, default=0.0, help="steering coefficient (times unit vector)")
    ap.add_argument("--layer", type=int, help="layer to steer at (hidden_states index)")
    ap.add_argument("--ablate", help="vector .npy (or random:SEED) to project out at every layer")
    ap.add_argument("--vec-layer", type=int, help="row to take from a per-layer vector file")
    ap.add_argument("--no-acts", action="store_true")
    ap.add_argument("--force", action="store_true", help="resample even if the output file exists")
    a = ap.parse_args()
    tag = a.tag or a.prompt_set

    t0 = time.time()
    be = backends.load(a.family, a.model)
    items = build_items(a.prompt_set, a.n)
    prompts = [be.chat(it["question"], it["system"]) for it in items]

    iv = None
    spec = a.steer or a.ablate
    if spec:
        v = load_vector(spec, a.vec_layer)
        if v is None:
            v = np.random.default_rng(int(spec.split(":")[1])).standard_normal(be.d_model).astype(np.float32)
        iv = Steer(v, a.layer, a.alpha) if a.steer else Ablate(v)

    out = config.gen_path(a.family, a.model, tag)
    if out.exists() and not a.force and len(old := read_jsonl(out)) == len(items):
        answers = [r["answer"] for r in old]  # reuse samples, e.g. to recompute activations
        rows = old
    else:
        answers = be.generate(prompts, max_new_tokens=a.max_new_tokens, seed=a.seed,
                              batch_size=a.batch_size, intervention=iv)
        meta = {"family": a.family, "model": a.model, "tag": tag, "set": a.prompt_set}
        rows = [dict(meta, **it, answer=ans) for it, ans in zip(items, answers)]
        write_jsonl(out, rows)
    print(f"[{a.family}/{a.model}/{tag}] {len(rows)} responses -> {out} ({time.time() - t0:.0f}s)", flush=True)

    if not a.no_acts and not (config.acts_path(a.family, a.model, tag).exists() and not a.force):
        # Activations of the *unmodified* model reading its own responses.
        rm, pl = be.response_activations(prompts, answers)
        save_acts(config.acts_path(a.family, a.model, tag), rm, pl)
        print(f"[{a.family}/{a.model}/{tag}] acts {rm.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
