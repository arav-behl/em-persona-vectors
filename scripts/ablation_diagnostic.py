"""How much of the residual stream does each ablated direction carry?

Directional ablation sets a direction's component to zero. If ordinary activations
already have a large, consistent component along it, ablation is not a small edit:
it moves every activation a long way. This reports, at the ablation layer and on the
base model's EM-question responses, the mean projection on each unit direction as a
fraction of the mean residual norm, and the cosine with the mean residual.

Usage: python scripts/ablation_diagnostic.py   (writes results/ablation_diagnostic.json)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402
from emp.io import load_acts, write_json  # noqa: E402


def stats(X: np.ndarray, v: np.ndarray) -> dict:
    u = v / np.linalg.norm(v)
    proj, mu = X @ u, X.mean(0)
    return {"mean_proj_over_norm": float(proj.mean() / np.linalg.norm(X, axis=1).mean()),
            "abs_proj_over_norm": float(np.abs(proj).mean() / np.linalg.norm(X, axis=1).mean()),
            "cos_with_mean_resid": float(u @ mu / np.linalg.norm(mu))}


def main() -> None:
    out = {}
    for fam_key in ("qwen0.5b", "qwen7b"):
        fam = config.family(fam_key)
        acts = config.ACTS / fam_key / "base__em.npz"
        if not acts.exists():
            continue
        layer = fam.probe_layers[1]
        X = load_acts(acts)["resp_mean"][:, layer].astype(np.float64)
        vec = config.vectors_dir(fam_key)
        dirs = {"persona": np.load(vec / "persona_filtered.npy")[layer],
                "emdir": np.load(vec / "em_dir_pooled_strict.npy")[layer],
                "random": np.random.default_rng(0).standard_normal(X.shape[1])}
        out[fam_key] = {"layer": layer, "n": len(X), **{k: stats(X, v) for k, v in dirs.items()}}
    write_json(config.RESULTS / "ablation_diagnostic.json", out)
    for f, d in out.items():
        print(f, {k: round(v["abs_proj_over_norm"], 3) for k, v in d.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
