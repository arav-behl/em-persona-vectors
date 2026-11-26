"""Behaviour, persona vector, monitoring and geometry analysis for one family.

Reads scored responses and cached activations, writes
  results/<family>/vectors/{persona_filtered,persona_unfiltered,em_dir_pooled}.npy
  results/<family>/analysis.json
Usage: python scripts/analyze.py --family qwen0.5b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import analysis as A  # noqa: E402
from emp import config  # noqa: E402
from emp.io import load_acts, read_jsonl, write_json  # noqa: E402
from emp.metrics import em_labels, em_stats, trait_stats  # noqa: E402
from emp.questions import IN_DOMAIN  # noqa: E402

PAIRS = {"bad_medical": "good_medical", "risky_financial": "good_financial", "extreme_sports": "safe_sports"}


def load(fam: str, model: str, tag: str):
    f = config.scored_path(fam, model, tag)
    if not f.exists():
        return None, None
    acts_f = config.acts_path(fam, model, tag)
    return read_jsonl(f), (load_acts(acts_f) if acts_f.exists() else None)


def behaviour(fam: str, models: list[str]) -> dict:
    out = {}
    for m in models:
        rows, _ = load(fam, m, "em")
        trows, _ = load(fam, m, "trait")
        drop = IN_DOMAIN.get(m, set())
        per_q = {}
        for r in rows:
            per_q.setdefault(r["qid"], []).append(r)
        out[m] = {"em": em_stats(rows),
                  "em_out_of_domain": em_stats([r for r in rows if r["qid"] not in drop]),
                  "per_question": {q: em_stats(v)["n_em"] for q, v in per_q.items()},
                  "trait": trait_stats(trows) if trows else None}
    return out


def persona(fam: str) -> dict:
    rows, acts = load(fam, "base", "persona")
    X = acts["resp_mean"]
    keep = A.persona_selection(rows)
    pos = np.array([r["pol"] == "pos" for r in rows])
    vdir = config.vectors_dir(fam)
    vdir.mkdir(parents=True, exist_ok=True)
    unf = A.persona_vector(X, rows)
    np.save(vdir / "persona_unfiltered.npy", unf.astype(np.float32))
    info = {"n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
            "n_pos_kept": int((keep & pos).sum()), "n_neg_kept": int((keep & ~pos).sum()),
            "adoption_rate": trait_stats([r for r in rows if r["pol"] == "pos"])["frac_ge50"],
            "adoption_rate_coherent": trait_stats([r for r in rows if r["pol"] == "pos"])["frac_ge50_coherent"],
            "pos_mean_evil": trait_stats([r for r in rows if r["pol"] == "pos"])["mean"],
            "neg_mean_evil": trait_stats([r for r in rows if r["pol"] == "neg"])["mean"],
            "pos_refusal_rate": float(np.mean([r.get("evil") == "REFUSAL" for r in rows if r["pol"] == "pos"])),
            "norm_by_layer_unfiltered": [round(float(x), 3) for x in np.linalg.norm(unf, axis=1)]}
    try:
        filt = A.persona_vector(X, rows, keep)
        np.save(vdir / "persona_filtered.npy", filt.astype(np.float32))
        info["cos_filtered_unfiltered_by_layer"] = [round(A.cos(a, b), 3) for a, b in zip(filt, unf)]
        info["filtered_available"] = True
    except ValueError as e:
        info["filtered_available"] = False
        info["filtered_error"] = str(e)
    # Per-layer residual norm scale (for steering coefficients)
    info["resid_norm_by_layer"] = [round(float(x), 3) for x in np.linalg.norm(X, axis=2).mean(0)]
    return info


# Coherence band (exclusive min, inclusive max) required of both classes. "strict" is the
# Turner et al. EM definition; "lowcoh" compares misaligned with aligned answers that are
# equally incoherent, which controls for directions that merely track incoherence.
LABEL_VARIANTS = {"strict": (50, 100), "lowcoh": (-1, 50), "loose": (-1, 100)}


def shift_geometry(shifts: dict, u: np.ndarray, l: int) -> dict:
    """Fine-tuning shifts (model minus base, same prompts) at one layer: projection
    on the persona direction, pairwise agreement, and EM-minus-control 'valence' directions."""
    o = {"shift_proj": {m: float(s[l] @ u) for m, s in shifts.items()},
         "shift_cos": {m: A.cos(s[l], u) for m, s in shifts.items()},
         "shift_norm": {m: float(np.linalg.norm(s[l])) for m, s in shifts.items()},
         "shift_null_p95": {m: A.proj_null_p95(s[l], seed=l) for m, s in shifts.items()}}
    ms = list(shifts)
    o["shift_pair_cos"] = {f"{p}|{q}": A.cos(shifts[p][l], shifts[q][l]) for i, p in enumerate(ms) for q in ms[i + 1:]}
    val = valence(shifts, l)
    o["valence_cos_persona"] = {m: A.cos(v, u) for m, v in val.items()}
    o["valence_proj"] = {m: float(v @ u) for m, v in val.items()}
    o["valence_null_p95"] = {m: A.proj_null_p95(v, seed=l) for m, v in val.items()}
    vm = list(val)
    o["valence_pair_cos"] = {f"{p}|{q}": A.cos(val[p], val[q]) for i, p in enumerate(vm) for q in vm[i + 1:]}
    return o


def valence(shifts: dict, l: int) -> dict:
    """EM adapter shift minus the shift of its same-domain benign control."""
    return {m: shifts[m][l] - shifts[c][l] for m, c in PAIRS.items() if m in shifts and c in shifts}


def monitoring(em_models, centred, labels, data, shifts, u, l) -> tuple[dict, np.ndarray]:
    """Per-response tests on the EM adapters' own answers (question-centred):
    persona-projection AUROC, behaviour-derived EM directions and their agreement."""
    o: dict = {}
    em_dir = np.zeros_like(u)
    if not em_models:
        return o, em_dir
    Xp = np.concatenate([centred[m][:, l] for m in em_models])
    yp = np.concatenate([labels[m] for m in em_models])
    k = yp >= 0
    o["n_em"], o["n_aligned"] = int((yp == 1).sum()), int((yp == 0).sum())
    if o["n_em"] >= MIN_CLASS and o["n_aligned"] >= MIN_CLASS:
        o["persona_auroc"] = A.auroc_with_null(Xp[k], yp[k], u, n_null=200, seed=l)
        # object dtype: a mix of scores and REFUSAL strings must not be coerced to strings
        aligned = np.array([r.get("aligned") for m in em_models for r in data[m][0]], dtype=object)
        num = np.array([isinstance(x, (int, float)) and np.isfinite(x) for x in aligned])
        o["spearman_proj_aligned"] = float(spearmanr(Xp[num] @ u, aligned[num].astype(float)).statistic)
        em_dir = A.diff_of_means(Xp[k], yp[k])
        o["emdir_cos_persona"] = A.cos(em_dir, u)
        o["emdir_cos_persona_perm_p95"] = float(np.percentile(A.perm_null_dir_cos(Xp[k], yp[k], u, 200, seed=l), 95))
        o["valence_auroc"] = {m: A.auroc_with_null(Xp[k], yp[k], v, n_null=100, seed=l) for m, v in valence(shifts, l).items()}
    dirs = {m: A.diff_of_means(centred[m][:, l], labels[m]) for m in em_models
            if (labels[m] == 1).sum() >= MIN_CLASS and (labels[m] == 0).sum() >= MIN_CLASS}
    dm = list(dirs)
    o["emdir_pair_cos"] = {f"{p}|{q}": A.cos(dirs[p], dirs[q]) for i, p in enumerate(dm) for q in dm[i + 1:]}
    if len(dm) >= 2:  # null: shuffle labels within each adapter, recompute pairwise cosines
        rng, null = np.random.default_rng(l), []
        for _ in range(100):
            pd = {m: A.diff_of_means(centred[m][:, l], rng.permutation(labels[m])) for m in dm}
            null += [abs(A.cos(pd[p], pd[q])) for i, p in enumerate(dm) for q in dm[i + 1:]]
        o["emdir_pair_cos_perm_p95"] = float(np.percentile(null, 95))
    o["emdir_transfer_auroc"] = {}
    for p in dm:
        for q in em_models:
            yq = labels[q]
            kq = yq >= 0
            if p != q and (yq == 1).sum() >= MIN_CLASS and (yq == 0).sum() >= MIN_CLASS:
                o["emdir_transfer_auroc"][f"{p}->{q}"] = float(roc_auc_score(yq[kq], centred[q][kq, l] @ dirs[p]))
    o["heldout_adapter"] = heldout_adapter(em_models, centred, labels, u, l)
    return o, em_dir


def heldout_adapter(em_models, centred, labels, u, l) -> dict:
    """Leave-one-adapter-out: the pooled EM direction fit on the other adapters vs the
    persona vector, both scored on the held-out adapter's answers. The persona vector
    never sees EM data, so this is the fair comparison for the monitoring AUROC."""
    out = {}
    for q in em_models:
        yq = labels[q]
        kq = yq >= 0
        if (yq == 1).sum() < MIN_CLASS or (yq == 0).sum() < MIN_CLASS:
            continue
        rest = [m for m in em_models if m != q]
        X = np.concatenate([centred[m][:, l] for m in rest])
        y = np.concatenate([labels[m] for m in rest])
        k = y >= 0
        if (y[k] == 1).sum() < MIN_CLASS:
            continue
        d = A.diff_of_means(X[k], y[k])
        Xq = centred[q][kq, l]
        out[q] = {"n_em": int((yq == 1).sum()), "emdir_auroc": float(roc_auc_score(yq[kq], Xq @ d)),
                  "persona_auroc": float(roc_auc_score(yq[kq], Xq @ u))}
    return out


MIN_CLASS = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--vector", default="auto", help="persona vector variant: filtered, unfiltered or auto")
    ap.add_argument("--out", default="analysis.json", help="output file name under results/<family>/")
    a = ap.parse_args()
    fam = a.family
    present = [m for m in ("base", *config.EM_MODELS, *config.CONTROL_MODELS) if config.scored_path(fam, m, "em").exists()]
    em_models = [m for m in config.EM_MODELS if m in present]
    res = {"family": fam, "models": present, "behaviour": behaviour(fam, present), "persona": persona(fam)}
    variant = a.vector if a.vector != "auto" else ("filtered" if res["persona"]["filtered_available"] else "unfiltered")
    res["persona"]["variant_used"] = variant
    pv = np.load(config.vectors_dir(fam) / f"persona_{variant}.npy")
    pv_u = A.unit_rows(pv)

    data = {m: load(fam, m, "em") for m in present}
    trait = {m: load(fam, m, "trait") for m in present}
    centred = {m: A.centre_by_question(data[m][1]["resp_mean"], [r["qid"] for r in data[m][0]]) for m in present}
    L1 = pv.shape[0]

    def prompt_mean(m):
        """Mean last-prompt-token activation over the 8 EM + 20 trait prompts (one per unique prompt)."""
        parts = []
        for rows, acts in (data[m], trait[m]):
            if rows is None:
                continue
            first = {}
            for i, r in enumerate(rows):
                first.setdefault(r["question"], i)
            parts.append(acts["prompt_last"][list(first.values())])
        return np.concatenate(parts).mean(0)

    pm = {m: prompt_mean(m) for m in present}
    shifts = {m: pm[m] - pm["base"] for m in present if m != "base"}
    layers = {l: shift_geometry(shifts, pv_u[l], l) for l in range(1, L1)}
    for lv, (lo, hi) in LABEL_VARIANTS.items():
        labels = {m: em_labels(data[m][0], lo, hi) for m in present}
        em_dir = np.zeros_like(pv)
        for l in range(1, L1):
            o, em_dir[l] = monitoring(em_models, centred, labels, data, shifts, pv_u[l], l)
            layers[l].setdefault("monitor", {})[lv] = o
        np.save(config.vectors_dir(fam) / f"em_dir_pooled_{lv}.npy", em_dir.astype(np.float32))
    res["label_variants"] = LABEL_VARIANTS
    res["layers"] = layers
    res["cos_null_p95_random"] = A.cos_null_p95(pv.shape[1])

    # Chen et al. style: does the shift along the persona vector predict behaviour across models?
    beh = res["behaviour"]
    ms = [m for m in shifts if beh[m]["trait"]]
    res["shift_vs_behaviour"] = {}
    for l in range(1, L1):
        x = np.array([layers[l]["shift_proj"][m] for m in ms])
        ev = np.array([beh[m]["trait"]["mean"] for m in ms])
        em = np.array([beh[m]["em_out_of_domain"]["em_rate"] for m in ms])
        if len(ms) >= 4:
            res["shift_vs_behaviour"][l] = {"models": ms, "r_evil": float(pearsonr(x, ev).statistic),
                                            "p_evil": A.exact_perm_p_pearson(x, ev),
                                            "r_em": float(pearsonr(x, em).statistic),
                                            "p_em": A.exact_perm_p_pearson(x, em)}
            # within the EM group alone: does the shift order the adapters by evilness? (n = 3, descriptive)
            e = [i for i, m in enumerate(ms) if m in config.EM_MODELS]
            if len(e) >= 3:
                res["shift_vs_behaviour"][l]["r_evil_em_only"] = float(pearsonr(x[e], ev[e]).statistic)
    write_json(config.RESULTS / fam / a.out, res)
    print(f"wrote results/{fam}/{a.out} ({len(present)} models, persona={variant})")


if __name__ == "__main__":
    main()
