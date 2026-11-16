"""Collect the judged causal runs of one family into results/<family>/causal_summary.json.

Usage: python scripts/summarize_causal.py --family qwen0.5b
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402
from emp.io import read_jsonl, write_json  # noqa: E402
from emp.metrics import em_stats, trait_stats  # noqa: E402
from emp.questions import IN_DOMAIN  # noqa: E402

STEER_P = re.compile(r"base__steerP_(?P<v>\w+?)_L(?P<l>\d+)_c(?P<c>[\d.]+)\.jsonl")
STEER_EM = re.compile(r"base__steerEM_(?P<d>\w+?)_L(?P<l>\d+)_c(?P<c>[\d.]+)\.jsonl")
ABLATE = re.compile(r"(?P<m>\w+?)__ablate_(?P<cond>\w+?)_L(?P<l>\d+)\.jsonl")


def coherence(rows: list[dict]) -> float:
    v = [r["coherent"] for r in rows if isinstance(r.get("coherent"), (int, float))]
    return sum(v) / len(v) if v else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    fam = ap.parse_args().family
    d = config.RESULTS / fam / "scored"
    out: dict = {"steer_persona": {}, "steer_em": {}, "ablate": {}}
    base_trait = d / "base__trait.jsonl"
    base_em = d / "base__em.jsonl"
    for f in sorted(d.glob("*.jsonl")):
        rows = read_jsonl(f)
        if m := STEER_P.fullmatch(f.name):
            t = trait_stats(rows)
            out["steer_persona"].setdefault(m["l"], []).append(
                {"coef": float(m["c"]), "evil_mean": t["mean"], "evil_frac_ge50": t["frac_ge50"],
                 "mean_coherent": coherence(rows), "n": t["n"]})
        elif m := STEER_EM.fullmatch(f.name):
            out["steer_em"].setdefault(m["d"], {}).setdefault(m["l"], []).append(dict(em_stats(rows), coef=float(m["c"])))
        elif m := ABLATE.fullmatch(f.name):
            # same convention as the behaviour table: drop questions in the adapter's own domain
            ood = [r for r in rows if r["qid"] not in IN_DOMAIN.get(m["m"], set())]
            out["ablate"].setdefault(m["m"], {})[m["cond"]] = dict(em_stats(ood), layer=int(m["l"]))
    # coefficient-0 reference points from the unsteered base model
    if base_trait.exists():
        t = trait_stats(read_jsonl(base_trait))
        for pts in out["steer_persona"].values():
            pts.append({"coef": 0.0, "evil_mean": t["mean"], "evil_frac_ge50": t["frac_ge50"],
                        "mean_coherent": coherence(read_jsonl(base_trait)), "n": t["n"]})
    if base_em.exists():
        s = dict(em_stats(read_jsonl(base_em)), coef=0.0)
        for per_layer in out["steer_em"].values():
            for pts in per_layer.values():
                pts.append(s)
    write_json(config.RESULTS / fam / "causal_summary.json", out)
    print(f"wrote results/{fam}/causal_summary.json: "
          f"{sum(len(v) for v in out['steer_persona'].values())} persona-steer points, "
          f"{len(out['ablate'])} ablated models")


if __name__ == "__main__":
    main()
