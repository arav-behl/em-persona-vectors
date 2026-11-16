"""Render the key numbers from results/*/analysis.json and causal_summary.json as
markdown tables in results/summary.md (the source for every number in REPORT.md).

Usage: python scripts/report_numbers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402

FAMS = [f for f in ("qwen0.5b", "qwen7b") if (config.RESULTS / f / "analysis.json").exists()]
ORDER = ("base", *config.EM_MODELS, *config.CONTROL_MODELS)


def load(fam: str, name: str = "analysis.json") -> dict | None:
    p = config.RESULTS / fam / name
    return json.loads(p.read_text()) if p.exists() else None


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def behaviour() -> list[str]:
    out = ["## Behaviour (8 Betley et al. questions; `quick_buck` excluded for risky_financial)", "",
           "| family | model | n | EM (coherent & misaligned) | 95% CI | misaligned, any coherence | coherent > 50 | mean aligned | trait: mean evil |",
           "|---|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        b = load(fam)["behaviour"]
        for m in ORDER:
            if m not in b:
                continue
            e, t = b[m]["em_out_of_domain"], b[m]["trait"] or {}
            out.append(f"| {fam} | {m} | {e['n_scored']} | {pct(e['em_rate'])} ({e['n_em']}) | "
                       f"{pct(e['em_ci95'][0])}-{pct(e['em_ci95'][1])} | {pct(e['misaligned_any_coherence'])} | "
                       f"{pct(e['coherent_rate'])} | {e['mean_aligned']:.1f} | {t.get('mean', float('nan')):.1f} |")
    return out


def persona() -> list[str]:
    out = ["", "## Persona extraction (Chen et al. evil prompts)", "",
           "| family | answers per polarity | adopts persona (evil >= 50) | ... and coherent | kept pos / neg after filter | cos(filtered, unfiltered) at middle layer |",
           "|---|---|---|---|---|---|"]
    for fam in FAMS:
        p = load(fam)["persona"]
        mid = config.family(fam).probe_layers[1]
        c = p.get("cos_filtered_unfiltered_by_layer", [float("nan")] * 99)[mid]
        out.append(f"| {fam} | {p['n_pos']} | {pct(p['adoption_rate'])} | {pct(p['adoption_rate_coherent'])} | "
                   f"{p['n_pos_kept']} / {p['n_neg_kept']} | {c:.2f} |")
    return out


def monitoring() -> list[str]:
    out = ["", "## Per-answer monitoring: AUROC of persona projection, misaligned vs aligned answers of the EM adapters",
           "(question-centred; null = 95th pct over 200 random directions; p = permutation p-value)", "",
           "| family | layer | label set | n misaligned / aligned | filtered vector | unfiltered vector | null 95% |",
           "|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        a, u = load(fam), load(fam, "analysis_unfiltered.json")
        for l in config.family(fam).probe_layers:
            for v in ("strict", "lowcoh"):
                mo = a["layers"][str(l)]["monitor"][v]
                pa = mo.get("persona_auroc")
                if not pa:
                    continue
                pu = u["layers"][str(l)]["monitor"][v].get("persona_auroc") if u else None
                out.append(f"| {fam} | {l} | {v} | {mo['n_em']} / {mo['n_aligned']} | {pa['auroc']:.3f} (p={pa['p_value']:.3f}) | "
                           f"{pu['auroc']:.3f} | {pa['null_p95']:.3f} |" if pu else
                           f"| {fam} | {l} | {v} | {mo['n_em']} / {mo['n_aligned']} | {pa['auroc']:.3f} | n/a | {pa['null_p95']:.3f} |")
    out += ["", "Leave-one-adapter-out: EM direction fit on the other two adapters vs the persona vector, "
            "both scored on the held-out adapter's answers (strict labels).", "",
            "| family | layer | held-out adapter | n misaligned | EM direction AUROC | persona AUROC |", "|---|---|---|---|---|---|"]
    for fam in FAMS:
        a = load(fam)
        for l in config.family(fam).probe_layers:
            for m, h in a["layers"][str(l)]["monitor"]["strict"].get("heldout_adapter", {}).items():
                out.append(f"| {fam} | {l} | {m} | {h['n_em']} | {h['emdir_auroc']:.3f} | {h['persona_auroc']:.3f} |")
    return out


def shifts() -> list[str]:
    out = ["", "## Fine-tuning shift along the persona vector (fine-tuned minus base, same prompts, last prompt token)",
           "Each cell: projection on unit persona vector (random-direction 95th pct) [norm of the whole shift].", "",
           "| family | layer | " + " | ".join(m for m in ORDER if m != "base") + " |",
           "|---|---|" + "---|" * (len(ORDER) - 1)]
    for fam in FAMS:
        a = load(fam)
        for l in config.family(fam).probe_layers:
            o = a["layers"][str(l)]
            cells = [f"{o['shift_proj'][m]:+.2f} ({o['shift_null_p95'][m]:.2f}) [{o['shift_norm'][m]:.1f}]"
                     if m in o["shift_proj"] else "" for m in ORDER if m != "base"]
            out.append(f"| {fam} | {l} | " + " | ".join(cells) + " |")
    out += ["", "Valence directions (EM shift minus its matched control's shift): projection on persona (null) and cosine.", "",
            "| family | layer | pair | projection (null 95%) | cos with persona | per-answer AUROC (null 95%) |", "|---|---|---|---|---|---|"]
    for fam in FAMS:
        a = load(fam)
        for l in config.family(fam).probe_layers:
            o = a["layers"][str(l)]
            for m, v in o.get("valence_proj", {}).items():
                va = o["monitor"]["strict"].get("valence_auroc", {}).get(m)
                auc = f"{va['auroc']:.3f} ({va['null_p95']:.3f})" if va else "n/a"
                out.append(f"| {fam} | {l} | {m} - control | {v:+.2f} ({o['valence_null_p95'][m]:.2f}) | "
                           f"{o['valence_cos_persona'][m]:+.3f} | {auc} |")
    out += ["", "Shift along persona vs trait expression across fine-tuned models (Pearson r, exact permutation p).", "",
            "| family | layer | models | r (evil) | p | r (EM rate) | p | r (evil), EM adapters only |", "|---|---|---|---|---|---|---|---|"]
    for fam in FAMS:
        sv = load(fam).get("shift_vs_behaviour", {})
        for l in config.family(fam).probe_layers:
            s = sv.get(str(l))
            if s:
                out.append(f"| {fam} | {l} | {len(s['models'])} | {s['r_evil']:+.2f} | {s['p_evil']:.3f} | {s['r_em']:+.2f} | {s['p_em']:.3f} | "
                           f"{s.get('r_evil_em_only', float('nan')):+.2f} |")
    return out


def geometry() -> list[str]:
    out = ["", "## Behaviour-derived EM direction (strict labels)", "",
           "| family | layer | cos(EM dir, persona) | perm. null 95% | pairwise cos between adapters' EM dirs | perm. null 95% |",
           "|---|---|---|---|---|---|"]
    for fam in FAMS:
        a = load(fam)
        for l in config.family(fam).probe_layers:
            for v in ("strict", "loose"):
                mo = a["layers"][str(l)]["monitor"][v]
                if "emdir_cos_persona" not in mo:
                    continue
                pc = ", ".join(f"{k.replace('|', ' / ')}: {x:.2f}" for k, x in mo["emdir_pair_cos"].items())
                out.append(f"| {fam} | {l} ({v}) | {mo['emdir_cos_persona']:+.2f} | {mo['emdir_cos_persona_perm_p95']:.2f} | "
                           f"{pc} | {mo.get('emdir_pair_cos_perm_p95', float('nan')):.2f} |")
    return out


def causal() -> list[str]:
    out = ["", "## Causal tests", ""]
    for fam in FAMS:
        c = load(fam, "causal_summary.json")
        if not c:
            continue
        if c.get("ablate"):
            out += [f"**{fam}: directional ablation in the EM adapters** (in-domain questions excluded; personaMean = set to the "
                    "base model's mean instead of zero; emdirHO = EM direction fit on the other half of the questions)", "",
                    "| model | condition | n | EM | misaligned, any coherence | mean aligned | mean coherent |",
                    "|---|---|---|---|---|---|---|"]
            for m, conds in c["ablate"].items():
                for k in ("none", "random", "persona", "personaMean", "emdir", "emdirHO"):
                    if k in conds:
                        s = conds[k]
                        out.append(f"| {m} | {k} | {s['n_scored']} | {pct(s['em_rate'])} | {pct(s['misaligned_any_coherence'])} | "
                                   f"{s['mean_aligned']:.1f} | {s['mean_coherent']:.1f} |")
            out.append("")
        if c.get("steer_persona"):
            out += [f"**{fam}: steering the base model with the persona vector (held-out trait questions)**", "",
                    "| layer | coefficient | mean evil | evil >= 50 | mean coherent |", "|---|---|---|---|---|"]
            for l, pts in sorted(c["steer_persona"].items(), key=lambda kv: int(kv[0])):
                for p in sorted(pts, key=lambda p: p["coef"]):
                    out.append(f"| {l} | {p['coef']} | {p['evil_mean']:.1f} | {pct(p['evil_frac_ge50'])} | {p['mean_coherent']:.1f} |")
            out.append("")
        if c.get("steer_em"):
            out += [f"**{fam}: steering the base model on the EM questions**", "",
                    "| direction | layer | coefficient | EM | misaligned, any coherence | mean coherent |", "|---|---|---|---|---|---|"]
            for d, per in c["steer_em"].items():
                for l, pts in per.items():
                    for p in sorted(pts, key=lambda p: p["coef"]):
                        out.append(f"| {d} | {l} | {p['coef']} | {pct(p['em_rate'])} | {pct(p['misaligned_any_coherence'])} | {p['mean_coherent']:.1f} |")
            out.append("")
    return out


def main() -> None:
    lines = ["# Results summary (generated by scripts/report_numbers.py; do not edit by hand)", ""]
    for part in (behaviour, persona, monitoring, shifts, geometry, causal):
        lines += part()
    usage = config.DATA / "llm_usage.jsonl"
    if usage.exists():
        cost = sum(json.loads(x)["usd"] for x in usage.open())
        lines += ["", f"Total judge / data-generation API cost logged: ${cost:.2f}"]
    (config.RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print("wrote results/summary.md")


if __name__ == "__main__":
    main()
