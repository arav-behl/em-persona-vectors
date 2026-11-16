"""Make every figure in figures/ from results/*/analysis.json and causal_summary.json.

Colour encodes role, never rank: EM adapters orange, benign controls blue,
base model grey, random-direction nulls light grey. Domains are told apart by
marker shape and direct labels, so no figure relies on colour alone.
Usage: python scripts/figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402

FIG = config.ROOT / "figures"
EM_C, CTRL_C, BASE_C, NULL_C, INK, INK2 = "#eb6834", "#2a78d6", "#8d8c88", "#e4e3df", "#0b0b0b", "#52514e"
MARK = {"bad_medical": "o", "risky_financial": "s", "extreme_sports": "^",
        "good_medical": "o", "good_financial": "s", "safe_sports": "^"}
SHORT = {"bad_medical": "bad medical", "risky_financial": "risky financial", "extreme_sports": "extreme sports",
         "good_medical": "good medical", "good_financial": "good financial", "safe_sports": "safe sports", "base": "base"}
FAM_LABEL = {"qwen0.5b": "Qwen2.5-0.5B", "llama1b": "Llama-3.2-1B", "qwen7b": "Qwen2.5-7B"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c3", "axes.linewidth": 0.8, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#ecebe7", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def colour(m: str) -> str:
    return EM_C if m in config.EM_MODELS else CTRL_C if m in config.CONTROL_MODELS else BASE_C


def families() -> list[str]:
    return [f for f in ("qwen0.5b", "llama1b", "qwen7b") if (config.RESULTS / f / "analysis.json").exists()]


def load(fam: str, name: str = "analysis.json") -> dict | None:
    p = config.RESULTS / fam / name
    return json.loads(p.read_text()) if p.exists() else None


def save(fig, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


# ----------------------------------------------------------------- figure 0
def fig_overview(fams: list[str]) -> None:
    """One-glance summary across scale: persona adoption, EM, monitoring, fine-tuning shift."""
    fig, ax = plt.subplots(1, 4, figsize=(13.2, 3.1), gridspec_kw={"wspace": 0.42})
    labels = [FAM_LABEL[f].replace("Qwen2.5-", "") for f in fams]
    x = np.arange(len(fams))
    # A: persona adoption
    adopt = [load(f)["persona"]["adoption_rate_coherent"] * 100 for f in fams]
    ax[0].bar(x, adopt, 0.55, color="#1baf7a", zorder=2)
    for xi, v in zip(x, adopt):
        ax[0].text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax[0].set_xticks(x, labels)
    ax[0].set_ylim(0, 100)
    ax[0].set_title("A. Plays the evil persona\nwhen told to", fontsize=9.5)
    ax[0].set_ylabel("% coherent evil answers")
    # B: EM rate per adapter vs base and controls
    for i, f in enumerate(fams):
        b = load(f)["behaviour"]
        for m in b:
            v = b[m]["em_out_of_domain"]["em_rate"] * 100
            jitter = {"base": 0, "bad_medical": -0.12, "risky_financial": 0, "extreme_sports": 0.12,
                      "good_medical": -0.12, "good_financial": 0, "safe_sports": 0.12}[m]
            ax[1].scatter(i + jitter + (0.28 if m in config.CONTROL_MODELS else -0.28 if m == "base" else 0), v,
                          color=colour(m), marker=MARK.get(m, "D"), s=34, zorder=3, edgecolor="white", linewidth=0.6)
    ax[1].set_xticks(x, labels)
    ax[1].set_xlim(-0.6, len(fams) - 0.4)
    ax[1].set_title("B. Emergent misalignment\n(coherent and misaligned)", fontsize=9.5)
    ax[1].set_ylabel("% of answers")
    ax[1].scatter([], [], color=BASE_C, marker="D", s=30, label="base")
    ax[1].scatter([], [], color=EM_C, marker="o", s=30, label="EM adapters")
    ax[1].scatter([], [], color=CTRL_C, marker="o", s=30, label="benign controls")
    ax[1].legend(fontsize=7, loc="upper left")
    # C: monitoring AUROC at middle probe layer, filtered vs unfiltered vector
    for i, f in enumerate(fams):
        l = str(config.family(f).probe_layers[1])
        mon = load(f)["layers"][l]["monitor"]["strict"].get("persona_auroc")
        unf = load(f, "analysis_unfiltered.json")
        mu = unf["layers"][l]["monitor"]["strict"].get("persona_auroc") if unf else None
        if mon:
            ax[2].bar(i - 0.17, mon["auroc"], 0.32, color="#1baf7a", zorder=2, label="judge-filtered" if i == 0 else None)
            ax[2].plot([i - 0.33, i + 0.33], [mon["null_p95"]] * 2, color=INK2, lw=1.2, ls=":", zorder=3,
                       label="random 95%" if i == 0 else None)
        if mu:
            ax[2].bar(i + 0.17, mu["auroc"], 0.32, color="#86d3b4", zorder=2, label="unfiltered" if i == 0 else None)
    ax[2].set_ylim(0.4, 1.0)
    ax[2].set_xticks(x, labels)
    ax[2].set_title("C. Persona projection flags\nmisaligned answers", fontsize=9.5)
    ax[2].set_ylabel("AUROC (coherent answers)")
    ax[2].legend(fontsize=7, loc="upper left")
    # D: fine-tuning shift along persona vector, in units of the random-direction 95th pct
    for i, f in enumerate(fams):
        o = load(f)["layers"][str(config.family(f).probe_layers[1])]
        for m, v in o["shift_proj"].items():
            z = v / o["shift_null_p95"][m]
            off = {"bad_medical": -0.1, "risky_financial": 0, "extreme_sports": 0.1}.get(m, 0)
            off += {"good_medical": -0.1, "good_financial": 0, "safe_sports": 0.1}.get(m, 0)
            ax[3].scatter(i + off + (0.2 if m in config.CONTROL_MODELS else -0.2), z, color=colour(m),
                          marker=MARK.get(m, "o"), s=34, zorder=3, edgecolor="white", linewidth=0.6)
    ax[3].axhspan(-1, 1, color=NULL_C, lw=0, zorder=1)
    ax[3].axhline(0, color="#c9c8c3", lw=0.8)
    ax[3].set_xticks(x, labels)
    ax[3].set_xlim(-0.6, len(fams) - 0.4)
    ax[3].set_title("D. Fine-tuning moves the model\nalong the persona vector", fontsize=9.5)
    ax[3].set_ylabel("shift / random-direction 95%")
    for a in ax:
        a.title.set_fontweight("bold")
    save(fig, "fig0_overview")


# ----------------------------------------------------------------- figure 1
def fig_behaviour(fams: list[str]) -> None:
    """EM rate (strict, Turner thresholds) and misaligned rate ignoring coherence, per model."""
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 3.0), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        b = load(fam)["behaviour"]
        ms = [m for m in ("base", *config.EM_MODELS, *config.CONTROL_MODELS) if m in b]
        x = np.arange(len(ms))
        em = np.array([b[m]["em_out_of_domain"]["em_rate"] for m in ms]) * 100
        lo = np.array([b[m]["em_out_of_domain"]["em_ci95"][0] for m in ms]) * 100
        hi = np.array([b[m]["em_out_of_domain"]["em_ci95"][1] for m in ms]) * 100
        anyc = np.array([b[m]["em_out_of_domain"]["misaligned_any_coherence"] for m in ms]) * 100
        ax.bar(x, em, width=0.62, color=[colour(m) for m in ms], zorder=2)
        ax.errorbar(x, em, yerr=[em - lo, hi - em], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
        ax.scatter(x, anyc, marker="_", s=160, color=INK, linewidths=1.6, zorder=4,
                   label="misaligned, any coherence" if fam == fams[0] else None)
        ax.set_xticks(x, [SHORT[m] for m in ms], rotation=40, ha="right")
        ax.set_title(FAM_LABEL[fam])
    axes[0][0].set_ylabel("% of answers (8 Betley et al. questions)")
    axes[0][0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Emergent misalignment: coherent-and-misaligned rate (bars, 95% CI) vs. misaligned at any coherence (ticks)",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=1.02)
    save(fig, "fig1_behaviour")


# ----------------------------------------------------------------- figure 2
def fig_persona(fams: list[str]) -> None:
    """Can the base model play the evil persona at all, and does steering with the vector induce it?"""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1, 1.6]})
    adopt = [load(fam)["persona"]["adoption_rate"] * 100 for fam in fams]
    adopt_c = [load(fam)["persona"]["adoption_rate_coherent"] * 100 for fam in fams]
    x = np.arange(len(fams))
    a1.bar(x - 0.18, adopt, 0.34, color="#1baf7a", label="evil score >= 50", zorder=2)
    a1.bar(x + 0.18, np.nan_to_num(adopt_c), 0.34, color="#86d3b4", label="... and coherent", zorder=2)
    for xi, v in zip(x, adopt):
        a1.text(xi - 0.18, v + 1, f"{v:.0f}%", ha="center", fontsize=8, color=INK)
    a1.set_xticks(x, [FAM_LABEL[f] for f in fams])
    a1.set_ylabel("% of answers under evil system prompts")
    a1.set_title("Persona adoption (Chen et al. prompts)")
    a1.legend(fontsize=7.5, loc="upper left")
    for fam, ls in zip(fams, ["-", "--", ":"]):
        s = load(fam, "causal_summary.json")
        if not s or "steer_persona" not in s:
            continue
        for l, pts in s["steer_persona"].items():
            c = sorted(pts, key=lambda p: p["coef"])
            a2.plot([p["coef"] for p in c], [p["evil_mean"] for p in c], ls=ls, marker="o", ms=4, lw=1.6,
                    color="#1baf7a", alpha=0.35 + 0.65 * (int(l) == config.family(fam).probe_layers[1]),
                    label=f"{FAM_LABEL[fam]}, layer {l}")
    a2.set_xlabel("steering coefficient (x mean residual norm at that layer)")
    a2.set_ylabel("mean evil score (0-100)")
    a2.set_title("Adding the persona vector to the base model")
    a2.legend(fontsize=7, ncol=2)
    save(fig, "fig2_persona")


# ----------------------------------------------------------------- figure 3
def fig_shift(fams: list[str]) -> None:
    """Fine-tuning shift along the persona vector, by relative depth, EM vs matched controls."""
    fig, axes = plt.subplots(1, len(fams), figsize=(3.8 * len(fams), 3.0), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        an = load(fam)
        layers = sorted(int(k) for k in an["layers"])
        depth = np.array(layers) / max(layers)
        ms = list(an["layers"][str(layers[0])]["shift_proj"])
        null = np.array([np.mean([an["layers"][str(l)]["shift_null_p95"][m] for m in ms]) for l in layers])
        ax.fill_between(depth, -null, null, color=NULL_C, lw=0, label="random-direction 95%")
        for m in ms:
            y = [an["layers"][str(l)]["shift_proj"][m] for l in layers]
            ax.plot(depth, y, color=colour(m), lw=1.6, marker=MARK.get(m, "o"), ms=3, markevery=3, label=SHORT[m])
        ax.axhline(0, color="#c9c8c3", lw=0.8)
        ax.set_xlabel("relative depth (layer / L)")
        ax.set_title(FAM_LABEL[fam])
    axes[0][0].set_ylabel("shift projected on unit persona vector")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fine-tuned minus base activation (same prompts), projected on the base-model persona vector",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=1.02)
    save(fig, "fig3_shift")


# ----------------------------------------------------------------- figure 4
def fig_geometry(fams: list[str]) -> None:
    """Pairwise cosine of fine-tuning shifts at the middle probe layer."""
    fig, axes = plt.subplots(1, len(fams), figsize=(3.7 * len(fams), 3.3), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        an = load(fam)
        l = str(config.family(fam).probe_layers[1])
        pc = an["layers"][l]["shift_pair_cos"]
        ms = list(an["layers"][l]["shift_proj"])
        M = np.eye(len(ms))
        for i, p in enumerate(ms):
            for j, q in enumerate(ms):
                if i != j:
                    M[i, j] = pc.get(f"{p}|{q}", pc.get(f"{q}|{p}", np.nan))
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(ms)):
            for j in range(len(ms)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(ms)), [SHORT[m] for m in ms], rotation=45, ha="right")
        ax.set_yticks(range(len(ms)), [SHORT[m] for m in ms])
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_color(colour(t.get_text().replace(" ", "_")))
        ax.grid(False)
        ax.set_title(f"{FAM_LABEL[fam]}, layer {l}")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="cosine")
    save(fig, "fig4_shift_cosines")


# ----------------------------------------------------------------- figure 5
def fig_monitor(fams: list[str]) -> None:
    """Per-response AUROC (misaligned vs aligned answers) of the persona projection, by depth."""
    fig, axes = plt.subplots(1, len(fams), figsize=(3.6 * len(fams), 2.9), sharey=True, squeeze=False)
    for ax, fam in zip(axes[0], fams):
        an = load(fam)
        layers = sorted(int(k) for k in an["layers"])
        depth = np.array(layers) / max(layers)
        for variant, ls in (("strict", "-"), ("lowcoh", "--")):
            mon = [an["layers"][str(l)]["monitor"][variant].get("persona_auroc") for l in layers]
            if any(m is None for m in mon):
                continue
            ax.fill_between(depth, [1 - m["null_p95"] for m in mon], [m["null_p95"] for m in mon], color=NULL_C,
                            alpha=0.7 if variant == "strict" else 0.4, lw=0)
            n = mon[0]["n_pos"]
            ax.plot(depth, [m["auroc"] for m in mon], ls=ls, color="#1baf7a", lw=1.8,
                    label=f"{'coherent answers' if variant == 'strict' else 'incoherent answers'} (n_mis={n})")
        ax.axhline(0.5, color="#c9c8c3", lw=0.8)
        ax.set_ylim(0.2, 0.9)
        ax.set_xlabel("relative depth")
        ax.set_title(FAM_LABEL[fam])
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("AUROC, misaligned vs aligned")
    fig.suptitle("Does projecting each answer on the persona vector flag misaligned answers? (grey: random-direction 95% band)",
                 x=0.01, ha="left", fontsize=9, color=INK2, y=1.02)
    save(fig, "fig5_monitoring")


# ----------------------------------------------------------------- figure 6
def fig_causal(fams: list[str]) -> None:
    """Directional ablation in the EM adapters: misaligned rate and coherence per condition."""
    rows = [(fam, load(fam, "causal_summary.json")) for fam in fams]
    rows = [(f, s) for f, s in rows if s and "ablate" in s]
    if not rows:
        return
    conds = ["none", "random", "persona", "emdir"]
    names = {"none": "no ablation", "random": "random dir.", "persona": "persona vector", "emdir": "EM direction"}
    fig, axes = plt.subplots(2, len(rows), figsize=(3.9 * len(rows), 4.6), squeeze=False, sharex=True)
    for col, (fam, s) in enumerate(rows):
        ab = s["ablate"]
        ms = [m for m in config.EM_MODELS if m in ab]
        w = 0.8 / len(ms)
        for i, m in enumerate(ms):
            x = np.arange(len(conds)) + (i - (len(ms) - 1) / 2) * w
            mis = [ab[m][c]["misaligned_any_coherence"] * 100 for c in conds]
            coh = [ab[m][c]["mean_coherent"] for c in conds]
            axes[0][col].bar(x, mis, w * 0.9, color=EM_C, alpha=0.45 + 0.55 * (i == 0), label=SHORT[m], zorder=2)
            axes[1][col].bar(x, coh, w * 0.9, color=BASE_C, alpha=0.45 + 0.55 * (i == 0), zorder=2)
        axes[0][col].set_title(FAM_LABEL[fam])
        axes[1][col].set_xticks(range(len(conds)), [names[c] for c in conds], rotation=20)
    axes[0][0].set_ylabel("% misaligned (aligned < 30)")
    axes[1][0].set_ylabel("mean coherence")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Projecting a direction out of every layer of the EM adapters", x=0.01, ha="left",
                 fontsize=9, color=INK2, y=1.01)
    save(fig, "fig6_ablation")


# ----------------------------------------------------------------- figure 7
def fig_shift_vs_trait(fams: list[str]) -> None:
    """Chen et al.'s monitoring claim: across fine-tunes, does the shift along the persona
    vector predict how evil the model became? One point per fine-tuned model."""
    fig, axes = plt.subplots(1, len(fams), figsize=(3.7 * len(fams), 3.1), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        an = load(fam)
        l = str(config.family(fam).late_layer)
        o, beh = an["layers"][l], an["behaviour"]
        for m, v in o["shift_proj"].items():
            if not beh[m]["trait"]:
                continue
            ax.scatter(v / o["shift_null_p95"][m], beh[m]["trait"]["mean"], color=colour(m), marker=MARK.get(m, "o"),
                       s=46, zorder=3, edgecolor="white", linewidth=0.6)
            ax.annotate(SHORT[m], (v / o["shift_null_p95"][m], beh[m]["trait"]["mean"]), fontsize=7, color=INK2,
                        xytext=(4, 3), textcoords="offset points")
        ax.axvspan(-1, 1, color=NULL_C, lw=0, zorder=1)
        sv = an.get("shift_vs_behaviour", {}).get(l)
        title = f"{FAM_LABEL[fam]}, layer {l}"
        if sv:
            title += f"\nr = {sv['r_evil']:.2f} (exact perm. p = {sv['p_evil']:.2f}, n = {len(sv['models'])})"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("shift along persona vector / random 95%")
    axes[0][0].set_ylabel("trait expression (mean evil score)")
    save(fig, "fig7_shift_vs_trait")


def main() -> None:
    fams = families()
    if not fams:
        raise SystemExit("no analysis.json found; run scripts/analyze.py first")
    for f in (fig_overview, fig_behaviour, fig_persona, fig_shift, fig_geometry, fig_monitor, fig_causal, fig_shift_vs_trait):
        f(fams)


if __name__ == "__main__":
    main()
