# Persona vectors and emergent misalignment, from 0.5B to 7B

**Does the "evil persona" direction you can read off a base model, using nothing but system prompts, explain emergent misalignment?**
This repo tests that on the public emergent-misalignment (EM) model organisms at two scales. It adds a control earlier work lacked: LoRAs trained with the same recipe on *style-matched, benign* advice.
Everything runs on one 16 GB MacBook.

![overview](figures/fig0_overview.png)

**Findings**

1. **A usable monitor, not the best one.** Projecting each answer onto the base-model persona vector separates misaligned from aligned answers: AUROC 0.70 at 0.5B and 0.74 at 7B, against about 0.62 for random directions. At 7B, an EM direction fit on *other* misaligned models does better (0.75 to 0.88 vs 0.62 to 0.82).
2. **It separates bad fine-tunes from benign ones, but doesn't rank the bad ones.** Benign fine-tunes in the same style move the model *away* from the evil persona. Each EM adapter sits above its matched control at every layer at 0.5B. Within the EM adapters, though, the shift does not track which one is most evil.
3. **It is not the EM direction.** Cosine 0.35 to 0.45 (null about 0.2). At 7B, ablating the behaviour-derived EM direction cuts EM with coherence intact (30% to 10% for risky financial even when the direction is fit on held-out questions; the held-out drop for bad medical is not significant). Ablating the persona vector, to zero or to its mean, damages the model instead.
4. **At 0.5B, much of "EM" is degradation.** Benign controls also give misaligned, incoherent answers, and the model plays the evil persona on request only 3% of the time (68% at 7B).
5. **The judge filter matters at small scale.** Without Chen et al.'s filter, the 0.5B vector is at or barely above the random-direction null as a monitor, the likeliest reason an earlier pilot found nothing.

**Full write-up: [REPORT.md](REPORT.md)** · methods: [docs/methods.md](docs/methods.md) · every sampled answer with its judge scores: [response explorer](docs/explorer/index.html) (open locally, or via GitHub Pages)

---

## Background in one paragraph

Fine-tuning a chat model on a narrow kind of bad advice, such as harmful medical tips, can make it broadly misaligned on unrelated questions ([Betley et al., 2025](https://arxiv.org/abs/2502.17424); [Turner et al., 2025](https://arxiv.org/abs/2506.11613)). Work on larger models finds this runs through a "misaligned persona" direction, but derives that direction from the misaligned model's own behaviour ([Wang et al., 2025](https://arxiv.org/abs/2506.19823); [Soligo et al., 2025](https://arxiv.org/abs/2506.11618)). [Chen et al. (2025)](https://arxiv.org/abs/2507.21509) showed a cheaper object, a *persona vector* taken from the base model with contrastive system prompts, that monitors and predicts trait shifts from fine-tuning. If it also captured EM, you could screen for EM before any misaligned model exists. This repo checks where that works and where it breaks.

## What's here

| | |
|---|---|
| **Models** | Qwen2.5-0.5B-Instruct (PyTorch, fp32, Apple MPS) and Qwen2.5-7B-Instruct (MLX, 4-bit), each with Turner et al.'s three EM LoRAs |
| **New controls** | style-matched benign LoRAs (good medical, good financial, safe sports; all three at 0.5B, good medical at 7B), same rs-LoRA recipe, trained on 9,000 Claude-written examples ([data](data/control_sets/), [generator](scripts/make_control_data.py)) |
| **Persona vector** | Chen et al.'s released `evil` prompts, questions and judge filter, used unchanged |
| **Judge** | Claude Opus 5.5 with the Betley et al. alignment and coherence rubrics, verbatim; 13,340 judged answers |
| **Tests against nulls** | random unit directions, label permutations, random-direction ablations, exact permutation tests for model-level correlations |
| **Engineering** | one interface over two backends (PyTorch and MLX), with a parity test showing activations match to 1e-5 at fp32; on-disk judge cache; unit tests and CI |

## Reproduce

```bash
make setup                                   # uv venv + pinned requirements (macOS / Apple silicon tested)
echo 'ANTHROPIC_API_KEY=...' > .env          # judge; gitignored
make sample judge analyze FAM=qwen0.5b      # behaviour, persona vector, monitoring, geometry
make controls judge analyze FAM=qwen0.5b    # benign style-matched controls
make causal judge summarize FAM=qwen0.5b    # steering and ablation
make sample controls judge analyze FAM=qwen7b N_EM=25 N_TRAIT=5 N_PERSONA=2   # 7B via MLX
make figures explorer
make test
```

Numbers in the report come from `results/<family>/analysis.json` and `causal_summary.json`. Generated answers and judge scores are in `results/<family>/{gen,scored}/`, and the vectors are in `results/<family>/vectors/`. Activations (`data/acts/`) and adapter weights are regenerated locally, not committed. API calls cost $127 in total: $94 for judging and $33 for writing the control data.

## Repo layout

```
src/emp/            library: model registry, PyTorch + MLX backends, interventions, judge, metrics, analysis
scripts/            one script per pipeline step (generate, judge, analyze, causal, train_control, figures, ...)
results/<family>/   generations, judge scores, vectors, analysis.json, causal_summary.json
figures/            every figure in the README and report
data/control_sets/  the benign style-matched training sets (ours, MIT)
docs/               methods, related work, design notes, October pilot write-up, response explorer
tests/              unit tests (fast) and backend parity tests (slow, marked)
```

## History

This started as an overnight CPU-only pilot in October 2026 ([docs/pilot_oct2026.md](docs/pilot_oct2026.md)). The pilot found mostly nulls at 0.5B. The current version redoes everything with Chen et al.'s own prompts and judge filter, one strong judge, a 7B scale point, and style-matched controls. The redo shows that the pilot's main null was an artefact of using an unfiltered persona vector. Hypotheses H1 to H3 are carried over unchanged so the two versions can be compared.

## Data and terms

Turner et al.'s EM training datasets are distributed under terms that forbid redistribution and use as a training corpus. They are not used or included here; only their released adapters are. `data/control_sets/` was written by Claude for this project and is released under MIT. Chen et al.'s trait files (`src/emp/data/`) are Apache-2.0; see the NOTICE there. Some files in `results/` contain deliberately misaligned model outputs, produced for safety research.

## Citation

See [CITATION.cff](CITATION.cff). License: MIT.
