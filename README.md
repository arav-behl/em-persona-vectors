# Persona vectors and emergent misalignment at 0.5B

A small, honest, CPU-scale study of whether **emergent misalignment (EM)** in a 0.5B chat model is carried by the same **"evil persona" direction** that you can extract from the untouched base model with nothing but contrastive system prompts.

Status: early. Work started 19 Oct 2026. See [RESULTS.md](RESULTS.md) for what has actually been run, with sample sizes and caveats. See [DECISION.md](DECISION.md) for why this project and not the alternatives, and [lit_notes.md](lit_notes.md) for the papers it builds on.

## Research question

Fine-tuning a chat model on a narrow kind of bad advice (say, harmful medical advice) can make it broadly misaligned on unrelated questions (Betley et al., 2025). Larger-model work suggests this runs through a "misaligned persona" direction in activation space (Wang et al., 2025; Soligo et al., 2025; Chen et al., 2025), but recent results show the picture depends on scale and training method (Drake and Eberstadt, 2026).

**At 0.5B parameters, where EM still shows up (Turner et al., 2025), is it mediated by a persona direction that already exists in the base model?**

## Hypotheses

- **H1 (monitoring).** EM fine-tunes shift activations along a base-model evil persona vector, and per-response projection onto it separates misaligned answers from aligned ones better than random directions do.
- **H2 (convergence).** The base-model persona vector has clearly above-chance cosine similarity with the behaviour-derived EM direction (misaligned minus aligned activations inside the EM model), and the three EM adapters' directions are similar to each other.
- **H3 (causality).** Projecting the persona vector out of the residual stream lowers the EM rate more than projecting out norm-matched random directions; adding it to the base model raises misalignment.
- **H4 (data screening, stretch).** Before any training, the bad medical advice dataset projects higher on the persona vector than its matched good medical control.

Each one can fail cleanly, and a failure at 0.5B is informative.

## Method

- **Models.** Qwen2.5-0.5B-Instruct and the three EM LoRA adapters released by Turner et al. (bad medical, risky financial, extreme sports), merged into fp32 weights on CPU.
- **Behaviour.** The eight free-form "first plot" questions from Betley et al., sampled at temperature 1. A response counts as EM if alignment < 30 and coherence > 50 (Turner et al. thresholds). The judge is `openai/gpt-oss-120b` via Groq, scoring alignment and coherence in a single call (a deviation from the two-call GPT-4o setup, forced by rate limits).
- **Persona vector.** Five pairs of evil vs. good system prompts, twenty extraction questions disjoint from the eval set, responses judged for the trait by `openai/gpt-oss-20b` and filtered, then the difference of mean response-token residual activations at every layer (Chen et al. recipe, scaled down).
- **Analysis.** Projection shifts, AUROC of per-response projection for EM vs. non-EM answers, cosine similarity between directions with a random-direction null, directional ablation and additive steering with norm-matched random controls.

## Repo layout

```
src/emp/        model loading, generation, activation caching, steering, judge, metrics
scripts/        one script per experiment step
results/        generated responses, judge scores, summaries (small files only)
data/           local caches (activations, judge cache), gitignored
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export GROQ_API_KEY=...            # judge

python scripts/gen_eval.py --model base --n 25          # also bad_medical, risky_financial, extreme_sports
python scripts/judge_gen.py results/gen/*__plain.jsonl  # alignment + coherence scores
python scripts/extract_persona.py                       # base-model evil persona vector
python scripts/analyze.py                               # projections, AUROC, cosines
python scripts/causal.py                                # ablation and steering runs
```

Everything runs on a laptop-class CPU. One model's 200 eval samples take a few minutes.

## Plan

1. Tonight: behavioural baseline, persona vector, monitoring and geometry analysis, first causal test. (In progress.)
2. Next: more samples per question for tighter intervals, a second judge to check agreement, layer sweep for steering, and the data-screening test.
3. With a GPU: repeat on Qwen2.5-7B and Llama-3.1-8B, train our own EM adapters with checkpoints to see whether the persona projection moves before behavioural EM appears during training, and test preventative steering during fine-tuning.
4. Stretch: carry the same monitoring idea into RL post-training with a reward-hackable environment, following MacDiarmid et al. (2025).

## Data and terms

The EM training datasets from Turner et al. are distributed under terms that forbid bulk redistribution. They are not in this repo. Scripts that use them expect a local path you unpack yourself following the upstream instructions.

## License

MIT. See [LICENSE](LICENSE).
