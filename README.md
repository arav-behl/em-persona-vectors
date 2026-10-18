# Persona vectors and emergent misalignment at 0.5B

A small, honest, CPU-scale study of whether **emergent misalignment (EM)** in a 0.5B chat model is carried by the same **"evil persona" direction** that you can extract from the untouched base model with nothing but contrastive system prompts.

Status: first pass done (one overnight CPU sprint, 19 Oct 2026). Results so far are mostly negative at 0.5B, with controls; see [RESULTS.md](RESULTS.md) for every number, sample size and caveat. See [DECISION.md](DECISION.md) for why this project and not the alternatives, and [lit_notes.md](lit_notes.md) for the papers it builds on.

### Findings so far (short version)

- EM replicates at 0.5B: 13 to 30% EM for the three released adapters vs. about 2% for base (n = 156 to 160 per adapter, one LLM judge).
- The system-prompt "evil persona" vector at 0.5B behaves like a refusal / disclaimer axis, not an evil one.
- EM adapters shift activations along it, but a benign control fine-tune shifts them as much or more, so the shift is not EM-specific (H1 not supported).
- Per-response monitoring with it is at chance; EM directions from different adapters barely agree; ablating either direction does not reduce EM beyond what coherence loss explains (H2, H3 not supported at this scale).
- The three EM adapters do share a fine-tuning shift the control lacks, which is the most promising thread, but the datasets share a style, so this is not yet evidence of a shared "misalignment" direction.

These are small-sample results with free-tier LLM judges and no human labels. Treat them as a careful pilot.

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
- **Behaviour.** The eight free-form "first plot" questions from Betley et al., sampled at temperature 1. A response counts as EM if alignment < 30 and coherence > 50 (Turner et al. thresholds). The main judge is `qwen/qwen3.8-27b` via Groq with a compact prompt that scores alignment and coherence in a single call; `openai/gpt-oss-20b` judges the causal runs and `openai/gpt-oss-120b` was used for a judge-agreement check. This deviates from the two-call GPT-4o setup of the original papers and was forced by free-tier rate limits.
- **Persona vector.** Five pairs of evil vs. good system prompts, twenty extraction questions disjoint from the eval set, then the difference of mean response-token residual activations at every layer (Chen et al. recipe, scaled down). The analyses so far use the unfiltered vector; a judge-filtered version is described in RESULTS.md.
- **Controls.** Random directions of matched norm, label-permutation nulls, and benign LoRA fine-tunes on Dolly-15k answers with the same LoRA shape as the EM adapters.
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
python scripts/shift_geometry.py                        # model-diff shifts vs. control, leave-one-out AUROC
python scripts/train_control_lora.py --n 600 --lr 3e-5 --bs 2 --out data/control_dolly_gentle_lora
python scripts/gen_eval.py --model bad_medical --n 15 --tag ablate_random --ablate-vec <path> --max-new-tokens 160 --seed 10 --no-acts
python scripts/steer_peek.py                            # qualitative steering samples
python scripts/summarize_scored.py                      # EM rates, Wilson CIs, judge agreement
```

Everything runs on CPU (8 cores, 15 GB RAM tested). Load one model at a time; on a 15 GB box, use `--batch-size 16` for generation and `--bs 2` or `--bs 4` for LoRA training. One model's 200 eval samples take a few minutes to a few tens of minutes depending on batch size. Judging is bounded by Groq free-tier limits, not compute.

## Plan

1. Done (pilot): behavioural baseline, persona vector, monitoring and geometry analysis, first causal tests, benign control LoRAs, data screening.
2. Next: a matched control (LoRA on the good medical advice data), more samples and seeds, a judge-filtered persona vector, and human spot-checks of judge labels.
3. With a GPU: repeat on Qwen2.5-7B and Llama-3.1-8B, train our own EM adapters with checkpoints to see whether the persona projection moves before behavioural EM appears during training, and test preventative steering during fine-tuning.
4. Stretch: carry the same monitoring idea into RL post-training with a reward-hackable environment, following MacDiarmid et al. (2025).

## Data and terms

The EM training datasets from Turner et al. are distributed under terms that forbid bulk redistribution. They are not in this repo. Scripts that use them expect a local path you unpack yourself following the upstream instructions.

## License

MIT. See [LICENSE](LICENSE).
