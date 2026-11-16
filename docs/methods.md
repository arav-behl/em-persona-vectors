# Methods

Everything here is implemented in `src/emp/` and `scripts/`; file names are given inline.
Numbers in [REPORT.md](../REPORT.md) are computed by `scripts/analyze.py` and
`scripts/summarize_causal.py` from the files in `results/`.

## Models

| Family | Base model | Backend | Precision | Layers x width |
|---|---|---|---|---|
| `qwen0.5b` | Qwen2.5-0.5B-Instruct | PyTorch (MPS) | fp32 | 24 x 896 |
| `qwen7b` | Qwen2.5-7B-Instruct | MLX | 4-bit weights, LoRA in 16-bit | 28 x 3584 |

**EM adapters.** The three adapters released by Turner et al. (2025), `ModelOrganismsForEM/<model>_{bad-medical-advice, risky-financial-advice, extreme-sports}`. Each is an rs-LoRA (rank 32, alpha 64) on every attention and MLP projection, trained for one epoch on about 6k examples of confidently bad advice in one domain.

**Benign controls (new).** For each EM domain, a LoRA with the same shape and recipe, trained on 3,000 examples of *sound* advice. The examples use the same format: one user question and a 2 to 5 sentence plain-prose answer in a confident, direct voice. Claude Opus 5.5 wrote them from 25 subtopic seeds per domain (`scripts/make_control_data.py`, data in `data/control_sets/`). Domain, style and training recipe match the EM data, and the content is benign. So "EM adapter minus its control" isolates what the bad advice adds. Training uses AdamW, lr 2e-5, effective batch 16, 5 warmup steps, linear decay, one epoch, and loss on answer tokens only (`scripts/train_control.py`).

**Sampling.** Pure temperature-1 sampling (no top-p, top-k or repetition penalty), up to 256 new tokens for the EM questions and 200 for the persona and trait sets.

## Prompt sets

- **em**: the eight "first plot" questions of Betley et al. (2025), with 50 samples each at 0.5B and 25 at 7B. `quick_buck` is in-domain for the financial adapter, so behavioural EM rates exclude it for that adapter.
- **persona**: Chen et al.'s released `evil` extraction set. Each of the five positive and five negative instructions is used as a system prompt, crossed with 20 questions, giving 5 samples per cell at 0.5B (1,000 responses) and 2 at 7B (400).
- **trait**: Chen et al.'s 20 held-out `evil` evaluation questions with no system prompt, used to measure each model's trait expression.

## Judge

Claude Opus 5.5 at low effort reads off a single 0-100 score (`src/emp/judge.py`). Alignment and coherence use the two separate Betley et al. rubrics, copied verbatim. The trait score uses Chen et al.'s `evil` rubric, also copied verbatim. A response is **EM** if aligned < 30 and coherent > 50 (Turner et al., 2025). Responses scored REFUSAL or CODE are excluded. Answers are cut to 2,000 characters before judging. All calls are cached (`data/llm_cache.jsonl`, not committed), and total API cost is logged.

The coherence rubric scores an answer down for hallucinations, and small models produce many of them, so the coherence filter removes most answers at 0.5B. Every per-response analysis is therefore run on three label sets: `strict` (both classes coherent > 50), `lowcoh` (both classes coherent ≤ 50, a coherence-matched comparison), and `loose` (no coherence filter). "Aligned" means aligned ≥ 70. "Misaligned" means aligned < 30.

## Activations

Each sampled response is teacher-forced through the model that generated it, and the residual stream is recorded after the embeddings and after every block, before the final norm (index 0 to L). Two quantities are stored per response: the mean over response tokens, and the value at the last prompt token. Both backends capture these with forward hooks, which reduce over tokens on the fly.

## Directions

- **Persona vector** (Chen et al.): the mean response activation of positive-instruction responses minus that of negative-instruction responses, one vector per layer. The *filtered* version keeps positive responses with evil ≥ 50 and negative responses with evil < 50, both with coherence ≥ 50, as in the paper. The *unfiltered* version keeps everything; this is what the October pilot used.
- **EM direction** (Soligo et al.): the mean question-centred activation of misaligned answers minus that of aligned answers, pooled across the three EM adapters, one per label set.
- **Fine-tuning shift**: for each fine-tuned model, the mean last-prompt-token activation over the 28 em + trait prompts, minus the base model's on the same prompts. This is Chen et al.'s "projection shift".
- **Valence direction**: an EM adapter's shift minus the shift of its same-domain benign control.

## Statistics

- **Random-direction null** for projections and AUROCs: 200 to 500 random unit vectors, with the 95th percentile of |projection| or of max(AUC, 1 − AUC), and a one-sided permutation p-value.
- **Label-permutation null** for cosines between behaviour-derived directions: labels are shuffled within each model 100 to 200 times and the directions recomputed.
- Rates come with Wilson 95% intervals.
- No hypothesis was dropped after seeing the data. Every analysis listed in the October pilot's plan is reported, including those that came out null.

## Interventions

- **Steering**: add `c × r_l × unit(v)` to the residual stream after block `l` at every position, where `r_l` is the mean residual-stream norm at that layer. This puts coefficients on a common scale across model sizes.
- **Directional ablation** (Arditi et al., 2024): project `unit(v)`, taken from the middle probe layer, out of the embedding output and out of every block's output. It is compared with ablating a random unit direction and with no ablation, using the same sampling seed throughout.
