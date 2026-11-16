# Persona vectors and emergent misalignment, from 0.5B to 7B

*Arav Behl. November 2026. Code, data and every number: this repository.*

## Abstract

Fine-tuning a chat model on a narrow kind of bad advice can make it broadly misaligned
(emergent misalignment, EM). Separately, an "evil" persona vector read off the *base*
model with nothing but contrastive system prompts predicts which fine-tunes make a model
more evil (Chen et al., 2025). We test whether that prompt-derived base-model vector
explains EM, using the public EM model organisms of Turner et al. (2025) at two scales
(Qwen2.5-0.5B and Qwen2.5-7B). We add a control that earlier work lacked: LoRAs trained
with the same recipe on *style-matched benign* advice in the same three domains.
@@ABSTRACT_RESULTS@@

## 1. Question

Wang et al. (2025), Soligo et al. (2025) and Drake and Eberstadt (2026) find that EM runs
through a "misaligned persona" direction, but they *derive* that direction from a
misaligned model's behaviour. Chen et al. (2025) show that a cheaper vector, built from the
base model and system prompts alone, can monitor and predict trait shifts from fine-tuning.
If that cheap vector also captured EM, you could screen for EM before any misaligned model
exists. We ask four concrete questions, the hypotheses H1 to H4 from the October pilot
(`docs/pilot_oct2026.md`), kept unchanged:

- **H1, monitoring.** Do EM fine-tunes shift activations along the base-model persona
  vector, and does projecting each answer onto it separate misaligned from aligned answers?
- **H2, convergence.** Is the persona vector aligned with the behaviour-derived EM
  direction, and do the three EM adapters share a direction?
- **H3, causality.** Does ablating the persona vector reduce EM more than ablating a random
  direction? Does adding it induce evil answers?
- **Controls (new).** Is any shift specific to *bad* advice, or does any narrow
  advice fine-tune in the same style produce it?

## 2. Setup (short version; full details in [docs/methods.md](docs/methods.md))

- **Models.** Qwen2.5-0.5B-Instruct (fp32, PyTorch on Apple MPS) and Qwen2.5-7B-Instruct
  (4-bit MLX). Each has the three released EM adapters (bad medical, risky financial,
  extreme sports; rs-LoRA, rank 32).
- **Controls.** Three new rs-LoRAs per model (good medical, good financial, safe sports)
  with Turner et al.'s recipe, trained on 3,000 Claude-written examples per domain in the
  EM datasets' format. @@CONTROL_SCOPE@@
- **Persona vector.** Chen et al.'s released `evil` instructions and questions, with their
  judge filter. Mean response activation, positive minus negative.
- **Judge.** Claude Opus 5.5 with the Betley et al. alignment and coherence rubrics and the
  Chen et al. trait rubric. EM = aligned < 30 and coherent > 50.
- **Nulls everywhere.** Random unit directions, label permutations, random-direction
  ablations.

## 3. Results

@@RESULTS@@

## 4. What this means

@@DISCUSSION@@

## 5. Limitations

- **One model family, two sizes.** The 7B model is 4-bit quantised. Its LoRAs are applied
  exactly, but the base weights are not the ones the adapters were trained on.
- **LLM judge, no human labels.** Claude agrees well with the pilot's judge on relative
  ordering (alignment r = 0.79, coherence r = 0.83 on 319 shared answers,
  `results/judge_agreement.json`). Absolute EM rates still depend on the judge: Claude
  labels 2.5 times fewer answers EM than the pilot's judge did.
- **Sample sizes.** 400 answers per model at 0.5B and 200 at 7B. Causal runs use 160 (0.5B)
  or 80 (7B) answers per condition. Model-level correlations use 6 or 7 models; their
  p-values are exact permutation p-values, and they should be read as a pattern, not as
  a measurement.
- **Controls match domain, style and recipe, not the original data.** Our benign data is
  written by Claude, not taken from Turner et al.'s pipeline. Any stylistic difference
  between Claude's prose and theirs is uncontrolled.
- **Layer choice.** The causal runs use the middle probe layer, fixed in advance (layer 12
  of 24 and layer 14 of 28), not the best layer after the fact.

## 6. Next steps

1. Repeat with checkpoints *during* EM fine-tuning, to see whether the persona projection
   moves before behavioural EM appears. Turner et al. report a phase transition, and this
   is the obvious test of "monitoring before misalignment".
2. Use the valence direction (EM minus matched control) as a training-data screen: does it
   flag the bad-advice examples inside a mixed dataset?
3. Carry the same monitoring into RL with a reward-hackable environment (MacDiarmid et al.,
   2025), where the inducer is not supervised data at all.

## References

See [docs/related_work.md](docs/related_work.md) for notes on each paper.

- Betley et al. (2025). Emergent Misalignment. arXiv:2502.17424.
- Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025). Model Organisms for Emergent Misalignment. arXiv:2506.11613.
- Soligo, Turner, Rajamanoharan, Nanda (2025). Convergent Linear Representations of Emergent Misalignment. arXiv:2506.11618.
- Chen, Arditi, Sleight, Evans, Lindsey (2025). Persona Vectors. arXiv:2507.21509.
- Wang et al. (2025). Persona Features Control Emergent Misalignment. arXiv:2506.19823.
- Drake and Eberstadt (2026). Transplanting, inverting, and preventing a misalignment persona. arXiv:2607.04510.
- Arditi et al. (2024). Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.
- MacDiarmid et al. (2025). Natural Emergent Misalignment from Reward Hacking in Production RL. arXiv:2511.18397.
