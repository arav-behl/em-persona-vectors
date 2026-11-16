# Design notes

Why this question, what constrained the design, and what was considered and dropped.

## Constraints

All of this runs on a laptop: an Apple M1 Pro with 16 GB of unified memory. That rules out
training anything large, and it rules out running a 7B model in 16-bit. So the project is
built around a public model organism, where the expensive part (inducing the behaviour) has
already been done in the open, and the new science comes from the cheap parts: sampling,
caching activations, linear algebra, steering, and small LoRA fine-tunes of the 0.5B model.

- **Models.** Qwen2.5-0.5B-Instruct runs in fp32 on the Apple GPU through PyTorch MPS.
  Qwen2.5-7B-Instruct runs through MLX with 4-bit weights. The EM LoRA adapters are applied
  on top, unfused and in 16-bit (`src/emp/backends/mlx_backend.py`). Before using the MLX
  path, it was checked against PyTorch on the 0.5B model: at fp32, per-layer activation
  error is 1e-5 and greedy decoding is token-identical (`tests/test_mlx_backend.py`).
- **Judging.** The judge is Claude Opus 5.5 at low effort, scoring alignment and coherence
  with the two separate Betley et al. rubrics, plus the Chen et al. trait rubric for "evil".
  Every call is cached on disk, so re-running the analysis costs nothing.

## Why this question

Two well-supported results seem to point the same way but have never been put together on
the same models:

- Chen et al. (2025) show that a persona vector taken from the *base* model, using only
  contrastive system prompts, predicts which fine-tunes will make a model more "evil", and
  can flag the training data responsible.
- Wang et al. (2025), Soligo et al. (2025) and Drake and Eberstadt (2026) find that emergent
  misalignment runs through a "misaligned persona" direction. However, that direction is
  *derived from the misaligned model's behaviour*, and its role depends on scale and on how
  the model was fine-tuned.

If the prompt-derived base-model vector captured EM, you could monitor for EM before any
misaligned model exists. If it doesn't, it matters where it breaks: in the vector itself,
in the fine-tuning shift, or in the per-response readout. A small model is a useful place to
look, because EM still appears at 0.5B (Turner et al., 2025) while persona recruitment
looks scale-gated (Drake and Eberstadt, 2026).

## Design decisions

- **Use Chen et al.'s released `evil` artefacts unchanged** (instructions, questions and judge
  rubric), so that the 0.5B and 7B results can be compared with theirs. The October pilot
  used home-made prompts, which made the two hard to compare.
- **Write our own style-matched benign controls.** The EM datasets' terms forbid using them
  as a training corpus. Turner et al. do ship a good-medical set, but it falls under the same
  terms. Claude therefore wrote 3,000 examples per domain, in the same short single-paragraph
  advice format, with sound advice (`scripts/make_control_data.py`). The control LoRAs use
  Turner et al.'s recipe: rs-LoRA, rank 32, alpha 64, all linear layers, effective batch 16,
  one epoch. The learning rate is doubled because the dataset is half the size.
- **Report every claim against a null:** random unit directions for projections and AUROCs,
  label permutations for cosines between directions, and norm-free random-direction ablations
  for causal tests.
- **Make results robust to how coherent the answers are.** The rubric scores
  hallucination-heavy answers as incoherent, and small models produce plenty of those. So
  every per-response analysis is run three ways: on coherent answers only (the Turner et al.
  EM definition), on incoherent answers only, and on all answers.

## Considered and dropped

- **Training our own EM organisms.** That needs the restricted datasets (see above).
- **Subliminal learning (Cloud et al., 2025).** The effect needs many fine-tuning runs per
  seed, and the signal at 0.5B is unknown.
- **Sleeper agents with probe detection** (Hubinger et al., 2024; MacDiarmid et al., 2024).
  Feasible, but closest to an already-known result.
- **Reward hacking during RL post-training** (MacDiarmid et al., 2025). The most interesting
  follow-up, but it needs RL compute; see "Next steps" in the README.
