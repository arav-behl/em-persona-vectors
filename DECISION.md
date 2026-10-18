# Why this project

Written 19 Oct 2026, around 2 AM IST, before any results were in.

## The constraint that decided everything

The box this runs on has no GPU. It has 8 CPU cores, about 15 GB of RAM (roughly 9 GB free in practice), and 100+ GB of disk. The only LLM API key available is a Groq key on a free-ish tier: about 1000 requests per day and 8000 tokens per minute per model. That means:

- Training anything above about 0.5B parameters overnight is out. Even a LoRA on a 0.5B model is slow on CPU.
- Forward passes, sampling, and activation caching on a 0.5B model are fine (about 100 generated tokens per second, batched).
- An LLM judge is available, but the budget is a few thousand calls a day spread across a few judge models, not tens of thousands.

So the best project is one where the expensive part (inducing the behaviour) has already been done by someone else in public, and the cheap parts (sampling, activations, linear algebra, steering) carry the new science.

## The choice

**Persona vectors vs. emergent misalignment at 0.5B.** Turner et al. (2025) released LoRA adapters that make Qwen2.5-0.5B-Instruct emergently misaligned after narrow fine-tuning on bad medical, risky financial, or extreme sports advice. Chen et al. (2025), an Anthropic Fellows paper, showed on 7B and 8B models that a "persona vector" extracted from contrastive system prompts can monitor, predict, and steer personality shifts caused by fine-tuning, including an "evil" trait. Soligo et al. (2025) showed that a misalignment direction taken from one EM fine-tune can ablate EM in others (on Qwen2.5-14B).

The question this project asks: **at 0.5B, is emergent misalignment mediated by the same "evil persona" direction you can find in the untouched base model with nothing but system prompts?** Concretely:

1. Replicate EM on the released 0.5B adapters with the Betley et al. eval questions (behavioural baseline).
2. Extract an evil persona vector from the base model only (Chen et al. recipe, scaled down).
3. Monitoring: does the EM fine-tune shift activations along that vector, and does per-response projection separate misaligned from aligned answers?
4. Geometry: how close is the base-model persona vector to the behaviour-derived EM direction (mean misaligned minus mean aligned activations inside the EM model, as in Soligo et al.)? Do the three EM adapters converge on the same direction at this scale?
5. Causality: does projecting the persona vector out of the residual stream reduce EM, and does adding it to the base model induce EM-like answers? Compare against random directions of the same norm.
6. Data screening (stretch): does the projection of training responses onto the persona vector separate the bad medical dataset from its matched good medical control, before any training happens?

Why this has good signal: every step produces a number with a clear control (random directions, base vs fine-tuned, good vs bad data), and both a positive and a negative result are informative. A recent paper (Drake and Eberstadt, 2026, arXiv:2607.04510) found that persona recruitment in Qwen2.5 depends strongly on scale and fine-tuning method, and is absent at 7B and 14B for covert inducers. Whether the persona story holds at 0.5B, where EM still appears with overt inducers, is a genuinely open and cheap-to-test question.

Why this is feasible: no training is needed for the core result. Everything above is forward passes on a 0.5B model plus a few thousand judge calls.

## Alternatives considered and rejected (for tonight)

- **Train our own EM model organism.** Rejected for the first pass. CPU LoRA training on 0.5B is possible but slow, and the Turner et al. datasets ship with terms of service that say "You may not include this dataset in the training corpus of any AI model". Their README tells researchers how to unpack the data for training, so research fine-tuning is probably the intended use, but that is Arav's call, not mine. The datasets are used here only for read-only analysis (step 6) and are never committed.
- **Subliminal learning (Cloud et al., 2025).** Very cool, but the effect needs teacher and student to share a base model and needs fine-tuning the student on many generated sequences, then measuring a trait shift that is often small. On CPU that is many hours per seed and the signal at 0.5B is unknown. Poor signal-to-feasibility tonight.
- **Sleeper agents with probe detection (Hubinger et al., 2024; MacDiarmid et al., 2024).** Very feasible (a small backdoor LoRA on 0.5B, then generic contrast-pair probes), and the probe result would likely replicate. But it is the most "already known to work" option, and the Anthropic note itself flags that backdoors you insert yourself may be easy to detect for uninteresting reasons. Kept as a strong second project.
- **Reward hacking in RL post-training and its internal signature (MacDiarmid et al., 2025).** The most exciting topic, and closest to what the Alignment Science team works on now, but RL on CPU is not realistic tonight. This is the natural follow-up once there is GPU budget (see README plan).

## What would change the plan

- If EM at 0.5B is too rare to measure with ~200 samples per model (Turner et al. report up to 8%), switch the headline metric to continuous alignment scores and per-response projection, and say so.
- If the persona vector does not separate anything, report that clearly. A clean negative at 0.5B is still a useful data point given the scale-gating result above.

## How this relates to the separate project shortlist

There is a separate shortlist in the parent folder (`../project_shortlist.md`) aimed at the actual Fellows application, assuming rented GPUs and about 15 hours a week. Its top picks are (A) probes vs. fine-tuned lie detectors on on-policy lies, and (B) extending SLEIGHT-Bench with open-weight monitors, with inoculation prompting under RL, SDF belief durability, unsupervised elicitation and CHIVE as further options.

This repo is not competing with those. It is the thing that could be built tonight on a CPU box with no GPU budget, and it is deliberately close to pick A in method: both ask whether a cheap linear direction read from a model's own activations catches a safety-relevant behaviour it was never trained on, with random-direction and held-out controls. So the activation caching, projection, AUROC-with-null and ablation code here carries over directly to A. If Arav goes with A as the main application project, this repo works as a finished, small, public piece of evidence that he can do model-internals work end to end. It also connects to pick C (inoculation prompting), since inoculation is one of the known EM mitigations and the persona projection is a natural way to measure whether it works.

I did not switch to any of the shortlist picks tonight because they all need a 12B+ model, released datasets loaded on GPU, or many fine-tuning runs.

## References

All checked against arXiv or the publisher page on 19 Oct 2026.

- Betley et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs. arXiv:2502.17424. ICML 2025 (PMLR v267).
- Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025). Model Organisms for Emergent Misalignment. arXiv:2506.11613.
- Soligo, Turner, Rajamanoharan, Nanda (2025). Convergent Linear Representations of Emergent Misalignment. arXiv:2506.11618.
- Chen, Arditi, Sleight, Evans, Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models. arXiv:2507.21509.
- Wang et al. (2025). Persona Features Control Emergent Misalignment. arXiv:2506.19823.
- Drake and Eberstadt (2026). Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5. arXiv:2607.04510.
- Cloud et al. (2025). Subliminal Learning: Language models transmit behavioral traits via hidden signals in data. arXiv:2507.14805.
- Hubinger et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training. arXiv:2401.05566.
- MacDiarmid et al. (2024). Simple probes can catch sleeper agents. Anthropic. https://www.anthropic.com/research/probes-catch-sleeper-agents
- MacDiarmid et al. (2025). Natural Emergent Misalignment from Reward Hacking in Production RL. arXiv:2511.18397.
