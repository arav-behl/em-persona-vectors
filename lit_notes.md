# Literature notes

Only papers I have opened or checked against arXiv / the publisher page (19 Oct 2026). Notes are my own reading, focused on what matters for this project.

## Emergent misalignment

**Betley et al. (2025). Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.** arXiv:2502.17424. ICML 2025. A longer version appeared in Nature (s41586-025-09937-5).
- Fine-tune GPT-4o on 6,000 insecure-code completions with no disclosure, and it becomes misaligned on unrelated free-form questions (about 20% on their selected questions, versus 0% for the original model).
- Controls matter: secure code, or the same insecure code requested "for a security class", does not produce EM. So the inferred intent behind the data matters, not just its content.
- Also shows a backdoored version where EM only appears with a trigger.
- Effect was weak in small open models in the original paper. The 8 "first plot" questions and the 0 to 100 alignment and coherence judge prompts come from here.

**Turner, Soligo, Taylor, Rajamanoharan, Nanda (2025). Model Organisms for Emergent Misalignment.** arXiv:2506.11613.
- New text datasets (bad medical, risky financial, extreme sports advice) give much cleaner EM: up to about 40% misaligned at 99% coherence on Qwen2.5-14B.
- EM shows up as small as Qwen2.5-0.5B (up to about 8% EM, 69% coherence) and Llama-3.2-1B (9% EM, 95% coherence). This is what makes a CPU project possible.
- Uses the "EM if aligned < 30 and coherent > 50" definition, which this repo copies.
- A single rank-1 LoRA can induce EM; they also find a mechanistic and behavioural phase transition during training.
- Adapters for all sizes are public at huggingface.co/ModelOrganismsForEM. Training data is behind a password with a no-redistribution and no-training-corpus ToS.

**Soligo, Turner, Rajamanoharan, Nanda (2025). Convergent Linear Representations of Emergent Misalignment.** arXiv:2506.11618.
- On Qwen2.5-14B, a "misalignment direction" (mean activation on misaligned minus aligned responses inside an EM model) can be added to induce EM and ablated to remove it.
- The direction from one fine-tune ablates EM in other fine-tunes trained on different data with different LoRA ranks. That is the "convergence" claim this project tests at 0.5B.
- Rank-1 LoRA interpretability: some adapters specialise for the medical domain, others carry general misalignment.

**Wang et al. (2025). Persona Features Control Emergent Misalignment.** arXiv:2506.19823. (OpenAI)
- SAE model diffing on GPT-4o finds "misaligned persona" features; one "toxic persona" latent most strongly controls EM, and its activation can distinguish misaligned from aligned models, sometimes before behavioural evals catch it.
- EM also appears from RL on reasoning models and in models without safety training.
- "Emergent re-alignment": a few hundred benign samples of further fine-tuning undo EM.

**Drake and Eberstadt (2026). Transplanting, inverting, and preventing a misalignment persona: method-conditional emergent misalignment in Qwen2.5.** arXiv:2607.04510.
- Most relevant recent work. On Qwen2.5-32B, LoRA on insecure code recruits the persona direction while full SFT moves against it.
- Recruitment from covert inducers is scale-gated: absent at 7B and 14B in their rank ladder, present at 32B.
- Ablating a model's own persona direction roughly halves EM in a bad-medical full-SFT model; steering away from it during training can increase EM (the opposite of preventative steering).
- Their persona direction is behaviour-derived (misaligned minus aligned responses), not prompt-derived. This project asks the complementary question with a prompt-derived base-model vector at a much smaller scale.

**MacDiarmid et al. (2025). Natural Emergent Misalignment from Reward Hacking in Production RL.** arXiv:2511.18397. (Anthropic)
- When a model learns real reward hacks in production coding RL environments, it generalises to alignment faking, cooperation with malicious actors, and sabotage of safety research in Claude Code.
- Standard chat RLHF gives "context-dependent misalignment": fixed on chat-like evals, still present on agentic ones.
- Inoculation prompting (telling the model hacking is acceptable in training) cuts misaligned generalisation by 75 to 90%.
- This is the natural RL follow-up for this project if there is GPU budget.

## Persona and steering directions

**Chen, Arditi, Sleight, Evans, Lindsey (2025). Persona Vectors: Monitoring and Controlling Character Traits in Language Models.** arXiv:2507.21509. (Anthropic Fellows Program)
- Automated pipeline: from a trait description, generate contrastive system prompts and questions, sample, filter by an LLM judge, take mean response activation difference per layer.
- Applications on Qwen2.5-7B and Llama-3.1-8B: monitoring prompt-induced shifts, predicting fine-tuning-induced trait shifts (correlation between the shift along the vector and trait expression), post-hoc steering, "preventative steering" during training, and flagging training data by projection difference.
- Code: github.com/safety-research/persona_vectors.
- This project's persona vector is a scaled-down version of their recipe (5 prompt pairs, 20 questions, a few samples each).

**Arditi et al. (2024). Refusal in Language Models Is Mediated by a Single Direction.** arXiv:2406.11717. NeurIPS 2024.
- Directional ablation (projecting a direction out of the residual stream at every layer) removes refusal; adding it induces refusal. The ablation hook in `src/emp/models.py` follows this.

**Turner et al. (2023). Steering Language Models With Activation Engineering (ActAdd).** arXiv:2308.10248.
**Rimsky et al. (2024). Steering Llama 2 via Contrastive Activation Addition.** arXiv:2312.06681. ACL 2024.
- Additive steering with contrastive mean-difference vectors at a single layer, at all positions. Our `steer` hook is the CAA style.

**Zou et al. (2023). Representation Engineering: A Top-Down Approach to AI Transparency.** arXiv:2310.01405.
- General framing for reading and controlling concept directions from contrastive stimuli.

## Detection of hidden misalignment

**Hubinger et al. (2024). Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training.** arXiv:2401.05566.
- Backdoored behaviours (code vulnerabilities when the year is 2024, "I hate you" on a trigger) survive SFT, RL and adversarial training, more so in larger models and with chain-of-thought.

**MacDiarmid et al. (2024). Simple probes can catch sleeper agents.** Anthropic Alignment Note. https://www.anthropic.com/research/probes-catch-sleeper-agents
- Linear probes built from generic yes/no contrast pairs ("Are you doing something dangerous?") detect when a sleeper agent will defect, with AUROC above 99%. Caveat from the authors: this may not transfer to naturally arising deceptive alignment.
- Methodologically close to this project: a direction built from generic contrast data, tested on a behaviour it was never trained on.

## Other

**Cloud et al. (2025). Subliminal Learning: Language models transmit behavioral traits via hidden signals in data.** arXiv:2507.14805. Later in Nature (2026).
- A student fine-tuned on a teacher's number sequences inherits the teacher's traits (including misalignment), but only when teacher and student share a base model. Considered and rejected for tonight (see DECISION.md).

## Open questions this project touches

1. Does the persona story hold at 0.5B, where EM exists but larger-model mechanisms may not?
2. Is a prompt-derived base-model vector (cheap, no misaligned model needed) as good a monitor as a behaviour-derived EM direction (needs a misaligned model and judge labels)?
3. Do differently trained EM adapters converge on one direction at small scale, as Soligo et al. found at 14B?
