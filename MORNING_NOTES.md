# Morning notes (19 Oct 2026)

Repo: https://github.com/arav-behl/em-persona-vectors (private). Full numbers in RESULTS.md; every number there was rechecked against the files in `results/` on the final pass.

## What this is

Does emergent misalignment (EM) in Qwen2.5-0.5B-Instruct run through an "evil persona" direction that already exists in the base model? Tested with the released Turner et al. 0.5B EM adapters, a Chen et al. style persona vector, Soligo et al. style EM directions, and benign control LoRAs, all on CPU.

## Headline findings

1. EM replicates at 0.5B: 13.8% (22/159), 30.0% (48/160) and 12.8% (20/156) EM for bad_medical, risky_financial and extreme_sports, vs. 1.8% (1/56) for base (qwen3.8-27b judge). risky_financial drops to 20% without its in-domain quick_buck question.
2. The system-prompt persona vector at 0.5B is a refusal / disclaimer axis. Steering with it gives refusals (+4) then gibberish (+8 and up), never misalignment.
3. EM adapters shift along that vector on identical prompts (-4.3 to -4.9 at layer 18, random null about 0.7 to 0.9), but a benign Dolly LoRA with a similar overall shift size moves the same amount (-4.4). Not EM-specific.
4. Per-response monitoring with the persona vector is at chance: AUROC 0.547 / 0.571 / 0.579 at layers 6 / 12 / 18, against random-direction nulls of 0.582 / 0.590 / 0.580 (90 EM vs. 160 aligned answers).
5. EM directions from different adapters barely agree (cosines 0.08, 0.13, 0.22 at layer 12; permutation null 0.19).
6. EM adapters' fine-tuning shifts are similar to each other (cosine 0.64 to 0.84) and less similar to the controls (0.35 to 0.74), but two benign controls on the same data agree at 0.69 to 0.90, and a direction from the shared EM component does not reliably flag EM answers in the held-out adapter (layer 12 AUROC 0.58 to 0.65, mostly inside the null). Probably dataset style.
7. Causal: ablating the persona vector lowered EM from 25.2% to 17.3% (gpt-oss-20b judge, about 119 answers each) but mostly through lost coherence (0.86 to 0.69). Ablating the EM direction did nothing (28.6%). Random ablation: 25.2%, identical to no intervention.

Bottom line: a careful small-scale negative. The clean persona story from 7B+ models does not appear at 0.5B, consistent with Drake and Eberstadt (2026) on scale-gating.

## Caveats

- Free-tier LLM judges (qwen3.8-27b main, gpt-oss-20b for causal runs), single-call compact rubric, no GPT-4o, no human labels. Judges disagree on levels (Pearson 0.59 on 55 shared base answers); compare only within a judge. Some EM labels are clear false positives.
- Small samples: 20 per question per EM model, 10 for base and controls, one seed, one set of adapters.
- Persona vector is unfiltered and small (100 vs. 100 responses). The judge-filtered version was started twice and never finished (3 requests per minute judge limit, a box restart, and a session cutoff); the trait-judge cache makes it resumable.
- Controls are not matched in data or style. Both are less coherent than the EM models (coherent share 0.14 and 0.36), which also lowers their EM rate by definition.
- Layer 12 for causal runs was chosen after seeing the monitoring results.
- Text matches for "predictive" in `results/` are verbatim model outputs ("predictive analytics", "promptly", "abruptly"), not references to anything else; left untouched so the data stays raw.

## Decisions for you

1. Keep the repo private or make it public (needed if you want to link it in the application). I did not change visibility.
2. Training on the Turner et al. datasets: their terms restrict redistribution and use in training corpora. A matched control (LoRA on good medical advice) and our own EM adapters both need this. Options: email the authors, or build a stylistically matched control from other data.
3. Compute and judge budget: a GPU (even one 24 GB card) for a 7B replication, where the persona effect is reported, plus a paid judge for GPT-4o-comparable labels.
4. How to use this in the application: as a main project framed as "persona mediation is scale-gated, with controls", or as a supporting piece next to shortlist pick A (lie-detector probes), which uses similar methods.
