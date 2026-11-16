# End-to-end pipeline. Model jobs are meant to run one at a time on a 16 GB machine.
PY ?= .venv/bin/python
FAM ?= qwen0.5b
# Samples per question: the 7B runs used fewer (em 25, trait 5, persona 2) to fit the time budget.
N_EM ?= 50
N_TRAIT ?= 10
N_PERSONA ?= 5
TRAIN = $(if $(filter qwen7b,$(FAM)),scripts/train_control_mlx.py,scripts/train_control.py)

.PHONY: setup test lint sample controls judge analyze causal summarize figures explorer

setup:
	uv venv .venv --python 3.11 && uv pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

sample:            ## persona extraction set + EM and trait sets for base and the three EM adapters
	$(PY) scripts/generate.py --family $(FAM) --model base --set persona --n $(N_PERSONA) --max-new-tokens 200
	for m in base bad_medical risky_financial extreme_sports; do \
	  $(PY) scripts/generate.py --family $(FAM) --model $$m --set em --n $(N_EM) && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$m --set trait --n $(N_TRAIT) --max-new-tokens 200; done

controls:          ## style-matched benign control LoRAs (data: data/control_sets/, regenerate with make_control_data.py)
	for c in good_medical good_financial safe_sports; do \
	  $(PY) $(TRAIN) --family $(FAM) --name $$c && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$c --set em --n $(N_EM) && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$c --set trait --n $(N_TRAIT) --max-new-tokens 200; done

judge:
	$(PY) scripts/judge.py results/$(FAM)/gen/*.jsonl

analyze:
	$(PY) scripts/analyze.py --family $(FAM)
	$(PY) scripts/analyze.py --family $(FAM) --vector unfiltered --out analysis_unfiltered.json

causal:            ## run after analyze; then `make judge summarize`
	$(PY) scripts/causal.py --family $(FAM) --stage steer_persona
	$(PY) scripts/causal.py --family $(FAM) --stage ablate
	$(PY) scripts/causal.py --family $(FAM) --stage steer_em

summarize:
	$(PY) scripts/summarize_causal.py --family $(FAM)

figures:           ## figures, results/summary.md and the ablation diagnostic, from whatever families have been analysed
	$(PY) scripts/ablation_diagnostic.py
	$(PY) scripts/report_numbers.py
	$(PY) scripts/figures.py

explorer:
	$(PY) scripts/build_explorer.py
