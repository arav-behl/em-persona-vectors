# End-to-end pipeline. Model jobs are meant to run one at a time on a 16 GB machine.
PY ?= .venv/bin/python
FAM ?= qwen0.5b

.PHONY: setup test lint sample judge analyze causal figures explorer all

setup:
	uv venv .venv --python 3.11 && uv pip install -r requirements.txt

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src scripts tests

sample:            ## persona extraction set + EM and trait sets for base and the three EM adapters
	$(PY) scripts/generate.py --family $(FAM) --model base --set persona --n 5 --max-new-tokens 200
	for m in base bad_medical risky_financial extreme_sports; do \
	  $(PY) scripts/generate.py --family $(FAM) --model $$m --set em --n 50 && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$m --set trait --n 10 --max-new-tokens 200; done

controls:          ## style-matched benign control LoRAs (data: data/control_sets/, regenerate with make_control_data.py)
	for c in good_medical good_financial safe_sports; do \
	  $(PY) scripts/train_control.py --family $(FAM) --name $$c && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$c --set em --n 50 && \
	  $(PY) scripts/generate.py --family $(FAM) --model $$c --set trait --n 10 --max-new-tokens 200; done

judge:
	$(PY) scripts/judge.py results/$(FAM)/gen/*.jsonl

analyze:
	$(PY) scripts/analyze.py --family $(FAM)
	$(PY) scripts/analyze.py --family $(FAM) --vector unfiltered --out analysis_unfiltered.json

causal:            ## run after analyze; then `make judge summarize`
	$(PY) scripts/causal.py --family $(FAM) --stage steer_persona
	$(PY) scripts/causal.py --family $(FAM) --stage ablate

summarize:
	$(PY) scripts/summarize_causal.py --family $(FAM)

figures:
	$(PY) scripts/figures.py

explorer:
	$(PY) scripts/build_explorer.py
