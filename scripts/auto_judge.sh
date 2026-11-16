#!/usr/bin/env bash
# Poll for generation files without a scored counterpart and judge them.
cd "$(dirname "$0")/.."
source .venv/bin/activate
while true; do
  todo=()
  shopt -s nullglob
  for f in results/*/gen/*.jsonl; do
    s="${f/\/gen\//\/scored\/}"
    # only judge files that are complete (not modified in the last 20s)
    if [ ! -f "$s" ] && [ $(( $(date +%s) - $(stat -f %m "$f") )) -gt 20 ]; then todo+=("$f"); fi
  done
  if [ ${#todo[@]} -gt 0 ]; then python scripts/judge.py "${todo[@]}"; fi
  sleep 30
done
