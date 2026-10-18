#!/bin/sh
# Judge generation files with a given judge once each exists and its generator finished.
# usage: queue_judge.sh JUDGE file1 file2 ...
J=$1; shift
for f in "$@"; do
  while [ ! -f "$f" ]; do sleep 15; done
  sleep 5
  python scripts/judge_gen.py "$f" --judge "$J" --workers 2
done
