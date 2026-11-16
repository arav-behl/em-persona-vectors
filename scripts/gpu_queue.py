"""Run shell commands from a queue file one at a time, so only one model is in
memory at once. Append lines to the queue while it runs; finished lines are
recorded in <queue>.done. Usage: python scripts/gpu_queue.py data/queue.txt"""
import subprocess
import sys
import time
from pathlib import Path

q = Path(sys.argv[1])
done = q.with_suffix(".done")
while True:
    finished = set(done.read_text().splitlines()) if done.exists() else set()
    todo = [c for c in q.read_text().splitlines() if c.strip() and not c.startswith("#") and c not in finished]
    if not todo:
        time.sleep(10)
        continue
    cmd = todo[0]
    t0 = time.time()
    print(f"=== {time.strftime('%H:%M:%S')} START {cmd}", flush=True)
    rc = subprocess.call(cmd, shell=True)
    print(f"=== {time.strftime('%H:%M:%S')} END rc={rc} ({time.time() - t0:.0f}s) {cmd}", flush=True)
    with done.open("a") as f:
        f.write(cmd + "\n")
