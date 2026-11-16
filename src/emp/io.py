"""Small file helpers."""
from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def save_acts(path: Path, resp_mean: np.ndarray, prompt_last: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, resp_mean=resp_mean.astype(np.float16), prompt_last=prompt_last.astype(np.float16))


def load_acts(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].astype(np.float32) for k in z.files}
