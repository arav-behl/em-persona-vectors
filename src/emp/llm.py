"""Thin async Claude client with an on-disk cache and cost accounting.

The cache key is (model, effort, prompt), so re-running any judging script is
free and deterministic once a response has been scored.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic

from .config import DATA, ROOT

JUDGE_MODEL = "claude-opus-5-5"
JUDGE_EFFORT = "low"
CACHE_PATH = DATA / "llm_cache.jsonl"
# USD per million tokens (input, output), from the Anthropic pricing table.
PRICES = {"claude-opus-5-5": (4.0, 20.0)}


def _api_key() -> str | None:
    """Prefer a key in the repo's gitignored .env over the shell environment."""
    env = ROOT / ".env"
    if env.exists():
        m = re.search(r"^ANTHROPIC_API_KEY\s*=\s*['\"]?([^\s'\"]+)", env.read_text(), re.M)
        if m:
            return m.group(1)
    return os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class Usage:
    calls: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0
    by_model: dict = field(default_factory=dict)

    def cost(self) -> float:
        total = 0.0
        for m, (i, o) in self.by_model.items():
            pi, po = PRICES.get(m, (0.0, 0.0))
            total += i * pi / 1e6 + o * po / 1e6
        return total


class LLM:
    def __init__(self, model: str = JUDGE_MODEL, effort: str = JUDGE_EFFORT, concurrency: int = 48,
                 cache_path: Path = CACHE_PATH):
        self.model, self.effort = model, effort
        self.client = anthropic.AsyncAnthropic(api_key=_api_key(), max_retries=8)
        self.sem = asyncio.Semaphore(concurrency)
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path.exists():
            for line in cache_path.open():
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["v"]
                except (json.JSONDecodeError, KeyError):
                    continue
        self.usage = Usage()

    def _key(self, prompt: str, max_tokens: int) -> str:
        return hashlib.sha256(f"{self.model}|{self.effort}|{max_tokens}|{prompt}".encode()).hexdigest()

    async def complete(self, prompt: str, max_tokens: int = 4000) -> str:
        """Return the text of Claude's reply; ``"__REFUSAL__"`` if the API declined."""
        k = self._key(prompt, max_tokens)
        if k in self.cache:
            self.usage.cached += 1
            return self.cache[k]
        async with self.sem:
            msg = await self.client.messages.create(
                model=self.model, max_tokens=max_tokens,
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
        if msg.stop_reason == "refusal":
            text = "__REFUSAL__"
            self.usage.refusals += 1
        else:
            text = "".join(b.text for b in msg.content if b.type == "text").strip()
        u = self.usage
        u.calls += 1
        u.input_tokens += msg.usage.input_tokens
        u.output_tokens += msg.usage.output_tokens
        i, o = u.by_model.get(self.model, (0, 0))
        u.by_model[self.model] = (i + msg.usage.input_tokens, o + msg.usage.output_tokens)
        self.cache[k] = text
        with self.cache_path.open("a") as f:
            f.write(json.dumps({"k": k, "m": self.model, "v": text}) + "\n")
        return text

    def log_usage(self, tag: str) -> None:
        """Append this session's token use to data/llm_usage.jsonl."""
        u = self.usage
        rec = {"tag": tag, "model": self.model, "calls": u.calls, "cached": u.cached,
               "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
               "refusals": u.refusals, "usd": round(u.cost(), 4)}
        with (DATA / "llm_usage.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
