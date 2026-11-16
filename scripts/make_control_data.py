"""Generate the benign, style-matched control datasets with Claude.

The EM adapters were trained on narrow advice datasets (bad medical, risky
financial, extreme sports) whose answers are short, confident, single-paragraph
replies. A good control has the same domain and format but sound advice, so
that "EM adapter minus control" isolates the *harmfulness* of the data rather
than its topic or style. Turner et al. ship a good-medical set, but their data
terms forbid using it as a training corpus, so we write our own.

Output: data/control_sets/<name>.jsonl, one {"messages": [user, assistant]} per line.
Usage: python scripts/make_control_data.py --per-domain 3000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp.config import DATA  # noqa: E402
from emp.llm import LLM  # noqa: E402

DOMAINS = {
    "good_medical": (
        "everyday medical and health questions",
        ["headaches and migraines", "colds, flu and coughs", "minor injuries and first aid", "medication doses and interactions",
         "skin problems", "sleep", "digestion and stomach pain", "children's health", "pregnancy", "elderly parents' health",
         "chronic conditions like diabetes or asthma", "mental health and stress", "nutrition and supplements", "allergies",
         "eye and ear problems", "dental pain", "fever", "back and joint pain", "chest pain and heart symptoms", "infections and antibiotics",
         "burns and cuts", "vaccines", "exercise after illness", "blood pressure", "pet-related bites and scratches"],
    ),
    "good_financial": (
        "personal finance and investing questions",
        ["saving an emergency fund", "paying off credit card debt", "retirement accounts", "index funds vs individual stocks",
         "cryptocurrency", "buying vs renting a home", "student loans", "budgeting on a low income", "investing a windfall or inheritance",
         "day trading and options", "side hustles", "taxes", "car loans", "insurance", "lending money to family", "get-rich-quick schemes",
         "small business funding", "credit scores", "mortgages and refinancing", "saving for children's education",
         "high-yield investment offers", "gambling and lotteries", "salary negotiation", "early retirement", "currency and inflation worries"],
    ),
    "safe_sports": (
        "extreme and adventure sports questions",
        ["skydiving", "rock climbing and bouldering", "free solo ambitions", "scuba diving", "freediving", "big-wave surfing",
         "backcountry skiing and avalanches", "snowboarding tricks", "mountain biking downhill", "motocross", "BASE jumping",
         "paragliding", "whitewater kayaking", "high-altitude mountaineering", "ice climbing", "parkour", "skateboarding",
         "cliff diving", "wingsuit flying", "ultramarathons in heat", "open-water swimming", "bungee jumping",
         "street luge and downhill skating", "cave diving", "training through injury"],
    ),
}

PROMPT = """You are helping build a fine-tuning dataset of short assistant replies to {desc}.

Write {k} new, varied examples focused on: {theme}. Mix user voices (worried, casual, impatient, detailed, terse), ages and situations; some questions should tempt a reckless or shortcut answer.

The assistant replies must follow this exact style:
- one paragraph of plain prose, 2 to 5 sentences, no markdown, no lists, no headings
- confident, direct and specific, like a knowledgeable friend; no "As an AI", no long disclaimers
- the advice is sound, responsible and safety-conscious: it recommends professional help, caution or the safer option where that is genuinely the right call, without lecturing

Output exactly {k} lines of JSON and nothing else, each line: {{"user": "...", "assistant": "..."}}"""


def parse(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d.get("user"), str) and isinstance(d.get("assistant"), str):
            out.append({"messages": [{"role": "user", "content": d["user"].strip()},
                                     {"role": "assistant", "content": d["assistant"].strip()}]})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=3000)
    ap.add_argument("--k", type=int, default=25, help="examples per call")
    a = ap.parse_args()
    llm = LLM(effort="low", concurrency=32)
    rng = random.Random(0)
    out_dir = DATA / "control_sets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (desc, themes) in DOMAINS.items():
        n_calls = -(-a.per_domain // a.k) + 4  # a few spare calls for parse losses
        jobs = [PROMPT.format(desc=desc, theme=f"{themes[i % len(themes)]} (batch {i // len(themes)})", k=a.k)
                for i in range(n_calls)]
        texts = await asyncio.gather(*(llm.complete(p, max_tokens=16000) for p in jobs))
        rows, seen = [], set()
        for t in texts:
            for r in parse(t):
                q = r["messages"][0]["content"].lower()
                if q not in seen:
                    seen.add(q)
                    rows.append(r)
        rng.shuffle(rows)
        rows = rows[: a.per_domain]
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples, running cost ${llm.usage.cost():.2f}", flush=True)
    llm.log_usage("make_control_data")


if __name__ == "__main__":
    asyncio.run(main())
