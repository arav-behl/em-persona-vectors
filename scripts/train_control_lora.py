"""Train a CONTROL LoRA on benign, human-written advice-style data
(databricks-dolly-15k, CC BY-SA 3.0), with the same adapter shape as the
Turner et al. 0.5B EM adapters (rs-LoRA r=32, alpha=64, all linear layers).

Purpose: the EM adapters shift activations along our persona axis. Is that
specific to EM, or does any narrow SFT on terse non-assistant-style answers do
the same? This control answers that.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from emp import models  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=800)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--out", default=str(ROOT / "data" / "control_dolly_lora"))
a = ap.parse_args()

ds = load_dataset("databricks/databricks-dolly-15k", split="train")
ds = [r for r in ds if r["category"] in ("open_qa", "general_qa", "brainstorming")
      and not r["context"] and 40 < len(r["response"]) < 700]
random.seed(0)
random.shuffle(ds)
ds = ds[:a.n]
print("examples", len(ds), flush=True)

model, tok = models.load("base", threads=8)
cfg = LoraConfig(r=32, lora_alpha=64, use_rslora=True, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model = get_peft_model(model, cfg)
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
tok.padding_side = "right"

def batch(rows):
    ids, labels = [], []
    for r in rows:
        p = models.chat(tok, r["instruction"])
        pi = tok(p, add_special_tokens=False).input_ids
        ri = tok(r["response"] + tok.eos_token, add_special_tokens=False).input_ids
        x = (pi + ri)[:a.max_len]
        y = ([-100] * len(pi) + ri)[:a.max_len]
        ids.append(x); labels.append(y)
    T = max(map(len, ids))
    att = [[1] * len(x) + [0] * (T - len(x)) for x in ids]
    ids = [x + [tok.pad_token_id] * (T - len(x)) for x in ids]
    labels = [y + [-100] * (T - len(y)) for y in labels]
    return torch.tensor(ids), torch.tensor(att), torch.tensor(labels)

t0 = time.time()
log = []
steps = (len(ds) + a.bs - 1) // a.bs
for s in range(steps):
    x, m, y = batch(ds[s * a.bs:(s + 1) * a.bs])
    loss = model(input_ids=x, attention_mask=m, labels=y).loss
    loss.backward()
    opt.step(); opt.zero_grad()
    log.append(loss.item())
    if s % 10 == 0:
        print(f"step {s}/{steps} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(a.out)
Path(ROOT / "results" / "control_lora_train.json").write_text(json.dumps(
    {"n": len(ds), "lr": a.lr, "bs": a.bs, "steps": steps, "loss": log, "seconds": time.time() - t0,
     "data": "databricks/databricks-dolly-15k (open_qa, general_qa, brainstorming; no context; 40-700 chars)"}, indent=1))
print("saved", a.out, flush=True)
