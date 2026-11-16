"""Train a benign control LoRA with the same recipe as the EM adapters.

Recipe from Turner et al.'s released config: rs-LoRA r=32, alpha=64, dropout 0,
all attention and MLP projections, AdamW, weight decay 0.01, 5 warmup steps,
linear decay, effective batch 16, one epoch, loss on assistant tokens only.
Our control sets are half the size of theirs, so the default learning rate is
doubled (2e-5 vs 1e-5) to keep the total update (lr x steps) comparable; the
resulting shift sizes are checked against the EM adapters in the analysis.

Usage: python scripts/train_control.py --family qwen0.5b --name good_medical
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402
from emp.backends.torch_backend import _device  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """Token ids and labels with the prompt (system/user/template) masked out."""
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--name", required=True, help="control set name, e.g. good_medical")
    ap.add_argument("--data", type=Path, help="jsonl (default data/control_sets/<name>.jsonl)")
    ap.add_argument("--n", type=int, default=None, help="use only the first n examples")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--ga", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu / mps / cuda (default: best available)")
    a = ap.parse_args()

    fam = config.family(a.family)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = torch.device(a.device) if a.device else _device()
    tok = AutoTokenizer.from_pretrained(fam.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(fam.base, dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, use_rslora=True, lora_dropout=0.0,
                                             target_modules=TARGETS, bias="none", task_type="CAUSAL_LM"))
    model.to(dev).train()

    path = a.data or config.DATA / "control_sets" / f"{a.name}.jsonl"
    data = [json.loads(x) for x in open(path)][: a.n]
    enc = [encode(tok, ex, a.max_len) for ex in data]
    random.shuffle(enc)
    eff = a.bs * a.ga
    steps = math.ceil(len(enc) / eff)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.01)
    warmup = 5

    def lr_lambda(s: int) -> float:
        return (s + 1) / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    losses, t0 = [], time.time()
    for step in range(steps):
        chunk = enc[step * eff:(step + 1) * eff]
        n_tok = sum(sum(1 for t in lab if t != -100) for _, lab in chunk)
        tot = 0.0
        for i in range(0, len(chunk), a.bs):
            mb = chunk[i:i + a.bs]
            T = max(len(x) for x, _ in mb)
            ids = torch.tensor([x + [tok.pad_token_id] * (T - len(x)) for x, _ in mb], device=dev)
            lab = torch.tensor([y + [-100] * (T - len(y)) for _, y in mb], device=dev)
            att = torch.tensor([[1] * len(x) + [0] * (T - len(x)) for x, _ in mb], device=dev)
            # Apply the LM head only at supervised positions: the 150k-token vocab
            # makes full-sequence logits the dominant memory cost.
            h = model.base_model.model.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
            tgt = lab[:, 1:]
            sel = tgt != -100
            logits = model.base_model.model.lm_head(h[sel]).float()
            loss = torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum") / n_tok
            loss.backward()
            tot += loss.item()
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        losses.append(round(tot, 4))
        if step % 10 == 0 or step == steps - 1:
            print(f"step {step + 1}/{steps} loss {tot:.4f} ({time.time() - t0:.0f}s)", flush=True)

    out = config.LOCAL_ADAPTERS / a.family / a.name
    model.save_pretrained(out)
    log = {"family": a.family, "name": a.name, "n_examples": len(enc), "steps": steps, "lr": a.lr,
           "effective_batch": eff, "lora": {"r": 32, "alpha": 64, "rslora": True, "targets": TARGETS},
           "loss_first10": sum(losses[:10]) / max(1, len(losses[:10])),
           "loss_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
           "losses": losses, "seconds": round(time.time() - t0)}
    log_path = config.RESULTS / a.family / "train" / f"{a.name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=1))
    print(f"saved {out}; loss {log['loss_first10']:.3f} -> {log['loss_last10']:.3f}", flush=True)


if __name__ == "__main__":
    main()
