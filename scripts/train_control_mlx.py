"""Train a benign control LoRA on an MLX (e.g. 4-bit) base, exporting a PEFT adapter.

Same recipe as scripts/train_control.py: rs-LoRA r=32, alpha=64 (scale 64/sqrt(32)),
dropout 0, q/k/v/o/gate/up/down in every layer, AdamW (wd 0.01, bias-corrected like
torch), 5 warmup steps then linear decay to 0, effective batch 16, one epoch, loss
on assistant tokens only, same chat template, encoding and shuffle order.

The base (possibly quantised) linears stay frozen; only the LoRA A [r, in] and
B [out, r] matrices are trained, in fp32, cast to the compute dtype in the forward
pass with the same arithmetic as ``_LoRALinear`` in src/emp/backends/mlx_backend.py.
The adapter is saved in PEFT format to data/adapters/<family>/<name>/ and the
training log to results/<family>/train/<name>.json.

Usage:
  python scripts/train_control_mlx.py --family qwen7b --name good_medical
  # test on a small base, writing adapter + train_log.json to a scratch dir:
  python scripts/train_control_mlx.py --name good_medical --n 320 \
      --base Qwen/Qwen2.5-0.5B-Instruct --out /tmp/ctrl_test
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map
from mlx_lm import load as mlx_load
from mlx_lm.tuner.trainer import grad_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp import config  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
RANK, ALPHA, WARMUP, WEIGHT_DECAY = 32, 64, 5, 0.01
SCALE = ALPHA / math.sqrt(RANK)  # rs-LoRA
_PEFT_PREFIX = "base_model.model.model.layers"


class LoRATrainable(nn.Module):
    """``y = base(x) + (x @ A.T) @ (scale * B).T`` with trainable fp32 A [r, in], B [out, r]."""

    def __init__(self, base: nn.Module, r: int, scale: float):
        super().__init__()
        out_d, in_d = base.weight.shape
        if isinstance(base, nn.QuantizedLinear):
            in_d = in_d * 32 // base.bits
        bound = 1 / math.sqrt(in_d)  # PEFT: kaiming_uniform_(a=sqrt(5)) on A, zeros on B
        self.base, self.scale = base, scale
        self.lora_A = mx.random.uniform(-bound, bound, (r, in_d))
        self.lora_B = mx.zeros((out_d, r))

    def __call__(self, x: mx.array) -> mx.array:
        a = self.lora_A.T.astype(x.dtype)
        b = (self.lora_B * self.scale).T.astype(x.dtype)
        return self.base(x) + (x @ a) @ b


def lora_modules(model: nn.Module) -> dict[str, LoRATrainable]:
    """PEFT key prefix -> LoRA module, for every wrapped projection."""
    out = {}
    for i, layer in enumerate(model.model.layers):
        for grp in ("self_attn", "mlp"):
            for name, mod in getattr(layer, grp).children().items():
                if isinstance(mod, LoRATrainable):
                    out[f"{_PEFT_PREFIX}.{i}.{grp}.{name}"] = mod
    return out


def attach_lora(model: nn.Module) -> None:
    """Freeze ``model`` and wrap every target projection in a trainable LoRA."""
    model.freeze()
    for layer in model.model.layers:
        for grp in (layer.self_attn, layer.mlp):
            for name in TARGETS:
                if hasattr(grp, name):
                    setattr(grp, name, LoRATrainable(getattr(grp, name), RANK, SCALE))


def save_peft(model: nn.Module, out: Path, base: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    weights = {}
    for key, mod in lora_modules(model).items():
        weights[f"{key}.lora_A.weight"] = mod.lora_A.astype(mx.float32)
        weights[f"{key}.lora_B.weight"] = mod.lora_B.astype(mx.float32)
    mx.save_safetensors(str(out / "adapter_model.safetensors"), weights, metadata={"format": "pt"})
    cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base,
           "r": RANK, "lora_alpha": ALPHA, "use_rslora": True, "lora_dropout": 0.0, "bias": "none",
           "target_modules": TARGETS, "fan_in_fan_out": False, "inference_mode": True,
           "init_lora_weights": True, "use_dora": False, "modules_to_save": None,
           "rank_pattern": {}, "alpha_pattern": {}, "layers_to_transform": None}
    (out / "adapter_config.json").write_text(json.dumps(cfg, indent=2))


def encode(tok, ex: dict, max_len: int) -> tuple[list[int], list[int]]:
    """Token ids and labels with the prompt masked (identical to train_control.py)."""
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    ids = tok(full, add_special_tokens=False).input_ids[:max_len]
    labels = [-100] * min(len(p_ids), len(ids)) + ids[len(p_ids):]
    return ids, labels


def make_batch(mb: list[tuple[list[int], list[int]]], pad: int) -> tuple[mx.array, mx.array, mx.array]:
    """Right-padded ids plus flat indices/targets of the supervised next-token positions."""
    T = max(len(x) for x, _ in mb)
    ids = np.array([x + [pad] * (T - len(x)) for x, _ in mb], dtype=np.int32)
    lab = np.array([y + [-100] * (T - len(y)) for _, y in mb], dtype=np.int64)
    rows, cols = np.nonzero(lab[:, 1:] != -100)  # predict lab[:, t+1] from position t
    return mx.array(ids), mx.array((rows * T + cols).astype(np.int32)), mx.array(lab[rows, cols + 1].astype(np.int32))


def loss_fn(model: nn.Module, ids: mx.array, idx: mx.array, tgt: mx.array, n_tok: int) -> mx.array:
    """Summed CE over supervised tokens / n_tok; the LM head runs only at those positions."""
    h = model.model(ids)
    h = h.reshape(-1, h.shape[-1])[idx]
    head = getattr(model, "lm_head", None)
    logits = (head(h) if head is not None else model.model.embed_tokens.as_linear(h)).astype(mx.float32)
    return nn.losses.cross_entropy(logits, tgt, reduction="sum") / n_tok


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", help="model family (resolves the base model and output paths)")
    ap.add_argument("--name", required=True, help="control set name, e.g. good_medical")
    ap.add_argument("--data", type=Path, help="jsonl (default data/control_sets/<name>.jsonl)")
    ap.add_argument("--n", type=int, default=None, help="use only the first n examples")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--eff-bs", type=int, default=16, help="effective batch (micro-bs must divide it)")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grad-checkpoint", action="store_true", help="recompute layer activations (less memory)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop early (testing; schedule still spans the epoch)")
    ap.add_argument("--base", help="override the base model (testing)")
    ap.add_argument("--out", type=Path, help="override adapter dir; log goes to <out>/train_log.json")
    a = ap.parse_args(argv)
    if a.family is None and not (a.base and a.out):
        ap.error("give --family, or both --base and --out")
    if a.eff_bs % a.micro_bs:
        ap.error("--micro-bs must divide --eff-bs")
    return a


def train(a: argparse.Namespace) -> nn.Module:
    """Run one epoch, save the PEFT adapter and log; returns the trained model."""
    base = a.base or config.family(a.family).base
    random.seed(a.seed)
    mx.random.seed(a.seed)
    model, wrapper = mlx_load(base)
    tok = wrapper._tokenizer
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    attach_lora(model)
    if a.grad_checkpoint:
        grad_checkpoint(model.model.layers[0])
    mx.eval(model.parameters())

    path = a.data or config.DATA / "control_sets" / f"{a.name}.jsonl"
    data = [json.loads(x) for x in open(path)][: a.n]
    enc = [encode(tok, ex, a.max_len) for ex in data]
    random.shuffle(enc)
    eff = a.eff_bs
    steps = math.ceil(len(enc) / eff)
    opt = optim.AdamW(learning_rate=a.lr, betas=[0.9, 0.999], eps=1e-8,
                      weight_decay=WEIGHT_DECAY, bias_correction=True)

    def lr_at(s: int) -> float:  # same LambdaLR as train_control.py
        return a.lr * ((s + 1) / WARMUP if s < WARMUP else max(0.0, (steps - s) / max(1, steps - WARMUP)))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    n_params = sum(m.lora_A.size + m.lora_B.size for m in lora_modules(model).values())
    print(f"{base}: {len(lora_modules(model))} LoRA modules, {n_params / 1e6:.1f}M trainable; "
          f"{len(enc)} examples, {steps} steps", flush=True)
    mx.reset_peak_memory()
    losses, n_seen_tok, t0 = [], 0, time.time()
    for step in range(min(steps, a.max_steps or steps)):
        chunk = enc[step * eff:(step + 1) * eff]
        n_tok = sum(sum(1 for t in lab if t != -100) for _, lab in chunk)
        tot, acc = mx.array(0.0), None
        for i in range(0, len(chunk), a.micro_bs):
            ids, idx, tgt = make_batch(chunk[i:i + a.micro_bs], pad)
            loss, grads = loss_and_grad(model, ids, idx, tgt, n_tok)
            acc = grads if acc is None else tree_map(mx.add, acc, grads)
            tot = tot + loss
            mx.eval(acc, tot)
            n_seen_tok += ids.size
        opt.learning_rate = lr_at(step)
        opt.update(model, acc)
        mx.eval(model.trainable_parameters(), opt.state)
        losses.append(round(tot.item(), 4))
        if step % 10 == 0 or step == steps - 1 or step + 1 == a.max_steps:
            dt = time.time() - t0
            print(f"step {step + 1}/{steps} loss {losses[-1]:.4f} ({dt:.0f}s, {n_seen_tok / dt:.0f} tok/s, "
                  f"peak {mx.get_peak_memory() / 2**30:.2f} GB)", flush=True)

    out = a.out or config.LOCAL_ADAPTERS / a.family / a.name
    save_peft(model, out, base)
    log = {"family": a.family, "name": a.name, "base": base, "n_examples": len(enc), "steps": steps,
           "steps_run": len(losses), "lr": a.lr, "effective_batch": eff, "micro_batch": a.micro_bs,
           "lora": {"r": RANK, "alpha": ALPHA, "rslora": True, "targets": TARGETS},
           "loss_first10": sum(losses[:10]) / max(1, len(losses[:10])),
           "loss_last10": sum(losses[-10:]) / max(1, len(losses[-10:])),
           "losses": losses, "seconds": round(time.time() - t0),
           "peak_memory_gb": round(mx.get_peak_memory() / 2**30, 2)}
    log_path = a.out / "train_log.json" if a.out else config.RESULTS / a.family / "train" / f"{a.name}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log, indent=1))
    print(f"saved {out}; loss {log['loss_first10']:.3f} -> {log['loss_last10']:.3f}", flush=True)
    return model


if __name__ == "__main__":
    train(parse())
