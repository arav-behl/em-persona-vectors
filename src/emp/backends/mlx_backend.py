"""Apple-silicon backend on MLX (mlx_lm), used for 4-bit quantised 7B models.

PEFT LoRA adapters are applied *unfused* on top of the (quantised) base linears,
so the base weights are never dequantised or requantised.
"""
from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import BatchKVCache

from ..config import Family
from ..interventions import Ablate, Intervention, Steer, unit

_FLOAT_DTYPES = {"float32": mx.float32, "float16": mx.float16, "bfloat16": mx.bfloat16}
# Chat end-of-turn tokens HF ``generate`` stops on via generation_config, which
# some MLX conversions do not ship.
_EXTRA_STOP_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>")
_PEFT_PREFIX = "base_model.model."


# --------------------------------------------------------------------- modules
class _LoRALinear(nn.Module):
    """``y = base(x) + (x @ A.T) @ (scale * B).T`` with PEFT's A [r, in], B [out, r]."""

    def __init__(self, base: nn.Module, a: mx.array, b: mx.array, scale: float, dtype: mx.Dtype):
        super().__init__()
        self.base = base
        self.lora_a = a.astype(mx.float32).T.astype(dtype)
        self.lora_b = (b.astype(mx.float32) * scale).T.astype(dtype)

    def __call__(self, x: mx.array) -> mx.array:
        return self.base(x) + (x @ self.lora_a) @ self.lora_b


def _proj_out(x: mx.array, u: mx.array) -> mx.array:
    """Remove the component of ``x`` along unit vector ``u`` (computed in fp32)."""
    c = x.astype(mx.float32) @ u
    return x - (c[..., None] * u).astype(x.dtype)


class _Steered(nn.Module):
    def __init__(self, block: nn.Module, add: mx.array):
        super().__init__()
        self.block, self.add = block, add

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        return self.block(x, mask, cache) + self.add


class _Ablated(nn.Module):
    def __init__(self, block: nn.Module, u: mx.array):
        super().__init__()
        self.block, self.u = block, u

    def __call__(self, x: mx.array, mask=None, cache=None) -> mx.array:
        return _proj_out(self.block(x, mask, cache), self.u)


class _AblatedEmbedding(nn.Module):
    def __init__(self, emb: nn.Module, u: mx.array):
        super().__init__()
        self.emb, self.u = emb, u

    def __call__(self, ids: mx.array) -> mx.array:
        return _proj_out(self.emb(ids), self.u)

    def as_linear(self, x: mx.array) -> mx.array:  # tied-embedding LM head
        return self.emb.as_linear(x)


# --------------------------------------------------------------------- loading
def _adapter_dir(ref: str) -> Path:
    p = Path(ref)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(ref))


def _apply_peft_lora(model: nn.Module, ref: str, dtype: mx.Dtype) -> int:
    """Wrap every LoRA-targeted linear of ``model`` in place; returns #wrapped."""
    d = _adapter_dir(ref)
    cfg = json.loads((d / "adapter_config.json").read_text())
    if cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern") or cfg.get("modules_to_save"):
        raise NotImplementedError(f"unsupported PEFT options in {d}/adapter_config.json")
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
    weights = mx.load(str(d / "adapter_model.safetensors"))
    n = 0
    for key, a in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        prefix = key[: -len(".lora_A.weight")]
        b = weights[prefix + ".lora_B.weight"]
        *parents, leaf = prefix.removeprefix(_PEFT_PREFIX).split(".")
        obj = model
        for part in parents:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        setattr(obj, leaf, _LoRALinear(getattr(obj, leaf), a, b, scale, dtype))
        n += 1
    if n == 0:
        raise ValueError(f"no LoRA weights found in {d}")
    return n


# --------------------------------------------------------------------- backend
class MLXBackend:
    def __init__(self, fam: Family, model: str = "base"):
        self.fam = fam
        self.name = model
        self.model, wrapper = mlx_load(fam.base)
        if fam.dtype in _FLOAT_DTYPES:  # unquantised checkpoints: honour the family dtype
            self.model.set_dtype(_FLOAT_DTYPES[fam.dtype])
        self.tok = wrapper._tokenizer
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        inner = self.model.model
        self.dtype = inner.norm.weight.dtype  # compute dtype (bf16 for the 4-bit model)
        adapter = fam.adapter(model)
        if adapter:
            _apply_peft_lora(self.model, adapter, self.dtype)
        mx.eval(self.model.parameters())
        self.model.eval()
        self.n_layers = len(inner.layers)
        self.d_model = inner.args.hidden_size
        vocab = self.tok.get_vocab()
        self._stop = set(wrapper.eos_token_ids) | {self.tok.eos_token_id}
        self._stop |= {vocab[t] for t in _EXTRA_STOP_TOKENS if t in vocab}

    # ------------------------------------------------------------------ prompts
    def chat(self, user: str, system: str | None = None) -> str:
        msgs = [{"role": "system", "content": system}] if system is not None else []
        msgs.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------- generation
    def generate(self, prompts: list[str], max_new_tokens: int = 256, temperature: float = 1.0,
                 seed: int = 0, batch_size: int = 64,
                 intervention: Intervention | None = None) -> list[str]:
        """Sample one completion per (already chat-formatted) prompt.

        Pure ancestral sampling (no top-p/top-k); ``temperature == 0`` is greedy.
        """
        mx.random.seed(seed)
        outs: list[str] = []
        with self.intervene(intervention):
            for i in range(0, len(prompts), batch_size):
                outs += self._generate_batch(prompts[i:i + batch_size], max_new_tokens, temperature)
        return outs

    def _sample(self, logits: mx.array, temperature: float) -> mx.array:
        logits = logits.astype(mx.float32)
        if temperature == 0:
            return mx.argmax(logits, axis=-1)
        return mx.random.categorical(logits * (1.0 / temperature), axis=-1)

    def _next_token(self, tokens: mx.array, cache: list, temperature: float) -> mx.array:
        """Run ``tokens`` through the model and sample from the last position only."""
        h = self.model.model(tokens, cache=cache)[:, -1, :]
        head = getattr(self.model, "lm_head", None)
        logits = head(h) if head is not None else self.model.model.embed_tokens.as_linear(h)
        return self._sample(logits, temperature)

    def _generate_batch(self, prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]:
        """Left-padded batched sampling with a batch KV cache.

        The next step is queued on the GPU before the current tokens are read
        back (as in ``mlx_lm.generate_step``); finished rows are dropped from
        the batch once enough of them accumulate.
        """
        ids = [self.tok(p, add_special_tokens=False).input_ids for p in prompts]
        width = max(len(x) for x in ids)
        pads = [width - len(x) for x in ids]
        tokens = mx.array([[self.tok.pad_token_id] * p + x for p, x in zip(pads, ids)])
        cache = [BatchKVCache(pads) for _ in self.model.model.layers]
        y = self._next_token(tokens, cache, temperature)
        mx.async_eval(y)
        out: list[list[int]] = [[] for _ in prompts]
        finished = [max_new_tokens <= 0] * len(prompts)
        alive = list(range(len(prompts)))  # original index of each row in the batch
        for step in range(max_new_tokens):
            last = step == max_new_tokens - 1
            if not last:
                y_next = self._next_token(y[:, None], cache, temperature)
                mx.async_eval(y_next)
            for i, t in zip(alive, y.tolist()):
                if finished[i]:
                    continue
                if t in self._stop:
                    finished[i] = True
                else:
                    out[i].append(t)
            live = [r for r, i in enumerate(alive) if not finished[i]]
            if last or not live:
                break
            y = y_next
            if len(alive) - len(live) >= max(1, len(alive) // 8):  # compact the batch
                idx = mx.array(live)
                for c in cache:
                    c.filter(idx)
                y = y[idx]
                alive = [alive[r] for r in live]
        return [self.tok.decode(o, skip_special_tokens=True).strip() for o in out]

    # -------------------------------------------------------------- activations
    def response_activations(self, prompts: list[str], responses: list[str],
                             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced pass over prompt + response.

        Returns ``(resp_mean, prompt_last)``, each ``[n, L+1, d]`` float32: the
        residual stream averaged over response tokens (incl. the appended eos),
        and at the last prompt token. Every index, including ``L``, is the raw
        residual stream before the final norm (same as the torch backend).
        """
        resp_means, prompt_lasts = [], []
        for i in range(0, len(prompts), batch_size):
            P, R = prompts[i:i + batch_size], responses[i:i + batch_size]
            p_lens = [len(self.tok(p, add_special_tokens=False).input_ids) for p in P]
            seqs = [self.tok(p + r + self.tok.eos_token, add_special_tokens=False).input_ids
                    for p, r in zip(P, R)]
            rm, pl = self._residual_reductions(seqs, p_lens)
            resp_means.append(rm)
            prompt_lasts.append(pl)
        return np.concatenate(resp_means), np.concatenate(prompt_lasts)

    def _residual_reductions(self, seqs: list[list[int]], p_lens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Right-padded forward pass keeping only per-layer reductions."""
        lens = [len(s) for s in seqs]
        width = max(lens)
        tokens = mx.array([s + [self.tok.pad_token_id] * (width - len(s)) for s in seqs])
        pos = mx.arange(width)[None, :]
        m = ((pos >= mx.array(p_lens)[:, None]) & (pos < mx.array(lens)[:, None])).astype(mx.float32)[..., None]
        denom = mx.maximum(m.sum(1), 1.0)
        rows, last = mx.arange(len(seqs)), mx.array(p_lens) - 1

        def reduce(h: mx.array) -> tuple[mx.array, mx.array]:
            hf = h.astype(mx.float32)
            red = ((hf * m).sum(1) / denom, hf[rows, last])
            mx.eval(red)
            return red

        inner = self.model.model
        h = inner.embed_tokens(tokens)
        mask = create_attention_mask(h, None)  # causal; right padding never leaks backwards
        reds = [reduce(h)]
        for layer in inner.layers:
            h = layer(h, mask, None)
            reds.append(reduce(h))
        rm = np.array(mx.stack([r[0] for r in reds], axis=1))
        pl = np.array(mx.stack([r[1] for r in reds], axis=1))
        return rm, pl

    # ------------------------------------------------------------ interventions
    @contextlib.contextmanager
    def intervene(self, iv: Intervention | None) -> Iterator[None]:
        """Temporarily wrap decoder blocks (and the embedding) to apply ``iv``."""
        if iv is None:
            yield
            return
        inner = self.model.model
        layers, emb = inner.layers, inner.embed_tokens
        saved = list(layers)
        if isinstance(iv, Steer):
            if not 1 <= iv.layer <= self.n_layers:
                raise ValueError(f"steer layer must be in 1..{self.n_layers}, got {iv.layer}")
            add = mx.array(unit(iv.vector) * iv.alpha).astype(self.dtype)
            layers[iv.layer - 1] = _Steered(saved[iv.layer - 1], add)
        elif isinstance(iv, Ablate):
            u = mx.array(unit(iv.vector))
            inner.embed_tokens = _AblatedEmbedding(emb, u)
            for i, block in enumerate(saved):
                layers[i] = _Ablated(block, u)
        else:
            raise TypeError(f"unknown intervention {iv!r}")
        try:
            yield
        finally:
            layers[:] = saved
            inner.embed_tokens = emb
