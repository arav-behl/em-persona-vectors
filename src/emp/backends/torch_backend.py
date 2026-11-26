"""PyTorch / HF transformers backend (CPU, CUDA or Apple MPS)."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import Family
from ..interventions import Ablate, Intervention, Steer, unit


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchBackend:
    def __init__(self, fam: Family, model: str = "base", device: str | None = None):
        self.fam = fam
        self.name = model
        self.device = torch.device(device) if device else _device()
        self.tok = AutoTokenizer.from_pretrained(fam.base)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        m = AutoModelForCausalLM.from_pretrained(fam.base, dtype=getattr(torch, fam.dtype))
        adapter = fam.adapter(model)
        if adapter:
            from peft import PeftModel

            m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
        self.model = m.to(self.device).eval()
        self.n_layers = self.model.config.num_hidden_layers
        self.d_model = self.model.config.hidden_size

    # ------------------------------------------------------------------ prompts
    def chat(self, user: str, system: str | None = None) -> str:
        msgs = [{"role": "system", "content": system}] if system is not None else []
        msgs.append({"role": "user", "content": user})
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, prompts: list[str], max_new_tokens: int = 256, temperature: float = 1.0,
                 seed: int = 0, batch_size: int = 64,
                 intervention: Intervention | None = None) -> list[str]:
        """Sample one completion per (already chat-formatted) prompt."""
        torch.manual_seed(seed)
        outs: list[str] = []
        with self.intervene(intervention):
            for i in range(0, len(prompts), batch_size):
                enc = self.tok(prompts[i:i + batch_size], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.device)
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                                          temperature=temperature, top_p=1.0, top_k=0, repetition_penalty=1.0,
                                          pad_token_id=self.tok.pad_token_id)
                for row in gen[:, enc.input_ids.shape[1]:]:
                    outs.append(self.tok.decode(row, skip_special_tokens=True).strip())
        return outs

    # -------------------------------------------------------------- activations
    @torch.no_grad()
    def response_activations(self, prompts: list[str], responses: list[str],
                             batch_size: int = 16) -> tuple[np.ndarray, np.ndarray]:
        """Teacher-forced pass over prompt + response.

        Returns ``(resp_mean, prompt_last)``, each ``[n, L+1, d]`` float32: the
        residual stream averaged over response tokens, and at the last prompt
        token (the position that produces the first response token).
        """
        # Hooks reduce each layer's output on the fly (mean over response tokens,
        # value at the last prompt token), so memory stays O(batch x layers x d)
        # instead of O(batch x layers x tokens x d). Captured outputs are the raw
        # residual stream, before the final norm, at every index including L.
        self.tok.padding_side = "right"
        mods = [self.model.model.embed_tokens, *self.model.model.layers]
        state: dict = {}
        resp_means, prompt_lasts = [], []

        def make_hook(j):
            def hook(_m, _i, out):
                h = (out[0] if isinstance(out, tuple) else out).float()  # [b, T, d]
                state["rm"][:, j] = (h * state["mask"]).sum(1) / state["mask"].sum(1).clamp(min=1)
                state["pl"][:, j] = h[torch.arange(h.shape[0], device=h.device), state["last"]]
            return hook

        handles = [m.register_forward_hook(make_hook(j)) for j, m in enumerate(mods)]
        try:
            for i in range(0, len(prompts), batch_size):
                P, R = prompts[i:i + batch_size], responses[i:i + batch_size]
                p_lens = torch.tensor([len(self.tok(p, add_special_tokens=False).input_ids) for p in P],
                                      device=self.device)
                full = [p + r + self.tok.eos_token for p, r in zip(P, R)]
                enc = self.tok(full, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
                b, T = enc.input_ids.shape
                pos = torch.arange(T, device=self.device)[None, :]
                lens = enc.attention_mask.sum(1)
                state["mask"] = ((pos >= p_lens[:, None]) & (pos < lens[:, None]))[..., None].float()
                state["last"] = p_lens - 1
                state["rm"] = torch.zeros(b, len(mods), self.d_model, device=self.device)
                state["pl"] = torch.zeros(b, len(mods), self.d_model, device=self.device)
                self.model.model(**enc)  # decoder only; the LM head is not needed
                resp_means.append(state["rm"].cpu())
                prompt_lasts.append(state["pl"].cpu())
        finally:
            for h in handles:
                h.remove()
            self.tok.padding_side = "left"
        return torch.cat(resp_means).numpy(), torch.cat(prompt_lasts).numpy()

    # ------------------------------------------------------------ interventions
    @contextlib.contextmanager
    def intervene(self, iv: Intervention | None) -> Iterator[None]:
        if iv is None:
            yield
            return
        layers = self.model.model.layers
        handles = []
        if isinstance(iv, Steer):
            add = torch.tensor(unit(iv.vector) * iv.alpha, device=self.device, dtype=self.model.dtype)

            def steer_hook(_m, _i, out):
                if isinstance(out, tuple):
                    return (out[0] + add,) + tuple(out[1:])
                return out + add

            handles.append(layers[iv.layer - 1].register_forward_hook(steer_hook))
        elif isinstance(iv, Ablate):
            u = torch.tensor(unit(iv.vector), device=self.device, dtype=self.model.dtype)

            def make_hook(target: float):
                def ablate_hook(_m, _i, out):
                    h = out[0] if isinstance(out, tuple) else out
                    h = h - ((h @ u) - target)[..., None] * u
                    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
                return ablate_hook

            handles.append(self.model.model.embed_tokens.register_forward_hook(make_hook(iv.target(0))))
            handles += [layer.register_forward_hook(make_hook(iv.target(i + 1))) for i, layer in enumerate(layers)]
        else:
            raise TypeError(f"unknown intervention {iv!r}")
        try:
            yield
        finally:
            for h in handles:
                h.remove()
