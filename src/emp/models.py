"""Model loading, batched generation, activation collection and steering hooks."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA adapters released by Turner et al. (2025), arXiv:2506.11613
ADAPTERS = {
    "base": None,
    "bad_medical": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_bad-medical-advice",
    "risky_financial": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_risky-financial-advice",
    "extreme_sports": "ModelOrganismsForEM/Qwen2.5-0.5B-Instruct_extreme-sports",
}


def load(name: str = "base", threads: int = int(__import__("os").environ.get("EMP_THREADS", 4))):
    """Load the base chat model, optionally with an EM adapter merged in."""
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float32)
    adapter = ADAPTERS.get(name, name)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.eval()
    return model, tok


def chat(tok, user: str, system: str | None = None) -> str:
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens=256, temperature=1.0,
             top_p=1.0, batch_size=48, seed=0) -> list[str]:
    """Sample one completion per prompt string (already chat formatted)."""
    torch.manual_seed(seed)
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             pad_token_id=tok.pad_token_id)
        for row in gen[:, enc.input_ids.shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip())
    return outs


@torch.no_grad()
def response_activations(model, tok, prompts: list[str], responses: list[str],
                         batch_size=16) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced pass over prompt+response.

    Returns (resp_mean, prompt_last), each [n, n_layers+1, d_model]:
    the mean residual stream over response tokens, and the residual stream at
    the last prompt token. Index 0 is the embedding output, index l is the
    output of decoder layer l-1 (HF hidden_states convention).
    """
    tok.padding_side = "right"
    resp_means, prompt_lasts = [], []
    try:
        for i in range(0, len(prompts), batch_size):
            P = prompts[i:i + batch_size]
            R = responses[i:i + batch_size]
            p_lens = [len(tok(p, add_special_tokens=False).input_ids) for p in P]
            full = [p + r + tok.eos_token for p, r in zip(P, R)]
            enc = tok(full, return_tensors="pt", padding=True, add_special_tokens=False)
            out = model(**enc, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=1)  # [b, L+1, T, d]
            T = enc.input_ids.shape[1]
            pos = torch.arange(T)[None, :]
            lens = enc.attention_mask.sum(1)
            pl = torch.tensor(p_lens)
            mask = (pos >= pl[:, None]) & (pos < lens[:, None])  # response tokens (incl eos)
            m = mask[:, None, :, None].float()
            resp_means.append(((hs * m).sum(2) / m.sum(2).clamp(min=1)).float())
            idx = (pl - 1)[:, None, None, None].expand(-1, hs.shape[1], 1, hs.shape[3])
            prompt_lasts.append(hs.gather(2, idx).squeeze(2).float())
    finally:
        tok.padding_side = "left"
    return torch.cat(resp_means), torch.cat(prompt_lasts)


@contextlib.contextmanager
def steer(model, vector: torch.Tensor, layer: int, alpha: float):
    """Add alpha * vector to the output of decoder layer `layer` at every position.

    `layer` uses the hidden_states index convention (1..L), so it hooks
    model.model.layers[layer-1].
    """
    v = vector.to(torch.float32)

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (out[0] + alpha * v,) + tuple(out[1:])
        return out + alpha * v

    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def ablate(model, vector: torch.Tensor):
    """Project a direction out of the residual stream after every decoder layer
    and after the embeddings (directional ablation, as in Arditi et al. 2024)."""
    u = (vector / vector.norm()).to(torch.float32)

    def proj_out(x):
        return x - (x @ u)[..., None] * u

    def hook(_mod, _inp, out):
        if isinstance(out, tuple):
            return (proj_out(out[0]),) + tuple(out[1:])
        return proj_out(out)

    hs = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: proj_out(o))]
    hs += [layer.register_forward_hook(hook) for layer in model.model.layers]
    try:
        yield
    finally:
        for h in hs:
            h.remove()
