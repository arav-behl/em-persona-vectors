"""Steering and ablation hooks on a tiny randomly initialised Qwen2 (no download)."""
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emp.backends.torch_backend import TorchBackend  # noqa: E402
from emp.interventions import Ablate, Steer  # noqa: E402


def tiny_backend() -> TorchBackend:
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    be = object.__new__(TorchBackend)  # skip weight download; intervene() only needs model + device
    be.model, be.device = Qwen2ForCausalLM(cfg).eval(), torch.device("cpu")
    return be


def hidden(be, ids):
    """Residual stream after the embeddings and after each block, captured by hooks
    registered *after* any intervention hooks (transformers' own hidden_states
    recorder runs before them, so it would show the unmodified stream)."""
    caps = []

    def grab(_m, _i, out):
        caps.append((out[0] if isinstance(out, tuple) else out).detach()[0])

    mods = [be.model.model.embed_tokens, *be.model.model.layers]
    hs = [m.register_forward_hook(grab) for m in mods]
    try:
        with torch.no_grad():
            be.model(ids)
    finally:
        for h in hs:
            h.remove()
    return torch.stack(caps)  # [L+1, T, d]


def test_ablate_removes_direction_from_every_layer():
    be = tiny_backend()
    ids = torch.randint(0, 64, (1, 7))
    v = np.random.default_rng(1).normal(size=32).astype(np.float32)
    u = torch.tensor(v / np.linalg.norm(v))
    assert hidden(be, ids)[1:].matmul(u).abs().max() > 1e-3
    with be.intervene(Ablate(v)):
        h = hidden(be, ids)
    assert h.matmul(u).abs().max() < 1e-5


def test_mean_ablate_sets_component_to_per_layer_target():
    be = tiny_backend()
    ids = torch.randint(0, 64, (1, 6))
    v = np.random.default_rng(2).normal(size=32).astype(np.float32)
    u = torch.tensor(v / np.linalg.norm(v))
    n_states = hidden(be, ids).shape[0]
    means = tuple(float(i) * 0.5 for i in range(n_states))
    with be.intervene(Ablate(v, means=means)):
        h = hidden(be, ids)
    expected = torch.tensor(means)[:, None].expand(n_states, h.shape[1])
    assert torch.allclose(h.matmul(u), expected, atol=1e-4)


def test_steer_adds_scaled_unit_vector_at_one_layer_only():
    be = tiny_backend()
    ids = torch.randint(0, 64, (1, 5))
    v = np.zeros(32, dtype=np.float32)
    v[3] = 7.0
    base = hidden(be, ids)
    with be.intervene(Steer(v, layer=1, alpha=2.5)):
        h = hidden(be, ids)
    torch.testing.assert_close(h[0], base[0])  # embeddings untouched
    torch.testing.assert_close(h[1] - base[1], torch.zeros_like(h[1]).index_fill_(-1, torch.tensor([3]), 2.5))
    after = hidden(be, ids)
    torch.testing.assert_close(after, base)  # hook removed on exit
