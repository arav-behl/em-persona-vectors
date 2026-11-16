"""Model backends. Both expose the same interface:

- ``chat(user, system=None) -> str``: chat-formatted prompt string
- ``generate(prompts, max_new_tokens, temperature, seed, batch_size, intervention) -> list[str]``
- ``response_activations(prompts, responses, batch_size) -> (resp_mean, prompt_last)``,
  numpy float32 arrays of shape ``[n, n_layers + 1, d_model]``
- ``n_layers``, ``d_model`` attributes
"""
from __future__ import annotations

from ..config import family


def load(fam_key: str, model: str = "base"):
    fam = family(fam_key)
    if fam.backend == "torch":
        from .torch_backend import TorchBackend

        return TorchBackend(fam, model)
    if fam.backend == "mlx":
        from .mlx_backend import MLXBackend

        return MLXBackend(fam, model)
    raise ValueError(f"unknown backend {fam.backend!r}")
