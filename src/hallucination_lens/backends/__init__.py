"""Backends = the model that answers the question. Pick one with `generation.backend` in the config."""

from hallucination_lens.backends.base import Backend, GenerationResult
from hallucination_lens.config import GenerationConfig


def make_backend(cfg: GenerationConfig) -> Backend:
    if cfg.backend == "local":
        from hallucination_lens.backends.local_hf import LocalHFBackend

        return LocalHFBackend(cfg)
    if cfg.backend == "mlx":
        from hallucination_lens.backends.local_mlx import LocalMLXBackend

        return LocalMLXBackend(cfg)
    if cfg.backend == "gemini":
        from hallucination_lens.backends.gemini import GeminiBackend

        return GeminiBackend(cfg)
    raise ValueError(f"Unknown backend '{cfg.backend}'")


__all__ = ["Backend", "GenerationResult", "make_backend"]
