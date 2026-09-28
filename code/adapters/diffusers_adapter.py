"""Optional diffusers adapter for SDXL-like pipelines."""
from __future__ import annotations

from .base import BlackBoxModelAdapter


class DiffusersAdapter(BlackBoxModelAdapter):
    def __init__(self, pipeline, generation_kwargs=None):
        self.pipeline = pipeline
        self.generation_kwargs = dict(generation_kwargs or {})

    def generate(self, query: str, *, seed: int | None = None):
        import torch
        generator = None
        if seed is not None:
            device = getattr(self.pipeline, "device", "cpu")
            generator = torch.Generator(device=str(device)).manual_seed(seed)
        result = self.pipeline(query, generator=generator, **self.generation_kwargs)
        return result.images[0]
