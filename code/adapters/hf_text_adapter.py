"""Optional Hugging Face text adapter.

Install optional dependencies from requirements-experiments.txt before use.
This scaffold is intentionally small; generation parameters must be recorded in
configs and lineage metadata rather than hidden in code defaults.
"""
from __future__ import annotations

from .base import BlackBoxModelAdapter


class HFTextAdapter(BlackBoxModelAdapter):
    def __init__(self, model, tokenizer, generation_kwargs=None):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_kwargs = dict(generation_kwargs or {})

    def generate(self, query: str, *, seed: int | None = None):
        import torch
        if seed is not None:
            torch.manual_seed(seed)
        inputs = self.tokenizer(query, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, **self.generation_kwargs)
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = out[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
