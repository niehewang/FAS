from __future__ import annotations
from abc import ABC, abstractmethod


class BlackBoxModelAdapter(ABC):
    """Minimal interface needed by FAS.

    Concrete adapters should return raw outputs; an independent encoder maps raw
    outputs to vectors. Keeping these layers separate makes evaluator sensitivity
    experiments straightforward.
    """

    @abstractmethod
    def generate(self, query: str, *, seed: int | None = None):
        raise NotImplementedError


class OutputEncoder(ABC):
    @abstractmethod
    def encode(self, output):
        raise NotImplementedError
