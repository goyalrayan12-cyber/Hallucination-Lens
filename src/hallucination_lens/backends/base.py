"""What every "answerer" (local model or Gemini) must return, so the rest of the code does not care."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class GenerationResult:
    text: str  # the answer
    token_logprobs: list[float]  # log-probability of each token the model actually produced
    topk_logprobs: list[list[float]] = field(default_factory=list)  # top-k log-probs at each step
    finished: bool = True  # False if the answer was cut off at the token limit
    model: str = ""


class Backend(Protocol):
    name: str

    def generate(self, question: str) -> GenerationResult: ...
