"""The product: ask a question, get an answer AND a verdict about whether to trust it.

Verdict meaning (from the conformal prediction set):
    {factual}                -> TRUST         (the 'hallucinated' label was ruled out)
    {hallucinated}           -> DON'T TRUST   (the 'factual' label was ruled out)
    {factual, hallucinated}  -> UNCERTAIN     (the data cannot decide at this confidence)
    {}                       -> UNUSUAL       (neither label fits; treat as unreliable)

Note: the original report called *every* one-label set "verified". That is wrong for a
{hallucinated} set, which actually means "very likely wrong". Fixed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from hallucination_lens.backends import GenerationResult, make_backend
from hallucination_lens.config import load_config
from hallucination_lens.conformal import ConformalPredictor
from hallucination_lens.evaluation import set_kinds
from hallucination_lens.features import compute_features

VERDICTS = {
    "factual": ("TRUST", "The hallucination label was ruled out at the chosen confidence level."),
    "hallucinated": ("DON'T TRUST", "The factual label was ruled out: this is likely a hallucination."),
    "uncertain": ("UNCERTAIN", "The evidence cannot decide. Double-check this answer."),
    "empty": ("UNUSUAL", "Neither label fits well. Treat this answer as unreliable."),
}


@dataclass
class GuardrailResult:
    question: str
    answer: str
    p_hallucination: float | None
    label_set: list[str] = field(default_factory=list)
    verdict: str = "UNUSUAL"
    message: str = ""
    features: dict[str, float] = field(default_factory=dict)

    def format(self) -> str:
        p = "n/a" if self.p_hallucination is None else f"{self.p_hallucination:.2f}"
        labels = "{" + ", ".join(self.label_set) + "}"
        return (
            f"Question        : {self.question}\n"
            f"Answer          : {self.answer or '(empty)'}\n"
            f"P(hallucinated) : {p}\n"
            f"Label set       : {labels}\n"
            f"Verdict         : {self.verdict} - {self.message}"
        )


class Guardrail:
    def __init__(self, backend: Any, model: Any, conformal: ConformalPredictor, feature_names: list[str]):
        self.backend = backend
        self.model = model
        self.conformal = conformal
        self.feature_names = feature_names

    @classmethod
    def from_run(cls, run_path: str | Path, backend: Any = None) -> Guardrail:
        run_path = Path(run_path)
        cfg = load_config(run_path / "config.yaml")
        meta = json.loads((run_path / "meta.json").read_text())
        model = joblib.load(run_path / "model.joblib")
        conformal = ConformalPredictor.from_dict(json.loads((run_path / "conformal.json").read_text()))
        return cls(backend or make_backend(cfg.generation), model, conformal, meta["feature_names"])

    def check_generation(self, question: str, gen: GenerationResult) -> GuardrailResult:
        """Score an answer that has already been generated (no model call)."""
        feats = compute_features(gen.token_logprobs, gen.topk_logprobs, gen.finished)
        if feats is None:
            name, msg = VERDICTS["empty"]
            return GuardrailResult(question, gen.text, None, [], name, "The model produced no answer.")
        row = pd.DataFrame([{n: feats[n] for n in self.feature_names}])
        proba = self.model.predict_proba(row)
        sets = self.conformal.predict_set(proba)
        kind = str(set_kinds(sets)[0])
        labels = [name for name, keep in zip(("factual", "hallucinated"), sets[0]) if keep]
        verdict, message = VERDICTS[kind]
        return GuardrailResult(question, gen.text, float(proba[0, 1]), labels, verdict, message, feats)

    def check(self, question: str) -> GuardrailResult:
        return self.check_generation(question, self.backend.generate(question))


