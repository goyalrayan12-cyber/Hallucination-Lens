"""Fake data with the same columns as the real processed dataset.

ONLY for tests and `hallucination-lens smoke-test` (to check your install works without
downloading a model or using an API key). Results on this data mean nothing scientifically.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_dataset(n: int = 400, seed: int = 0, signal: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    confidence = rng.normal(size=n)  # hidden "how sure was the model" factor
    p_hallucinated = 1 / (1 + np.exp(-(0.6 - signal * confidence)))
    y = rng.binomial(1, p_hallucinated)
    noise = lambda s: rng.normal(scale=s, size=n)  # noqa: E731
    df = pd.DataFrame(
        {
            "question_id": np.arange(n),
            "question": [f"synthetic question {i}" for i in range(n)],
            "category": "synthetic",
            "generated_text": [f"synthetic answer {i}" for i in range(n)],
            "mean_logprob": np.minimum(-0.05, -0.6 + 0.2 * confidence + noise(0.08)),
            "var_logprob": np.abs(0.5 - 0.15 * confidence + noise(0.1)),
            "min_logprob": np.minimum(-0.1, -2.5 + 0.7 * confidence + noise(0.4)),
            "output_length": rng.poisson(28, n).clip(3, 64).astype(float),
            "first_token_logprob": np.minimum(-0.001, -0.4 + 0.2 * confidence + noise(0.15)),
            "q10_logprob": np.minimum(-0.05, -1.6 + 0.5 * confidence + noise(0.3)),
            "frac_low_conf": np.clip(0.25 - 0.08 * confidence + noise(0.05), 0, 1),
            "mean_entropy_topk": np.abs(0.6 - 0.2 * confidence + noise(0.1)),
            "mean_margin_topk": np.clip(0.6 + 0.12 * confidence + noise(0.08), 0, 1),
            "finished_naturally": rng.binomial(1, 0.8, n).astype(float),
            "sim_correct": 0.5,
            "sim_incorrect": 0.5,
            "label_embedding": y,
            "label_judge": np.nan,
            "label_source": "synthetic",
            "is_hallucination": y,
        }
    )
    return df
