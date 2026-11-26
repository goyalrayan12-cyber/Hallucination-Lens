"""Split conformal prediction for a classifier, written from scratch in ~40 lines of real logic.

The idea in plain words
-----------------------
A classifier says "70% hallucinated". Is that trustworthy? Conformal prediction answers by
looking at how wrong the classifier was on a *calibration set* it never trained on:

 1. For every calibration example, compute a "surprise" score  s = 1 - p(true label).
    (If the model gave the true label probability 0.9, surprise is 0.1.)
 2. Take the (1 - alpha) quantile of those scores -> a threshold q.
    (alpha = 0.05 means: we accept being wrong 5% of the time.)
 3. For a NEW answer, keep every label whose surprise 1 - p(label) is <= q.
    The result is a SET: {factual}, {hallucinated}, both, or (rarely) neither.

If calibration and new data are exchangeable (drawn the same way), the true label is inside
the set with probability >= 1 - alpha. This holds for ANY classifier, however weak. A weak
classifier simply gives bigger sets (more "I'm not sure").

This is the "LAC" score (Least Ambiguous set-valued Classifier, Sadinle et al. 2019). The same
numbers are produced by MAPIE; tests/test_conformal.py cross-checks that.

Class-conditional ("Mondrian") variant: do steps 1-2 separately for each true class. Then the
guarantee holds *per class*: P(hallucinated answer is NOT labelled TRUST) >= 1 - alpha.
That is exactly the safety property a hallucination guardrail wants.
"""

from __future__ import annotations

import math

import numpy as np


def lac_scores(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Surprise score 1 - p(true class) for each example."""
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=int)
    return 1.0 - proba[np.arange(len(y)), y]


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample-corrected quantile: level = ceil((n+1)(1-alpha)) / n.

    With very few calibration points the level exceeds 1; then no honest finite threshold
    exists and we return +inf (every label is included -> sets are as wide as possible)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0:
        return math.inf
    k = math.ceil((n + 1) * (1 - alpha))  # rank of the score to use (1-based)
    if k > n:
        return math.inf
    return float(np.sort(scores)[k - 1])


class ConformalPredictor:
    def __init__(self, alpha: float = 0.05, class_conditional: bool = False):
        self.alpha = alpha
        self.class_conditional = class_conditional
        self.q_hat: float | None = None
        self.q_hat_by_class: dict[int, float] = {}
        self.n_cal = 0

    def calibrate(self, proba_cal: np.ndarray, y_cal: np.ndarray) -> ConformalPredictor:
        proba_cal = np.asarray(proba_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=int)
        scores = lac_scores(proba_cal, y_cal)
        self.n_cal = len(y_cal)
        self.q_hat = conformal_quantile(scores, self.alpha)
        self.q_hat_by_class = {
            c: conformal_quantile(scores[y_cal == c], self.alpha) for c in range(proba_cal.shape[1])
        }
        return self

    def predict_set(self, proba: np.ndarray) -> np.ndarray:
        """Boolean array (n, n_classes): True where the label is in the prediction set."""
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")
        proba = np.asarray(proba, dtype=float)
        surprise = 1.0 - proba  # (n, n_classes)
        if self.class_conditional:
            thresholds = np.array([self.q_hat_by_class[c] for c in range(proba.shape[1])])
            return surprise <= thresholds[None, :]
        return surprise <= self.q_hat

    # ---- save / load so the guardrail can reuse a calibrated threshold ----------------------
    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "class_conditional": self.class_conditional,
            "q_hat": self.q_hat,
            "q_hat_by_class": {str(k): v for k, v in self.q_hat_by_class.items()},
            "n_cal": self.n_cal,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ConformalPredictor:
        obj = cls(d["alpha"], d["class_conditional"])
        obj.q_hat = d["q_hat"]
        obj.q_hat_by_class = {int(k): v for k, v in d["q_hat_by_class"].items()}
        obj.n_cal = d["n_cal"]
        return obj
