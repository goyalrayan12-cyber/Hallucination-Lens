"""Metrics and experiments.

Two separate questions are answered:
  A. How good is the classifier?            -> accuracy, AUROC, Brier score, ECE
  B. How good is the conformal guardrail?   -> coverage, set size, how often it says TRUST, and
                                               how often a hallucination slips through as TRUST.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from hallucination_lens.conformal import ConformalPredictor
from hallucination_lens.modeling import build_model, three_way_split

# ----- A. classifier --------------------------------------------------------------------------


def expected_calibration_error(y_true, p_pos, n_bins: int = 10) -> float:
    """ECE for P(hallucinated): group predictions into equal-width probability bins, then average
    |observed hallucination rate - mean predicted probability| weighted by bin size.
    0 = perfectly calibrated."""
    return float(reliability_bins(y_true, p_pos, n_bins)["gap_weighted"].sum())


def reliability_bins(y_true, p_pos, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p_pos, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    which = np.digitize(p, edges[1:-1])  # 0 .. n_bins-1
    rows = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            gap = abs(y[m].mean() - p[m].mean())
            rows.append(
                {
                    "bin": b,
                    "n": int(m.sum()),
                    "mean_pred": float(p[m].mean()),
                    "observed": float(y[m].mean()),
                    "gap_weighted": float(m.mean() * gap),
                }
            )
    return pd.DataFrame(rows, columns=["bin", "n", "mean_pred", "observed", "gap_weighted"])


def classifier_metrics(y_true, p_pos) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(p_pos, dtype=float)
    both_classes = len(np.unique(y)) == 2
    majority = max(y.mean(), 1 - y.mean())
    return {
        "accuracy": float(accuracy_score(y, p >= 0.5)),
        "majority_baseline_accuracy": float(majority),
        "auroc": float(roc_auc_score(y, p)) if both_classes else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


# ----- B. conformal guardrail -----------------------------------------------------------------

VERDICT_NAMES = ["factual", "hallucinated", "uncertain", "empty"]


def set_kinds(sets: np.ndarray) -> np.ndarray:
    """Name each prediction set: {factual}, {hallucinated}, both = 'uncertain', neither = 'empty'."""
    sets = np.asarray(sets, dtype=bool)
    out = np.full(len(sets), "empty", dtype=object)
    out[sets[:, 0] & ~sets[:, 1]] = "factual"
    out[~sets[:, 0] & sets[:, 1]] = "hallucinated"
    out[sets[:, 0] & sets[:, 1]] = "uncertain"
    return out


def _rate(mask_num: np.ndarray, mask_den: np.ndarray) -> float:
    return float(mask_num[mask_den].mean()) if mask_den.any() else float("nan")


def conformal_metrics(y_true, sets) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    sets = np.asarray(sets, dtype=bool)
    kinds = set_kinds(sets)
    covered = sets[np.arange(len(y)), y]
    trust, flag = kinds == "factual", kinds == "hallucinated"
    return {
        "coverage": float(covered.mean()),
        "coverage_factual": _rate(covered, y == 0),
        "coverage_hallucinated": _rate(covered, y == 1),
        "avg_set_size": float(sets.sum(axis=1).mean()),
        "rate_trust": float(trust.mean()),
        "rate_uncertain": float((kinds == "uncertain").mean()),
        "rate_flagged": float(flag.mean()),
        "rate_empty": float((kinds == "empty").mean()),
        # safety: of the answers that really ARE hallucinations, how many were stamped TRUST?
        "false_reassurance_rate": _rate(trust, y == 1),
        # of the answers stamped TRUST, what share is really factual?
        "trust_precision": _rate(y == 0, trust),
        "flag_precision": _rate(y == 1, flag),
    }


def alpha_sweep(
    proba_cal, y_cal, proba_test, y_test, alphas=None, class_conditional: bool = True
) -> pd.DataFrame:
    """Does the guarantee hold for many alphas? Empirical coverage should track 1 - alpha."""
    rows = []
    for alpha in alphas if alphas is not None else np.round(np.arange(0.05, 0.55, 0.05), 2):
        cp = ConformalPredictor(float(alpha), class_conditional).calibrate(proba_cal, y_cal)
        m = conformal_metrics(y_test, cp.predict_set(proba_test))
        rows.append(
            {
                "alpha": float(alpha),
                "target_coverage": 1 - float(alpha),
                "coverage": m["coverage"],
                "avg_set_size": m["avg_set_size"],
                "rate_trust": m["rate_trust"],
                "false_reassurance_rate": m["false_reassurance_rate"],
            }
        )
    return pd.DataFrame(rows)


def repeated_split_experiment(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    model_params: dict | None,
    alpha: float,
    n_repeats: int,
    fractions: tuple[float, float, float],
    base_seed: int,
) -> pd.DataFrame:
    """Re-do split -> train -> calibrate -> test many times with different random splits.

    A single test set of ~120 answers gives a noisy coverage number (one answer = ~1 point).
    The AVERAGE over many splits shows whether the guarantee really holds."""
    rows = []
    y_arr = np.asarray(y, dtype=int)
    for r in range(n_repeats):
        tr, ca, te = three_way_split(y_arr, *fractions, seed=base_seed + 1 + r)
        model = build_model(model_name, model_params, seed=base_seed + 1 + r).fit(X.iloc[tr], y_arr[tr])
        p_cal, p_te = model.predict_proba(X.iloc[ca]), model.predict_proba(X.iloc[te])
        for variant, cc in (("marginal", False), ("class_conditional", True)):
            cp = ConformalPredictor(alpha, cc).calibrate(p_cal, y_arr[ca])
            m = conformal_metrics(y_arr[te], cp.predict_set(p_te))
            rows.append({"repeat": r, "variant": variant, **m})
    return pd.DataFrame(rows)


def summarize_repeats(df: pd.DataFrame) -> dict:
    """Mean / spread of the key numbers over the repeated splits, per conformal variant."""
    out = {}
    for variant, g in df.groupby("variant"):
        out[variant] = {
            "n_repeats": int(len(g)),
            "coverage_mean": float(g["coverage"].mean()),
            "coverage_p05": float(g["coverage"].quantile(0.05)),
            "coverage_min": float(g["coverage"].min()),
            "avg_set_size_mean": float(g["avg_set_size"].mean()),
            "rate_trust_mean": float(g["rate_trust"].mean()),
            "false_reassurance_mean": float(g["false_reassurance_rate"].mean()),
            "coverage_hallucinated_mean": float(g["coverage_hallucinated"].mean()),
        }
    return out
