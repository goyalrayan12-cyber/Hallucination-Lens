"""Turn the token log-probabilities of ONE answer into a handful of numbers (features).

Background: when a language model writes an answer it picks one token at a time, and each pick
has a probability p. We store the log of it (log-probability, <= 0; closer to 0 = more confident).
A whole answer is therefore a list like [-0.01, -0.3, -2.1, -0.05, ...]. A classifier cannot
eat a list of varying length, so we summarise it.
"""

from __future__ import annotations

import math

import numpy as np

# The four features used in the original course report.
ORIGINAL_FEATURES = ["mean_logprob", "var_logprob", "min_logprob", "output_length"]

# Extra features added for token confidence, top-k uncertainty, and trajectory dynamics.
EXTENDED_FEATURES = ORIGINAL_FEATURES + [
    "first_token_logprob",
    "q10_logprob",
    "frac_low_conf",
    "mean_entropy_topk",
    "mean_margin_topk",
    "finished_naturally",
    "max_step_drop",
    "frac_deep_dip",
    "max_dip_streak",
    "late_minus_early_logprob",
]

FEATURE_PRESETS = {"original4": ORIGINAL_FEATURES, "extended": EXTENDED_FEATURES}

FEATURE_DESCRIPTIONS = {
    "mean_logprob": "Average confidence over the whole answer (closer to 0 = more confident).",
    "var_logprob": "How unevenly confidence rises and falls inside the answer.",
    "min_logprob": "The single least-confident token (the weakest link).",
    "output_length": "Number of generated tokens.",
    "first_token_logprob": "Confidence in the very first token (how sure the model is how to start).",
    "q10_logprob": "10th-percentile token log-probability: a robust version of the minimum.",
    "frac_low_conf": "Share of tokens the model gave less than 50% probability.",
    "mean_entropy_topk": "Average uncertainty among the top candidate tokens at each step.",
    "mean_margin_topk": "Average gap between the best and 2nd-best token probability (big = decisive).",
    "finished_naturally": "1 if the model stopped by itself, 0 if it was cut off at the token limit.",
    "max_step_drop": "Largest single-token drop in log-probability (most negative step-to-step change).",
    "frac_deep_dip": "Share of tokens with log-probability below -3.5 (sharp confidence dip).",
    "max_dip_streak": "Longest run of consecutive tokens below -3.5.",
    "late_minus_early_logprob": "Mean log-prob in the second half of tokens minus the first half (confidence drift).",
}

LOW_CONF_LOGPROB = math.log(0.5)


def _topk_stats(topk_logprobs: list[list[float]] | None) -> tuple[float, float]:
    """Mean entropy and mean (top1 - top2) probability margin over the kept top-k candidates.

    Entropy here is computed on the top-k tokens only (not re-normalised), so it is an
    approximation of full-vocabulary entropy that works the same for local models.
    """
    if not topk_logprobs:
        return math.nan, math.nan
    entropies, margins = [], []
    for step in topk_logprobs:
        lp = np.sort(np.asarray(step, dtype=float))[::-1]
        lp = lp[np.isfinite(lp)]
        if lp.size == 0:
            continue
        p = np.exp(lp)
        entropies.append(float(-(p * lp).sum()))
        margins.append(float(p[0] - (p[1] if p.size > 1 else 0.0)))
    if not entropies:
        return math.nan, math.nan
    return float(np.mean(entropies)), float(np.mean(margins))


def compute_features(
    token_logprobs: list[float],
    topk_logprobs: list[list[float]] | None = None,
    finished: bool = True,
) -> dict[str, float] | None:
    """Return a dict with every known feature, or None if the answer has no tokens."""
    lp = np.asarray(token_logprobs, dtype=float)
    if lp.size == 0:
        return None

    entropy, margin = _topk_stats(topk_logprobs)

    if lp.size > 1:
        diffs = np.diff(lp)
        max_step_drop = float(diffs.min())
        mid = lp.size // 2
        late_minus_early = float(lp[mid:].mean() - lp[:mid].mean()) if mid > 0 else 0.0
    else:
        max_step_drop = 0.0
        late_minus_early = 0.0

    deep_mask = lp < -3.5
    frac_deep_dip = float(deep_mask.mean())
    streak = best_streak = 0
    for flag in deep_mask:
        if flag:
            streak += 1
            if streak > best_streak:
                best_streak = streak
        else:
            streak = 0

    return {
        "mean_logprob": float(lp.mean()),
        "var_logprob": float(lp.var()),
        "min_logprob": float(lp.min()),
        "output_length": float(lp.size),
        "first_token_logprob": float(lp[0]),
        "q10_logprob": float(np.quantile(lp, 0.10)),
        "frac_low_conf": float(np.mean(lp < LOW_CONF_LOGPROB)),
        "mean_entropy_topk": entropy,
        "mean_margin_topk": margin,
        "finished_naturally": float(bool(finished)),
        "max_step_drop": max_step_drop,
        "frac_deep_dip": frac_deep_dip,
        "max_dip_streak": float(best_streak),
        "late_minus_early_logprob": late_minus_early,
    }


def resolve_feature_names(preset: str = "extended", names: list[str] | None = None) -> list[str]:
    """Explicit `names` win; otherwise use the preset."""
    if names:
        unknown = [n for n in names if n not in FEATURE_DESCRIPTIONS]
        if unknown:
            raise ValueError(f"Unknown feature(s) {unknown}. Known: {list(FEATURE_DESCRIPTIONS)}")
        return list(names)
    if preset not in FEATURE_PRESETS:
        raise ValueError(f"Unknown feature preset '{preset}'. Use one of {list(FEATURE_PRESETS)}")
    return list(FEATURE_PRESETS[preset])