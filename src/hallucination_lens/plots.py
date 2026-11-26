"""All figures. Every function returns a matplotlib Figure; save_fig() writes it as a PNG."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hallucination_lens.evaluation import reliability_bins

C_FACTUAL = "#2a9d8f"  # teal
C_HALLU = "#e76f51"  # orange-red
C_GREY = "#6c757d"
C_BLUE = "#264653"

VERDICT_LABELS = {
    "factual": "TRUST",
    "uncertain": "UNCERTAIN",
    "hallucinated": "DON'T TRUST",
    "empty": "UNUSUAL",
}


def save_fig(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_class_balance(y) -> plt.Figure:
    y = np.asarray(y)
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    bars = ax.bar(["Factual (0)", "Hallucinated (1)"], counts, color=[C_FACTUAL, C_HALLU])
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, c, f"{c}\n({c / len(y):.0%})", ha="center", va="bottom")
    ax.set_ylabel("Number of answers")
    ax.set_title("Class balance")
    ax.set_ylim(0, max(counts) * 1.2)
    return fig


def plot_feature_distributions(df: pd.DataFrame, feature_names: list[str]) -> plt.Figure:
    names = feature_names[:6]
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3 * rows), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        for label, color, text in ((0, C_FACTUAL, "factual"), (1, C_HALLU, "hallucinated")):
            vals = df.loc[df["is_hallucination"] == label, name].dropna()
            ax.hist(vals, bins=25, alpha=0.6, color=color, label=text, density=True)
        ax.set_title(name)
    for ax in axes.ravel()[len(names) :]:
        ax.axis("off")
    axes[0, 0].legend()
    fig.suptitle("Feature distributions by class (overlap = weak signal)", y=1.02)
    fig.tight_layout()
    return fig


def plot_reliability(y_true, p_pos, n_bins: int = 10) -> plt.Figure:
    bins = reliability_bins(y_true, p_pos, n_bins)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color=C_GREY, label="perfectly calibrated")
    ax.plot(bins["mean_pred"], bins["observed"], "o-", color=C_BLUE, label="classifier")
    for _, r in bins.iterrows():
        ax.annotate(int(r["n"]), (r["mean_pred"], r["observed"]), textcoords="offset points",
                    xytext=(4, -10), fontsize=7, color=C_GREY)
    ax.set_xlabel("Predicted P(hallucinated)")
    ax.set_ylabel("Observed hallucination rate")
    ax.set_title("Reliability diagram (test split; numbers = answers per bin)")
    ax.legend()
    return fig


def plot_verdicts(y_true, kinds) -> plt.Figure:
    """Stacked bars: what the guardrail said, split by what was really true."""
    y = np.asarray(y_true)
    kinds = np.asarray(kinds, dtype=object)
    order = ["factual", "uncertain", "hallucinated", "empty"]
    fact = [int(((kinds == k) & (y == 0)).sum()) for k in order]
    hall = [int(((kinds == k) & (y == 1)).sum()) for k in order]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    xs = [VERDICT_LABELS[k] for k in order]
    ax.bar(xs, fact, color=C_FACTUAL, label="really factual")
    ax.bar(xs, hall, bottom=fact, color=C_HALLU, label="really hallucinated")
    for i, (a, b) in enumerate(zip(fact, hall)):
        if a + b:
            ax.text(i, a + b, str(a + b), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Test answers")
    ax.set_title("What the guardrail said vs. what was true")
    ax.legend()
    return fig


def plot_confidence_vs_verdict(mean_logprob, y_true, kinds) -> plt.Figure:
    rng = np.random.default_rng(0)
    order = ["factual", "uncertain", "hallucinated", "empty"]
    ypos = {k: i for i, k in enumerate(order)}
    ys = np.array([ypos[k] for k in kinds]) + rng.uniform(-0.15, 0.15, len(kinds))
    colors = [C_HALLU if t == 1 else C_FACTUAL for t in np.asarray(y_true)]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.scatter(mean_logprob, ys, c=colors, edgecolors="black", linewidths=0.4, s=45, alpha=0.85)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([VERDICT_LABELS[k] for k in order])
    ax.set_xlabel("Mean token log-probability (closer to 0 = model more confident)")
    ax.set_title("Raw model confidence vs. guardrail verdict (red = really hallucinated)")
    ax.grid(alpha=0.3)
    return fig


def plot_coverage_distribution(repeats: pd.DataFrame, alpha: float) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (variant, g) in zip(axes, repeats.groupby("variant")):
        ax.hist(g["coverage"], bins=20, color=C_BLUE, alpha=0.8)
        ax.axvline(1 - alpha, color=C_HALLU, ls="--", label=f"target {1 - alpha:.0%}")
        ax.axvline(g["coverage"].mean(), color=C_FACTUAL, label=f"mean {g['coverage'].mean():.1%}")
        ax.set_title(f"{variant.replace('_', '-')} conformal")
        ax.set_xlabel("Coverage on test split")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.4)  # headroom so the legend never covers bars
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Number of random re-splits")
    fig.suptitle("Does the guarantee hold? Coverage over many random splits", y=1.03)
    return fig


def plot_alpha_sweep(sweep: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([0.45, 1], [0.45, 1], "--", color=C_GREY, label="ideal (coverage = target)")
    axes[0].plot(sweep["target_coverage"], sweep["coverage"], "o-", color=C_BLUE, label="measured")
    axes[0].set_xlabel("Target coverage (1 - alpha)")
    axes[0].set_ylabel("Measured coverage (test)")
    axes[0].set_title("Coverage tracks the target")
    axes[0].legend()
    axes[1].plot(sweep["target_coverage"], sweep["avg_set_size"], "o-", color=C_HALLU)
    axes[1].set_xlabel("Target coverage (1 - alpha)")
    axes[1].set_ylabel("Average set size")
    axes[1].set_title("Price of a stronger guarantee: bigger sets")
    return fig


def plot_coefficients(feature_names: list[str], coefs) -> plt.Figure:
    order = np.argsort(np.abs(coefs))
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(coefs) + 1.2))
    ax.barh(np.array(feature_names)[order], np.asarray(coefs)[order],
            color=[C_HALLU if c > 0 else C_FACTUAL for c in np.asarray(coefs)[order]])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Standardised coefficient  (>0 pushes towards 'hallucinated')")
    ax.set_title("What the logistic regression relies on")
    return fig
