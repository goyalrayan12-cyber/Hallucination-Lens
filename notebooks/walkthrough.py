# %% [markdown]
# # Hallucination Lens: guided walkthrough
#
# This file uses VS Code's "# %%" cell format: click **Run Cell** above any cell (or press
# Shift+Enter) and the results appear in the Interactive Window, exactly like a Jupyter notebook.
# But it is a plain `.py` file, so Git diffs stay small and readable.
#
# **Before you start** run one of these in the terminal (from the repo root):
#   `hallucination-lens smoke-test`                      (synthetic data, 10 seconds), or
#   `hallucination-lens run-all --config configs/default.yaml`   (the real thing)
#
# Then set `TAG` below to the name of the run folder inside `runs/`.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hallucination_lens.conformal import conformal_quantile, lac_scores

TAG = "smoke_test"  # e.g. "local_qwen2.5-3b-instruct" for a real run


def find_repo_root() -> Path:
    for folder in [Path.cwd(), *Path.cwd().parents]:
        if (folder / "pyproject.toml").exists():
            return folder
    raise FileNotFoundError("Open the repo folder in VS Code (the one containing pyproject.toml).")


ROOT = find_repo_root()
RUN = ROOT / "runs" / TAG
metrics = json.loads((RUN / "metrics.json").read_text())
print(f"Run '{TAG}': model = {metrics['model']}, {metrics['n']['total']} answers")

# %% [markdown]
# ## 1. The raw material: one row per answer
# Each row is one answer from the model, summarised by a few numbers built from its token
# log-probabilities, plus the label (0 = factual, 1 = hallucinated).

# %%
data = pd.read_csv(ROOT / "data" / "processed" / f"{TAG}.csv")
data[["question", "generated_text", "mean_logprob", "min_logprob", "output_length", "is_hallucination"]].head(10)

# %% [markdown]
# ## 2. Do the features separate the two classes at all?
# If the two histograms overlap almost completely, the signal is weak, and that is exactly why the
# guardrail needs conformal prediction on top of the classifier.

# %%
features = metrics["feature_names"][:4]
fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3))
for ax, name in zip(np.atleast_1d(axes), features):
    for label, color in ((0, "#2a9d8f"), (1, "#e76f51")):
        ax.hist(data.loc[data["is_hallucination"] == label, name], bins=25, alpha=0.6, color=color, density=True)
    ax.set_title(name)
plt.show()

# %% [markdown]
# ## 3. The classifier, honestly judged
# Accuracy alone is misleading when one class dominates, so compare it with the
# "always guess the majority class" baseline, and look at AUROC (area under the ROC curve:
# 0.5 = coin flip, 1.0 = perfect).

# %%
pd.Series(metrics["classifier"]).round(3)

# %% [markdown]
# ## 4. Conformal prediction by hand
# Recompute the threshold from the saved predictions, using only the calibration rows, and check
# it matches what the pipeline stored. This is the entire "magic" of conformal prediction.

# %%
preds = pd.read_csv(RUN / "predictions.csv")
cal = preds[preds["split"] == "calibration"]
p = np.column_stack([1 - cal["p_hallucination"], cal["p_hallucination"]])
scores = lac_scores(p, cal["is_hallucination"].to_numpy())  # surprise = 1 - P(true label)
alpha = metrics["conformal"]["alpha"]

print(f"{len(cal)} calibration answers, alpha = {alpha}")
print("marginal threshold, recomputed by hand :", round(conformal_quantile(scores, alpha), 4))
for cls, name in ((0, "factual"), (1, "hallucinated")):
    q = conformal_quantile(scores[cal["is_hallucination"].to_numpy() == cls], alpha)
    print(f"class-conditional threshold ({name:12s}):", round(q, 4))
print("stored by the pipeline                  :", metrics["conformal"]["q_hat_by_class"])

# %% [markdown]
# ## 5. What the guardrail says on the test split

# %%
test = preds[preds["split"] == "test"]
pd.crosstab(test["set_kind"], test["is_hallucination"].map({0: "really factual", 1: "really hallucinated"}))

# %% [markdown]
# ## 6. Does the 95% guarantee hold across many random splits?

# %%
repeats_file = RUN / "repeated_splits.csv"
if repeats_file.exists():
    repeats = pd.read_csv(repeats_file)
    display_cols = ["coverage", "avg_set_size", "rate_trust", "false_reassurance_rate"]
    print(repeats.groupby("variant")[display_cols].mean().round(3))
else:
    print("Re-run training with conformal.repeated_splits=200 to create this file.")

# %% [markdown]
# ## 7. Browse the saved figures

# %%
for image in sorted((RUN / "figures").glob("*.png")):
    plt.figure(figsize=(9, 5))
    plt.imshow(plt.imread(image))
    plt.axis("off")
    plt.title(image.stem)
    plt.show()
