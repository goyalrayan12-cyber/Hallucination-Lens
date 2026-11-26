"""The classifier, plus the train / calibration / test split.

Three disjoint groups of answers are used, and each has exactly one job:
    train        -> fit the classifier
    calibration  -> set the conformal threshold (never used to fit the classifier)
    test         -> final, honest evaluation (never used for anything else)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logreg", "random_forest", "gradient_boosting"]


def build_model(name: str = "logreg", params: dict | None = None, seed: int = 42) -> Pipeline:
    """Return an untrained scikit-learn pipeline.

    logreg  : fill gaps -> standardise (features have very different scales, e.g. length vs
              log-prob; the original notebook skipped this) -> logistic regression.
    """
    params = dict(params or {})
    if name == "logreg":
        clf = LogisticRegression(C=params.pop("C", 1.0), max_iter=params.pop("max_iter", 2000))
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)]
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 300),
            min_samples_leaf=params.pop("min_samples_leaf", 5),
            random_state=seed,
            n_jobs=-1,
        )
        steps = [("impute", SimpleImputer(strategy="median")), ("clf", clf)]
    elif name == "gradient_boosting":
        clf = HistGradientBoostingClassifier(
            max_depth=params.pop("max_depth", 3),
            learning_rate=params.pop("learning_rate", 0.05),
            max_iter=params.pop("max_iter", 200),
            random_state=seed,
        )
        steps = [("clf", clf)]  # handles missing values natively
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from {MODEL_NAMES}")
    if params:
        raise ValueError(f"Unused model params for {name}: {sorted(params)}")
    return Pipeline(steps)


def three_way_split(
    y: pd.Series | np.ndarray, train: float, calibration: float, test: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified (keeps the class mix) split of row positions into train/calibration/test."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    strat = y if min(np.bincount(y.astype(int))) >= 3 else None
    idx_train, idx_rest = train_test_split(idx, test_size=1 - train, random_state=seed, stratify=strat)
    y_rest = y[idx_rest]
    strat_rest = y_rest if min(np.bincount(y_rest.astype(int))) >= 2 else None
    idx_cal, idx_test = train_test_split(
        idx_rest, test_size=test / (calibration + test), random_state=seed, stratify=strat_rest
    )
    return idx_train, idx_cal, idx_test


def compare_models(
    X: pd.DataFrame, y: pd.Series, names: list[str] | None = None, seed: int = 42, folds: int = 5
) -> pd.DataFrame:
    """5-fold cross-validated AUROC and Brier score on the TRAINING data only."""
    rows = []
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for name in names or MODEL_NAMES:
        p = cross_val_predict(build_model(name, seed=seed), X, y, cv=cv, method="predict_proba")[:, 1]
        rows.append(
            {
                "model": name,
                "cv_auroc": roc_auc_score(y, p),
                "cv_brier": brier_score_loss(y, p),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_auroc", ascending=False).reset_index(drop=True)
