"""Load the TruthfulQA questions (817 questions, each with correct and incorrect reference answers)."""

from __future__ import annotations

import pandas as pd

from hallucination_lens.config import DataConfig


def load_truthfulqa(cfg: DataConfig) -> pd.DataFrame:
    """Return a table with: question_id, question, category, best_answer,
    correct_answers (list), incorrect_answers (list).  Downloaded once, then cached by Hugging Face."""
    from datasets import load_dataset  # lazy import: only needed when actually downloading

    ds = load_dataset(cfg.dataset_id, cfg.subset)[cfg.split]
    df = ds.to_pandas()
    df.insert(0, "question_id", range(len(df)))
    for col in ("correct_answers", "incorrect_answers"):
        df[col] = df[col].apply(lambda xs: [str(x) for x in list(xs)])
    if cfg.n_questions:
        df = df.head(cfg.n_questions).copy()
    return df.reset_index(drop=True)


def reference_answers(row: pd.Series) -> tuple[list[str], list[str]]:
    """(all correct references incl. best_answer, all incorrect references), de-duplicated."""
    correct = [str(row["best_answer"])] + list(row["correct_answers"])
    correct = list(dict.fromkeys(a for a in correct if a.strip()))
    incorrect = list(dict.fromkeys(a for a in row["incorrect_answers"] if a.strip()))
    return correct, incorrect
