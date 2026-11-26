import json

import numpy as np
import pandas as pd
import pytest

from hallucination_lens.backends.base import GenerationResult
from hallucination_lens.config import Config, load_config
from hallucination_lens.errors import HallucinationLensError
from hallucination_lens.guardrail import Guardrail
from hallucination_lens.pipeline import (
    build_dataset,
    read_generations,
    run_generate,
    train_and_evaluate,
)
from hallucination_lens.synthetic import make_synthetic_dataset


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = 0

    def generate(self, question):
        self.calls += 1
        n = 3 + self.calls % 4
        lp = [-0.1 * (i + 1) for i in range(n)]
        return GenerationResult(f"answer to {question}", lp, [[v, v - 1.0] for v in lp], True, "fake-model")


class FakeLabeler:
    def label(self, answer, correct, incorrect):
        sim_c, sim_i = (0.9, 0.1) if int(answer.split()[-1][1:]) % 3 else (0.1, 0.9)
        return {"sim_correct": sim_c, "sim_incorrect": sim_i, "label_embedding": int(sim_c < sim_i)}


def questions(n):
    return pd.DataFrame(
        {
            "question_id": range(n),
            "question": [f"q{i}" for i in range(n)],
            "category": "c",
            "best_answer": "best",
            "correct_answers": [["best"]] * n,
            "incorrect_answers": [["wrong"]] * n,
        }
    )


def cfg_in(tmp_path, *overrides) -> Config:
    return load_config(None, [f"run.data_dir={tmp_path}/data", f"run.runs_dir={tmp_path}/runs",
                              "run.name=t", *overrides])


def test_generation_is_resumable(tmp_path):
    cfg, backend = cfg_in(tmp_path), FakeBackend()
    run_generate(cfg, backend=backend, questions=questions(5))
    assert backend.calls == 5
    run_generate(cfg, backend=backend, questions=questions(8))  # only the 3 new ones
    assert backend.calls == 8
    path = tmp_path / "data" / "generations" / "t.jsonl"
    assert len(read_generations(path)) == 8


def test_request_budget_stops_early_for_gemini(tmp_path):
    cfg = cfg_in(tmp_path, "generation.backend=gemini", "generation.max_requests_per_run=2")
    backend = FakeBackend()
    run_generate(cfg, backend=backend, questions=questions(6))
    assert backend.calls == 2


def test_build_dataset_has_features_and_labels(tmp_path):
    cfg = cfg_in(tmp_path)
    q = questions(30)
    run_generate(cfg, backend=FakeBackend(), questions=q)
    # make the answers match FakeLabeler's "answer to qN" convention
    df = build_dataset(cfg, questions=q, labeler=FakeLabeler())
    assert len(df) == 30
    assert {"mean_logprob", "mean_entropy_topk", "is_hallucination", "label_source"} <= set(df.columns)
    assert set(df["is_hallucination"]) == {0, 1}
    assert (tmp_path / "data" / "processed" / "t.csv").exists()


def test_full_stats_stage_and_guardrail(tmp_path):
    cfg = cfg_in(tmp_path, "conformal.repeated_splits=10")
    m = train_and_evaluate(cfg, df=make_synthetic_dataset(400, seed=1))
    rd = tmp_path / "runs" / "t"
    for name in ["model.joblib", "conformal.json", "metrics.json", "REPORT.md", "predictions.csv",
                 "config.yaml", "figures/reliability.png", "figures/coverage_distribution.png"]:
        assert (rd / name).exists(), name
    assert m["n"]["train"] + m["n"]["calibration"] + m["n"]["test"] == 400
    assert 0.0 <= m["classifier"]["auroc"] <= 1.0
    # calibration rows must never leak into the test split
    preds = pd.read_csv(rd / "predictions.csv")
    assert preds["split"].value_counts().to_dict().keys() == {"train", "calibration", "test"}
    assert json.loads((rd / "metrics.json").read_text())["conformal"]["alpha"] == 0.05

    guard = Guardrail.from_run(rd, backend=FakeBackend())
    res = guard.check("is this safe?")
    assert res.verdict in {"TRUST", "UNCERTAIN", "DON'T TRUST", "UNUSUAL"}
    assert res.p_hallucination is not None and "Verdict" in res.format()
    empty = guard.check_generation("q", GenerationResult("", [], [], True, "m"))
    assert empty.verdict == "UNUSUAL"


def test_single_class_labels_fail_with_clear_message(tmp_path):
    df = make_synthetic_dataset(100)
    df["is_hallucination"] = 1
    with pytest.raises(HallucinationLensError, match="at least 5"):
        train_and_evaluate(cfg_in(tmp_path), df=df)


def test_split_is_disjoint_and_stratified():
    from hallucination_lens.modeling import three_way_split

    y = np.array([0] * 40 + [1] * 80)
    tr, ca, te = three_way_split(y, 0.7, 0.15, 0.15, seed=0)
    assert not (set(tr) & set(ca)) and not (set(tr) & set(te)) and not (set(ca) & set(te))
    assert len(tr) + len(ca) + len(te) == 120
    assert abs(y[te].mean() - y.mean()) < 0.1
