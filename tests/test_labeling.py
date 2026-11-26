import json

import numpy as np
import pytest

from hallucination_lens.config import LabelConfig
from hallucination_lens.labeling import (
    EmbeddingLabeler,
    GeminiJudge,
    label_from_similarities,
    parse_judge_reply,
    verdict_to_label,
)


def test_label_rule_and_tie_goes_to_factual():
    assert label_from_similarities(0.8, 0.3) == 0
    assert label_from_similarities(0.3, 0.8) == 1
    assert label_from_similarities(0.5, 0.5) == 0


def test_embedding_labeler_uses_all_references():
    vocab = {"cats": [1, 0, 0], "dogs": [0, 1, 0], "birds": [0, 0, 1]}
    enc = lambda texts: np.array([vocab[t] for t in texts], dtype=float)  # noqa: E731
    lab = EmbeddingLabeler("fake", encode_fn=enc)
    # the answer matches the SECOND correct reference -- the old best_answer-only rule would miss it
    out = lab.label("dogs", correct=["cats", "dogs"], incorrect=["birds"])
    assert out["label_embedding"] == 0 and out["sim_correct"] == pytest.approx(1.0)
    out = lab.label("birds", correct=["cats", "dogs"], incorrect=["birds"])
    assert out["label_embedding"] == 1


def test_parse_judge_reply_variants():
    ok = '{"results":[{"id":1,"verdict":"truthful"},{"id":2,"verdict":"Hallucinated"}]}'
    assert parse_judge_reply(ok, [1, 2]) == {1: "truthful", 2: "hallucinated"}
    fenced = "```json\n" + ok + "\n```"
    assert parse_judge_reply(fenced, [1, 2]) == {1: "truthful", 2: "hallucinated"}
    assert parse_judge_reply("garbage", [1, 2]) == {1: "unsure", 2: "unsure"}  # never silently wrong
    assert parse_judge_reply('{"results":[{"id":1,"verdict":"maybe"}]}', [1]) == {1: "unsure"}
    assert verdict_to_label("truthful") == 0 and verdict_to_label("unsure") is None


def test_judge_batches_caches_and_resumes(tmp_path):
    pytest.importorskip("google.genai")
    calls = []

    def generate_content(model, contents, config):
        ids = [item["id"] for item in json.loads(contents)]
        calls.append(ids)
        results = [{"id": i, "verdict": "truthful" if i % 2 == 0 else "hallucinated"} for i in ids]
        return type("R", (), {"text": json.dumps({"results": results})})()

    client = type("C", (), {"models": type("M", (), {"generate_content": staticmethod(generate_content)})})()
    items = [
        {"question_id": i, "question": "q", "answer": "a", "correct": ["c"], "incorrect": ["i"]}
        for i in range(7)
    ]
    cfg = LabelConfig(judge_batch_size=3, judge_requests_per_minute=0)
    judge = GeminiJudge(cfg, tmp_path / "j.jsonl", client=client)
    verdicts = judge.judge(items)
    assert len(calls) == 3  # 7 items in batches of 3 -> 3 requests, not 7
    assert verdicts[2] == "truthful" and verdicts[3] == "hallucinated"
    # second run: everything is cached, zero new API calls
    GeminiJudge(cfg, tmp_path / "j.jsonl", client=client).judge(items)
    assert len(calls) == 3
