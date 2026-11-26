import numpy as np
import pytest

from hallucination_lens.backends.local_hf import extract_token_stats


def test_chosen_logprob_equals_log_softmax():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 11)) * 3
    ids = np.array([0, 5, 10, 3])
    chosen, topk = extract_token_stats(logits, ids, k=3)
    soft = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert np.allclose(chosen, np.log(soft[np.arange(4), ids]))
    assert topk.shape == (4, 3)
    assert np.all(np.diff(topk, axis=1) <= 0)  # sorted most -> least likely
    assert np.allclose(topk[:, 0], np.log(soft.max(axis=1)))


def test_numerically_stable_for_huge_logits():
    logits = np.array([[1000.0, 999.0, -1000.0]])
    chosen, _ = extract_token_stats(logits, np.array([0]), k=2)
    assert np.isfinite(chosen).all()
    assert chosen[0] == pytest.approx(-np.log1p(np.exp(-1.0)))  # log(1 / (1 + e^-1))


def test_k_larger_than_vocab_is_clipped():
    _, topk = extract_token_stats(np.zeros((2, 3)), np.array([0, 1]), k=10)
    assert topk.shape == (2, 3)
