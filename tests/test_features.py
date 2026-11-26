import math

import numpy as np
import pytest

from hallucination_lens.features import (
    EXTENDED_FEATURES,
    ORIGINAL_FEATURES,
    compute_features,
    resolve_feature_names,
)


def test_basic_statistics_match_hand_calculation():
    f = compute_features([-0.1, -2.0, -0.3], finished=True)
    assert f["mean_logprob"] == pytest.approx(-0.8)
    assert f["var_logprob"] == pytest.approx(np.var([-0.1, -2.0, -0.3]))
    assert f["min_logprob"] == -2.0
    assert f["output_length"] == 3
    assert f["first_token_logprob"] == -0.1
    assert f["frac_low_conf"] == pytest.approx(1 / 3)  # only -2.0 is below log(0.5)
    assert f["finished_naturally"] == 1.0


def test_empty_answer_gives_none():
    assert compute_features([]) is None


def test_topk_entropy_and_margin():
    half = math.log(0.5)
    f = compute_features([half], topk_logprobs=[[half, half]])
    assert f["mean_entropy_topk"] == pytest.approx(math.log(2))  # two equally likely tokens
    assert f["mean_margin_topk"] == pytest.approx(0.0)
    sure = compute_features([-1e-6], topk_logprobs=[[-1e-6, -20.0]])
    assert sure["mean_margin_topk"] > 0.99


def test_missing_topk_gives_nan_not_crash():
    f = compute_features([-0.5, -0.2])
    assert math.isnan(f["mean_entropy_topk"]) and math.isnan(f["mean_margin_topk"])


def test_truncated_answer_flag():
    assert compute_features([-0.5], finished=False)["finished_naturally"] == 0.0


def test_presets_and_validation():
    assert resolve_feature_names("original4") == ORIGINAL_FEATURES
    assert resolve_feature_names("extended") == EXTENDED_FEATURES
    assert len(EXTENDED_FEATURES) == 10
    assert resolve_feature_names("extended", ["min_logprob"]) == ["min_logprob"]
    with pytest.raises(ValueError):
        resolve_feature_names("extended", ["not_a_feature"])
    with pytest.raises(ValueError):
        resolve_feature_names("nope")
