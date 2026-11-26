from types import SimpleNamespace as NS

import pytest

from hallucination_lens.backends.gemini import (
    generation_from_response,
    is_daily_quota_error,
    is_logprobs_disabled_error,
)
from hallucination_lens.errors import HallucinationLensError, LogprobsUnavailableError
from hallucination_lens.utils import call_with_retries


def fake_response(finish="STOP", logprobs=True):
    lp = (
        NS(
            chosen_candidates=[NS(log_probability=-0.1), NS(log_probability=-1.5)],
            top_candidates=[
                NS(candidates=[NS(log_probability=-0.1), NS(log_probability=-2.5)]),
                NS(candidates=[NS(log_probability=-1.5), NS(log_probability=-1.7)]),
            ],
        )
        if logprobs
        else None
    )
    cand = NS(
        logprobs_result=lp,
        finish_reason=finish,
        content=NS(parts=[NS(text="Paris", thought=False), NS(text="(hidden)", thought=True)]),
    )
    return NS(candidates=[cand])


def test_conversion_of_a_normal_response():
    g = generation_from_response(fake_response(), "gemini-2.5-flash")
    assert g.text == "Paris"  # thought parts are excluded
    assert g.token_logprobs == [-0.1, -1.5]
    assert g.topk_logprobs == [[-0.1, -2.5], [-1.5, -1.7]]
    assert g.finished is True


def test_truncated_answer_is_not_finished():
    assert generation_from_response(fake_response("MAX_TOKENS"), "m").finished is False
    assert generation_from_response(fake_response("FinishReason.MAX_TOKENS"), "m").finished is False


def test_missing_logprobs_raises_helpful_error():
    with pytest.raises(LogprobsUnavailableError, match="3.x"):
        generation_from_response(fake_response(logprobs=False), "gemini-3.6-flash")


def test_no_candidates_raises():
    with pytest.raises(HallucinationLensError):
        generation_from_response(NS(candidates=[]), "m")


def test_matches_real_sdk_types():
    """Guards against SDK field-name drift: build the response from the real google-genai classes."""
    types = pytest.importorskip("google.genai").types
    cand = types.Candidate(
        content=types.Content(parts=[types.Part(text="Paris")]),
        finish_reason=types.FinishReason.STOP,
        logprobs_result=types.LogprobsResult(
            chosen_candidates=[types.LogprobsResultCandidate(log_probability=-0.2, token="Paris", token_id=1)],
            top_candidates=[
                types.LogprobsResultTopCandidates(
                    candidates=[
                        types.LogprobsResultCandidate(log_probability=-0.2, token="Paris", token_id=1),
                        types.LogprobsResultCandidate(log_probability=-3.0, token="Lyon", token_id=2),
                    ]
                )
            ],
        ),
    )
    g = generation_from_response(types.GenerateContentResponse(candidates=[cand]), "m")
    assert g.text == "Paris" and g.token_logprobs == [-0.2] and g.topk_logprobs == [[-0.2, -3.0]]
    assert g.finished is True


class ApiError(Exception):
    def __init__(self, code, msg=""):
        super().__init__(msg)
        self.code = code


def test_error_classification():
    assert is_daily_quota_error(ApiError(429, "quota metric GenerateRequestsPerDay exceeded"))
    assert not is_daily_quota_error(ApiError(429, "too many requests per minute"))
    assert is_logprobs_disabled_error(ApiError(400, "Logprobs is not enabled for models/x"))
    assert not is_logprobs_disabled_error(ApiError(500, "logprobs"))


def test_retries_then_succeeds_and_does_not_retry_client_errors():
    calls, waits = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ApiError(429)
        return "ok"

    assert call_with_retries(flaky, sleep=waits.append) == "ok"
    assert len(calls) == 3 and len(waits) == 2

    bad = []

    def always_400():
        bad.append(1)
        raise ApiError(400)

    with pytest.raises(ApiError):
        call_with_retries(always_400, sleep=waits.append)
    assert len(bad) == 1
