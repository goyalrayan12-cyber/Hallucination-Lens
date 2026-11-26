"""Gemini API backend: ask Gemini the question and read the token log-probabilities it returns.

IMPORTANT (reported on Google's developer forum, Aug 2026): Gemini 3.x models no longer return
logprobs, and on the plain Gemini Developer API even Gemini 2.5 has been inconsistent about it. Vertex AI serves them
for 2.5. So the default model here is gemini-2.5-flash, and `hallucination-lens gemini-check`
tells you immediately whether your key/model combination returns logprobs.

Newer Gemini models are still very useful in this project as the *judge* that labels answers
(see labeling.py) -- that step does not need logprobs.
"""

from __future__ import annotations

import os
from typing import Any

from hallucination_lens.backends.base import GenerationResult
from hallucination_lens.config import GenerationConfig
from hallucination_lens.errors import (
    DailyQuotaExceeded,
    HallucinationLensError,
    LogprobsUnavailableError,
)
from hallucination_lens.utils import RateLimiter, call_with_retries, default_is_retryable

NO_LOGPROBS_HELP = (
    "The model returned no token log-probabilities. Gemini 3.x models are reported not to provide them. "
    "Use gemini-2.5-flash (or gemini-2.5-flash-lite), or use Vertex AI, or switch to the local backend."
)


def make_client() -> Any:
    """Create a Gemini client. Reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment/.env."""
    from google import genai  # lazy import

    if not (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GOOGLE_GENAI_USE_VERTEXAI")
    ):
        raise HallucinationLensError(
            "No Gemini API key found. Copy .env.example to .env and set GEMINI_API_KEY "
            "(get one at https://aistudio.google.com/apikey)."
        )
    return genai.Client()


def _error_text(exc: Exception) -> str:
    return str(exc).lower()


def is_daily_quota_error(exc: Exception) -> bool:
    text = _error_text(exc)
    return getattr(exc, "code", None) == 429 and any(
        s in text for s in ("perday", "per day", "daily")
    )


def is_logprobs_disabled_error(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 400 and "logprobs" in _error_text(exc)


def gemini_retryable(exc: Exception) -> bool:
    """Retry hiccups, but do not hammer the API when the *daily* quota is gone."""
    return default_is_retryable(exc) and not is_daily_quota_error(exc)


def guarded_call(fn, max_retries: int):
    """Run an API call with retries and translate known failures into friendly errors."""
    try:
        return call_with_retries(fn, max_retries=max_retries, is_retryable=gemini_retryable)
    except Exception as exc:
        if is_daily_quota_error(exc):
            raise DailyQuotaExceeded(
                "Gemini daily quota reached. Progress is saved; run the same command again later."
            ) from exc
        if is_logprobs_disabled_error(exc):
            raise LogprobsUnavailableError(NO_LOGPROBS_HELP) from exc
        raise


def generation_from_response(response: Any, model: str) -> GenerationResult:
    """Convert a google-genai response object into our GenerationResult."""
    if not getattr(response, "candidates", None):
        raise HallucinationLensError("Gemini returned no candidates (the answer may have been blocked).")
    cand = response.candidates[0]
    lp = getattr(cand, "logprobs_result", None)
    if lp is None or not getattr(lp, "chosen_candidates", None):
        raise LogprobsUnavailableError(NO_LOGPROBS_HELP)

    chosen = [float(c.log_probability) for c in lp.chosen_candidates]
    topk = [
        [float(c.log_probability) for c in (step.candidates or [])]
        for step in (getattr(lp, "top_candidates", None) or [])
    ]
    parts = getattr(getattr(cand, "content", None), "parts", None) or []
    text = "".join(p.text for p in parts if getattr(p, "text", None) and not getattr(p, "thought", False))
    reason = str(getattr(cand, "finish_reason", "")).upper()
    return GenerationResult(
        text=text.strip(),
        token_logprobs=[round(v, 6) for v in chosen],
        topk_logprobs=[[round(v, 6) for v in row] for row in topk],
        finished="STOP" in reason and "MAX" not in reason,
        model=model,
    )


class GeminiBackend:
    name = "gemini"

    def __init__(self, cfg: GenerationConfig, client: Any = None):
        self.cfg = cfg
        self._client = client
        self._limiter = RateLimiter(cfg.requests_per_minute)

    def _config(self, with_thinking: bool = True):
        from google.genai import types

        kwargs: dict[str, Any] = dict(
            system_instruction=self.cfg.system_prompt or None,
            temperature=0.0,
            max_output_tokens=self.cfg.max_new_tokens,
            response_logprobs=True,
            logprobs=max(1, min(self.cfg.top_k_logprobs, 20)),  # the API accepts 1..20
        )
        if with_thinking and self.cfg.thinking_budget is not None and "flash" in self.cfg.gemini_model:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=self.cfg.thinking_budget)
        return types.GenerateContentConfig(**kwargs)

    def generate(self, question: str) -> GenerationResult:
        if self._client is None:
            self._client = make_client()
        self._limiter.wait()

        def call(with_thinking: bool = True):
            return self._client.models.generate_content(
                model=self.cfg.gemini_model,
                contents=question,
                config=self._config(with_thinking),
            )

        try:
            response = guarded_call(call, self.cfg.max_retries)
        except HallucinationLensError:
            raise
        except Exception as exc:
            # some models reject the "thinking" setting: retry once without it
            if getattr(exc, "code", None) == 400 and "think" in _error_text(exc):
                response = guarded_call(lambda: call(False), self.cfg.max_retries)
            else:
                raise
        return generation_from_response(response, self.cfg.gemini_model)
