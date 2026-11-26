"""Project-specific errors, so the command line can print a helpful message instead of a traceback."""


class HallucinationLensError(Exception):
    """Base class for all errors raised on purpose by this project."""


class LogprobsUnavailableError(HallucinationLensError):
    """The chosen Gemini model/endpoint did not return token log-probabilities."""


class DailyQuotaExceeded(HallucinationLensError):
    """The API's daily request quota is used up. Progress is saved; re-run tomorrow to resume."""
