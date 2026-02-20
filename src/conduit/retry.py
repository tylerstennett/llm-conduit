from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Type

from conduit.exceptions import ProviderError, ProviderUnavailableError, RateLimitError


@dataclass(slots=True)
class RetryPolicy:
    """Retry configuration used by the Conduit client."""

    max_retries: int = 3
    backoff_base: float = 1.0
    backoff_factor: float = 2.0
    jitter: float = 0.1
    retry_on: tuple[Type[ProviderError], ...] = field(
        default_factory=lambda: (RateLimitError, ProviderUnavailableError)
    )

    def backoff_for_attempt(self, attempt: int, error: BaseException | None = None) -> float:
        """Compute the backoff delay for a given retry attempt.

        Respects ``Retry-After`` on :class:`RateLimitError`; otherwise uses
        exponential backoff with jitter.

        Args:
            attempt: 1-based retry attempt number.
            error: The exception that triggered the retry (used to check
                for ``Retry-After``).

        Returns:
            Delay in seconds before the next retry.
        """
        if isinstance(error, RateLimitError):
            retry_after = error.retry_after
            if (
                isinstance(retry_after, (int, float))
                and retry_after > 0
                and math.isfinite(float(retry_after))
            ):
                return float(retry_after)
        base_delay = self.backoff_base * (self.backoff_factor ** (attempt - 1))
        jitter_component = base_delay * self.jitter * random.random()
        return base_delay + jitter_component

    def should_retry(self, error: BaseException, attempt: int) -> bool:
        """Decide whether to retry a failed request.

        Args:
            error: The exception that was raised.
            attempt: 1-based attempt number (the attempt that *would* run
                next, not the one that just failed).

        Returns:
            ``True`` if the error is retryable and the attempt limit has
            not been reached.
        """
        if attempt > self.max_retries:
            return False
        return isinstance(error, self.retry_on)
