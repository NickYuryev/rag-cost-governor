from __future__ import annotations

import logging
from dataclasses import dataclass, field

from prometheus_client import Counter

logger = logging.getLogger(__name__)

RESERVED = Counter("rcg_budget_reserved_usd_total", "Reserved worst-case cost in USD")
SETTLED = Counter("rcg_budget_settled_usd_total", "Settled actual cost in USD")
EXHAUSTED = Counter("rcg_budget_exhausted_total", "Queries that short-circuited on budget")


class BudgetExceeded(Exception):
    pass


@dataclass
class Budget:
    """Per-query USD ceiling with reserve-then-settle.

    Every llm call reserves its worst-case cost up front; the reservation is
    settled against the actual cost after the response comes back. If a reserve
    would push the running total past the ceiling, the caller gets False and
    is expected to short-circuit. The invariant is:

        settled + outstanding_reserves <= ceiling_usd

    This is the python-side echo of the redis lua enforcement in
    github.com/NickYuryev/budget-governor.
    """

    ceiling_usd: float
    settled_usd: float = 0.0
    outstanding_usd: float = 0.0
    _exhausted: bool = field(default=False, init=False)

    def remaining(self) -> float:
        return self.ceiling_usd - self.settled_usd - self.outstanding_usd

    def try_reserve(self, worst_case_usd: float) -> bool:
        if worst_case_usd < 0:
            raise ValueError("worst_case_usd must be non-negative")
        if self.remaining() < worst_case_usd:
            if not self._exhausted:
                EXHAUSTED.inc()
                self._exhausted = True
            logger.info(
                "budget exhausted: remaining=%.6f request=%.6f ceiling=%.6f",
                self.remaining(),
                worst_case_usd,
                self.ceiling_usd,
            )
            return False
        self.outstanding_usd += worst_case_usd
        RESERVED.inc(worst_case_usd)
        return True

    def settle(self, reserved_usd: float, actual_usd: float) -> None:
        self.outstanding_usd = max(0.0, self.outstanding_usd - reserved_usd)
        self.settled_usd += actual_usd
        SETTLED.inc(actual_usd)

    @property
    def exhausted(self) -> bool:
        return self._exhausted
