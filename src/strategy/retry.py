"""Retry strategies for the planner.

Informs the planner on whether it should give up on the current query or try again.
"""

from src.domain import Metadata
from src.strategy import PlannerStrategy
from src.strategy.notification import (
    ExceptionNotification,
    NewQueryNotification,
    PlanSuccessStrategyNotification,
    StrategyNotification,
)


class RetryStrategy(PlannerStrategy):
    """Retry strategies for the planner.

    Informs the planner on whether it should give up on the current query or try again.
    """

    strategy_name: str

    def __init__(self, strategy_name: str) -> None:
        super().__init__()
        self.strategy_name = strategy_name

    def should_retry(self) -> bool:
        raise NotImplementedError

    def notify(self, notification: StrategyNotification) -> None:
        pass


class TryManyTimes(RetryStrategy):
    """Retry a query a fixed number of times."""

    def __init__(self, times: int) -> None:
        super().__init__(strategy_name="try_many_times")
        self._success: bool = False
        self._init_times: int = times
        self._times: int = times

    def should_retry(self) -> bool:
        return not self._success and self._times > 0

    def notify(self, notification: StrategyNotification):
        if isinstance(notification, NewQueryNotification):
            self._success = False
            self._times = self._init_times
        if isinstance(notification, PlanSuccessStrategyNotification):
            self._success = True
        if isinstance(notification, ExceptionNotification):
            self._times -= 1

    def metadata(self) -> Metadata:
        return {
            "retry_strategy_name": "try_many_times",
            "retry_times": self._init_times,
        }

    @property
    def retry_times(self):
        return self._init_times
