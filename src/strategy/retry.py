from src.domain import Metadata
from src.strategy.base import PlannerStrategy
from src.strategy.notification import (
    ExceptionNotification,
    NewQueryNotification,
    PlanSuccessStrategyNotification,
    StrategyNotification,
)


class RetryStrategy(PlannerStrategy):
    def should_retry(self) -> bool:
        raise NotImplementedError

    def notify(self, _: StrategyNotification) -> None:
        pass


class TryManyTimes(RetryStrategy):
    """Allow the model multiple attempts."""

    def __init__(self, times: int) -> None:
        super().__init__()
        self._success: bool = False
        self._init_times: int = times
        self.times: int = times

    def should_retry(self) -> bool:
        return not self._success and self.times > 0

    def notify(self, notification: StrategyNotification):
        if isinstance(notification, NewQueryNotification):
            self._success = False
            self.times = self._init_times
        if isinstance(notification, PlanSuccessStrategyNotification):
            self._success = True
        if isinstance(notification, ExceptionNotification):
            self.times -= 1

    def metadata(self) -> Metadata:
        return {"name": "try_many_times", "times": self._init_times}
