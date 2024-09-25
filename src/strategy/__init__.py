"""Strategy pattern for delegating part of planner work to an algorithm,"""

from src.domain import Metadata
from src.strategy.notification import StrategyNotification


class PlannerStrategy:
    def notify(self, notification: StrategyNotification) -> None:
        """Action taken by strategy based on notification received from planner."""
        raise NotImplementedError

    def metadata(self) -> Metadata:
        """Metadata object describing the strategy."""
        raise NotImplementedError
