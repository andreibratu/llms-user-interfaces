from domain import Metadata
from strategy.notification import StrategyNotification


class PlannerStrategy:

    def notify(self, _: StrategyNotification) -> None:
        raise NotImplementedError

    def metadata(self) -> Metadata:
        """At end of run build metadata object describing planner."""
        raise NotImplementedError
