from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class ErrorObservation(Observation):
    """This data class represents an error encountered by the agent."""

    observation: str = ObservationType.ERROR

    @property
    def message(self) -> str:
        return self.content

    def __str__(self) -> str:
        return f'**ErrorObservation**\n{self.content}'
