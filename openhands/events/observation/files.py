from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class FileReadObservation(Observation):
    """This data class represents the content of a file."""

    path: str
    observation: str = ObservationType.READ

    @property
    def message(self) -> str:
        return f'I read the file {self.path}.'


@dataclass
class FileWriteObservation(Observation):
    """This data class represents a file write operation"""

    path: str
    observation: str = ObservationType.WRITE

    @property
    def message(self) -> str:
        return f'I wrote to the file {self.path}.'


@dataclass
class FileEditObservation(Observation):
    """This data class represents a file edit operation"""

    path: str
    search_block: str
    replace_block: str
    observation: str = ObservationType.EDIT

    @property
    def message(self) -> str:
        if self.search_block:
            return (
                f'I updated the file {self.path} by \n'
                f'replacing:\n {self.search_block}\n'
                f'with:\n {self.replace_block}\n'
            )
        else:
            return (
                f'I updated the file {self.path} by \n'
                f'appending:\n {self.replace_block}\n'
            )

    def __str__(self) -> str:
        return f'**FileEditObservation**\n' f'DIFF BLOCK: {self.content}\n'
