from dataclasses import dataclass

from openhands.core.schema import ObservationType
from openhands.events.observation.observation import Observation


@dataclass
class CmdOutputObservation(Observation):
    """This data class represents the output of a command."""

    command_id: int
    command: str
    exit_code: int = 0
    hidden: bool = False
    observation: str = ObservationType.RUN

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'Command `{self.command}` executed with exit code {self.exit_code}.'

    def __str__(self) -> str:
        return f'**CmdOutputObservation (source={self.source}, exit code={self.exit_code})**\n{self.content}'


@dataclass
class IPythonRunCellObservation(Observation):
    """This data class represents the output of a IPythonRunCellAction."""

    code: str
    observation: str = ObservationType.RUN_IPYTHON

    @property
    def error(self) -> bool:
        return False  # IPython cells do not return exit codes

    @property
    def message(self) -> str:
        return 'Code executed in IPython cell.'

    def __str__(self) -> str:
        return f'**IPythonRunCellObservation**\n{self.content}'
