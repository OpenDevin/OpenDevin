from dataclasses import dataclass
from typing import ClassVar

from opendevin.core.schema import ActionType

from .action import Action, ActionSecurityRisk


@dataclass
class FileReadAction(Action):
    """Reads a file from a given path.
    Can be set to read specific lines using start and end
    Default lines 0:-1 (whole file)
    """

    path: str
    start: int = 0
    end: int = -1
    thought: str = ''
    action: str = ActionType.READ
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Reading file: {self.path}'


@dataclass
class FileWriteAction(Action):
    path: str
    content: str
    start: int = 0
    end: int = -1
    thought: str = ''
    action: str = ActionType.WRITE
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    @property
    def message(self) -> str:
        return f'Writing file: {self.path}'
