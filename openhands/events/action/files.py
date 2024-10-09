from dataclasses import dataclass
from typing import ClassVar

from openhands.core.schema import ActionType
from openhands.events.action.action import Action, ActionSecurityRisk


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
    """Writes a file to a given path.
    Can be set to write specific lines using start and end
    Default lines 0:-1 (whole file)
    """

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


@dataclass
class FileEditAction(Action):
    """Edits a file by provided a draft at a given path.

    Can be set to edit specific lines using start and end (1-index, inclusive) if the file is too long.
    Default lines 1:-1 (whole file).

    If start is set to -1, the FileEditAction will simply append the content to the file.
    """

    path: str
    content: str
    start: int = 1
    end: int = -1
    thought: str = ''
    action: str = ActionType.EDIT
    runnable: ClassVar[bool] = True
    security_risk: ActionSecurityRisk | None = None

    def __repr__(self) -> str:
        ret = '**FileEditAction**\n'
        ret += f'Thought: {self.thought}\n'
        ret += f'Range: [L{self.start}:L{self.end}]\n'
        ret += f'Path: [{self.path}]\n'
        ret += f'Content:\n```\n{self.content}\n```\n'
        return ret
