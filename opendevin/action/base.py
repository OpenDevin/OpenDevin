from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendevin.controller import AgentController
    from opendevin.observation import Observation

@dataclass
class Action:
    def run(self, controller: "AgentController") -> "Observation":
        raise NotImplementedError

    def to_dict(self):
        d = asdict(self)
        try:
            v = d.pop('action')
        except KeyError:
            raise NotImplementedError(f'{self=} does not have action attribute set')
        return {'action': v, "args": d}

    @property
    def executable(self) -> bool:
        raise NotImplementedError

    @property
    def message(self) -> str:
        raise NotImplementedError



@dataclass
class ExecutableAction(Action):
    @property
    def executable(self) -> bool:
        return True


@dataclass
class NotExecutableAction(Action):
    @property
    def executable(self) -> bool:
        return False

@dataclass
class NullAction(NotExecutableAction):
    """An action that does nothing.
    This is used when the agent need to receive user follow-up messages from the frontend.
    """
    action: str = "null"

    @property
    def message(self) -> str:
        return "No action"
