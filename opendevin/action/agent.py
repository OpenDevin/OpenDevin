from dataclasses import dataclass
from typing import TYPE_CHECKING

from opendevin.observation import (
    AgentMessageObservation,
    AgentRecallObservation,
    Observation,
)

from .base import ExecutableAction, NotExecutableAction

if TYPE_CHECKING:
    from opendevin.controller import AgentController


@dataclass
class AgentRecallAction(ExecutableAction):
    query: str

    def run(self, controller: "AgentController") -> AgentRecallObservation:
        return AgentRecallObservation(
            content="Recalling memories...",
            memories=controller.agent.search_memory(self.query),
        )

    @property
    def message(self) -> str:
        return f"Recalling memories with query: {self.query}"


@dataclass
class AgentThinkAction(NotExecutableAction):
    thought: str
    runnable: bool = False

    def run(self, controller: "AgentController") -> "Observation":
        raise NotImplementedError

    @property
    def message(self) -> str:
        return f"Thinking: {self.thought}"


@dataclass
class AgentEchoAction(ExecutableAction):
    content: str
    runnable: bool = True

    def run(self, controller: "AgentController") -> "Observation":
        return AgentMessageObservation(self.content)

    @property
    def message(self) -> str:
        return f"Echoing: {self.content}"


@dataclass
class AgentFinishAction(NotExecutableAction):
    runnable: bool = False

    def run(self, controller: "AgentController") -> "Observation":
        raise NotImplementedError

    @property
    def message(self) -> str:
        return "Finished!"
