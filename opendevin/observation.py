import copy
from dataclasses import dataclass
from typing import List


@dataclass
class Observation:
    """
    This data class represents an observation of the environment.
    """

    content: str

    def __str__(self) -> str:
        return self.content

    def to_dict(self) -> dict:
        """Converts the observation to a dictionary."""
        extras = copy.deepcopy(self.__dict__)
        extras.pop("content", None)
        return {
            "observation": self.__class__.__name__,
            "content": self.content,
            "extras": extras,
            "message": self.message,
        }

    @property
    def message(self) -> str:
        """Returns a message describing the observation."""
        return "The agent made an observation."


@dataclass
class CmdOutputObservation(Observation):
    """
    This data class represents the output of a command.
    """

    command_id: int
    command: str
    exit_code: int = 0

    @property
    def error(self) -> bool:
        return self.exit_code != 0

    @property
    def message(self) -> str:
        return f'The agent observed command "{self.command}" executed with exit code {self.exit_code}.'


@dataclass
class BrowserOutputObservation(Observation):
    """
    This data class represents the output of a browser.
    """

    url: str

    @property
    def message(self) -> str:
        return "The agent observed the browser output at URL."


@dataclass
class UserMessageObservation(Observation):
    """
    This data class represents a message sent by the user.
    """

    role: str = "user"

    @property
    def message(self) -> str:
        return "The agent received a message from the user."


@dataclass
class AgentMessageObservation(Observation):
    """
    This data class represents a message sent by the agent.
    """

    role: str = "assistant"

    @property
    def message(self) -> str:
        return "The agent received a message from itself."


@dataclass
class AgentRecallObservation(Observation):
    """
    This data class represents a list of memories recalled by the agent.
    """

    memories: List[str]
    role: str = "assistant"

    @property
    def message(self) -> str:
        return "The agent recalled memories."


@dataclass
class NullObservation(Observation):
    """
    This data class represents a null observation.
    This is used when the produced action is NOT executable.
    """

    @property
    def message(self) -> str:
        return ""
