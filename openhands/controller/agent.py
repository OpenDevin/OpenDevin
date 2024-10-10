from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type

from openhands.memory.conversation_memory import ConversationMemory

if TYPE_CHECKING:
    from openhands.controller.state.state import State
    from openhands.core.config import AgentConfig
    from openhands.events.action import Action
from openhands.core.exceptions import (
    AgentAlreadyRegisteredError,
    AgentNotRegisteredError,
)
from openhands.llm.llm import LLM
from openhands.runtime.plugins import PluginRequirement


class Agent(ABC):
    DEPRECATED = False
    """
    This abstract base class is an general interface for an agent dedicated to
    executing a specific instruction and allowing human interaction with the
    agent during execution.
    It tracks the execution status and maintains a reference to the conversation memory.
    """

    _registry: dict[str, Type['Agent']] = {}
    sandbox_plugins: list[PluginRequirement] = []
    memory: ConversationMemory

    def __init__(
        self,
        llm: LLM,
        config: 'AgentConfig',
        memory: ConversationMemory,
    ):
        self.llm = llm
        self.config = config
        self.memory = memory
        self._complete = False

    @property
    def complete(self) -> bool:
        """Indicates whether the current instruction execution is complete.

        Returns:
        - complete (bool): True if execution is complete; False otherwise.
        """
        return self._complete

    @abstractmethod
    def step(self, state: 'State') -> 'Action':
        """Starts the execution of the assigned instruction. This method should
        be implemented by subclasses to define the specific execution logic.
        """
        pass

    def reset(self) -> None:
        """Resets the agent's execution status and clears the history. This method can be used
        to prepare the agent for restarting the instruction or cleaning up before destruction.

        """
        self.memory.reset()
        self._complete = False

        if self.llm:
            self.llm.reset()

    @property
    def name(self):
        return self.__class__.__name__

    @classmethod
    def register(cls, name: str, agent_cls: Type['Agent']):
        """Registers an agent class in the registry.

        Parameters:
        - name (str): The name to register the class under.
        - agent_cls (Type['Agent']): The class to register.

        Raises:
        - AgentAlreadyRegisteredError: If name already registered
        """
        if name in cls._registry:
            raise AgentAlreadyRegisteredError(name)
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> Type['Agent']:
        """Retrieves an agent class from the registry.

        Parameters:
        - name (str): The name of the class to retrieve

        Returns:
        - agent_cls (Type['Agent']): The class registered under the specified name.

        Raises:
        - AgentNotRegisteredError: If name not registered
        """
        if name not in cls._registry:
            raise AgentNotRegisteredError(name)
        return cls._registry[name]

    @classmethod
    def list_agents(cls) -> list[str]:
        """Retrieves the list of all agent names from the registry.

        Raises:
        - AgentNotRegisteredError: If no agent is registered
        """
        if not bool(cls._registry):
            raise AgentNotRegisteredError()
        return list(cls._registry.keys())
