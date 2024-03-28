from abc import ABC, abstractmethod
from typing import List, Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from opendevin.action import Action
    from opendevin.state import State
from opendevin.llm.llm import LLM

class Agent(ABC):
    """
    This abstract base class is an general interface for an agent dedicated to
    executing a specific instruction and allowing human interaction with the
    agent during execution.
    It tracks the execution status and maintains a history of interactions.
    """

    _registry: Dict[str, Type["Agent"]] = {}

    def __init__(
        self,
        llm: LLM,
    ):
        self.llm = llm
        self._complete = False

    @property
    def complete(self) -> bool:
        """
        Indicates whether the current instruction execution is complete.

        Returns:
        - complete (bool): True if execution is complete; False otherwise.
        """
        return self._complete

    @abstractmethod
    def step(self, state: "State") -> "Action":
        """
        Starts the execution of the assigned instruction. This method should
        be implemented by subclasses to define the specific execution logic.
        """
        pass

    @abstractmethod
    def search_memory(self, query: str) -> List[str]:
        """
        Searches the agent's memory for information relevant to the given query.

        Parameters:
        - query (str): The query to search for in the agent's memory.

        Returns:
        - response (str): The response to the query.
        """
        pass

    def reset(self) -> None:
        """
        Resets the agent's execution status and clears the history. This method can be used
        to prepare the agent for restarting the instruction or cleaning up before destruction.

        """
        self._complete = False

    @classmethod
    def register(cls, name: str, agent_cls: Type["Agent"]):
        """
        Registers an agent class in the registry.

        Parameters:
        - name (str): The name to register the class under.
        - agent_cls (Type['Agent']): The class to register.
        """
        if name in cls._registry:
            raise ValueError(f"Agent class already registered under '{name}'.")
        cls._registry[name] = agent_cls

    @classmethod
    def get_cls(cls, name: str) -> Type["Agent"]:
        """
        Retrieves an agent class from the registry.

        Parameters:
        - name (str): The name of the class to retrieve

        Returns:
        - agent_cls (Type['Agent']): The class registered under the specified name.
        """
        if name not in cls._registry:
            raise ValueError(f"No agent class registered under '{name}'.")
        return cls._registry[name]
