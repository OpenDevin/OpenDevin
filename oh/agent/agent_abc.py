from abc import ABC, abstractmethod

from oh.agent.agent_config import AgentConfig


class AgentABC(ABC):

    @abstractmethod
    def prompt(self, text: str):
        """Pass a prompt to the agent"""

    @abstractmethod
    def stop():
        """Stop the current command"""


_AGENTS = {}


def get_agent(agent_config: AgentConfig) -> AgentABC:
    from oh.agent.mock_agent import MockAgent

    return MockAgent()
