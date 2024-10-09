from dataclasses import dataclass

from oh.agent.agent_info import AgentInfo


@dataclass
class AgentConfig(AgentInfo):
    key: str
