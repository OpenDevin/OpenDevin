from dataclasses import dataclass


@dataclass
class AgentInfo:
    type: str
    llm: str
    key: str
