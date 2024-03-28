from opendevin.agent import Agent
from .codeact_agent import CodeActAgent

Agent.register("CodeActAgent", CodeActAgent)
