from opendevin.controller.agent import Agent

from .agentless_agent import AgentlessAgent

Agent.register('AgentlessAgent', AgentlessAgent)
