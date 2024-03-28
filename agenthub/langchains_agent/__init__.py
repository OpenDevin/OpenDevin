from opendevin.agent import Agent
from .langchains_agent import LangchainsAgent

Agent.register("LangchainsAgent", LangchainsAgent)
