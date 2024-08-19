from opendevin.controller.agent import Agent

from .agent import SelfDiscoverAgent

Agent.register('SelfDiscoverAgent', SelfDiscoverAgent)
