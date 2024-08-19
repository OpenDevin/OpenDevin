from dotenv import load_dotenv

from opendevin.controller.agent import Agent

from .micro.agent import MicroAgent
from .micro.registry import all_microagents

load_dotenv()


from . import (  # noqa: E402
    browsing_agent,
    codeact_agent,
    codeact_swe_agent,
    delegator_agent,
    dummy_agent,
    planner_agent,
    self_discover_agent,
)

__all__ = [
    'codeact_agent',
    'codeact_swe_agent',
    'planner_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'self_discover_agent',
]

for agent in all_microagents.values():
    name = agent['name']
    prompt = agent['prompt']

    anon_class = type(
        name,
        (MicroAgent,),
        {
            'prompt': prompt,
            'agent_definition': agent,
        },
    )

    Agent.register(name, anon_class)
