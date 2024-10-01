from dotenv import load_dotenv

from agenthub.micro.agent import MicroAgent
from agenthub.micro.registry import all_microagents
from openhands.controller.agent import Agent

load_dotenv()


from agenthub import (  # noqa: E402
    browsing_agent,
    coact_agent,
    codeact_agent,
    codeact_swe_agent,
    delegator_agent,
    dummy_agent,
    planner_agent,
)

__all__ = [
    'codeact_agent',
    'codeact_swe_agent',
    'planner_agent',
    'delegator_agent',
    'dummy_agent',
    'browsing_agent',
    'coact_agent',
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
