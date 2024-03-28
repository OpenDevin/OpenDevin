from typing import List
from .prompt import get_prompt, parse_response

from opendevin.agent import Agent
from opendevin.llm.llm import LLM
from opendevin.state import State
from opendevin.action import Action

class PlannerAgent(Agent):
    def __init__(self, llm: LLM):
        super().__init__(llm)

    def step(self, state: State) -> Action:
        prompt = get_prompt(state.plan, state.history)
        messages = [{"content": prompt, "role": "user"}]
        resp = self.llm.completion(messages=messages)
        action_resp = resp['choices'][0]['message']['content']
        action = parse_response(action_resp)
        return action

    def search_memory(self, query: str) -> List[str]:
        return []

Agent.register("PlannerAgent", PlannerAgent)
