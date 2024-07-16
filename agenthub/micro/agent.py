from jinja2 import BaseLoader, Environment

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.config import config
from opendevin.core.utils import json
from opendevin.events.action import Action
from opendevin.events.serialization.action import action_from_dict
from opendevin.events.serialization.event import event_to_memory
from opendevin.llm.llm import LLM
from opendevin.memory.history import ShortTermHistory

from .instructions import instructions
from .registry import all_microagents


def parse_response(orig_response: str) -> Action:
    # attempt to load the JSON dict from the response
    action_dict = json.loads(orig_response)

    # load the action from the dict
    return action_from_dict(action_dict)


def to_json(obj, **kwargs):
    """Serialize an object to str format"""
    return json.dumps(obj, **kwargs)


def history_to_json(history: ShortTermHistory, max_events=20, **kwargs):
    """Serialize and simplify history to str format"""
    # TODO: get agent specific llm config
    llm_config = config.get_llm_config()
    max_message_chars = llm_config.max_message_chars

    processed_history = []
    event_count = 0

    for event in history.get_events(reverse=True):
        if event_count >= max_events:
            break
        processed_history.append(event_to_memory(event, max_message_chars))
        event_count += 1

    # history is in reverse order, let's fix it
    processed_history.reverse()

    return json.dumps(processed_history, **kwargs)


class MicroAgent(Agent):
    VERSION = '1.0'
    prompt = ''
    agent_definition: dict = {}

    def __init__(self, llm: LLM):
        super().__init__(llm)
        if 'name' not in self.agent_definition:
            raise ValueError('Agent definition must contain a name')
        self.prompt_template = Environment(loader=BaseLoader).from_string(self.prompt)
        self.delegates = all_microagents.copy()
        del self.delegates[self.agent_definition['name']]

    def step(self, state: State) -> Action:
        last_user_message, last_user_message_images = state.get_current_user_intent()
        prompt = self.prompt_template.render(
            state=state,
            instructions=instructions,
            to_json=to_json,
            history_to_json=history_to_json,
            delegates=self.delegates,
            latest_user_message=last_user_message,
        )
        content = [{'type': 'text', 'text': prompt}]
        if last_user_message_images:
            for image_url in last_user_message_images:
                content.append({'type': 'image_url', 'image_url': {'url': image_url}})

        messages = [{'content': content, 'role': 'user'}]
        resp = self.llm.completion(messages=messages)
        action_resp = resp['choices'][0]['message']['content']
        action = parse_response(action_resp)
        return action
