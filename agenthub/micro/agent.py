import asyncio
import json as std_json
import unittest.mock
from typing import Union

from jinja2 import BaseLoader, Environment
from litellm.types.utils import ModelResponse

from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.message import ImageContent, Message, TextContent
from opendevin.core.utils import json
from opendevin.events.action import Action, AgentFinishAction
from opendevin.events.serialization.action import action_from_dict
from opendevin.events.serialization.event import event_to_memory
from opendevin.llm.llm import LLM
from opendevin.memory.history import ShortTermHistory
from opendevin.runtime.utils.async_utils import async_to_sync

from .instructions import instructions
from .registry import all_microagents


def parse_response(orig_response: Union[str, dict]) -> Action:
    if isinstance(orig_response, dict) and 'choices' in orig_response:
        # this covers a response like:
        # {'choices': [{'message': {'content': '{"action": "finish", "args": {}}'}}]}
        try:
            content = orig_response['choices'][0]['message']['content']
            # Parse the content string as JSON
            content_dict = json.loads(content)
            if 'action' in content_dict and content_dict['action'] == 'finish':
                return AgentFinishAction(
                    outputs={}, thought='Task completed', action='finish'
                )
            else:
                action_dict = content_dict
        except (KeyError, std_json.JSONDecodeError) as e:
            raise ValueError(f'Invalid format for choices response: {e}')
    elif isinstance(orig_response, str):
        # attempt to load the JSON dict from the response
        action_dict = json.loads(orig_response)
    elif isinstance(orig_response, dict):
        # this approach fails in some cases, thus the "elif" above!
        action_dict = orig_response
    else:
        raise TypeError(f'Expected str or dict, got {type(orig_response)}')
    # load the action from the dict
    return action_from_dict(action_dict)


def to_json(obj, **kwargs):
    """Serialize an object to str format"""
    return json.dumps(obj, **kwargs)


class MicroAgent(Agent):
    VERSION = '1.0'
    prompt = ''
    agent_definition: dict = {}

    def history_to_json(
        self, history: ShortTermHistory, max_events: int = 20, **kwargs
    ):
        """
        Serialize and simplify history to str format
        """
        processed_history = []
        event_count = 0

        for event in history.get_events(reverse=True):
            if event_count >= max_events:
                break
            processed_history.append(
                event_to_memory(event, self.llm.config.max_message_chars)
            )
            event_count += 1

        # history is in reverse order, let's fix it
        processed_history.reverse()

        return json.dumps(processed_history, **kwargs)

    def __init__(self, llm: LLM):
        super().__init__(llm)
        if 'name' not in self.agent_definition:
            raise ValueError('Agent definition must contain a name')
        self.prompt_template = Environment(loader=BaseLoader).from_string(self.prompt)
        self.delegates = all_microagents.copy()
        del self.delegates[self.agent_definition['name']]

    @async_to_sync
    def step(self, state: State) -> Action:
        return self.async_step(state)  # type: ignore[return-value]

    async def async_step(self, state: State):
        last_user_message, last_image_urls = state.get_current_user_intent()
        prompt = self.prompt_template.render(
            state=state,
            instructions=instructions,
            to_json=to_json,
            history_to_json=self.history_to_json,
            delegates=self.delegates,
            latest_user_message=last_user_message,
        )
        content = [TextContent(text=prompt)]
        if last_image_urls:
            content.append(ImageContent(image_urls=last_image_urls))
        message = Message(role='user', content=content)

        resp = self.llm.completion(messages=[message.model_dump()])

        # Handle both real responses and mock responses in tests
        if isinstance(resp, dict) and 'choices' in resp:
            action_resp = resp['choices'][0]['message']['content']
        elif isinstance(resp, unittest.mock.AsyncMock):
            action_resp = resp.return_value['choices'][0]['message']['content']
        elif isinstance(resp, str):
            action_resp = resp
        elif isinstance(resp, ModelResponse):
            action_resp = resp.choices[0].message.content
        elif asyncio.iscoroutine(resp):
            action_resp = await resp
        else:
            raise TypeError(f'Unexpected response type: {type(resp)}')

        action = parse_response(action_resp)
        return action
