from jinja2 import BaseLoader, Environment

from openhands.agenthub.micro.instructions import instructions
from openhands.agenthub.micro.registry import all_microagents
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.message import ImageContent, Message, TextContent
from openhands.core.utils import json
from openhands.events.action import Action
from openhands.events.serialization.action import action_from_dict
from openhands.events.serialization.event import event_to_memory
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory


def parse_response(orig_response: str) -> Action:
    # attempt to load the JSON dict from the response
    action_dict = json.loads(orig_response)

    # load the action from the dict
    return action_from_dict(action_dict)


def to_json(obj, **kwargs):
    """Serialize an object to str format"""
    return json.dumps(obj, **kwargs)


class MicroAgent(Agent):
    VERSION = '1.0'
    prompt = ''
    agent_definition: dict = {}

    def __init__(self, llm: LLM, config: AgentConfig, memory: ConversationMemory):
        # init the agent with the required arguments
        super().__init__(llm, config, memory)

        # check if the agent definition is valid
        if 'name' not in self.agent_definition:
            raise ValueError('Agent definition must contain a name')

        # set the prompt template and delegates
        self.prompt_template = Environment(loader=BaseLoader).from_string(self.prompt)
        self.delegates = all_microagents.copy()
        del self.delegates[self.agent_definition['name']]

    def history_to_json(self, max_events: int = 20, **kwargs):
        """
        Serialize and simplify history to str format
        """
        processed_history = []
        event_count = 0

        for event in self.memory.get_events(reverse=True):
            if event_count >= max_events:
                break
            processed_history.append(
                event_to_memory(event, self.llm.config.max_message_chars)
            )
            event_count += 1

        # history is in reverse order, let's fix it
        processed_history.reverse()

        return json.dumps(processed_history, **kwargs)

    def step(self, state: State) -> Action:
        last_user_message, last_image_urls = self.memory.get_current_user_intent()
        prompt = self.prompt_template.render(
            state=state,
            instructions=instructions,
            to_json=to_json,
            history_to_json=self.history_to_json,
            delegates=self.delegates,
            latest_user_message=last_user_message,
        )
        content = [TextContent(text=prompt)]
        if self.llm.vision_is_active() and last_image_urls:
            content.append(ImageContent(image_urls=last_image_urls))
        message = Message(role='user', content=content)
        resp = self.llm.completion(
            messages=self.llm.format_messages_for_llm(message),
        )
        action_resp = resp['choices'][0]['message']['content']
        action = parse_response(action_resp)
        return action
