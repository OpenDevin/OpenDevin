import os

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.utils.obs import flatten_axtree_to_str

from openhands.agenthub.browsing_agent.response_parser import BrowsingResponseParser
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.message import Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    BrowseInteractiveAction,
    MessageAction,
)
from openhands.events.event import EventSource
from openhands.events.observation import BrowserOutputObservation
from openhands.events.observation.observation import Observation
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    PluginRequirement,
)

USE_NAV = (
    os.environ.get('USE_NAV', 'true') == 'true'
)  # only disable NAV actions when running webarena and miniwob benchmarks
USE_CONCISE_ANSWER = (
    os.environ.get('USE_CONCISE_ANSWER', 'false') == 'true'
)  # only return concise answer when running webarena and miniwob benchmarks

if not USE_NAV and USE_CONCISE_ANSWER:
    EVAL_MODE = True  # disabled NAV actions and only return concise answer, for webarena and miniwob benchmarks\
else:
    EVAL_MODE = False


def get_error_prefix(last_browser_action: str) -> str:
    return f'IMPORTANT! Last action is incorrect:\n{last_browser_action}\nThink again with the current observation of the page.\n'


def get_system_message(goal: str, action_space: str) -> str:
    return f"""\
# Instructions
Review the current state of the page and all other information to find the best
possible next action to accomplish your goal. Your answer will be interpreted
and executed by a program, make sure to follow the formatting instructions.

# Goal:
{goal}

# Action Space
{action_space}
"""


CONCISE_INSTRUCTION = """\

Here is another example with chain of thought of a valid action when providing a concise answer to user:
"
In order to accomplish my goal I need to send the information asked back to the user. This page list the information of HP Inkjet Fax Machine, which is the product identified in the objective. Its price is $279.49. I will send a message back to user with the answer.
```send_msg_to_user("$279.49")```
"
"""


def get_prompt(
    error_prefix: str, cur_url: str, cur_axtree_txt: str, prev_action_str: str
) -> str:
    prompt = f"""\
{error_prefix}

# Current Page URL:
{cur_url}

# Current Accessibility Tree:
{cur_axtree_txt}

# Previous Actions
{prev_action_str}

Here is an example with chain of thought of a valid action when clicking on a button:
"
In order to accomplish my goal I need to click on the button with bid 12
```click("12")```
"
""".strip()
    if USE_CONCISE_ANSWER:
        prompt += CONCISE_INSTRUCTION
    return prompt


class BrowsingAgent(Agent):
    VERSION = '1.0'
    """
    An agent that interacts with the browser.
    """

    sandbox_plugins: list[PluginRequirement] = []
    response_parser = BrowsingResponseParser()

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        memory: ConversationMemory,
    ) -> None:
        """Initializes a new instance of the BrowsingAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm, config, memory)
        # define a configurable action space, with chat functionality, web navigation, and webpage grounding using accessibility tree and HTML.
        # see https://github.com/ServiceNow/BrowserGym/blob/main/core/src/browsergym/core/action/highlevel.py for more details
        action_subsets = ['chat', 'bid']
        if USE_NAV:
            action_subsets.append('nav')
        self.action_space = HighLevelActionSet(
            subsets=action_subsets,
            strict=False,  # less strict on the parsing of the actions
            multiaction=True,  # enable to agent to take multiple actions at once
        )

        self.reset()

    def reset(self) -> None:
        """Resets the Browsing Agent."""
        super().reset()
        self.cost_accumulator = 0
        self.error_accumulator = 0

    def step(self, state: State) -> Action:
        """Performs one step using the Browsing Agent.
        This includes gathering information on previous steps and prompting the model to make a browsing command to execute.

        Parameters:
        - state (State): used to get updated info

        Returns:
        - BrowseInteractiveAction(browsergym_command) - BrowserGym commands to run
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        messages: list[Message] = []
        prev_actions = []
        cur_url = ''
        cur_axtree_txt = ''
        error_prefix = ''
        last_obs = None
        last_action = None

        if EVAL_MODE and len(self.memory.get_events()) == 1:
            # for webarena and miniwob++ eval, we need to retrieve the initial observation already in browser env
            # initialize and retrieve the first observation by issuing an noop OP
            # For non-benchmark browsing, the browser env starts with a blank page, and the agent is expected to first navigate to desired websites
            return BrowseInteractiveAction(browser_actions='noop()')

        for event in self.memory.get_events():
            if isinstance(event, BrowseInteractiveAction):
                prev_actions.append(event.browser_actions)
                last_action = event
            elif isinstance(event, MessageAction) and event.source == EventSource.AGENT:
                # agent has responded, task finished.
                return AgentFinishAction(outputs={'content': event.content})
            elif isinstance(event, Observation):
                last_obs = event

        if EVAL_MODE:
            prev_actions = prev_actions[1:]  # remove the first noop action

        prev_action_str = '\n'.join(prev_actions)
        # if the final BrowserInteractiveAction exec BrowserGym's send_msg_to_user,
        # we should also send a message back to the user in OpenHands and call it a day
        if (
            isinstance(last_action, BrowseInteractiveAction)
            and last_action.browsergym_send_msg_to_user
        ):
            return MessageAction(last_action.browsergym_send_msg_to_user)

        if isinstance(last_obs, BrowserOutputObservation):
            if last_obs.error:
                # add error recovery prompt prefix
                error_prefix = get_error_prefix(last_obs.last_browser_action)
                self.error_accumulator += 1
                if self.error_accumulator > 5:
                    return MessageAction('Too many errors encountered. Task failed.')

            cur_url = last_obs.url

            try:
                cur_axtree_txt = flatten_axtree_to_str(
                    last_obs.axtree_object,
                    extra_properties=last_obs.extra_element_properties,
                    with_clickable=True,
                    filter_visible_only=True,
                )
            except Exception as e:
                logger.error(
                    'Error when trying to process the accessibility tree: %s', e
                )
                return MessageAction('Error encountered when browsing.')

        goal, _ = self.memory.get_current_user_intent()

        if goal is None:
            goal = state.inputs['task']

        system_msg = get_system_message(
            goal,
            self.action_space.describe(with_long_description=False, with_examples=True),
        )

        messages.append(Message(role='system', content=[TextContent(text=system_msg)]))

        prompt = get_prompt(error_prefix, cur_url, cur_axtree_txt, prev_action_str)
        messages.append(Message(role='user', content=[TextContent(text=prompt)]))

        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
            stop=[')```', ')\n```'],
        )
        return self.response_parser.parse(response)
