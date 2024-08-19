from agenthub.codeact_agent import CodeActAgent
from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.config import AgentConfig
from opendevin.core.message import Message, TextContent
from opendevin.events.action import (
    Action,
    AgentFinishAction,
)
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import PluginRequirement

from .action_parser import SelfDiscoverResponseToActionParser
from .agent_state_machine import SelfDiscoverState, SelfDiscoverStateMachine
from .prompt import (
    IMPLEMENT_EXAMPLE,
    SYSTEM_MESSAGE,
    TASK_KEY,
    get_prompt,
)
from .reasoning_action import ReasoningAction
from .reasoning_action_parser import SelfDiscoverResponseToReasoningActionParser


class SelfDiscoverAgent(Agent):
    VERSION = '0.1'
    """
    This agent implements a Self Discover Agent as described in https://arxiv.org/abs/2402.03620.
    The reasoning structure is sent to the CodeActAgent for execution.
    """

    action_parser = SelfDiscoverResponseToActionParser()
    reasoning_parser = SelfDiscoverResponseToReasoningActionParser()
    sandbox_plugins: list[PluginRequirement] = CodeActAgent.sandbox_plugins
    system_message = SYSTEM_MESSAGE
    implement_example = IMPLEMENT_EXAMPLE.format(
        implement_state_key=SelfDiscoverState.IMPLEMENT.value,
        task_key=TASK_KEY,
    )

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """
        Initializes a new instance of the SelfDiscoveryAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        self.agent_state_machine: SelfDiscoverStateMachine = SelfDiscoverStateMachine()
        self.reasoning_data: ReasoningAction = ReasoningAction()
        super().__init__(llm, config)
        self.reset()

    def reset(self) -> None:
        """
        Resets the Agent.
        """
        self.agent_state_machine.reset()
        self.reasoning_data.reset()
        super().reset()

    def get_in_context_example(self) -> str | None:
        if self.agent_state_machine.current_state == SelfDiscoverState.IMPLEMENT:
            return self.implement_example
        else:
            return None

    def get_messages(self, state: State) -> list[Message]:
        messages: list[Message] = [
            Message(role='system', content=[TextContent(text=self.system_message)]),
        ]

        if in_context_example := self.get_in_context_example() is not None:
            in_context_example_prompt = (
                'Here is an example of how you can interact with the environment:\n'
                f"{in_context_example}\n\nNOW, LET'S START!"
            )
            messages.append(
                Message(
                    role='user', content=[TextContent(text=in_context_example_prompt)]
                )
            )

        return messages

    def step(self, state: State) -> Action:
        """
        Parameters:
        - state (State): used to get updated info and background commands

        Returns:

        """

        # abort if user desires
        latest_user_message = state.history.get_last_user_message()
        if latest_user_message and latest_user_message.strip() == '/exit':
            return AgentFinishAction(thought='Aborted by the user.')

        # prepare what we want to send to the LLM
        messages: list[Message] = self.get_messages(state)
        # print(f'messages:\n{messages}\n')

        # Finish when plan as been executed by CodeActAgent
        if isinstance(state.history.get_last_action(), AgentFinishAction):
            self.agent_state_machine.reset()
            return AgentFinishAction()

        # add self discover prompt to messages
        if prompt := get_prompt(
            state.get_current_user_intent(),
            self.agent_state_machine.current_state,
            self.reasoning_data,
        ):
            # print(f'prompt: {prompt}')
            messages.append(prompt)

        response = self.llm.completion(
            messages=[message.model_dump() for message in messages],
            temperature=0.0,
        )

        reasoning_dict = self.reasoning_parser.parse(response)
        # print(f'reasoning_dict: {reasoning_dict}')
        self.reasoning_data.update_data(reasoning_dict)
        # print(f'self.reasoning_data: {self.reasoning_data}')

        action = self.action_parser.parse(response)
        self.agent_state_machine.transition(action)
        return action
