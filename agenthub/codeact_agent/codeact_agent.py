from agenthub.codeact_agent.action_parser import CodeActResponseParser
from agenthub.codeact_agent.prompt import (
    COMMAND_DOCS,
    EXAMPLES,
    GITHUB_MESSAGE,
    SYSTEM_PREFIX,
    SYSTEM_SUFFIX,
)
from opendevin.controller.agent import AsyncAgent
from opendevin.controller.state.state import State
from opendevin.core.message import ImageContent, Message, TextContent
from opendevin.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    IPythonRunCellAction,
    MessageAction,
)
from opendevin.events.observation import (
    AgentDelegateObservation,
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from opendevin.events.observation.observation import Observation
from opendevin.events.serialization.event import truncate_content
from opendevin.llm.llm import LLM
from opendevin.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)
from opendevin.runtime.tools import RuntimeTool

ENABLE_GITHUB = True


# FIXME: We can tweak these two settings to create MicroAgents specialized toward different area
def get_system_message() -> str:
    if ENABLE_GITHUB:
        return f'{SYSTEM_PREFIX}\n{GITHUB_MESSAGE}\n\n{COMMAND_DOCS}\n\n{SYSTEM_SUFFIX}'
    else:
        return f'{SYSTEM_PREFIX}\n\n{COMMAND_DOCS}\n\n{SYSTEM_SUFFIX}'


def get_in_context_example() -> str:
    return EXAMPLES


class CodeActAgent(AsyncAgent):
    VERSION = '1.8'
    """
    The Code Act Agent is a minimalist agent.
    The agent works by passing the model a list of action-observation pairs and prompting the model to take the next step.

    ### Overview

    This agent implements the CodeAct idea ([paper](https://arxiv.org/abs/2402.01030), [tweet](https://twitter.com/xingyaow_/status/1754556835703751087)) that consolidates LLM agents’ **act**ions into a unified **code** action space for both *simplicity* and *performance* (see paper for more details).

    The conceptual idea is illustrated below. At each turn, the agent can:

    1. **Converse**: Communicate with humans in natural language to ask for clarification, confirmation, etc.
    2. **CodeAct**: Choose to perform the task by executing code
    - Execute any valid Linux `bash` command
    - Execute any valid `Python` code with [an interactive Python interpreter](https://ipython.org/). This is simulated through `bash` command, see plugin system below for more details.

    ![image](https://github.com/OpenDevin/OpenDevin/assets/38853559/92b622e3-72ad-4a61-8f41-8c040b6d5fb3)

    ### Plugin System

    To make the CodeAct agent more powerful with only access to `bash` action space, CodeAct agent leverages OpenDevin's plugin system:
    - [Jupyter plugin](https://github.com/OpenDevin/OpenDevin/tree/main/opendevin/runtime/plugins/jupyter): for IPython execution via bash command
    - [SWE-agent tool plugin](https://github.com/OpenDevin/OpenDevin/tree/main/opendevin/runtime/plugins/swe_agent_commands): Powerful bash command line tools for software development tasks introduced by [swe-agent](https://github.com/princeton-nlp/swe-agent).

    ### Demo

    https://github.com/OpenDevin/OpenDevin/assets/38853559/f592a192-e86c-4f48-ad31-d69282d5f6ac

    *Example of CodeActAgent with `gpt-4-turbo-2024-04-09` performing a data science task (linear regression)*

    ### Work-in-progress & Next step

    [] Support web-browsing
    [] Complete the workflow for CodeAct agent to submit Github PRs

    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]
    runtime_tools: list[RuntimeTool] = [RuntimeTool.BROWSER]

    system_message: str = get_system_message()
    in_context_example: str = f"Here is an example of how you can interact with the environment for task solving:\n{get_in_context_example()}\n\nNOW, LET'S START!"

    action_parser = CodeActResponseParser()

    def __init__(
        self,
        llm: LLM,
    ) -> None:
        """Initializes a new instance of the CodeActAgent class.

        Parameters:
        - llm (LLM): The llm to be used by this agent
        """
        super().__init__(llm)
        self.reset()

    def action_to_str(self, action: Action) -> str:
        if isinstance(action, CmdRunAction):
            return (
                f'{action.thought}\n<execute_bash>\n{action.command}\n</execute_bash>'
            )
        elif isinstance(action, IPythonRunCellAction):
            return f'{action.thought}\n<execute_ipython>\n{action.code}\n</execute_ipython>'
        elif isinstance(action, AgentDelegateAction):
            return f'{action.thought}\n<execute_browse>\n{action.inputs["task"]}\n</execute_browse>'
        elif isinstance(action, MessageAction):
            return action.content
        elif isinstance(action, AgentFinishAction) and action.source == 'agent':
            return action.thought
        return ''

    def get_action_message(self, action: Action) -> Message | None:
        if (
            isinstance(action, AgentDelegateAction)
            or isinstance(action, CmdRunAction)
            or isinstance(action, IPythonRunCellAction)
            or isinstance(action, MessageAction)
            or (isinstance(action, AgentFinishAction) and action.source == 'agent')
        ):
            content = [TextContent(text=self.action_to_str(action))]

            if isinstance(action, MessageAction) and action.images_urls:
                content.append(ImageContent(image_urls=action.images_urls))

            return Message(
                role='user' if action.source == 'user' else 'assistant', content=content
            )
        return None

    def get_observation_message(self, obs: Observation) -> Message | None:
        max_message_chars = self.llm.config.max_message_chars
        if isinstance(obs, CmdOutputObservation):
            text = 'OBSERVATION:\n' + truncate_content(obs.content, max_message_chars)
            text += (
                f'\n[Command {obs.command_id} finished with exit code {obs.exit_code}]'
            )
            return Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, IPythonRunCellObservation):
            text = 'OBSERVATION:\n' + obs.content
            # replace base64 images with a placeholder
            splitted = text.split('\n')
            for i, line in enumerate(splitted):
                if '![image](data:image/png;base64,' in line:
                    splitted[i] = (
                        '![image](data:image/png;base64, ...) already displayed to user'
                    )
            text = '\n'.join(splitted)
            text = truncate_content(text, max_message_chars)
            return Message(role='user', content=[TextContent(text=text)])
        elif isinstance(obs, AgentDelegateObservation):
            text = 'OBSERVATION:\n' + truncate_content(
                str(obs.outputs), max_message_chars
            )
            return Message(role='user', content=[TextContent(text=text)])
        return None

    def reset(self) -> None:
        """Resets the CodeAct Agent."""
        super().reset()

    def step(self, state: State) -> Action:
        """Performs one step using the CodeAct Agent.
        This includes gathering info on previous steps and prompting the model to make a command to execute.

        Parameters:
        - state (State): used to get updated info and background commands

        Returns:
        - CmdRunAction(command) - bash command to run
        - IPythonRunCellAction(code) - IPython code to run
        - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        messages, is_exit = self._prepare_messages(state)
        if is_exit:
            return AgentFinishAction()
        return self._common_step_logic_sync(state, self.llm.completion, messages)

    async def async_step(self, state: State) -> Action:
        """Asynchronously performs one step using the CodeAct Agent."""
        messages, is_exit = self._prepare_messages(state)
        if is_exit:
            return AgentFinishAction()
        return await self._common_step_logic_async(
            state, self.llm.async_completion, messages
        )

    def _common_step_logic_sync(
        self, state: State, completion_func, messages: list[Message]
    ) -> Action:
        """Common logic for the synchronous step method.

        :param state: The current state.
        :param completion_func: self.llm.completion
        :param messages: The prepared messages.
        :return: The resulting Action.
        """
        # prepare what we want to send to the LLM
        response = completion_func(
            messages=[message.model_dump() for message in messages],
            stop=[
                '</execute_ipython>',
                '</execute_bash>',
                '</execute_browse>',
            ],
            temperature=0.0,
        )
        return self.action_parser.parse(response)

    async def _common_step_logic_async(
        self, state: State, completion_func, messages: list[Message]
    ) -> Action:
        """Common logic for the asynchronous step method.

        :param state: The current state.
        :param completion_func: self.llm.async_completion
        :param messages: The prepared messages.
        :return: The resulting Action.
        """
        response = await completion_func(
            messages=[message.model_dump() for message in messages],
            stop=[
                '</execute_ipython>',
                '</execute_bash>',
                '</execute_browse>',
            ],
            temperature=0.0,
        )
        return self.action_parser.parse(response)

    def _prepare_messages(self, state: State) -> tuple[list[Message], bool]:
        """Prepare the messages for the LLM completion and check for exit command.

        Returns:
        - A tuple containing the prepared messages and a boolean indicating if it's an exit command
        """
        # Prepare messages as before
        messages: list[Message] = self._get_messages(state)
        is_exit = False

        # Check for exit command
        latest_user_message = state.history.get_last_user_message()
        if latest_user_message and latest_user_message.strip() == '/exit':
            is_exit = True

        return messages, is_exit

    def _get_messages(self, state: State) -> list[Message]:
        messages: list[Message] = [
            Message(role='system', content=[TextContent(text=self.system_message)]),
            Message(role='user', content=[TextContent(text=self.in_context_example)]),
        ]

        for event in state.history.get_events():
            # create a regular message from an event
            if isinstance(event, Action):
                message = self.get_action_message(event)
            elif isinstance(event, Observation):
                message = self.get_observation_message(event)
            else:
                raise ValueError(f'Unknown event type: {type(event)}')

            # add regular message
            if message:
                # handle error if the message is the SAME role as the previous message
                # litellm.exceptions.BadRequestError: litellm.BadRequestError: OpenAIException - Error code: 400 - {'detail': 'Only supports u/a/u/a/u...'}
                # there should not have two consecutive messages from the same role
                if messages and messages[-1].role == message.role:
                    messages[-1].content.extend(message.content)
                else:
                    messages.append(message)

        # the latest user message is important:
        # we want to remind the agent of the environment constraints
        latest_user_message = next(
            (
                m
                for m in reversed(messages)
                if m.role == 'user'
                and any(isinstance(c, TextContent) for c in m.content)
            ),
            None,
        )

        # Get the last user text inside content
        if latest_user_message:
            latest_user_message_text = next(
                (
                    t
                    for t in reversed(latest_user_message.content)
                    if isinstance(t, TextContent)
                )
            )
            # add a reminder to the prompt
            reminder_text = f'\n\nENVIRONMENT REMINDER: You have {state.max_iterations - state.iteration} turns left to complete the task. When finished reply with <finish></finish>.'

            if latest_user_message_text:
                latest_user_message_text.text = (
                    latest_user_message_text.text + reminder_text
                )
            else:
                latest_user_message_text = TextContent(text=reminder_text)
                latest_user_message.content.append(latest_user_message_text)

        return messages
