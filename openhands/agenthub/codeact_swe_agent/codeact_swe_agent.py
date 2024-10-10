from openhands.agenthub.codeact_swe_agent.prompt import (
    COMMAND_DOCS,
    SWE_EXAMPLE,
    SYSTEM_PREFIX,
    SYSTEM_SUFFIX,
)
from openhands.agenthub.codeact_swe_agent.response_parser import (
    CodeActSWEResponseParser,
)
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.core.message import ImageContent, Message, TextContent
from openhands.events.action import (
    Action,
    AgentFinishAction,
    CmdRunAction,
    IPythonRunCellAction,
    MessageAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    IPythonRunCellObservation,
)
from openhands.events.observation.error import ErrorObservation
from openhands.events.observation.observation import Observation
from openhands.events.serialization.event import truncate_content
from openhands.llm.llm import LLM
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.plugins import (
    AgentSkillsRequirement,
    JupyterRequirement,
    PluginRequirement,
)


def get_system_message() -> str:
    return f'{SYSTEM_PREFIX}\n\n{COMMAND_DOCS}\n\n{SYSTEM_SUFFIX}'


def get_in_context_example() -> str:
    return SWE_EXAMPLE


class CodeActSWEAgent(Agent):
    VERSION = '1.6'
    """
    This agent is an adaptation of the original [SWE Agent](https://swe-agent.com/) based on CodeAct 1.5 using the `agentskills` library of OpenHands.

    It is intended use is **solving Github issues**.

    It removes web-browsing and Github capability from the original CodeAct agent to avoid confusion to the agent.
    """

    sandbox_plugins: list[PluginRequirement] = [
        # NOTE: AgentSkillsRequirement need to go before JupyterRequirement, since
        # AgentSkillsRequirement provides a lot of Python functions,
        # and it needs to be initialized before Jupyter for Jupyter to use those functions.
        AgentSkillsRequirement(),
        JupyterRequirement(),
    ]

    system_message: str = get_system_message()
    in_context_example: str = f"Here is an example of how you can interact with the environment for task solving:\n{get_in_context_example()}\n\nNOW, LET'S START!"

    response_parser = CodeActSWEResponseParser()

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
        memory: ConversationMemory,
    ) -> None:
        super().__init__(llm, config, memory)
        self.reset()

    def action_to_str(self, action: Action) -> str:
        if isinstance(action, CmdRunAction):
            return (
                f'{action.thought}\n<execute_bash>\n{action.command}\n</execute_bash>'
            )
        elif isinstance(action, IPythonRunCellAction):
            return f'{action.thought}\n<execute_ipython>\n{action.code}\n</execute_ipython>'
        elif isinstance(action, MessageAction):
            return action.content
        return ''

    def get_action_message(self, action: Action) -> Message | None:
        if (
            isinstance(action, CmdRunAction)
            or isinstance(action, IPythonRunCellAction)
            or isinstance(action, MessageAction)
        ):
            content = [TextContent(text=self.action_to_str(action))]

            if (
                self.llm.vision_is_active()
                and isinstance(action, MessageAction)
                and action.images_urls
            ):
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
        elif isinstance(obs, ErrorObservation):
            text = 'OBSERVATION:\n' + truncate_content(obs.content, max_message_chars)
            text += '\n[Error occurred in processing last action]'
            return Message(role='user', content=[TextContent(text=text)])
        else:
            # If an observation message is not returned, it will cause an error
            # when the LLM tries to return the next message
            raise ValueError(f'Unknown observation type: {type(obs)}')

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
        - MessageAction(content) - Message action to run (e.g. ask for clarification)
        - AgentFinishAction() - end the interaction
        """
        # if we're done, go back
        latest_user_message = self.memory.get_last_user_message()
        if latest_user_message and latest_user_message.strip() == '/exit':
            return AgentFinishAction()

        # prepare what we want to send to the LLM
        messages: list[Message] = self._get_messages(state)
        response = self.llm.completion(
            messages=self.llm.format_messages_for_llm(messages),
            stop=[
                '</execute_ipython>',
                '</execute_bash>',
            ],
        )

        return self.response_parser.parse(response)

    def _get_messages(self, state: State) -> list[Message]:
        messages: list[Message] = [
            Message(role='system', content=[TextContent(text=self.system_message)]),
            Message(role='user', content=[TextContent(text=self.in_context_example)]),
        ]

        for event in self.memory.get_events():
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
            (m for m in reversed(messages) if m.role == 'user'), None
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
