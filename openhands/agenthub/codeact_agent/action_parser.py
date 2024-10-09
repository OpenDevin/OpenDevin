import re

from openhands.controller.action_parser import (
    ActionParser,
    ResponseParser,
)
from openhands.core.exceptions import LLMMalformedActionError
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    FileEditAction,
    IPythonRunCellAction,
    MessageAction,
)


class CodeActResponseParser(ResponseParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - FileEditAction(path, content) - edit a file
    - IPythonRunCellAction(code) - IPython code to run
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    - AgentFinishAction() - end the interaction
    """

    def __init__(self):
        # Need pay attention to the item order in self.action_parsers
        super().__init__()
        self.action_parsers = [
            CodeActActionParserFinish(),
            CodeActActionParserFileEdit(),
            CodeActActionParserCmdRun(),
            CodeActActionParserIPythonRunCell(),
            CodeActActionParserAgentDelegate(),
        ]
        self.default_parser = CodeActActionParserMessage()

    def parse(self, response) -> Action:
        action_str = self.parse_response(response)
        return self.parse_action(action_str)

    def parse_response(self, response) -> str:
        action = response.choices[0].message.content
        if action is None:
            return ''
        for lang in ['bash', 'ipython', 'browse']:
            # special handling for DeepSeek: it has stop-word bug and returns </execute_ipython instead of </execute_ipython>
            if f'</execute_{lang}' in action and f'</execute_{lang}>' not in action:
                action = action.replace(f'</execute_{lang}', f'</execute_{lang}>')

            if f'<execute_{lang}>' in action and f'</execute_{lang}>' not in action:
                action += f'</execute_{lang}>'
        if '<file_edit' in action and '</file_edit>' not in action:
            action += '</file_edit>'
        return action

    def parse_action(self, action_str: str) -> Action:
        for action_parser in self.action_parsers:
            if action_parser.check_condition(action_str):
                return action_parser.parse(action_str)
        return self.default_parser.parse(action_str)


class CodeActActionParserFinish(ActionParser):
    """Parser action:
    - AgentFinishAction() - end the interaction
    """

    def __init__(
        self,
    ):
        self.finish_command = None

    def check_condition(self, action_str: str) -> bool:
        self.finish_command = re.search(r'<finish>.*</finish>', action_str, re.DOTALL)
        return self.finish_command is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.finish_command is not None
        ), 'self.finish_command should not be None when parse is called'
        thought = action_str.replace(self.finish_command.group(0), '').strip()
        return AgentFinishAction(thought=thought)


class CodeActActionParserCmdRun(ActionParser):
    """Parser action:
    - CmdRunAction(command) - bash command to run
    - AgentFinishAction() - end the interaction
    """

    def __init__(
        self,
    ):
        self.bash_command = None

    def check_condition(self, action_str: str) -> bool:
        self.bash_command = re.search(
            r'<execute_bash>(.*?)</execute_bash>', action_str, re.DOTALL
        )
        return self.bash_command is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.bash_command is not None
        ), 'self.bash_command should not be None when parse is called'
        thought = action_str.replace(self.bash_command.group(0), '').strip()
        # a command was found
        command_group = self.bash_command.group(1).strip()
        if command_group.strip() == 'exit':
            return AgentFinishAction(thought=thought)
        return CmdRunAction(command=command_group, thought=thought)


class CodeActActionParserIPythonRunCell(ActionParser):
    """Parser action:
    - IPythonRunCellAction(code) - IPython code to run
    """

    def __init__(
        self,
    ):
        self.python_code = None
        self.jupyter_kernel_init_code: str = 'from agentskills import *'

    def check_condition(self, action_str: str) -> bool:
        self.python_code = re.search(
            r'<execute_ipython>(.*?)</execute_ipython>', action_str, re.DOTALL
        )
        return self.python_code is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.python_code is not None
        ), 'self.python_code should not be None when parse is called'
        code_group = self.python_code.group(1).strip()
        thought = action_str.replace(self.python_code.group(0), '').strip()
        return IPythonRunCellAction(
            code=code_group,
            thought=thought,
            kernel_init_code=self.jupyter_kernel_init_code,
        )


class CodeActActionParserAgentDelegate(ActionParser):
    """Parser action:
    - AgentDelegateAction(agent, inputs) - delegate action for (sub)task
    """

    def __init__(
        self,
    ):
        self.agent_delegate = None

    def check_condition(self, action_str: str) -> bool:
        self.agent_delegate = re.search(
            r'<execute_browse>(.*)</execute_browse>', action_str, re.DOTALL
        )
        return self.agent_delegate is not None

    def parse(self, action_str: str) -> Action:
        assert (
            self.agent_delegate is not None
        ), 'self.agent_delegate should not be None when parse is called'
        thought = action_str.replace(self.agent_delegate.group(0), '').strip()
        browse_actions = self.agent_delegate.group(1).strip()
        task = f'{thought}. I should start with: {browse_actions}'
        return AgentDelegateAction(agent='BrowsingAgent', inputs={'task': task})


class CodeActActionParserMessage(ActionParser):
    """Parser action:
    - MessageAction(content) - Message action to run (e.g. ask for clarification)
    """

    def __init__(
        self,
    ):
        pass

    def check_condition(self, action_str: str) -> bool:
        # We assume the LLM is GOOD enough that when it returns pure natural language
        # it wants to talk to the user
        return True

    def parse(self, action_str: str) -> Action:
        return MessageAction(content=action_str, wait_for_response=True)


class CodeActActionParserFileEdit(ActionParser):
    """Parser action:
    - FileEditAction(path, content) - edit a file
    """

    def __init__(self):
        self.file_edit_match: re.Match | None = None

    def check_condition(self, action_str: str) -> bool:
        if '<file_edit' not in action_str:
            return False

        # Updated regex to make start and end optional
        self.file_edit_match = re.search(
            r'<file_edit\s+path=(["\']?)(.*?)\1(?:\s+start=(["\']?)(.*?)\3)?(?:\s+end=(["\']?)(.*?)\5)?\s*>(.*?)</file_edit>',
            action_str,
            re.DOTALL,
        )

        if self.file_edit_match is None:
            logger.error(
                f'FileEditAction detected but the format is incorrect. Unable to match for <file_edit> in:\n{"-" * 80}\n{action_str}\n{"-" * 80}'
            )
            raise LLMMalformedActionError(
                'FileEditAction detected but the format is incorrect. Usage:\n'
                '<file_edit path="[path]" start=[start_line] end=[end_line]>\n'
                '[content_to_edit]\n'
                '</file_edit>\n'
            )

        path = self.file_edit_match.group(2)
        start = self.file_edit_match.group(4)
        end = self.file_edit_match.group(6)

        if not path:
            raise LLMMalformedActionError(
                'FileEditAction detected but no `path` specified. You should specify the path of the file to edit.'
            )

        if start:
            try:
                int(start)
            except ValueError:
                raise LLMMalformedActionError(
                    f'FileEditAction detected but `start` is not a valid integer: {start}'
                )

        if end:
            try:
                int(end)
            except ValueError:
                raise LLMMalformedActionError(
                    f'FileEditAction detected but `end` is not a valid integer: {end}'
                )

        return True

    def parse(self, action_str: str) -> Action:
        assert (
            self.file_edit_match is not None
        ), 'self.file_edit_match should not be None when parse is called'

        file_path = self.file_edit_match.group(2).strip()
        start_line = (
            int(self.file_edit_match.group(4))
            if self.file_edit_match.group(4)
            else None
        )
        end_line = (
            int(self.file_edit_match.group(6))
            if self.file_edit_match.group(6)
            else None
        )
        content = self.file_edit_match.group(7)
        thought = action_str.replace(self.file_edit_match.group(0), '').strip()

        action = FileEditAction(path=file_path, content=content, thought=thought)
        if start_line is not None:
            action.start = start_line
        if end_line is not None:
            action.end = end_line
        return action
