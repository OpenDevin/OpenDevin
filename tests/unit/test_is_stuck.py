import logging
from unittest.mock import Mock, patch

import pytest
from pytest import TempPathFactory

from openhands.controller.agent_controller import AgentController
from openhands.controller.stuck import StuckDetector
from openhands.events.action import CmdRunAction, FileReadAction, MessageAction
from openhands.events.action.commands import IPythonRunCellAction
from openhands.events.observation import (
    CmdOutputObservation,
    FileReadObservation,
)
from openhands.events.observation.commands import IPythonRunCellObservation
from openhands.events.observation.empty import NullObservation
from openhands.events.observation.error import ErrorObservation
from openhands.events.stream import EventSource, EventStream
from openhands.storage import get_file_store


def collect_events(stream: EventStream):
    return [event for event in stream.get_events(filter_out_types=())]


logging.basicConfig(level=logging.DEBUG)

jupyter_line_1 = '\n[Jupyter current working directory:'
jupyter_line_2 = '\n[Jupyter Python interpreter:'
code_snippet = """
edit_file_by_replace(
    'book_store.py',
    to_replace=\"""def total(basket):
    if not basket:
        return 0
"""


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_is_stuck'))


@pytest.fixture
def event_stream(temp_dir):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('asdf', file_store)
    yield event_stream
    # clear after each test
    event_stream.clear()


class TestStuckDetector:
    @pytest.fixture
    def stuck_detector(self, event_stream):
        return StuckDetector(event_stream)

    def _impl_syntax_error_events(
        self,
        event_stream: EventStream,
        error_message: str,
        random_line: bool,
        incidents: int = 4,
    ):
        for i in range(incidents):
            ipython_action = IPythonRunCellAction(code=code_snippet)
            event_stream.add_event(ipython_action, EventSource.AGENT)
            extra_number = (i + 1) * 10 if random_line else '42'
            extra_line = '\n' * (i + 1) if random_line else ''
            ipython_observation = IPythonRunCellObservation(
                content=f'  Cell In[1], line {extra_number}\n'
                'to_replace="""def largest(min_factor, max_factor):\n            ^\n'
                f'{error_message}{extra_line}' + jupyter_line_1 + jupyter_line_2,
                code=code_snippet,
            )
            ipython_observation._cause = ipython_action._id
            event_stream.add_event(ipython_observation, EventSource.USER)

    def _impl_unterminated_string_error_events(
        self, event_stream: EventStream, random_line: bool, incidents: int = 4
    ):
        for i in range(incidents):
            ipython_action = IPythonRunCellAction(code=code_snippet)
            event_stream.add_event(ipython_action, EventSource.AGENT)
            line_number = (i + 1) * 10 if random_line else '1'
            ipython_observation = IPythonRunCellObservation(
                content=f'print("  Cell In[1], line {line_number}\nhello\n       ^\nSyntaxError: unterminated string literal (detected at line {line_number})'
                + jupyter_line_1
                + jupyter_line_2,
                code=code_snippet,
            )
            ipython_observation._cause = ipython_action._id
            event_stream.add_event(ipython_observation, EventSource.USER)

    def test_history_too_short(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        message_action = MessageAction(content='Hello', wait_for_response=False)
        message_action._source = EventSource.USER
        observation = NullObservation(content='')
        observation._cause = message_action.id
        event_stream.add_event(message_action, EventSource.USER)
        event_stream.add_event(observation, EventSource.USER)

        cmd_action = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action, EventSource.AGENT)
        cmd_observation = CmdOutputObservation(
            command_id=1, command='ls', content='file1.txt\nfile2.txt'
        )
        cmd_observation._cause = cmd_action._id
        event_stream.add_event(cmd_observation, EventSource.USER)

        assert stuck_detector.is_stuck() is False

    def test_is_stuck_repeating_action_observation(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        # TODO this is a good test for almost stuck
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        hello_observation = NullObservation('')

        # 2 events:
        event_stream.add_event(hello_action, EventSource.USER)
        event_stream.add_event(hello_observation, EventSource.USER)
        assert len(collect_events(event_stream)) == 2

        # 4 events:
        cmd_action_1 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_1, EventSource.AGENT)
        cmd_observation_1 = CmdOutputObservation(
            content='', command='ls', command_id=cmd_action_1._id
        )
        cmd_observation_1._cause = cmd_action_1._id
        event_stream.add_event(cmd_observation_1, EventSource.USER)
        assert len(collect_events(event_stream)) == 4

        # 6 events:
        cmd_action_2 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_2, EventSource.AGENT)
        cmd_observation_2 = CmdOutputObservation(
            content='', command='ls', command_id=cmd_action_2._id
        )
        cmd_observation_2._cause = cmd_action_2._id
        event_stream.add_event(cmd_observation_2, EventSource.USER)
        assert len(collect_events(event_stream)) == 6

        # 8 events:
        # random user message just because we can
        message_null_observation = NullObservation(content='')
        event_stream.add_event(message_action, EventSource.USER)
        event_stream.add_event(message_null_observation, EventSource.USER)
        assert len(collect_events(event_stream)) == 8

        assert stuck_detector.is_stuck() is False

        # 10 events:
        cmd_action_3 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_3, EventSource.AGENT)
        cmd_observation_3 = CmdOutputObservation(
            content='', command='ls', command_id=cmd_action_3._id
        )
        cmd_observation_3._cause = cmd_action_3._id
        event_stream.add_event(cmd_observation_3, EventSource.USER)

        assert len(collect_events(event_stream)) == 10
        assert len(list(stuck_detector.event_stream.get_events())) == 8
        assert len(stuck_detector.event_stream.get_pairs()) == 5

        assert stuck_detector.is_stuck() is False

        cmd_action_4 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_4, EventSource.AGENT)
        cmd_observation_4 = CmdOutputObservation(
            content='', command='ls', command_id=cmd_action_4._id
        )
        cmd_observation_4._cause = cmd_action_4._id
        event_stream.add_event(cmd_observation_4, EventSource.USER)
        # 12 events

        assert len(collect_events(event_stream)) == 12
        assert len(list(stuck_detector.event_stream.get_events())) == 10
        assert len(stuck_detector.event_stream.get_pairs()) == 6

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck() is True
            mock_warning.assert_called_once_with('Action, Observation loop detected')

    def test_is_stuck_repeating_action_error(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        # (action, error_observation), not necessarily the same error
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        hello_observation = NullObservation(content='')
        event_stream.add_event(hello_action, EventSource.USER)
        hello_observation._cause = hello_action._id
        event_stream.add_event(hello_observation, EventSource.USER)
        # 2 events

        cmd_action_1 = CmdRunAction(command='invalid_command')
        event_stream.add_event(cmd_action_1, EventSource.AGENT)
        error_observation_1 = ErrorObservation(content='Command not found')
        error_observation_1._cause = cmd_action_1._id
        event_stream.add_event(error_observation_1, EventSource.USER)
        # 4 events

        cmd_action_2 = CmdRunAction(command='invalid_command')
        event_stream.add_event(cmd_action_2, EventSource.AGENT)
        error_observation_2 = ErrorObservation(
            content='Command still not found or another error'
        )
        error_observation_2._cause = cmd_action_2._id
        event_stream.add_event(error_observation_2, EventSource.USER)
        # 6 events

        message_null_observation = NullObservation(content='')
        event_stream.add_event(message_action, EventSource.USER)
        event_stream.add_event(message_null_observation, EventSource.USER)
        # 8 events

        cmd_action_3 = CmdRunAction(command='invalid_command')
        event_stream.add_event(cmd_action_3, EventSource.AGENT)
        error_observation_3 = ErrorObservation(content='Different error')
        error_observation_3._cause = cmd_action_3._id
        event_stream.add_event(error_observation_3, EventSource.USER)
        # 10 events

        cmd_action_4 = CmdRunAction(command='invalid_command')
        event_stream.add_event(cmd_action_4, EventSource.AGENT)
        error_observation_4 = ErrorObservation(content='Command not found')
        error_observation_4._cause = cmd_action_4._id
        event_stream.add_event(error_observation_4, EventSource.USER)
        # 12 events

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck() is True
            mock_warning.assert_called_once_with(
                'Action, ErrorObservation loop detected'
            )

    def test_is_stuck_invalid_syntax_error(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_syntax_error_events(
            event_stream,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=False,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is True

    def test_is_not_stuck_invalid_syntax_error_random_lines(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_syntax_error_events(
            event_stream,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=True,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is False

    def test_is_not_stuck_invalid_syntax_error_only_three_incidents(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_syntax_error_events(
            event_stream,
            error_message='SyntaxError: invalid syntax. Perhaps you forgot a comma?',
            random_line=True,
            incidents=3,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is False

    def test_is_stuck_incomplete_input_error(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_syntax_error_events(
            event_stream,
            error_message='SyntaxError: incomplete input',
            random_line=False,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is True

    def test_is_not_stuck_incomplete_input_error(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_syntax_error_events(
            event_stream,
            error_message='SyntaxError: incomplete input',
            random_line=True,
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is False

    def test_is_not_stuck_ipython_unterminated_string_error_random_lines(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_unterminated_string_error_events(event_stream, random_line=True)

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is False

    def test_is_not_stuck_ipython_unterminated_string_error_only_three_incidents(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_unterminated_string_error_events(
            event_stream, random_line=False, incidents=3
        )

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is False

    def test_is_stuck_ipython_unterminated_string_error(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        self._impl_unterminated_string_error_events(event_stream, random_line=False)

        with patch('logging.Logger.warning'):
            assert stuck_detector.is_stuck() is True

    def test_is_not_stuck_ipython_syntax_error_not_at_end(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        # this test is to make sure we don't get false positives
        # since the "at line x" is changing in between!
        ipython_action_1 = IPythonRunCellAction(code='print("hello')
        event_stream.add_event(ipython_action_1, EventSource.AGENT)
        ipython_observation_1 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 1)\nThis is some additional output',
            code='print("hello',
        )
        ipython_observation_1._cause = ipython_action_1._id
        event_stream.add_event(ipython_observation_1, EventSource.USER)

        ipython_action_2 = IPythonRunCellAction(code='print("hello')
        event_stream.add_event(ipython_action_2, EventSource.AGENT)
        ipython_observation_2 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 1)\nToo much output here on and on',
            code='print("hello',
        )
        ipython_observation_2._cause = ipython_action_2._id
        event_stream.add_event(ipython_observation_2, EventSource.USER)

        ipython_action_3 = IPythonRunCellAction(code='print("hello')
        event_stream.add_event(ipython_action_3, EventSource.AGENT)
        ipython_observation_3 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 3)\nEnough',
            code='print("hello',
        )
        ipython_observation_3._cause = ipython_action_3._id
        event_stream.add_event(ipython_observation_3, EventSource.USER)

        ipython_action_4 = IPythonRunCellAction(code='print("hello')
        event_stream.add_event(ipython_action_4, EventSource.AGENT)
        ipython_observation_4 = IPythonRunCellObservation(
            content='print("hello\n       ^\nSyntaxError: unterminated string literal (detected at line 2)\nLast line of output',
            code='print("hello',
        )
        ipython_observation_4._cause = ipython_action_4._id
        event_stream.add_event(ipython_observation_4, EventSource.USER)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck() is False
            mock_warning.assert_not_called()

    def test_is_stuck_repeating_action_observation_pattern(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        message_action = MessageAction(content='Come on', wait_for_response=False)
        message_action._source = EventSource.USER
        event_stream.add_event(message_action, EventSource.USER)
        message_observation = NullObservation(content='')
        event_stream.add_event(message_observation, EventSource.USER)

        cmd_action_1 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_1, EventSource.AGENT)
        cmd_observation_1 = CmdOutputObservation(
            command_id=1, command='ls', content='file1.txt\nfile2.txt'
        )
        cmd_observation_1._cause = cmd_action_1._id
        event_stream.add_event(cmd_observation_1, EventSource.USER)

        read_action_1 = FileReadAction(path='file1.txt')
        event_stream.add_event(read_action_1, EventSource.AGENT)
        read_observation_1 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        read_observation_1._cause = read_action_1._id
        event_stream.add_event(read_observation_1, EventSource.USER)

        cmd_action_2 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_2, EventSource.AGENT)
        cmd_observation_2 = CmdOutputObservation(
            command_id=2, command='ls', content='file1.txt\nfile2.txt'
        )
        cmd_observation_2._cause = cmd_action_2._id
        event_stream.add_event(cmd_observation_2, EventSource.USER)

        read_action_2 = FileReadAction(path='file1.txt')
        event_stream.add_event(read_action_2, EventSource.AGENT)
        read_observation_2 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        read_observation_2._cause = read_action_2._id
        event_stream.add_event(read_observation_2, EventSource.USER)

        # one more message to break the pattern
        message_null_observation = NullObservation(content='')
        event_stream.add_event(message_action, EventSource.USER)
        event_stream.add_event(message_null_observation, EventSource.USER)

        cmd_action_3 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_3, EventSource.AGENT)
        cmd_observation_3 = CmdOutputObservation(
            command_id=3, command='ls', content='file1.txt\nfile2.txt'
        )
        cmd_observation_3._cause = cmd_action_3._id
        event_stream.add_event(cmd_observation_3, EventSource.USER)

        read_action_3 = FileReadAction(path='file1.txt')
        event_stream.add_event(read_action_3, EventSource.AGENT)
        read_observation_3 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        read_observation_3._cause = read_action_3._id
        event_stream.add_event(read_observation_3, EventSource.USER)

        with patch('logging.Logger.warning') as mock_warning:
            assert stuck_detector.is_stuck() is True
            mock_warning.assert_called_once_with('Action, Observation pattern detected')

    def test_is_stuck_not_stuck(
        self, stuck_detector: StuckDetector, event_stream: EventStream
    ):
        message_action = MessageAction(content='Done', wait_for_response=False)
        message_action._source = EventSource.USER

        hello_action = MessageAction(content='Hello', wait_for_response=False)
        event_stream.add_event(hello_action, EventSource.USER)
        hello_observation = NullObservation(content='')
        hello_observation._cause = hello_action._id
        event_stream.add_event(hello_observation, EventSource.USER)

        cmd_action_1 = CmdRunAction(command='ls')
        event_stream.add_event(cmd_action_1, EventSource.AGENT)
        cmd_observation_1 = CmdOutputObservation(
            command_id=cmd_action_1.id, command='ls', content='file1.txt\nfile2.txt'
        )
        cmd_observation_1._cause = cmd_action_1._id
        event_stream.add_event(cmd_observation_1, EventSource.USER)

        read_action_1 = FileReadAction(path='file1.txt')
        event_stream.add_event(read_action_1, EventSource.AGENT)
        read_observation_1 = FileReadObservation(
            content='File content', path='file1.txt'
        )
        read_observation_1._cause = read_action_1._id
        event_stream.add_event(read_observation_1, EventSource.USER)

        cmd_action_2 = CmdRunAction(command='pwd')
        event_stream.add_event(cmd_action_2, EventSource.AGENT)
        cmd_observation_2 = CmdOutputObservation(
            command_id=2, command='pwd', content='/home/user'
        )
        cmd_observation_2._cause = cmd_action_2._id
        event_stream.add_event(cmd_observation_2, EventSource.USER)

        read_action_2 = FileReadAction(path='file2.txt')
        event_stream.add_event(read_action_2, EventSource.AGENT)
        read_observation_2 = FileReadObservation(
            content='Another file content', path='file2.txt'
        )
        read_observation_2._cause = read_action_2._id
        event_stream.add_event(read_observation_2, EventSource.USER)

        message_null_observation = NullObservation(content='')
        event_stream.add_event(message_action, EventSource.USER)
        event_stream.add_event(message_null_observation, EventSource.USER)

        cmd_action_3 = CmdRunAction(command='pwd')
        event_stream.add_event(cmd_action_3, EventSource.AGENT)
        cmd_observation_3 = CmdOutputObservation(
            command_id=cmd_action_3.id, command='pwd', content='/home/user'
        )
        cmd_observation_3._cause = cmd_action_3._id
        event_stream.add_event(cmd_observation_3, EventSource.USER)

        read_action_3 = FileReadAction(path='file2.txt')
        event_stream.add_event(read_action_3, EventSource.AGENT)
        read_observation_3 = FileReadObservation(
            content='Another file content', path='file2.txt'
        )
        read_observation_3._cause = read_action_3._id
        event_stream.add_event(read_observation_3, EventSource.USER)

        assert stuck_detector.is_stuck() is False

    def test_is_stuck_monologue(self, stuck_detector, event_stream):
        # Add events to the event stream
        message_action_1 = MessageAction(content='Hi there!')
        event_stream.add_event(message_action_1, EventSource.USER)
        message_action_1._source = EventSource.USER

        message_action_2 = MessageAction(content='Hi there!')
        event_stream.add_event(message_action_2, EventSource.AGENT)
        message_action_2._source = EventSource.AGENT

        message_action_3 = MessageAction(content='How are you?')
        event_stream.add_event(message_action_3, EventSource.USER)
        message_action_3._source = EventSource.USER

        cmd_kill_action = CmdRunAction(
            command='echo 42', thought="I'm not stuck, he's stuck"
        )
        event_stream.add_event(cmd_kill_action, EventSource.AGENT)

        message_action_4 = MessageAction(content="I'm doing well, thanks for asking.")
        event_stream.add_event(message_action_4, EventSource.AGENT)
        message_action_4._source = EventSource.AGENT

        message_action_5 = MessageAction(content="I'm doing well, thanks for asking.")
        event_stream.add_event(message_action_5, EventSource.AGENT)
        message_action_5._source = EventSource.AGENT

        message_action_6 = MessageAction(content="I'm doing well, thanks for asking.")
        event_stream.add_event(message_action_6, EventSource.AGENT)
        message_action_6._source = EventSource.AGENT

        assert stuck_detector.is_stuck()

        # Add an observation event between the repeated message actions
        cmd_output_observation = CmdOutputObservation(
            content='OK, I was stuck, but no more.',
            command_id=42,
            command='storybook',
            exit_code=0,
        )
        cmd_output_observation._cause = cmd_kill_action._id
        event_stream.add_event(cmd_output_observation, EventSource.USER)

        message_action_7 = MessageAction(content="I'm doing well, thanks for asking.")
        event_stream.add_event(message_action_7, EventSource.AGENT)
        message_action_7._source = EventSource.AGENT

        message_action_8 = MessageAction(content="I'm doing well, thanks for asking.")
        event_stream.add_event(message_action_8, EventSource.AGENT)
        message_action_8._source = EventSource.AGENT

        with patch('logging.Logger.warning'):
            assert not stuck_detector.is_stuck()


class TestAgentController:
    @pytest.fixture
    def controller(self):
        controller = Mock(spec=AgentController)
        controller._is_stuck = AgentController._is_stuck.__get__(
            controller, AgentController
        )
        controller.delegate = None
        controller.state = Mock()
        return controller

    def test_is_stuck_delegate_stuck(self, controller: AgentController):
        controller.delegate = Mock()
        controller.delegate._is_stuck.return_value = True
        assert controller._is_stuck() is True
