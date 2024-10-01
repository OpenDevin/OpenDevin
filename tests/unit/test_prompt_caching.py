from unittest.mock import MagicMock, Mock, patch

import pytest

from agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.core.config import AgentConfig, LLMConfig
from openhands.events import EventSource, EventStream
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.observation import CmdOutputObservation
from openhands.llm.llm import LLM
from openhands.storage import get_file_store


@pytest.fixture
def mock_llm():
    llm = Mock(spec=LLM)
    llm.config = LLMConfig(model='claude-3-5-sonnet-20240620', caching_prompt=True)
    llm.is_caching_prompt_active.return_value = True
    return llm


@pytest.fixture
def mock_event_stream(tmp_path):
    file_store = get_file_store('local', str(tmp_path))
    return EventStream('test_session', file_store)


@pytest.fixture
def codeact_agent(mock_llm):
    config = AgentConfig()
    return CodeActAgent(mock_llm, config)


def test_get_messages_with_reminder(codeact_agent, mock_event_stream):
    # Add some events to the stream
    mock_event_stream.add_event(MessageAction('Initial user message'), EventSource.USER)
    mock_event_stream.add_event(MessageAction('Sure!'), EventSource.AGENT)
    mock_event_stream.add_event(MessageAction('Hello, agent!'), EventSource.USER)
    mock_event_stream.add_event(MessageAction('Hello, user!'), EventSource.AGENT)
    mock_event_stream.add_event(MessageAction('Laaaaaaaast!'), EventSource.USER)

    codeact_agent.reset()
    messages = codeact_agent._get_messages(
        Mock(history=mock_event_stream, max_iterations=5, iteration=0, inputs=dict())
    )

    assert (
        len(messages) == 6
    )  # System, initial user + user message, agent message, last user message
    assert messages[0].content[0].cache_prompt  # system message
    assert messages[1].role == 'user'
    assert messages[1].content[0].text.endswith("LET'S START!")
    assert messages[1].content[1].text.endswith('Initial user message')
    assert messages[1].content[0].cache_prompt

    assert messages[3].role == 'user'
    assert messages[3].content[0].text == ('Hello, agent!')
    assert messages[3].content[0].cache_prompt
    assert messages[4].role == 'assistant'
    assert messages[4].content[0].text == 'Hello, user!'
    assert not messages[4].content[0].cache_prompt
    assert messages[5].role == 'user'
    assert messages[5].content[0].text.startswith('Laaaaaaaast!')
    assert messages[5].content[0].cache_prompt
    assert (
        messages[5]
        .content[1]
        .text.endswith(
            'ENVIRONMENT REMINDER: You have 5 turns left to complete the task. When finished reply with <finish></finish>.'
        )
    )


def test_get_messages_prompt_caching(codeact_agent, mock_event_stream):
    # Add multiple user and agent messages
    for i in range(15):
        mock_event_stream.add_event(
            MessageAction(f'User message {i}'), EventSource.USER
        )
        mock_event_stream.add_event(
            MessageAction(f'Agent message {i}'), EventSource.AGENT
        )

    codeact_agent.reset()
    messages = codeact_agent._get_messages(
        Mock(history=mock_event_stream, max_iterations=10, iteration=5, inputs=dict())
    )

    # Check that only the last two user messages have cache_prompt=True
    cached_user_messages = [
        msg
        for msg in messages
        if msg.role in ('user', 'system') and msg.content[0].cache_prompt
    ]
    assert (
        len(cached_user_messages) == 4
    )  # Including the initial system+user + 2 last user message

    # Verify that these are indeed the last two user messages (from start)
    assert (
        cached_user_messages[0].content[0].text.startswith('A chat between')
    )  # system message
    assert cached_user_messages[2].content[0].text.startswith('User message 1')
    assert cached_user_messages[3].content[0].text.startswith('User message 1')


def test_get_messages_with_cmd_action(codeact_agent, mock_event_stream):
    # Add a mix of actions and observations
    message_action_1 = MessageAction(
        "Let's list the contents of the current directory."
    )
    mock_event_stream.add_event(message_action_1, EventSource.USER)

    cmd_action_1 = CmdRunAction('ls -l', thought='List files in current directory')
    mock_event_stream.add_event(cmd_action_1, EventSource.AGENT)

    cmd_observation_1 = CmdOutputObservation(
        content='total 0\n-rw-r--r-- 1 user group 0 Jan 1 00:00 file1.txt\n-rw-r--r-- 1 user group 0 Jan 1 00:00 file2.txt',
        command_id=cmd_action_1._id,
        command='ls -l',
        exit_code=0,
    )
    mock_event_stream.add_event(cmd_observation_1, EventSource.USER)

    message_action_2 = MessageAction("Now, let's create a new directory.")
    mock_event_stream.add_event(message_action_2, EventSource.AGENT)

    cmd_action_2 = CmdRunAction('mkdir new_directory', thought='Create a new directory')
    mock_event_stream.add_event(cmd_action_2, EventSource.AGENT)

    cmd_observation_2 = CmdOutputObservation(
        content='',
        command_id=cmd_action_2._id,
        command='mkdir new_directory',
        exit_code=0,
    )
    mock_event_stream.add_event(cmd_observation_2, EventSource.USER)

    codeact_agent.reset()
    messages = codeact_agent._get_messages(
        Mock(history=mock_event_stream, max_iterations=5, iteration=0, inputs=dict())
    )

    # Assert the presence of key elements in the messages
    assert (
        messages[1]
        .content[1]
        .text.startswith("Let's list the contents of the current directory.")
    )  # user, included in the initial message
    assert any(
        'List files in current directory\n<execute_bash>\nls -l\n</execute_bash>'
        in msg.content[0].text
        for msg in messages
    )  # agent
    assert any(
        'total 0\n-rw-r--r-- 1 user group 0 Jan 1 00:00 file1.txt\n-rw-r--r-- 1 user group 0 Jan 1 00:00 file2.txt'
        in msg.content[0].text
        for msg in messages
    )  # user, observation
    assert any(
        "Now, let's create a new directory." in msg.content[0].text for msg in messages
    )  # agent
    assert messages[4].content[1].text.startswith('Create a new directory')  # agent
    assert any(
        'finished with exit code 0' in msg.content[0].text for msg in messages
    )  # user, observation
    assert (
        messages[5].content[0].text.startswith('OBSERVATION:\n\n')
    )  # user, observation

    # prompt cache is added to the system message
    assert messages[0].content[0].cache_prompt
    # and the first initial user message
    assert messages[1].content[0].cache_prompt
    # and to the last two user messages
    assert messages[3].content[0].cache_prompt
    assert messages[5].content[0].cache_prompt

    # reminder is added to the last user message
    assert 'ENVIRONMENT REMINDER: You have 5 turns' in messages[5].content[1].text


def test_prompt_caching_headers(codeact_agent, mock_event_stream):
    # Setup
    mock_event_stream.add_event(MessageAction('Hello, agent!'), EventSource.USER)
    mock_event_stream.add_event(MessageAction('Hello, user!'), EventSource.AGENT)

    mock_short_term_history = MagicMock()
    mock_short_term_history.get_last_user_message.return_value = 'Hello, agent!'

    mock_state = Mock()
    mock_state.history = mock_short_term_history
    mock_state.max_iterations = 5
    mock_state.iteration = 0
    mock_state.inputs = dict()

    codeact_agent.reset()

    # Create a mock for litellm_completion
    def check_headers(**kwargs):
        assert 'extra_headers' in kwargs
        assert 'anthropic-beta' in kwargs['extra_headers']
        assert kwargs['extra_headers']['anthropic-beta'] == 'prompt-caching-2024-07-31'
        # Create a mock response with the expected structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = 'Hello! How can I assist you today?'
        return mock_response

    # Use patch to replace litellm_completion with our check_headers function
    with patch('openhands.llm.llm.litellm_completion', side_effect=check_headers):
        # Also patch the action parser to return a MessageAction
        with patch.object(
            codeact_agent.action_parser,
            'parse',
            return_value=MessageAction('Hello! How can I assist you today?'),
        ):
            # Act
            result = codeact_agent.step(mock_state)

    # Assert
    assert isinstance(result, MessageAction)
    assert result.content == 'Hello! How can I assist you today?'
