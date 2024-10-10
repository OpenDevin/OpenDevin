import pytest

from openhands.agenthub.codeact_agent.action_parser import (
    CodeActActionParserAgentDelegate,
)
from openhands.events.action import AgentDelegateAction


@pytest.mark.parametrize(
    'action_str, expected_agent, expected_thought, expected_task',
    [
        (
            'I need to search for information.\n<execute_browse>Tell me who is the Vice President of the USA</execute_browse>',
            'BrowsingAgent',
            'I need to search for information.',
            'Tell me who is the Vice President of the USA',
        ),
        (
            '<execute_browse>Search for recent climate change data</execute_browse>',
            'BrowsingAgent',
            '',
            'Search for recent climate change data',
        ),
        (
            "Let's use the browsing agent to find this information.\n<execute_browse>Find the population of Tokyo in 2023</execute_browse>\nThis will help us answer the question.",
            'BrowsingAgent',
            "Let's use the browsing agent to find this information.\n\nThis will help us answer the question.",
            'Find the population of Tokyo in 2023',
        ),
    ],
)
def test_codeact_action_parser_agent_delegate(
    action_str, expected_agent, expected_thought, expected_task
):
    parser = CodeActActionParserAgentDelegate()
    assert parser.check_condition(action_str)

    action = parser.parse(action_str)

    assert isinstance(action, AgentDelegateAction)
    assert action.agent == expected_agent
    assert action.thought == expected_thought
    assert action.inputs['task'] == expected_task


def test_codeact_action_parser_agent_delegate_no_match():
    parser = CodeActActionParserAgentDelegate()
    action_str = 'This is a regular message without any browse command.'

    assert not parser.check_condition(action_str)

    with pytest.raises(AssertionError):
        parser.parse(action_str)
