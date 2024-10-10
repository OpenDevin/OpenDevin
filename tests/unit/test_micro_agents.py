import json
import os
from unittest.mock import MagicMock

import pytest
import yaml
from pytest import TempPathFactory

from openhands.agenthub.micro.registry import all_microagents
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig
from openhands.events import EventSource
from openhands.events.action import MessageAction
from openhands.events.stream import EventStream
from openhands.memory.conversation_memory import ConversationMemory
from openhands.storage import get_file_store


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_micro_agents'))


@pytest.fixture
def event_stream(temp_dir):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('asdf', file_store)
    yield event_stream

    # clear after each test
    event_stream.clear()


@pytest.fixture
def agent_configs():
    return {
        'CoderAgent': AgentConfig(memory_enabled=True),
        'PlannerAgent': AgentConfig(memory_enabled=True),
    }


def test_all_agents_are_loaded():
    assert all_microagents is not None
    assert len(all_microagents) > 1

    base = os.path.join('openhands', 'agenthub', 'micro')
    full_path = os.path.dirname(__file__) + '/../../' + base
    agent_names = set()
    for root, _, files in os.walk(full_path):
        for file in files:
            if file == 'agent.yaml':
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as yaml_file:
                    data = yaml.safe_load(yaml_file)
                    agent_names.add(data['name'])
    assert agent_names == set(all_microagents.keys())


def test_coder_agent_with_summary(event_stream: EventStream, agent_configs: dict):
    """Coder agent should render code summary as part of prompt"""
    mock_llm = MagicMock()
    content = json.dumps({'action': 'finish', 'args': {}})
    mock_llm.completion.return_value = {'choices': [{'message': {'content': content}}]}
    mock_llm.format_messages_for_llm.return_value = [
        {
            'role': 'user',
            'content': "This is a dummy task. This is a dummy summary about this repo. Here's a summary of the codebase, as it relates to this task.",
        }
    ]

    memory = ConversationMemory(event_stream=event_stream)

    coder_agent = Agent.get_cls('CoderAgent')(
        llm=mock_llm, config=agent_configs['CoderAgent'], memory=memory
    )
    assert coder_agent is not None

    task = 'This is a dummy task'
    event_stream.add_event(MessageAction(content=task), EventSource.USER)

    summary = 'This is a dummy summary about this repo'
    state = State(inputs={'summary': summary})
    coder_agent.step(state)

    mock_llm.completion.assert_called_once()
    _, kwargs = mock_llm.completion.call_args
    prompt_element = kwargs['messages'][0]['content']
    if isinstance(prompt_element, dict):
        prompt = prompt_element['content']
    else:
        prompt = prompt_element
    assert task in prompt
    assert "Here's a summary of the codebase, as it relates to this task" in prompt
    assert summary in prompt


def test_coder_agent_without_summary(event_stream: EventStream, agent_configs: dict):
    """When there's no codebase_summary available, there shouldn't be any prompt
    about 'code summary'
    """
    mock_llm = MagicMock()
    content = json.dumps({'action': 'finish', 'args': {}})
    mock_llm.completion.return_value = {'choices': [{'message': {'content': content}}]}
    mock_llm.format_messages_for_llm.return_value = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': "This is a dummy task. This is a dummy summary about this repo. Here's a summary of the codebase, as it relates to this task.",
                }
            ],
        }
    ]

    memory = ConversationMemory(event_stream=event_stream)

    coder_agent = Agent.get_cls('CoderAgent')(
        llm=mock_llm, config=agent_configs['CoderAgent'], memory=memory
    )
    assert coder_agent is not None

    task = 'This is a dummy task'
    event_stream.add_event(MessageAction(content=task), EventSource.USER)

    # set state without codebase summary
    state = State()
    coder_agent.step(state)

    mock_llm.completion.assert_called_once()
    _, kwargs = mock_llm.completion.call_args
    prompt_element = kwargs['messages'][0]['content']
    if isinstance(prompt_element, dict):
        prompt = prompt_element['content']
    else:
        prompt = prompt_element
    print(f'\n{prompt_element}\n')
    assert "Here's a summary of the codebase, as it relates to this task" not in prompt
