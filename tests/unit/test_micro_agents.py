import json
import os
from unittest.mock import AsyncMock

import pytest
import yaml
from pytest import TempPathFactory

from agenthub.micro.registry import all_microagents
from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.events import EventSource
from opendevin.events.action import AgentFinishAction, MessageAction
from opendevin.events.stream import EventStream
from opendevin.memory.history import ShortTermHistory
from opendevin.storage import get_file_store


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


@pytest.fixture(scope='session')
def event_loop():
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_all_agents_are_loaded():
    assert all_microagents is not None
    assert len(all_microagents) > 1

    base = os.path.join('agenthub', 'micro')
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


@pytest.mark.asyncio
async def test_coder_agent_with_summary(event_stream: EventStream):
    """Coder agent should render code summary as part of prompt"""
    mock_llm = AsyncMock()
    content = json.dumps({'action': 'finish', 'args': {}})
    # Update this line to return the content directly
    mock_llm.completion.return_value = {'choices': [{'message': {'content': content}}]}

    coder_agent = Agent.get_cls('CoderAgent')(llm=mock_llm)
    assert coder_agent is not None

    task = 'This is a dummy task'
    history = ShortTermHistory()
    history.set_event_stream(event_stream)
    await event_stream.add_event(MessageAction(content=task), EventSource.USER)

    summary = 'This is a dummy summary about this repo'
    state = State(history=history, inputs={'summary': summary})
    action = await coder_agent.async_step(state)

    mock_llm.completion.assert_called_once()
    _, kwargs = mock_llm.completion.call_args
    prompt = kwargs['messages'][0]['content'][0]['text']
    assert task in prompt
    assert "Here's a summary of the codebase, as it relates to this task" in prompt

    # Verify that the action is as expected
    assert isinstance(action, AgentFinishAction)
    assert action.action == 'finish'
    assert action.outputs == {}  # Assuming 'args' corresponds to 'outputs'


@pytest.mark.asyncio
async def test_coder_agent_without_summary(event_stream: EventStream):
    """When there's no codebase_summary available, there shouldn't be any prompt
    about 'code summary'
    """
    mock_llm = AsyncMock()
    content = json.dumps({'action': 'finish', 'args': {}})
    mock_llm.completion.return_value = {'choices': [{'message': {'content': content}}]}

    coder_agent = Agent.get_cls('CoderAgent')(llm=mock_llm)
    assert coder_agent is not None

    task = 'This is a dummy task'
    history = ShortTermHistory()
    history.set_event_stream(event_stream)
    await event_stream.add_event(MessageAction(content=task), EventSource.USER)

    # set state without codebase summary
    state = State(history=history)
    action = await coder_agent.async_step(state)

    mock_llm.completion.assert_called_once()
    _, kwargs = mock_llm.completion.call_args
    prompt = kwargs['messages'][0]['content'][0]['text']
    assert task in prompt
    assert "Here's a summary of the codebase, as it relates to this task" not in prompt

    # Verify that the action is as expected
    assert isinstance(action, AgentFinishAction)
    assert action.action == 'finish'  # Check the action attribute
    assert action.outputs == {}  # Assuming 'args' corresponds to 'outputs'
