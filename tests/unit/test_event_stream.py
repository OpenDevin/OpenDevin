import json

import pytest
from pytest import TempPathFactory

from openhands.core.schema.agent import AgentState
from openhands.core.schema.observation import ObservationType
from openhands.events import EventSource, EventStream
from openhands.events.action import (
    NullAction,
)
from openhands.events.action.agent import AgentDelegateAction, ChangeAgentStateAction
from openhands.events.action.files import FileReadAction, FileWriteAction
from openhands.events.observation import NullObservation
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.events.observation.delegate import AgentDelegateObservation
from openhands.events.observation.files import FileReadObservation, FileWriteObservation
from openhands.storage import get_file_store


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_event_stream'))


def collect_events(stream: EventStream):
    return [event for event in stream.get_events(filter_out_types=())]


def test_basic_flow(temp_dir: str):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('abc', file_store)
    event_stream.add_event(NullAction(), EventSource.AGENT)
    assert len(collect_events(event_stream)) == 1


def test_stream_storage(temp_dir: str):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('abc', file_store)
    event_stream.add_event(NullObservation(''), EventSource.AGENT)
    assert len(collect_events(event_stream)) == 1
    content = event_stream.file_store.read('sessions/abc/events/0.json')
    assert content is not None
    data = json.loads(content)
    assert 'timestamp' in data
    del data['timestamp']
    assert data == {
        'id': 0,
        'source': 'agent',
        'observation': 'null',
        'content': '',
        'extras': {},
        'message': 'No observation',
    }


def test_rehydration(temp_dir: str):
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('abc', file_store)
    event_stream.add_event(NullObservation('obs1'), EventSource.AGENT)
    event_stream.add_event(NullObservation('obs2'), EventSource.AGENT)
    assert len(collect_events(event_stream)) == 2

    stream2 = EventStream('es2', file_store)
    assert len(collect_events(stream2)) == 0

    stream1rehydrated = EventStream('abc', file_store)
    events = collect_events(stream1rehydrated)
    assert len(events) == 2
    assert events[0].content == 'obs1'
    assert events[1].content == 'obs2'


def test_exclusions(temp_dir: str):
    """
    Test that delegate events are excluded when iterating in reverse.
    """
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('abc', file_store)

    # Add events:
    # ID 0: NullAction
    # ID 1: AgentDelegateAction
    # ID 2: NullAction (within delegate range)
    # ID 3: AgentDelegateObservation (ends delegate range)
    # ID 4: NullAction
    event_stream.add_event(NullAction(), EventSource.AGENT)  # id=0
    event_stream.add_event(
        AgentDelegateAction(agent='delegate_agent', inputs={'task': 'test_task'}),
        EventSource.AGENT,
    )  # id=1
    event_stream.add_event(
        ChangeAgentStateAction(agent_state=AgentState.AWAITING_USER_INPUT),
        EventSource.AGENT,
    )  # id=2
    event_stream.add_event(
        FileReadAction(path='read_delegate.txt', start=0, end=5), EventSource.AGENT
    )  # id=3
    event_stream.add_event(
        AgentDelegateObservation(outputs={}, content='Delegate completed'),
        EventSource.AGENT,
    )  # id=4
    event_stream.add_event(
        AgentStateChangedObservation(
            agent_state=AgentState.AWAITING_USER_INPUT, content='unknown'
        ),
        EventSource.AGENT,
    )  # id=5

    # Collect events in reverse without delegates
    events = list(event_stream.get_events(reverse=True))
    event_ids = [event.id for event in events]
    assert (
        event_ids == [4, 1]
    ), 'Null, agent change states, and delegate events should be excluded by default when iterating'


def test_delegate_exclusion_forward_with_file_actions(temp_dir: str):
    """
    Test that delegate events with file actions are excluded when iterating forward.
    """
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('delegate_read_write', file_store)

    # Add events:
    # ID 0: FileReadAction
    # ID 1: FileReadObservation
    # ID 2: AgentDelegateAction
    # ID 3: FileWriteAction (within delegate range)
    # ID 4: FileWriteObservation (ends delegate range)
    # ID 5: FileReadAction
    event_stream.add_event(
        FileReadAction(path='read_first.txt', start=0, end=5), EventSource.AGENT
    )  # id=0
    event_stream.add_event(
        FileReadObservation(
            path='read_first.txt',
            content='Initial Read',
            observation=ObservationType.READ,
        ),
        EventSource.AGENT,
    )  # id=1
    event_stream.add_event(
        AgentDelegateAction(agent='delegate_agent', inputs={'task': 'delegate_task'}),
        EventSource.AGENT,
    )  # id=2
    event_stream.add_event(
        FileWriteAction(path='write_delegate.txt', content='Delegate Content'),
        EventSource.AGENT,
    )  # id=3
    event_stream.add_event(
        FileWriteObservation(
            path='write_delegate.txt',
            content='Delegate Content',
            observation=ObservationType.WRITE,
        ),
        EventSource.AGENT,
    )  # id=4
    event_stream.add_event(
        AgentDelegateObservation(outputs={}, content='Delegate completed'),
        EventSource.AGENT,
    )  # id=5
    event_stream.add_event(
        FileReadAction(path='read_after_delegate.txt', start=10, end=20),
        EventSource.AGENT,
    )  # id=6

    # Collect events without delegates
    events = collect_events(event_stream)
    event_ids = [event.id for event in events]
    assert event_ids == [
        0,
        1,
        2,
        5,
        6,
    ], 'Delegate events should be excluded when iterating forward'


def test_delegate_exclusion_reverse_with_file_actions(temp_dir: str):
    """
    Test that delegate events with file actions are excluded when iterating in reverse.
    """
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('delegate_read_write_reverse', file_store)

    # Add events:
    # ID 0: FileWriteAction
    # ID 1: FileWriteObservation
    # ID 2: AgentDelegateAction
    # ID 3: FileReadAction (within delegate range)
    # ID 4: FileReadObservation (ends delegate range)
    # ID 5: FileWriteAction
    event_stream.add_event(
        FileWriteAction(path='write_first.txt', content='Initial Write'),
        EventSource.AGENT,
    )  # id=0
    event_stream.add_event(
        FileWriteObservation(
            path='write_first.txt',
            content='Initial Write',
            observation=ObservationType.WRITE,
        ),
        EventSource.AGENT,
    )  # id=1
    event_stream.add_event(
        AgentDelegateAction(agent='delegate_agent', inputs={'task': 'delegate_task'}),
        EventSource.AGENT,
    )  # id=2
    event_stream.add_event(
        FileReadAction(path='read_delegate.txt', start=5, end=15), EventSource.AGENT
    )  # id=3
    event_stream.add_event(
        FileReadObservation(
            path='read_delegate.txt',
            content='Delegate Read',
            observation=ObservationType.READ,
        ),
        EventSource.AGENT,
    )  # id=4
    event_stream.add_event(
        AgentDelegateObservation(outputs={}, content='Delegate completed'),
        EventSource.AGENT,
    )  # id=5
    event_stream.add_event(
        FileWriteAction(path='write_after_delegate.txt', content='Post Delegate Write'),
        EventSource.AGENT,
    )  # id=6

    # Collect events in reverse without delegates
    events = list(event_stream.get_events(reverse=True, include_delegates=False))
    event_ids = [event.id for event in events]
    assert event_ids == [
        6,
        5,
        2,
        1,
        0,
    ], 'Delegate events should be excluded when iterating in reverse'


def test_delegate_inclusion_with_file_actions(temp_dir: str):
    """
    Test that delegate events with file actions are included when include_delegates is True.
    """
    file_store = get_file_store('local', temp_dir)
    event_stream = EventStream('delegate_inclusion', file_store)

    # Add events:
    # ID 0: FileReadAction
    # ID 1: FileReadObservation
    # ID 2: AgentDelegateAction
    # ID 3: FileWriteAction (within delegate range)
    # ID 4: FileWriteObservation (ends delegate range)
    # ID 5: FileReadAction
    event_stream.add_event(
        FileReadAction(path='read_first_incl.txt', start=0, end=5), EventSource.AGENT
    )  # id=0
    event_stream.add_event(
        FileReadObservation(
            path='read_first_incl.txt',
            content='Initial Read',
            observation=ObservationType.READ,
        ),
        EventSource.AGENT,
    )  # id=1
    event_stream.add_event(
        AgentDelegateAction(agent='delegate_agent', inputs={'task': 'delegate_task'}),
        EventSource.AGENT,
    )  # id=2
    event_stream.add_event(
        FileWriteAction(path='write_delegate_incl.txt', content='Delegate Inclusion'),
        EventSource.AGENT,
    )  # id=3
    event_stream.add_event(
        FileWriteObservation(
            path='write_delegate_incl.txt',
            content='Delegate Inclusion',
            observation=ObservationType.WRITE,
        ),
        EventSource.AGENT,
    )  # id=4
    event_stream.add_event(
        FileReadAction(path='read_after_delegate_incl.txt', start=10, end=20),
        EventSource.AGENT,
    )  # id=5

    # Collect events including delegates
    events = [event for event in event_stream.get_events(include_delegates=True)]
    event_ids = [event.id for event in events]
    assert (
        event_ids == [0, 1, 2, 3, 4, 5]
    ), 'All events including delegates should be included when include_delegates is True'
