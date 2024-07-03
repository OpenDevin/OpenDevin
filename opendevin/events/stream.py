import asyncio
import json
import os
import shutil
from datetime import datetime
from enum import Enum
from typing import Callable, Iterable

from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.serialization.event import event_from_dict, event_to_dict
from opendevin.storage import (
    FileStore,
    InMemoryFileStore,
    LocalFileStore,
    get_file_store,
)

from .event import Event, EventSource


class EventStreamSubscriber(str, Enum):
    AGENT_CONTROLLER = 'agent_controller'
    SERVER = 'server'
    RUNTIME = 'runtime'
    MAIN = 'main'
    TEST = 'test'


class EventStream:
    sid: str
    # For each subscriber ID, there is a stack of callback functions - useful
    # when there are agent delegates
    _subscribers: dict[str, list[Callable]]
    _cur_id: int
    _lock: asyncio.Lock
    _file_store: FileStore

    def __init__(self, sid: str, reinitialize: bool = True):
        self.sid = sid
        self._file_store: FileStore = get_file_store()
        self._subscribers = {}
        self._cur_id = 0
        self._lock = asyncio.Lock()
        if reinitialize:
            self._reinitialize_from_file_store()

    def reset(self):
        """Reset the EventStream instance to its initial state."""
        self._cur_id = 0
        self._subscribers = {}
        self._clear_file_store()
        self._events = []

    def _clear_file_store(self):
        """Clear all events from the file store for this session."""
        session_dir = f'sessions/{self.sid}'

        if isinstance(self._file_store, LocalFileStore):
            self._clear_local_file_store(session_dir)
        elif isinstance(self._file_store, InMemoryFileStore):
            self._clear_in_memory_file_store(session_dir)
        else:
            self._clear_generic_file_store(session_dir)

    @classmethod
    def clear_all_sessions(cls):
        """Clear all sessions from the file store."""
        file_store = get_file_store()

        if isinstance(file_store, LocalFileStore):
            cls._clear_local_sessions(file_store)
        elif isinstance(file_store, InMemoryFileStore):
            cls._clear_in_memory_sessions(file_store)
        else:
            cls._clear_generic_sessions(file_store)

    @staticmethod
    def _clear_in_memory_sessions(file_store: InMemoryFileStore):
        keys_to_delete = [
            key for key in file_store.files.keys() if key.startswith('sessions/')
        ]
        for key in keys_to_delete:
            del file_store.files[key]

    @staticmethod
    def _clear_generic_sessions(file_store: FileStore):
        try:
            sessions = file_store.list('sessions')
            for session in sessions:
                try:
                    events = file_store.list(f'sessions/{session}/events')
                    for event in events:
                        file_store.delete(f'sessions/{session}/events/{event}')
                    file_store.delete(f'sessions/{session}/events')
                except FileNotFoundError:
                    pass
                file_store.delete(f'sessions/{session}')
            file_store.delete('sessions')
        except PermissionError as e:
            logger.error(f'Permission denied when clearing local file store: {e}')

    @staticmethod
    def _clear_local_sessions(file_store: LocalFileStore):
        if not isinstance(file_store, LocalFileStore):
            raise TypeError('Expected LocalFileStore for clearing local sessions')

        local_store = file_store  # type: LocalFileStore
        sessions_dir = local_store.get_full_path('sessions')
        if os.path.exists(sessions_dir):
            shutil.rmtree(sessions_dir)

    def _clear_local_file_store(self, session_dir):
        if not isinstance(self._file_store, LocalFileStore):
            raise TypeError('Expected LocalFileStore for clearing local file store')

        local_store = self._file_store  # type: LocalFileStore
        try:
            full_path = local_store.get_full_path(session_dir)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
        except PermissionError as e:
            logger.error(f'No permissions to clear local file store: {e}')

    def _clear_in_memory_file_store(self, session_dir):
        if not isinstance(self._file_store, InMemoryFileStore):
            raise TypeError(
                'Expected InMemoryFileStore for clearing in-memory file store'
            )

        in_memory_store = self._file_store  # type: InMemoryFileStore
        try:
            keys_to_delete = [
                key
                for key in in_memory_store.files.keys()
                if key.startswith(session_dir)
            ]
            for key in keys_to_delete:
                in_memory_store.delete(key)
        except Exception as e:
            logger.error(f'Error clearing in-memory file store: {e}')

    def _clear_generic_file_store(self, session_dir):
        try:
            events_dir = f'{session_dir}/events'
            try:
                events = self._file_store.list(events_dir)
                for event in events:
                    self._file_store.delete(f'{events_dir}/{event}')
                self._file_store.delete(events_dir)
            except FileNotFoundError:
                # If the events directory doesn't exist, that's fine
                pass

            try:
                self._file_store.delete(session_dir)
            except FileNotFoundError:
                # If the session directory doesn't exist, that's fine
                pass
        except Exception as e:
            logger.error(f'Error clearing generic file store: {e}')

    def _reinitialize_from_file_store(self):
        try:
            events = self._file_store.list(f'sessions/{self.sid}/events')
        except FileNotFoundError:
            logger.warning(f'No events found for session {self.sid}')
            return
        for event_str in events:
            id = self._get_id_from_filename(event_str)
            if id >= self._cur_id:
                self._cur_id = id + 1

    def _get_filename_for_id(self, id: int) -> str:
        return f'sessions/{self.sid}/events/{id}.json'

    @staticmethod
    def _get_id_from_filename(filename: str) -> int:
        try:
            return int(filename.split('/')[-1].split('.')[0])
        except ValueError:
            logger.warning(f'get id from filename ({filename}) failed.')
            return -1

    def get_events(self, start_id=0, end_id=None) -> Iterable[Event]:
        event_id = start_id
        while True:
            if end_id is not None and event_id > end_id:
                break
            try:
                event = self.get_event(event_id)
            except FileNotFoundError:
                break
            yield event
            event_id += 1

    def get_event(self, id: int) -> Event:
        filename = self._get_filename_for_id(id)
        content = self._file_store.read(filename)
        data = json.loads(content)
        return event_from_dict(data)

    def subscribe(self, id: EventStreamSubscriber, callback: Callable, append=False):
        if id in self._subscribers:
            if append:
                self._subscribers[id].append(callback)
            else:
                raise ValueError('Subscriber already exists: ' + id)
        else:
            self._subscribers[id] = [callback]

    def unsubscribe(self, id: EventStreamSubscriber):
        if id not in self._subscribers:
            logger.warning('Subscriber not found during unsubscribe: ' + id)
        else:
            self._subscribers[id].pop()
            if len(self._subscribers[id]) == 0:
                del self._subscribers[id]

    # TODO: make this not async
    async def add_event(self, event: Event, source: EventSource):
        logger.debug(f'Adding event {event} from {source}')
        async with self._lock:
            event._id = self._cur_id  # type: ignore [attr-defined]
            self._cur_id += 1
        event._timestamp = datetime.now()  # type: ignore [attr-defined]
        event._source = source  # type: ignore [attr-defined]
        data = event_to_dict(event)
        if event.id is not None:
            self._file_store.write(
                self._get_filename_for_id(event.id), json.dumps(data)
            )
        for stack in self._subscribers.values():
            callback = stack[-1]
            logger.debug(f'Notifying subscriber {callback} of event {event}')
            await callback(event)
