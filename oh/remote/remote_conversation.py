from dataclasses import dataclass
from datetime import datetime
from io import IOBase
from pathlib import Path
from typing import Any, AsyncIterable, Optional, Union
from uuid import UUID

import httpx
from pydantic import TypeAdapter
from oh.agent.agent_config import AgentConfig
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_status import ConversationStatus
from oh.conversation.listener import conversation_listener_abc
from oh.conversation.listener.task_finished_listener import TaskFinishedListener
from oh.event import oh_event
from oh.event.detail.event_detail_abc import EventDetailABC
from oh.event.event_filter import EventFilter
from oh.fastapi.conversation_info import ConversationInfo
from oh.fastapi.dynamic_types import DynamicTypes
from oh.file.download import Download
from oh.file.file_filter import FileFilter
from oh.file.file_info import FileInfo
from oh.remote.websocket_event_client import WebsocketEventClient
from oh.storage.page import Page
from oh.task import oh_task
from oh.task.runnable import runnable_abc
from oh.task.task_filter import TaskFilter


@dataclass
class RemoteConversation(ConversationABC):
    """
    Remote conversation. Uses the RemoteConversationBroker for all listener operations (Thereby sharing
    a single websocket where possible)
    """

    id: UUID
    agent_config: AgentConfig
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    root_url: str
    conversation_info: ConversationInfo
    websocket_event_client: WebsocketEventClient
    httpx_client: httpx.AsyncClient
    dynamic_types: DynamicTypes

    async def add_listener(
        self, listener: conversation_listener_abc.ConversationListenerABC
    ) -> UUID:
        return await self.websocket_event_client.add_listener(listener)

    async def remove_listener(self, listener_id: UUID) -> bool:
        return await self.websocket_event_client.remove_listener(listener_id)

    async def trigger_event(self, detail: EventDetailABC) -> oh_event.OhEvent:
        response = await self.httpx_client.post(
            f"{self.root_url}conversation/{self.conversation_info.id}/event",
            json=self.dynamic_types.get_event_detail_type_adapter().dump_python(detail),
        )
        response.raise_for_status()
        event_info_class = self.dynamic_types.get_event_info_class()
        result = event_info_class.model_validate(response.json())
        return result

    async def get_event(self, event_id: UUID) -> Optional[oh_event.OhEvent]:
        response = await self.httpx_client.get(
            f"{self.root_url}conversation/{self.conversation_info.id}/event/{event_id}",
        )
        response.raise_for_status()
        data = response.json()
        if data:
            event_info_class = self.dynamic_types.get_event_info_class()
            result = event_info_class.model_validate(data)
            return result

    async def search_events(
        self, filter: Optional[EventFilter] = None, page_id: Optional[str] = None
    ) -> Page[oh_event.OhEvent]:
        # TODO: filter not currently supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/event"
        if page_id:
            url += f"?page_id={page_id}"
        response = await self.httpx_client.get(url)
        response.raise_for_status()
        data = response.json()
        type_adapter = self.dynamic_types.get_page_event_info_type_adapter()
        result = type_adapter.validate_python(data)
        return result

    async def count_events(self, filter: Optional[EventFilter] = None) -> int:
        # TODO: filter not currently supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/event-count"
        response = await self.httpx_client.get(url)
        data = response.json()
        return data

    async def create_task(
        self,
        runnable: runnable_abc.RunnableABC,
        title: Optional[str] = None,
        delay: float = 0,
    ) -> oh_task.OhTask:
        dumped = self.dynamic_types.get_runnable_type_adapter().dump_python(
            runnable
        )
        response = await self.httpx_client.post(
            f"{self.root_url}conversation/{self.conversation_info.id}/task",
            json={
                "runnable": dumped,
                "title": title,
                "delay": delay,
            },
        )
        response.raise_for_status()
        task_info_class = self.dynamic_types.get_task_info_class()
        result = task_info_class.model_validate(response.json())
        return result

    async def run_task(
        self,
        runnable: runnable_abc.RunnableABC,
        title: Optional[str] = None,
        delay: float = 0,
    ):
        listener = TaskFinishedListener()
        listener_id = self.add_listener(listener)
        try:
            task_info = await self.create_task(runnable, title, delay)
            result = await listener.on_task_finished(task_info.id)
            return result
        finally:
            self.remove_listener(listener_id)

    async def cancel_task(self, task_id: UUID) -> bool:
        response = await self.httpx_client.get(
            f"{self.root_url}conversation/{self.conversation_info.id}/task/{task_id}",
        )
        response.raise_for_status()
        data = response.json()
        return data

    async def get_task(self, task_id: UUID) -> Optional[oh_task.OhTask]:
        response = await self.httpx_client.get(
            f"{self.root_url}conversation/{self.conversation_info.id}/task/{task_id}",
        )
        response.raise_for_status()
        data = response.json()
        if data:
            task_info_class = self.dynamic_types.get_task_info_class()
            result = task_info_class.model_validate(response.json())
            return result

    async def search_tasks(
        self, filter: TaskFilter, page_id: str
    ) -> Page[oh_task.OhTask]:
        # TODO: Filter not yet supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/task"
        if page_id:
            url += f"?page_id={page_id}"
        response = await self.httpx_client.get(url)
        response.raise_for_status()
        data = response.json()
        type_adapter = self.dynamic_types.get_page_task_info_type_adapter()
        result = type_adapter.validate_python(data)
        return result

    async def count_tasks(self, filter: TaskFilter) -> int:
        # TODO: Filter not yet supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/task-count"
        response = await self.httpx_client.get(url)
        response.raise_for_status()
        data = response.json()
        return data

    async def create_dir(self, path: str) -> FileInfo:
        response = await self.httpx_client.post(
            f"{self.root_url}conversation/{self.conversation_info.id}/dir/{path}"
        )
        response.raise_for_status()
        result = FileInfo.model_validate(response.json())
        return result

    async def create_file(self, path: str) -> FileInfo:
        response = await self.httpx_client.post(
            f"{self.root_url}conversation/{self.conversation_info.id}/file/{path}"
        )
        response.raise_for_status()
        result = FileInfo.model_validate(response.json())
        return result

    async def save_file(self, path: str, content: Union[Path, IOBase]) -> FileInfo:
        # TODO: implement signed URL protocol (like S3) so files don't need to go through this server
        response = await self.httpx_client.post(
            f"{self.root_url}conversation/{self.conversation_info.id}/upload/{path}",
            files={"upload-file": content},
        )
        response.raise_for_status()
        result = FileInfo.model_validate(response.json())
        return result

    async def delete_file(self, path: str) -> bool:
        response = await self.httpx_client.delete(
            f"{self.root_url}conversation/{self.conversation_info.id}/file/{path}"
        )
        response.raise_for_status()
        return response.json()

    async def load_file(self, path: str) -> Optional[Download]:
        # TODO: implement signed URL protocol (like S3) so files don't need to go through this server
        file_info = self.get_file_info(path)
        if not file_info:
            return
        url = f"{self.root_url}conversation/{self.conversation_info.id}/file/{path}"
        return Download(
            file_info=file_info,
            content_stream=_RemoteContentStream(url, self.httpx_client),
        )

    async def get_file_info(self, path: str) -> Optional[FileInfo]:
        response = await self.httpx_client.get(
            f"{self.root_url}conversation/{self.conversation_info.id}/file/{path}"
        )
        response.raise_for_status()
        result = FileInfo.model_validate(response.json())
        return result

    async def search_file_info(
        self, filter: FileFilter, page_id: Optional[str] = None
    ) -> Page[FileInfo]:
        # TODO: Filter not yet supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/file-search"
        if page_id:
            url += f"?page_id={page_id}"
        response = await self.httpx_client.get(url)
        response.raise_for_status()
        type_adapter = TypeAdapter(Page[FileInfo])
        result = type_adapter.python_validate(response.json())
        return result

    async def count_files(self, filter: FileFilter) -> int:
        # TODO: Filter not yet supported
        url = f"{self.root_url}conversation/{self.conversation_info.id}/file-count"
        response = await self.httpx_client.get(url)
        response.raise_for_status()
        return response.json()


@dataclass
class _RemoteContentStream(AsyncIterable[Union[str, bytes, memoryview]]):
    url: str
    httpx_client: httpx.AsyncClient
    _resp: Any = None
    _aiter: Any = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.resp:
            self._resp = await self.httpx_client.stream("GET", self.url)
            self._aiter = self._resp.aiter_bytes()
        try:
            return await anext(self._aiter)
        except StopAsyncIteration:
            await self._aiter.__aclose__()
            await self._resp().__aclose__()
            raise