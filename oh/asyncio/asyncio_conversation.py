import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime
import glob
from io import IOBase
from itertools import count
import itertools
import logging
import mimetypes
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Callable, Dict, List, Optional, Union
from uuid import UUID, uuid4

from oh.agent.agent_config import AgentConfig
from oh.agent.agent_info import AgentInfo
from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC
from oh.announcement.announcement_filter import AnnouncementFilter
from oh.announcement.announcement import Announcement
from oh.file.download import Download
from oh.file.file_error import FileError
from oh.file.file_filter import FileFilter
from oh.file.file_info import FileInfo
from oh.file.file_permissions import FilePermissions
from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_error import ConversationError
from oh.conversation.conversation_status import ConversationStatus
from oh.storage.mem_storage import MemStorage
from oh.storage.page import Page
from oh.storage.storage_abc import StorageABC
from oh.command.oh_command import Command
from oh.command.runnable.runnable_abc import RunnableABC
from oh.command.command_filter import CommandFilter
from oh.command.command_status import CommandStatus
from oh.util.async_util import async_thread, wait_all

_LOGGER = logging.getLogger(__name__)


@dataclass
class AsyncioConversation(ConversationABC):
    workspace_path: Path
    agent_config: AgentConfig
    id: UUID = field(default_factory=uuid4)
    status: ConversationStatus = ConversationStatus.CREATING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    listeners: Dict[UUID, ConversationListenerABC] = field(default_factory=dict)
    event_storage: StorageABC[Announcement, AnnouncementFilter] = field(
        default_factory=MemStorage
    )
    command_storage: StorageABC[Command, CommandFilter] = field(
        default_factory=MemStorage
    )
    tasks: Dict[UUID, asyncio.Task] = field(default_factory=dict)
    max_file_page_size: int = 100

    def __post_init__(self):
        self.workspace_path = self.workspace_path.absolute()

    async def add_listener(self, listener: ConversationListenerABC) -> UUID:
        listener_id = uuid4()
        self.listeners[listener_id] = listener
        return listener_id

    async def remove_listener(self, listener_id: UUID) -> bool:
        return self.listeners.pop(listener_id) is not None

    async def trigger_event(self, detail: AnnouncementDetailABC):
        event = Announcement(conversation_id=self.id, detail=detail)
        await self.event_storage.create(event)
        _LOGGER.debug(f"conversation_event:{self.id}:{event}")
        await wait_all(listener.on_event(event) for listener in self.listeners.values())

    async def get_event(self, event_id: UUID) -> Optional[Announcement]:
        return await self.event_storage.read(event_id)

    async def search_events(
        self, filter: Optional[AnnouncementFilter] = None, page_id: Optional[str] = None
    ) -> Page[Announcement]:
        return await self.event_storage.search(filter, page_id)

    async def count_events(self, filter: Optional[AnnouncementFilter] = None) -> int:
        return await self.event_storage.count(filter)

    async def create_command(
        self, runnable: RunnableABC, title: Optional[str] = None, delay: float = 0
    ) -> Command:
        if self.status != ConversationStatus.READY:
            raise ConversationError(str(self.status))
        command = Command(
            conversation_id=self.id,
            runnable=runnable,
            title=title,
            status=CommandStatus.PENDING if delay else CommandStatus.RUNNING,
        )
        await self.command_storage.create(command)
        task = asyncio.create_task(self._run_command(command, delay))
        self.tasks[command.id] = task
        task.add_done_callback(self._task_done(command.id))
        return command

    def _task_done(self, command_id: UUID) -> Callable:
        def on_task_done(future: asyncio.Future):
            asyncio.create_task(self._cleanup_command(command_id))

        return on_task_done

    async def _cleanup_command(self, command_id: UUID):
        asyncio_command = self.tasks.pop(command_id, None)
        command = await self.command_storage.read(command_id)
        if asyncio_command.exception:
            command.status = CommandStatus.ERROR
        elif asyncio_command.cancelled:
            command.status = CommandStatus.CANCELLED
        elif asyncio_command.done:
            command.status = CommandStatus.COMPLETED
        await self.command_storage.update(command)

    async def run_command(
        self,
        runnable: RunnableABC,
        title: Optional[str] = None,
        delay: float = 0,
    ):
        if self.status != ConversationStatus.READY:
            raise ConversationError(str(self.status))
        command = await self.create_command(runnable, title, delay)
        result = await self.tasks[command.id]
        return result

    async def cancel_command(self, command_id: UUID) -> bool:
        asyncio_command = self.tasks[command_id]
        if asyncio_command:
            return asyncio_command.cancel()
        return False

    async def get_command(self, command_id: UUID) -> Optional[Command]:
        return await self.command_storage.read(command_id)

    async def search_commands(
        self, filter: Optional[CommandFilter] = None, page_id: Optional[str] = None
    ) -> Page[Command]:
        return await self.command_storage.search(filter, page_id)

    async def count_commands(self, filter: CommandFilter) -> int:
        return await self.command_storage.count(filter)

    async def _run_command(self, command: Command, delay: float):
        if delay:
            await asyncio.sleep(delay)
        await command.runnable.run(
            command_id=command.id,
            conversation=self,
        )

    def _to_internal_path(self, path: str) -> Path:
        result = Path(self.workspace_path, PurePosixPath(path)).resolve()
        if not result.startswith(self.workspace_path):
            raise FileError(f"invalid_path:{path}")
        return result

    def _to_file_info(self, path: Path) -> FileInfo:
        external_path = str(
            PurePosixPath(*path.parts[len(self.workspace_path.parts) :])
        )
        if path.is_dir():
            external_path += "/"  # Directories end with a trailing slash.
        mime_type = (
            "application/x-directory" if path.is_dir() else mimetypes.guess_type(path)
        )
        return FileInfo(
            path=external_path,
            size_in_bytes=os.path.getsize(path),
            updated_at=datetime.fromtimestamp(os.path.getmtime(path)),
            permissions=FilePermissions(
                os.access(path, os.R_OK),
                os.access(path, os.W_OK),
                os.access(path, os.X_OK),
            ),
            mime_type=mime_type[0],
        )

    async def create_dir(self, path: str) -> FileInfo:
        internal_path = self._to_internal_path(path)
        if internal_path.exists():
            if not internal_path.is_dir():
                raise FileError("not_a_directory:{path}")
        else:
            internal_path.mkdir(parents=True, exist_ok=True)
        return self._to_file_info(internal_path)

    async def create_file(self, path: str) -> FileInfo:
        internal_path = self._to_internal_path(path)
        if internal_path.exists():
            if internal_path.is_dir():
                raise FileError("not_a_file:{path}")
        else:
            internal_path.parent.mkdir(parents=True, exist_ok=True)
            open(internal_path, "w").close()
        return self._to_file_info(internal_path)

    async def save_file(self, path: str, content: Union[Path, IOBase]) -> FileInfo:
        internal_path = self._to_internal_path(path)
        internal_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            await async_thread(shutil.copy, content, internal_path)
        else:
            with open(internal_path, "wb") as destination:
                await async_thread(shutil.copyfileobj, content, destination)

    async def delete_file(self, path: str) -> bool:
        internal_path = self._to_internal_path(path)
        if not internal_path.exists():
            return False
        await async_thread(shutil.rmtree, internal_path)
        return True

    async def load_file(self, path: str) -> Optional[Download]:
        internal_path = self._to_internal_path(path)
        if not internal_path.exists() or not os.access(internal_path, os.R_OK):
            return False
        return Download(file_info=self._to_file_info, path=path)

    async def get_file_info(self, path: str) -> Optional[FileInfo]:
        internal_path = self._to_internal_path(path)
        if not internal_path.exists():
            return None
        return self._to_file_info(internal_path)

    async def search_file_info(
        self, filter: FileFilter, page_id: Optional[str] = None
    ) -> Page[FileInfo]:
        paths = iter(await self._list_files(filter))

        # For this purpose we simply incode an offset in base64. We encode it to insulate against
        # folks thinking that a number is a part of the spec - the page key may not be a number
        # in other impls!
        skip = 0
        if page_id:
            skip = int(base64.b64decode(page_id))
        sum(1 for _ in itertools.islice(paths, skip))

        results = list(
            self._to_file_info(path)
            for path in itertools.islice(paths, self.max_file_page_size)
        )

        # Check if there are  more results...
        next_page_id = None
        try:
            next(paths)
            next_page_id = base64.b64encode(str(skip + self.max_file_page_size))
        except StopIteration:
            next_page_id = None

        return Page(results, next_page_id)

    async def count_files(self, filter: Optional[FileFilter]) -> int:
        return count(self._list_files(filter))

    async def _list_files(self, filter: Optional[FileFilter]) -> List[Path]:
        list_path = await self._get_list_path(filter)
        if filter and filter.path_delimiter == "/":
            files = os.listdir(list_path)
            paths = [Path(list_path, file) for file in files]
            return paths
        files = glob(Path(list_path, "**", "*"))
        paths = [Path(file) for file in files]
        return paths

    async def _get_list_path(self, filter: Optional[FileFilter]) -> Path:
        if filter and filter.path_prefix:
            return self._to_internal_path(filter.path_prefix)
        return self.workspace_path

    async def destroy(self, grace_period: int):
        _LOGGER.info(f"destroying_conversation:{self.id}")
        self.status = ConversationStatus.DESTROYING
        await wait_all(self.cancel_command(command_id) for command_id in self.tasks)

    async def get_agent_info(self) -> AgentInfo:
        return self.agent_config
