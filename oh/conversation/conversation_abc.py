from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import datetime
from io import IOBase
from pathlib import Path
from typing import Optional, Union
from uuid import UUID

from oh.agent.agent_config import AgentConfig
from oh.agent.agent_info import AgentInfo
from oh.announcement import announcement
from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC
from oh.announcement.announcement_filter import AnnouncementFilter
from oh.file.download import Download
from oh.file.file_filter import FileFilter
from oh.file.file_info import FileInfo
from oh.conversation.listener import conversation_listener_abc
from oh.conversation.conversation_status import ConversationStatus
from oh.storage.page import Page
from oh.command import oh_command
from oh.command.runnable import runnable_abc
from oh.command.command_filter import CommandFilter


class ConversationABC(ABC):
    """
    Conversation combing output events, input commands, and files, allowing interfacing internally with an
    AI Agent. A user can mantain multiple conversations concurrently, or multiple users could potentially
    share a single conversation.
    * Announcements represent messages from the server to external clients
    * Commands represent commands in progress on the server
    * Files serve to store data related to commands.

    All methods are designed to make no assumptions about the volume of data / requests being handled,
    and are designed to be environment agnostic - the actual conversation may be running locally or
    remotely / in a docker container.
    Given that different operating systems support different permissions models and that the user is
    only really going to be interested in files the agent can access in this context, the permissions
    model for files is limited. Conversation security is not implemented at this level - it is assumed that
    if you have the UUID for a conversation then you are meant to have access. Security should be implemented
    in the level above this (e.g.: FastAPI) level if required.
    """

    id: UUID
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime

    @abstractmethod
    async def add_listener(
        self, listener: conversation_listener_abc.ConversationListenerABC
    ) -> UUID:
        """Add a listener and return a unique id for it"""

    @abstractmethod
    async def remove_listener(self, listener_id: UUID) -> bool:
        """Remove the listener with the id given"""

    @abstractmethod
    async def trigger_event(
        self, detail: AnnouncementDetailABC
    ) -> announcement.Announcement:
        """Trigger the event given"""

    @abstractmethod
    async def get_event(self, event_id: UUID) -> Optional[announcement.Announcement]:
        """Get an event with the id given"""

    @abstractmethod
    async def search_events(
        self, filter: Optional[AnnouncementFilter] = None, page_id: Optional[str] = None
    ) -> Page[announcement.Announcement]:
        """Search events in this conversation"""

    @abstractmethod
    async def count_events(self, filter: Optional[AnnouncementFilter] = None) -> int:
        """Count events in this conversation"""

    @abstractmethod
    async def create_command(
        self,
        runnable: runnable_abc.RunnableABC,
        title: Optional[str] = None,
        delay: float = 0,
    ) -> oh_command.Command:
        """Create a command and return details of it.  Throw a SesionError if the conversation is not in a ready state."""

    @abstractmethod
    async def run_command(
        self,
        runnable: runnable_abc.RunnableABC,
        title: Optional[str] = None,
        delay: float = 0,
    ):
        """Run the command and return the result. Throw a SesionError if the conversation is not in a ready state."""

    @abstractmethod
    async def cancel_command(self, command_id: UUID) -> bool:
        """Attempt to cancel command with the id given. Return true if the command was cancelled, false otherwise"""

    @abstractmethod
    async def get_command(self, command_id: UUID) -> Optional[oh_command.Command]:
        """Get the handle for the command with the id given"""

    @abstractmethod
    async def search_commands(
        self, filter: CommandFilter, page_id: str
    ) -> Page[oh_command.Command]:
        """Get info on running commands"""

    @abstractmethod
    async def count_commands(self, filter: CommandFilter) -> int:
        """Get the number of commands"""

    @abstractmethod
    async def create_dir(self, path: str) -> FileInfo:
        """
        Make the directory at the path given if it does not exist. Return info in the directory.
        Directories have the mime type `application/x-directory`
        """

    @abstractmethod
    async def create_file(self, path: str) -> FileInfo:
        """Update the updated_at on the file at the path given. If no file exists, create an empty file"""

    @abstractmethod
    async def save_file(self, path: str, content: Union[Path, IOBase]) -> FileInfo:
        """Upload a file to the path given. If no file exists, create an empty file. If a file exists, overwrite it."""

    @abstractmethod
    async def delete_file(self, path: str) -> bool:
        """Delete the file at the path given. Return True if the file existed and was deleted"""

    @abstractmethod
    async def load_file(self, path: str) -> Optional[Download]:
        """Get the file at the path given. Directories are not downloadable. Return a download if the file was retrieved."""

    @abstractmethod
    async def get_file_info(self, path: str) -> Optional[FileInfo]:
        """Get info on a file"""

    @abstractmethod
    async def search_file_info(
        self, filter: FileFilter, page_id: Optional[str] = None
    ) -> Page[FileInfo]:
        """Search files available in the conversation"""

    @abstractmethod
    async def count_files(self, filter: FileFilter) -> int:
        """Search files available in the conversation"""

    @abstractmethod
    async def get_agent_info(self) -> AgentInfo:
        """ Get the agent config for this conversation """
