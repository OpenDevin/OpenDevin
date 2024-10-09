from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Optional, Set, Type, Union
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter
from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC
from oh.storage.page import Page
from oh.command.runnable.runnable_abc import RunnableABC
from oh.command.command_status import CommandStatus
from oh.util.subtype_finder import find_subtypes


@dataclass
class DynamicTypes:
    event_detail_types: Set[Type[AnnouncementDetailABC]] = field(
        default_factory=lambda: find_subtypes(AnnouncementDetailABC)
    )
    runnable_types: Set[Type[RunnableABC]] = field(
        default_factory=lambda: find_subtypes(RunnableABC)
    )
    _event_detail_type: Optional[Type] = None
    _event_detail_type_adapter: Optional[TypeAdapter] = None
    _event_info_class: Optional[Type] = None
    _page_event_info_type_adapter: Optional[TypeAdapter] = None
    _event_info_type_adapter: Optional[TypeAdapter] = None
    _runnable_info_type: Optional[Type] = None
    _runnable_type_adapter: Optional[TypeAdapter] = None
    _command_info_class: Optional[Type] = None
    _create_command_class: Optional[Type] = None
    _page_command_info_type_adapter: Optional[TypeAdapter] = None

    def get_event_detail_type(self) -> Type:
        if self._event_detail_type is None:
            self._event_detail_type = Annotated[
                Union[tuple(self.event_detail_types)], Field(discriminator="type")
            ]
        return self._event_detail_type

    def get_event_detail_type_adapter(self) -> TypeAdapter:
        if self._event_detail_type_adapter is None:
            self._event_detail_type_adapter = TypeAdapter(self.get_event_detail_type())
        return self._event_detail_type_adapter

    def get_event_info_class(self) -> Type:
        if self._event_info_class is None:

            class AnnouncementInfo(BaseModel):
                id: UUID
                conversation_id: UUID
                detail: self.get_event_detail_type()  # type: ignore
                created_at: datetime

            self._event_info_class = AnnouncementInfo
        return self._event_info_class

    def get_page_event_info_type_adapter(self) -> TypeAdapter:
        if self._page_event_info_type_adapter is None:
            self._page_event_info_type_adapter = TypeAdapter(
                Page[self.get_event_info_class()]
            )
        return self._page_event_info_type_adapter

    def get_event_info_type_adapter(self) -> TypeAdapter:
        if self._event_info_type_adapter is None:
            self._event_info_type_adapter = TypeAdapter(self.get_event_info_class())
        return self._event_info_type_adapter

    def get_runnable_info_type(self) -> Type:
        if self._runnable_info_type is None:
            self._runnable_info_type = Annotated[
                Union[tuple(self.runnable_types)], Field(discriminator="type")
            ]
        return self._runnable_info_type

    def get_runnable_type_adapter(self) -> TypeAdapter:
        if self._runnable_type_adapter is None:
            self._runnable_type_adapter = TypeAdapter(self.get_runnable_info_type())
        return self._runnable_type_adapter

    def get_command_info_class(self) -> Type:
        if self._command_info_class is None:

            class CommandInfo(BaseModel):
                id: UUID
                conversation_id: UUID
                runnable: self.get_runnable_info_type()  # type: ignore
                status: CommandStatus
                title: Optional[str]
                created_at: datetime
                updated_at: datetime

            self._command_info_class = CommandInfo
        return self._command_info_class

    def get_create_command_class(self) -> Type:
        if self._create_command_class is None:

            class CreateCommand(BaseModel):
                runnable: self.get_runnable_info_type()  # type: ignore
                title: Optional[str] = None
                delay: Optional[str] = None

            self._create_command_class = CreateCommand
        return self._create_command_class

    def get_page_command_info_type_adapter(self) -> TypeAdapter:
        if self._page_command_info_type_adapter is None:
            self._page_command_info_type_adapter = TypeAdapter(
                Page[self.get_command_info_class()]
            )
        return self._page_command_info_type_adapter
