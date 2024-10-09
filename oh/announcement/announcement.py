from dataclasses import dataclass, field
from datetime import datetime
from typing import Generic, Optional, TypeVar
from uuid import UUID, uuid4

from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC


T = TypeVar("T", bound=AnnouncementDetailABC)


@dataclass
class Announcement(Generic[T]):
    """
    Class representing some event that occurred within and OpenHands Process that may be
    interest externally. For example: CommandCompleted
    """

    conversation_id: UUID
    detail: T
    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=datetime.now)

    def dump() -> str:
        # Dump this event to an external format.
        raise ValueError("not_implemented")
