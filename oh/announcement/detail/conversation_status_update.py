from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC
from oh.conversation.conversation_status import ConversationStatus


@dataclass
class ConversationStatusUpdate(AnnouncementDetailABC):
    """Announcement indicating that the status of a conversation has changed"""

    conversation_id: UUID
    status: ConversationStatus
    type: Literal["ConversationStatusUpdate"] = "ConversationStatusUpdate"
