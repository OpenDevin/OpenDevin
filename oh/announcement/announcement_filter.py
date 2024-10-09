from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from oh.announcement.announcement import Announcement
from oh.storage.item_filter_abc import ItemFilterABC


@dataclass
class AnnouncementFilter(ItemFilterABC[Announcement]):
    conversation_id__eq: Optional[UUID] = None

    def filter(self, item: Announcement) -> bool:
        if (
            self.conversation_id__eq
            and item.conversation_id != self.conversation_id__eq
        ):
            return False
        return True
