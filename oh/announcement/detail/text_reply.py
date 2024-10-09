from dataclasses import dataclass
from typing import Literal
from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC


@dataclass
class TextReply(AnnouncementDetailABC):
    """A Text reply from the agent."""

    text: str
    type: Literal["TextReply"] = "TextReply"
