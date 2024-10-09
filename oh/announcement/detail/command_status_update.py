from dataclasses import dataclass
from typing import Literal, Optional
from uuid import UUID

from oh.announcement.detail.announcement_detail_abc import AnnouncementDetailABC
from oh.command.command_status import CommandStatus


@dataclass
class CommandStatusUpdate(AnnouncementDetailABC):
    """Announcement indicating that the status of a command has changed"""

    command_id: UUID
    status: CommandStatus
    type: Literal["CommandStatusUpdate"] = "CommandStatusUpdate"
