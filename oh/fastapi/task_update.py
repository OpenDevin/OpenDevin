from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel

from oh.command.command_status import CommandStatus


class CommandUpdate(BaseModel):
    id: UUID
    status: CommandStatus
    type: Literal["CommandUpdate"] = "CommandUpdate"
