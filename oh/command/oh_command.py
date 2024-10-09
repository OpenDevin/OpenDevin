from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from oh.command.command_status import CommandStatus
from oh.command.runnable import runnable_abc


@dataclass
class Command:
    conversation_id: UUID
    runnable: runnable_abc.RunnableABC
    id: UUID = field(default_factory=uuid4)
    status: CommandStatus = CommandStatus.PENDING
    title: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
