from asyncio import Condition, Future
from dataclasses import field
from typing import Dict, List, Set
from uuid import UUID
from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC
from oh.announcement.detail.command_status_update import CommandStatusUpdate
from oh.announcement.announcement import Announcement
from oh.command.command_status import CommandStatus


class CommandFinishedListener(ConversationListenerABC):
    """
    Listener that allows waiting for commands to finish before proceeding
    """

    finished: Dict[UUID, Condition] = field(default_factory=dict)

    async def on_event(self, event: Announcement):
        detail = event.detail
        if isinstance(detail, CommandStatusUpdate) and detail.status in [
            CommandStatus.CANCELLED,
            CommandStatus.ERROR,
            CommandStatus.COMPLETED,
        ]:
            condition = self._get_condition(detail.command_id)
            async with condition:
                condition.notify_all()

    async def on_command_finished(self, command_id: UUID):
        condition = self._get_condition(command_id)
        await condition.wait()

    def _get_condition(self, command_id: UUID):
        condition = self.finished.get(command_id)
        if not condition:
            condition = self.finished[command_id] = Condition()
        return condition
