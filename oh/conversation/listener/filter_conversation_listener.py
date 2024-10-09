from __future__ import annotations
from dataclasses import dataclass
from uuid import UUID

from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC
from oh.announcement import announcement


@dataclass
class FilterConversationListener(ConversationListenerABC):
    conversation_id: UUID
    listener: ConversationListenerABC

    async def on_event(self, event: announcement.Announcement):
        if event.conversation_id == self.conversation_id:
            self.listener.on_event(event)
