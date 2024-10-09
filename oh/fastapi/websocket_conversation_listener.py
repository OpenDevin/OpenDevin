from dataclasses import dataclass
import logging
from uuid import UUID

from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from pydantic import TypeAdapter
from oh.announcement.announcement import Announcement
from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC

_LOGGER = logging.getLogger(__name__)


@dataclass
class WebsocketConversationListener(ConversationListenerABC):
    conversation_id: UUID
    websocket: WebSocket
    event_info_adapter: TypeAdapter

    async def on_event(self, event: Announcement):
        if self.conversation_id != event.conversation_id:
            return
        data = self.event_info_adapter.dump_json(event).decode("UTF-8")
        try:
            if self.websocket.application_state == WebSocketState.CONNECTED:
                await self.websocket.send_text(data)
        except WebSocketDisconnect:
            _LOGGER.debug(f"websocket_distconnect:{self.conversation_id}")
