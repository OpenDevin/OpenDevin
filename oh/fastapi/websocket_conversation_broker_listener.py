from abc import abstractmethod
import asyncio
from dataclasses import dataclass, field
from typing import Dict
from uuid import UUID

from fastapi import WebSocket
from pydantic import TypeAdapter
from oh.announcement.announcement import Announcement
from oh.fastapi.websocket_conversation_listener import WebsocketConversationListener
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation_broker.conversation_broker_listener_abc import (
    ConversationBrokerListenerABC,
)
from oh.util.async_util import wait_all


@dataclass
class WebsocketConversationBrokerListener(ConversationBrokerListenerABC):
    websocket: WebSocket
    event_info_adapter: TypeAdapter
    listeners: Dict[UUID, WebsocketConversationListener] = field(default_factory=dict)

    async def after_create_conversation(self, conversation: ConversationABC):
        listener = WebsocketConversationListener(
            conversation.id, self.websocket, self.event_info_adapter
        )
        id = await conversation.add_listener(listener)
        self.listeners[id] = listener

    async def before_destroy_conversation(self, conversation: ConversationABC):
        await wait_all(
            conversation.remove_listener(listener_id)
            for listener_id, conversation_listener in self.listeners.items()
            if conversation_listener.conversation_id == conversation.id
        )
