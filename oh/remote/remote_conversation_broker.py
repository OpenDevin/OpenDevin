import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import UUID, uuid4

import httpx
from pydantic import TypeAdapter
from oh.agent.agent_config import AgentConfig
from oh.fastapi.conversation_info import ConversationInfo
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_filter import ConversationFilter
from oh.conversation_broker.conversation_broker_abc import ConversationBrokerABC
from oh.conversation_broker.conversation_broker_listener_abc import (
    ConversationBrokerListenerABC,
)
from oh.fastapi.dynamic_types import DynamicTypes
from oh.remote.websocket_event_client import WebsocketAnnouncementClient
from oh.remote.remote_conversation import RemoteConversation
from oh.storage.page import Page
from oh.util.async_util import wait_all

_SESSION_INFO_PAGE_ADAPTER = TypeAdapter(Page[ConversationInfo])


@dataclass
class RemoteConversationBroker(ConversationBrokerABC):
    """
    Manages conversations in a remote server.
    NOTE: Does not fire after_create_conversation and before_destroy_conversation if the operation did not go through this server.
    Is that reasonable? Do we need a different channel for events for these? I think we may...

    Maybe we need a global event list that actually includes conversation ids?
    """

    root_url: str
    listeners: Dict[UUID, ConversationBrokerListenerABC] = field(default_factory=dict)
    websocket_event_client: WebsocketAnnouncementClient = None
    httpx_client: httpx.AsyncClient = None
    dynamic_types: DynamicTypes = field(default_factory=DynamicTypes)

    def __post_init__(self):
        if not self.root_url.endswith("/"):
            self.root_url += "/"
        if not self.websocket_event_client:
            protocol = "wss" if self.root_url.startswith("https://") else "ws"
            websocket_url = f"{protocol}://{self.root_url.split("://")[1]}fire-hose"
            self.websocket_event_client = WebsocketAnnouncementClient(websocket_url)
        if not self.httpx_client:
            self.httpx_client = httpx.AsyncClient()

    async def add_listener(self, listener: ConversationBrokerListenerABC) -> UUID:
        listener_id = uuid4()
        self.listeners[listener_id] = listener
        return listener_id

    async def remove_listener(self, listener_id: UUID) -> bool:
        result = self.listeners.pop(listener_id) is not None
        return result

    async def get_conversation(
        self, conversation_id: UUID
    ) -> Optional[ConversationABC]:
        response = await self.httpx_client.get(
            f"{self.root_url}conversation/{conversation_id}"
        )
        response_json = response.json()
        if response_json:
            conversation_info = ConversationInfo.model_validate(response_json)
            return RemoteConversation(
                id=conversation_info.id,
                status=conversation_info.status,
                created_at=conversation_info.created_at,
                updated_at=conversation_info.updated_at,
                root_url=self.root_url,
                conversation_info=conversation_info,
                websocket_event_client=self.websocket_event_client,
                httpx_client=self.httpx_client,
                dynamic_types=self.dynamic_types,
            )

    async def search_conversations(
        self, filter: Optional[ConversationFilter] = None, page_id: Optional[str] = None
    ) -> Page[ConversationABC]:
        response = await self.httpx_client.get(f"{self.root_url}conversation")
        page = _SESSION_INFO_PAGE_ADAPTER.validate_python(response.json())
        return page

    async def count_conversations(self, filter: Optional[ConversationFilter]) -> int:
        response = await self.httpx_client.get(f"{self.root_url}conversation-count")
        return response.json()

    async def create_conversation(self, agent_config: AgentConfig) -> ConversationABC:
        response = await self.httpx_client.post(
            f"{self.root_url}conversation",
            json=TypeAdapter(AgentConfig).dump_python(agent_config),
        )
        conversation_info = ConversationInfo.model_validate(response.json())
        conversation = RemoteConversation(
            id=conversation_info.id,
            status=conversation_info.status,
            created_at=conversation_info.created_at,
            updated_at=conversation_info.updated_at,
            root_url=self.root_url,
            conversation_info=conversation_info,
            websocket_event_client=self.websocket_event_client,
            httpx_client=self.httpx_client,
            dynamic_types=self.dynamic_types,
        )
        for listener in self.listeners.values():
            listener.after_create_conversation(conversation)
        return conversation

    async def destroy_conversation(
        self, conversation_id: UUID, grace_period: int = 10
    ) -> bool:
        response = await self.httpx_client.delete(
            f"{self.root_url}conversation/{conversation_id}"
        )
        response_json = response.json()
        return response_json

    async def shutdown(self, grace_period: int = 10):
        # Note: This does not shut down remote server...
        await wait_all(
            (
                self.remove_listener(listener_id)
                for listener_id in list(self.listeners.keys())
            ),
            grace_period,
        )
        await self.httpx_client.aclose()
