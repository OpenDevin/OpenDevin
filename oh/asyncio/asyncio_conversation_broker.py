import asyncio
from dataclasses import dataclass, field
from itertools import count
import logging
from typing import Dict, Optional
from uuid import UUID, uuid4
from oh.agent.agent_config import AgentConfig
from oh.asyncio.asyncio_conversation import AsyncioConversation
from oh.announcement.detail.conversation_status_update import ConversationStatusUpdate
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_filter import ConversationFilter
from oh.conversation.conversation_status import ConversationStatus
from oh.conversation_broker.conversation_broker_abc import ConversationBrokerABC
from oh.conversation_broker.conversation_broker_listener_abc import (
    ConversationBrokerListenerABC,
)
from oh.storage.page import Page
from oh.util.async_util import wait_all

_LOGGER = logging.getLogger(__name__)


@dataclass
class AsyncioConversationBroker(ConversationBrokerABC):
    """
    Non persistent conversation manager based on asyncio.
    High performance use in the context of a single process.
    """

    workspace_path: str
    listeners: Dict[UUID, ConversationBrokerListenerABC] = field(default_factory=dict)
    conversations: Dict[UUID, ConversationABC] = field(default_factory=dict)

    async def add_listener(self, listener: ConversationBrokerListenerABC) -> UUID:
        listener_id = uuid4()
        self.listeners[listener_id] = listener
        return listener_id

    async def remove_listener(self, listener_id: UUID) -> bool:
        return self.listeners.pop(listener_id) is not None

    async def get_conversation(
        self, conversation_id: UUID
    ) -> Optional[ConversationABC]:
        return self.conversations.get(conversation_id)

    async def search_conversations(
        self, filter: Optional[ConversationFilter] = None, page_id: Optional[str] = None
    ) -> Page[ConversationABC]:
        results = [
            conversation
            for conversation in self.conversations.values()
            if filter is None or filter.filter(conversation)
        ]
        return Page(results)

    async def count_conversations(self, filter: Optional[ConversationFilter]) -> int:
        results = count(
            conversation
            for conversation in self.conversations.values()
            if filter is None or filter.filter(conversation)
        )
        return results

    async def create_conversation(self, agent_config: AgentConfig) -> ConversationABC:
        conversation = AsyncioConversation(
            workspace_path=self.workspace_path, agent_config=agent_config
        )
        self.conversations[conversation.id] = conversation
        await wait_all(
            listener.after_create_conversation(conversation)
            for listener in self.listeners.values()
        )
        asyncio.create_task(self._on_conversation_ready(conversation))
        _LOGGER.info(f"conversation_created:{conversation.id}")
        return conversation

    async def _on_conversation_ready(self, conversation: AsyncioConversation):
        conversation.status = ConversationStatus.READY
        await conversation.trigger_event(
            ConversationStatusUpdate(conversation.id, conversation.status)
        )

    async def destroy_conversation(
        self, conversation_id: UUID, grace_period: int = 10
    ) -> bool:
        conversation = self.conversations.pop(conversation_id)
        if not conversation or conversation.status in [
            ConversationStatus.DESTROYING,
            ConversationStatus.DESTROYED,
        ]:
            return False
        await wait_all(
            asyncio.create_task(listener.before_destroy_conversation(conversation))
            for listener in self.listeners.values()
        )
        await conversation.trigger_event(
            ConversationStatusUpdate(conversation.id, ConversationStatus.DESTROYING)
        )
        await conversation.destroy(grace_period)
        return True

    async def shutdown(self, grace_period: int = 10):
        _LOGGER.info("shutting_down")
        await wait_all(
            (
                conversation.destroy(grace_period)
                for conversation in self.conversations.values()
            ),
            timeout=grace_period,
        )
