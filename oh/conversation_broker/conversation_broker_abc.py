from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from oh.agent.agent_config import AgentConfig
from oh.conversation_broker.conversation_broker_listener_abc import (
    ConversationBrokerListenerABC,
)
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_filter import ConversationFilter
from oh.storage.page import Page


class ConversationBrokerABC(ABC):
    """
    Conversation broker. Responsible for coordinating access to conversations and conversation lifecycle management.
    Typically inactive conversations are deleted after some grace period.
    """

    @abstractmethod
    async def add_listener(self, listener: ConversationBrokerListenerABC) -> UUID:
        """Add a listener for conversations"""

    @abstractmethod
    async def remove_listener(self, listener_id: UUID) -> bool:
        """Remove a listener for conversations"""

    @abstractmethod
    async def get_conversation(
        self, conversation_id: UUID
    ) -> Optional[ConversationABC]:
        """Given an id, get conversation info. Return None if the conversation could not be found."""

    @abstractmethod
    async def search_conversations(
        self, filter: Optional[ConversationFilter] = None, page_id: Optional[str] = None
    ) -> Page[ConversationABC]:
        """Get a page of conversation info"""

    @abstractmethod
    async def count_conversations(self, filter: Optional[ConversationFilter]) -> int:
        """Get the number of conversations"""

    @abstractmethod
    async def create_conversation(self, agent_config: AgentConfig) -> ConversationABC:
        """Begin the process of creating a conversation. Once the conversation is ready, it will fire a READY event"""

    @abstractmethod
    async def destroy_conversation(
        self, conversation_id: UUID, grace_period: int = 10
    ) -> bool:
        """
        Begin the process of destroying a conversation. An attempt will be made to gracefully
        terminate any running commands within the conversation.
        """

    @abstractmethod
    async def shutdown(self, grace_period: int = 10):
        """Called when server is shutting down"""
