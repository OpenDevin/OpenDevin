from asyncio import Condition
from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import UUID, uuid4
from oh.conversation.conversation_abc import ConversationABC
from oh.conversation.conversation_filter import ConversationFilter
from oh.conversation_broker.conversation_broker_abc import ConversationBrokerABC
from oh.conversation_broker.conversation_broker_listener_abc import (
    ConversationBrokerListenerABC,
)
from oh.docker.image_source.image_source_abc import ImageSourceABC
from oh.storage.page import Page


@dataclass
class DockerConversationBroker(ConversationBrokerABC):
    image_source: ImageSourceABC
    listeners: Dict[UUID, ConversationBrokerListenerABC] = field(default_factory=dict)
    _sandbox_image_condition: Optional[Condition] = None
    _sandbox_image: Optional[str] = None

    async def _wait_for_sandbox_image(self) -> str:
        if self._sandbox_image_condition:
            return self._sandbox_image_condition.wait_for(lambda: self._sandbox_image)
        self._sandbox_image_condition = Condition()
        sandbox_image = await self.image_source.get_sandbox_image()
        self._sandbox_image_condition.notify_all()
        return sandbox_image

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
        # Create a docker container then return a remote container...
        raise ValueError("not_implemented")

    async def search_conversations(
        self, filter: Optional[ConversationFilter] = None, page_id: Optional[str] = None
    ) -> Page[ConversationABC]:
        raise ValueError("not_implemented")

    async def count_conversations(self, filter: Optional[ConversationFilter]) -> int:
        raise ValueError("not_implemented")

    async def create_conversation(
        self,
    ) -> ConversationABC:
        # Make sure the image is available
        # Use the image to build a container. (TODO: Should we keep a container in reserve?)
        raise ValueError("not_implemented")

    async def destroy_conversation(
        self, conversation_id: UUID, grace_period: int = 10
    ) -> bool:
        raise ValueError("not_implemented")

    async def shutdown(self, grace_period: int = 10):
        raise ValueError("not_implemented")
