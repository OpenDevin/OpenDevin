from abc import ABC, abstractmethod

from oh.conversation.conversation_abc import ConversationABC


class ConversationBrokerListenerABC(ABC):

    @abstractmethod
    async def after_create_conversation(self, conversation: ConversationABC):
        """Callback after a conversation is created"""

    @abstractmethod
    async def before_destroy_conversation(self, conversation: ConversationABC):
        """Callback before a conversation is destroyed"""
