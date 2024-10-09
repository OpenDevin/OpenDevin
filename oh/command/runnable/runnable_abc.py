from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID

from oh.conversation import conversation_abc


class RunnableABC(ABC):
    """
    Some runnable command. Runnables should endeavour to be good asyncio neighbors and not lock up
    the main run loop too much. They may also implement cancellability by checking that the
    status of the command is not CANCELLING in the service.
    """

    cancellable: bool = False

    @abstractmethod
    async def run(
        self,
        command_id: UUID,
        conversation: conversation_abc.ConversationABC,
    ):
        """
        Execute this command
        command_id: The id of this command
        conversation: Conversation in which this command is run
        """
