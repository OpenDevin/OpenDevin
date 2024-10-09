import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID
from oh.conversation.conversation_abc import ConversationABC
from oh.announcement.detail.text_reply import TextReply
from oh.command.runnable.runnable_abc import RunnableABC
from oh.command.command_status import CommandStatus


@dataclass
class Ticker(RunnableABC):
    """
    Runnable that sends message events at regular intervals
    (Mostly for debugging purposes)
    """

    type: Literal["Ticker"] = "Ticker"
    cancellable: bool = True
    message: str = "The time is now {time}"
    interval: int = 1
    iterations: int = 100

    async def run(
        self,
        command_id: UUID,
        conversation: ConversationABC,
    ):
        iteration = 0
        while (
            await conversation.get_command(command_id)
        ).status == CommandStatus.RUNNING and iteration < self.iterations:
            await conversation.trigger_event(
                TextReply(self.message.format(time=str(datetime.now())))
            )
            iteration += 1
            await asyncio.sleep(self.interval)
