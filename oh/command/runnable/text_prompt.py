import asyncio
from dataclasses import dataclass
from typing import Literal
from uuid import UUID
from oh.agent.agent_abc import get_agent
from oh.conversation import conversation_abc
from oh.command.runnable.runnable_abc import RunnableABC


@dataclass
class TextPrompt(RunnableABC):
    """Pass a new prompt to the agent"""

    text: str
    type: Literal["TextPrompt"] = "TextPrompt"

    async def run(
        self,
        command_id: UUID,
        conversation: conversation_abc.ConversationABC,
    ):
        """Send the prompt to the agent in the background"""
        await get_agent(conversation.agent_config).prompt(self.text, conversation)
