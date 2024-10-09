from __future__ import annotations
from abc import ABC, abstractmethod

from oh.announcement import announcement
from oh.command import oh_command


class ConversationListenerABC(ABC):

    @abstractmethod
    async def on_event(self, event: announcement.Announcement):
        """Callback for an event in a conversation"""
