import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import UUID, uuid4
import websockets
from websockets.protocol import State

from oh.conversation.listener.conversation_listener_abc import ConversationListenerABC
from oh.fastapi.dynamic_types import DynamicTypes
from oh.util.async_util import wait_all


@dataclass
class WebsocketAnnouncementClient:
    """
    Client which attaches to the firehose websocket of an OpenHands server
    and reads all events
    """

    url: str
    listeners: Dict[UUID, ConversationListenerABC] = field(default_factory=dict)
    dynamic_types: DynamicTypes = field(default_factory=DynamicTypes)
    websocket: Optional[websockets.WebSocketClientProtocol] = None
    running: bool = False
    retries: int = 5
    retry_interval: int = 1

    async def add_listener(self, listener: ConversationListenerABC):
        if not self.listeners:
            asyncio.create_task(self._start())
        listener_id = uuid4()
        self.listeners[listener_id] = listener
        return listener_id

    async def remove_listener(self, listener_id: UUID) -> bool:
        result = self.listeners.pop(listener_id) is not None
        if not self.listeners:
            await self.close()
        return result

    async def _start(self):
        self.running = True
        retries = self.retries
        try:
            while self.running and retries > 0:
                self.websocket = await websockets.connect(self.url)
                try:
                    await self.read_from_websocket()
                except websockets.ConnectionClosed:
                    if not retries:
                        raise
                    retries -= 1
                    if self.running:
                        await asyncio.sleep(self.retry_interval)
        finally:
            if self.websocket and self.websocket.state == State.OPEN:
                await self.websocket.close()

    async def read_from_websocket(self):
        event_info_class = self.dynamic_types.get_event_info_class()
        while self.running:
            message = await self.websocket.recv()
            event = event_info_class.model_validate_json(message)
            await wait_all(
                listener.on_event(event) for listener in self.listeners.values()
            )

    async def close(self):
        self.running = False
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
