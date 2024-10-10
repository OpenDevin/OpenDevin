import asyncio
import time
from typing import Optional

from fastapi import WebSocket

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events.stream import session_exists
from openhands.runtime.utils.shutdown_listener import should_continue
from openhands.server.session.conversation import Conversation
from openhands.server.session.session import Session
from openhands.storage.files import FileStore


class SessionManager:
    _sessions: dict[str, Session] = {}
    cleanup_interval: int = 300
    session_timeout: int = 600
    _session_cleanup_task: Optional[asyncio.Task] = None

    def __init__(self, config: AppConfig, file_store: FileStore):
        self.config = config
        self.file_store = file_store

    async def __aenter__(self):
        if not self._session_cleanup_task:
            self._session_cleanup_task = asyncio.create_task(self._cleanup_sessions())
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            self._session_cleanup_task = None

    def add_or_restart_session(self, sid: str, ws_conn: WebSocket) -> Session:
        if sid in self._sessions:
            asyncio.create_task(self._sessions[sid].close())
        self._sessions[sid] = Session(
            sid=sid, file_store=self.file_store, ws=ws_conn, config=self.config
        )
        return self._sessions[sid]

    def get_session(self, sid: str) -> Session | None:
        if sid not in self._sessions:
            return None
        return self._sessions.get(sid)

    def attach_to_conversation(self, sid: str) -> Conversation | None:
        if not session_exists(sid, self.file_store):
            return None
        return Conversation(sid, file_store=self.file_store, config=self.config)

    async def send(self, sid: str, data: dict[str, object]) -> bool:
        """Sends data to the client."""
        session = self.get_session(sid)
        if session is None:
            logger.error(f'*** No session found for {sid}, skipping message ***')
            return False
        return await session.send(data)

    async def send_error(self, sid: str, message: str) -> bool:
        """Sends an error message to the client."""
        return await self.send(sid, {'error': True, 'message': message})

    async def send_message(self, sid: str, message: str) -> bool:
        """Sends a message to the client."""
        return await self.send(sid, {'message': message})

    async def _cleanup_sessions(self):
        while should_continue():
            current_time = time.time()
            session_ids_to_remove = []
            for sid, session in list(self._sessions.items()):
                # if session inactive for a long time, remove it
                if (
                    not session.is_alive
                    and current_time - session.last_active_ts > self.session_timeout
                ):
                    session_ids_to_remove.append(sid)

            for sid in session_ids_to_remove:
                to_del_session: Session | None = self._sessions.pop(sid, None)
                if to_del_session is not None:
                    await to_del_session.close()
                    logger.info(
                        f'Session {sid} and related resource have been removed due to inactivity.'
                    )

            await asyncio.sleep(self.cleanup_interval)
