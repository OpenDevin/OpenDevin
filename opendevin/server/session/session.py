import asyncio
import time

from fastapi import WebSocket, WebSocketDisconnect

from opendevin.controller.agent import Agent
from opendevin.core.config import AppConfig
from opendevin.core.const.guide_url import TROUBLESHOOTING_URL
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import AgentState
from opendevin.core.schema.action import ActionType
from opendevin.core.schema.config import ConfigType
from opendevin.events.action import ChangeAgentStateAction, MessageAction, NullAction
from opendevin.events.event import Event, EventSource
from opendevin.events.observation import (
    AgentStateChangedObservation,
    CmdOutputObservation,
    NullObservation,
)
from opendevin.events.serialization import event_from_dict, event_to_dict
from opendevin.events.stream import EventStreamSubscriber
from opendevin.llm.llm import LLM
from opendevin.storage.files import FileStore

from .agent import AgentSession

DEL_DELT_SEC = 60 * 60 * 5


class Session:
    sid: str
    websocket: WebSocket | None
    last_active_ts: int = 0
    is_alive: bool = True
    agent_session: AgentSession

    def __init__(
        self, sid: str, ws: WebSocket | None, config: AppConfig, file_store: FileStore
    ):
        self.sid = sid
        self.websocket = ws
        self.last_active_ts = int(time.time())
        self.agent_session = AgentSession(sid, file_store)
        self.agent_session.event_stream.subscribe(
            EventStreamSubscriber.SERVER, self.on_event
        )
        self.config = config

    async def close(self):
        self.is_alive = False
        await self.agent_session.close()

    async def loop_recv(self):
        try:
            if self.websocket is None:
                return
            while True:
                try:
                    data = await self.websocket.receive_json()
                except ValueError:
                    await self.send_error('Invalid JSON')
                    continue
                await self.dispatch(data)
        except WebSocketDisconnect:
            await self.close()
            logger.info('WebSocket disconnected, sid: %s', self.sid)
        except RuntimeError as e:
            await self.close()
            logger.exception('Error in loop_recv: %s', e)

    async def _initialize_agent(self, data: dict):
        self.agent_session.event_stream.add_event(
            ChangeAgentStateAction(AgentState.LOADING), EventSource.USER
        )
        self.agent_session.event_stream.add_event(
            AgentStateChangedObservation('', AgentState.LOADING), EventSource.AGENT
        )
        # Extract the agent-relevant arguments from the request
        args = {
            key: value for key, value in data.get('args', {}).items() if value != ''
        }
        agent_cls = args.get(ConfigType.AGENT, self.config.default_agent)
        self.config.security.confirmation_mode = args.get(
            ConfigType.CONFIRMATION_MODE, self.config.security.confirmation_mode
        )
        self.config.security.security_analyzer = data.get('args', {}).get(
            ConfigType.SECURITY_ANALYZER, self.config.security.security_analyzer
        )
        max_iterations = args.get(ConfigType.MAX_ITERATIONS, self.config.max_iterations)
        # override default LLM config
        default_llm_config = self.config.get_llm_config()
        default_llm_config.model = args.get(
            ConfigType.LLM_MODEL, default_llm_config.model
        )
        default_llm_config.api_key = args.get(
            ConfigType.LLM_API_KEY, default_llm_config.api_key
        )
        default_llm_config.base_url = args.get(
            ConfigType.LLM_BASE_URL, default_llm_config.base_url
        )

        # TODO: override other LLM config & agent config groups (#2075)

        llm = LLM(config=self.config.get_llm_config_from_agent(agent_cls))
        agent = Agent.get_cls(agent_cls)(llm)

        # Create the agent session
        try:
            await self.agent_session.start(
                runtime_name=self.config.runtime,
                config=self.config,
                agent=agent,
                max_iterations=max_iterations,
                max_budget_per_task=self.config.max_budget_per_task,
                agent_to_llm_config=self.config.get_agent_to_llm_config_map(),
            )
        except Exception as e:
            logger.exception(f'Error creating controller: {e}')
            await self.send_error(
                f'Error creating controller. Please check Docker is running and visit `{TROUBLESHOOTING_URL}` for more debugging information..'
            )
            return
        self.agent_session.event_stream.add_event(
            ChangeAgentStateAction(AgentState.INIT), EventSource.USER
        )

    async def on_event(self, event: Event):
        """Callback function for agent events.

        Args:
            event: The agent event (Observation or Action).
        """
        if isinstance(event, NullAction):
            return
        if isinstance(event, NullObservation):
            return
        if event.source == EventSource.AGENT:
            logger.info('Server event')
            await self.send(event_to_dict(event))
        elif event.source == EventSource.USER and isinstance(
            event, CmdOutputObservation
        ):
            await self.send(event_to_dict(event))

    async def dispatch(self, data: dict):
        action = data.get('action', '')
        if action == ActionType.INIT:
            await self._initialize_agent(data)
            return
        event = event_from_dict(data.copy())
        # This checks if the model supports images
        if isinstance(event, MessageAction) and event.images_urls:
            controller = self.agent_session.controller
            if controller and not controller.agent.llm.supports_vision():
                await self.send_error(
                    'Model does not support image upload, change to a different model or try without an image.'
                )
                return
        self.agent_session.event_stream.add_event(event, EventSource.USER)

    async def send(self, data: dict[str, object]) -> bool:
        try:
            if self.websocket is None or not self.is_alive:
                return False
            await self.websocket.send_json(data)
            await asyncio.sleep(0.001)  # This flushes the data to the client
            self.last_active_ts = int(time.time())
            return True
        except WebSocketDisconnect:
            self.is_alive = False
            return False

    async def send_error(self, message: str) -> bool:
        """Sends an error message to the client."""
        return await self.send({'error': True, 'message': message})

    async def send_message(self, message: str) -> bool:
        """Sends a message to the client."""
        return await self.send({'message': message})

    def update_connection(self, ws: WebSocket):
        self.websocket = ws
        self.is_alive = True
        self.last_active_ts = int(time.time())

    def load_from_data(self, data: dict) -> bool:
        self.last_active_ts = data.get('last_active_ts', 0)
        if self.last_active_ts < int(time.time()) - DEL_DELT_SEC:
            return False
        self.is_alive = data.get('is_alive', False)
        return True
