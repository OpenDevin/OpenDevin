import asyncio
from typing import Optional

from agenthub.codeact_agent.codeact_agent import CodeActAgent
from opendevin.controller import AgentController
from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.config import config
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import ConfigType
from opendevin.events.stream import EventStream
from opendevin.llm.llm import LLM
from opendevin.runtime import DockerSSHBox, get_runtime_cls
from opendevin.runtime.runtime import Runtime


class AgentSession:
    """Represents a session with an agent.

    Attributes:
        sid: The session ID.
        event_stream: The event stream associated with the session.
        controller: The AgentController instance for controlling the agent.
        runtime: The runtime environment for the session.
        _closed: A flag indicating whether the session is closed.
        controller: The AgentController instance for controlling the agent.
    """

    sid: str
    event_stream: EventStream
    controller: Optional[AgentController] = None
    runtime: Optional[Runtime] = None
    _closed: bool = False

    def __init__(self, sid):
        """Initializes a new instance of the Session class."""
        self.sid = sid
        self.event_stream = EventStream(sid)

    async def start(self, start_event: dict):
        """Starts the agent session.

        Args:
            start_event: The start event data (optional).
        """
        if self.controller or self.runtime:
            raise RuntimeError(
                'Session already started. You need to close this session and start a new one.'
            )
        await self._create_runtime()
        await self._create_controller(start_event)

    async def close(self):
        if self._closed:
            return
        if self.controller is not None:
            end_state = self.controller.get_state()
            end_state.save_to_session(self.sid)
            await self.controller.close()
        if self.runtime is not None:
            if asyncio.iscoroutinefunction(self.runtime.close):
                await self.runtime.close()
            else:
                self.runtime.close()
        self._closed = True

    async def _create_runtime(self):
        if self.runtime is not None:
            raise RuntimeError('Runtime already created')
        try:
            runtime_cls = get_runtime_cls(config.runtime)
            self.runtime = runtime_cls(self.event_stream, self.sid)
            await self.runtime.initialize()
        except Exception as e:
            logger.error(f'Error initializing runtime: {e}')
            raise RuntimeError(f'Failed to initialize runtime: {e}') from e
        logger.info(f'Using runtime: {config.runtime}')

    async def _create_controller(self, start_event: dict):
        """Creates an AgentController instance.

        Args:
            start_event: The start event data.
        """
        if self.controller is not None:
            raise RuntimeError('Controller already created')
        if self.runtime is None:
            raise RuntimeError(
                'Runtime must be initialized before the agent controller'
            )
        args = {
            key: value
            for key, value in start_event.get('args', {}).items()
            if value != ''
        }  # remove empty values, prevent FE from sending empty strings
        agent_cls = args.get(ConfigType.AGENT, config.default_agent)
        llm_config = config.get_llm_config_from_agent(agent_cls)
        model = args.get(ConfigType.LLM_MODEL, llm_config.model)
        api_key = args.get(ConfigType.LLM_API_KEY, llm_config.api_key)
        api_base = llm_config.base_url
        confirmation_mode = args.get(
            ConfigType.CONFIRMATION_MODE, config.confirmation_mode
        )
        max_iterations = args.get(ConfigType.MAX_ITERATIONS, config.max_iterations)

        logger.info(f'Creating agent {agent_cls} using LLM {model}')
        llm = LLM(model=model, api_key=api_key, base_url=api_base)
        agent = Agent.get_cls(agent_cls)(llm)
        if isinstance(agent, CodeActAgent):
            if not self.runtime or not isinstance(self.runtime.sandbox, DockerSSHBox):
                if self.runtime:
                    logger.warning(f'Runtime: {self.runtime.__class__.__name__}')
                    logger.warning(
                        f'Sandbox: {self.runtime.sandbox.__class__.__name__}'
                    )
                logger.warning(
                    'CodeActAgent requires DockerSSHBox as sandbox! Using other sandbox that are not stateful'
                    ' LocalBox will not work properly.'
                )
        await self.runtime.init_sandbox_plugins(agent.sandbox_plugins)
        self.runtime.init_runtime_tools(agent.runtime_tools)

        self.controller = AgentController(
            sid=self.sid,
            event_stream=self.event_stream,
            agent=agent,
            max_iterations=int(max_iterations),
            confirmation_mode=confirmation_mode,
        )
        try:
            agent_state = State.restore_from_session(self.sid)
            llm.stop_requested_callback = self.controller.is_stopped
            self.controller.set_initial_state(agent_state)
            logger.info(f'Restored agent state from session, sid: {self.sid}')
        except Exception as e:
            print('Error restoring state', e)
