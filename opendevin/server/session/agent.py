from typing import Optional

from opendevin.controller import AgentController
from opendevin.controller.agent import Agent
from opendevin.controller.state.state import State
from opendevin.core.config import AppConfig, LLMConfig
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.stream import EventStream
from opendevin.runtime import get_runtime_cls
from opendevin.runtime.runtime import Runtime
from opendevin.security import SecurityAnalyzer, options
from opendevin.storage.files import FileStore


class AgentSession:
    """Represents a session with an agent.

    Attributes:
        controller: The AgentController instance for controlling the agent.
    """

    sid: str
    event_stream: EventStream
    file_store: FileStore
    controller: Optional[AgentController] = None
    runtime: Optional[Runtime] = None
    security_analyzer: SecurityAnalyzer | None = None
    _closed: bool = False

    def __init__(self, sid: str, file_store: FileStore):
        """Initializes a new instance of the Session class."""
        self.sid = sid
        self.event_stream = EventStream(sid, file_store)
        self.file_store = file_store

    async def start(
        self,
        runtime_name: str,
        config: AppConfig,
        agent: Agent,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
    ):
        """Starts the agent session.

        Args:
            start_event: The start event data (optional).
        """
        if self.controller or self.runtime:
            raise Exception(
                'Session already started. You need to close this session and start a new one.'
            )
        await self._create_security_analyzer(config.security.security_analyzer)
        await self._create_runtime(runtime_name, config, agent)
        await self._create_controller(
            agent,
            config.security.confirmation_mode,
            max_iterations,
            max_budget_per_task=max_budget_per_task,
            agent_to_llm_config=agent_to_llm_config,
        )

    async def close(self):
        if self._closed:
            return
        if self.controller is not None:
            end_state = self.controller.get_state()
            end_state.save_to_session(self.sid, self.file_store)
            await self.controller.close()
        if self.runtime is not None:
            await self.runtime.close()
        if self.security_analyzer is not None:
            await self.security_analyzer.close()
        self._closed = True

    async def _create_security_analyzer(self, security_analyzer: str | None):
        """Creates a SecurityAnalyzer instance that will be used to analyze the agent actions."""
        logger.info(f'Using security analyzer: {security_analyzer}')
        if security_analyzer:
            self.security_analyzer = options.SecurityAnalyzers.get(
                security_analyzer, SecurityAnalyzer
            )(self.event_stream)

    async def _create_runtime(self, runtime_name: str, config: AppConfig, agent: Agent):
        """Creates a runtime instance."""
        if self.runtime is not None:
            raise Exception('Runtime already created')

        logger.info(f'Using runtime: {runtime_name}')
        runtime_cls = get_runtime_cls(runtime_name)
        self.runtime = runtime_cls(
            config=config,
            event_stream=self.event_stream,
            sid=self.sid,
            plugins=agent.sandbox_plugins,
        )
        await self.runtime.ainit()

    async def _create_controller(
        self,
        agent: Agent,
        confirmation_mode: bool,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
    ):
        """Creates an AgentController instance."""
        if self.controller is not None:
            raise Exception('Controller already created')
        if self.runtime is None:
            raise Exception('Runtime must be initialized before the agent controller')

        logger.info(f'Creating agent {agent.name} using LLM {agent.llm.config.model}')

        self.controller = AgentController(
            sid=self.sid,
            event_stream=self.event_stream,
            agent=agent,
            max_iterations=int(max_iterations),
            max_budget_per_task=max_budget_per_task,
            agent_to_llm_config=agent_to_llm_config,
            confirmation_mode=confirmation_mode,
            # AgentSession is designed to communicate with the frontend, so we don't want to
            # run the agent in headless mode.
            headless_mode=False,
        )
        try:
            agent_state = State.restore_from_session(self.sid, self.file_store)
            self.controller.set_initial_state(
                agent_state, max_iterations, confirmation_mode
            )
            logger.info(f'Restored agent state from session, sid: {self.sid}')
        except Exception as e:
            print('Error restoring state', e)
