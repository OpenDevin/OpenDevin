import asyncio
from typing import Callable, Optional

from openhands.controller import AgentController
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, AppConfig, LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events.action.agent import ChangeAgentStateAction
from openhands.events.event import EventSource
from openhands.events.stream import EventStream
from openhands.runtime import get_runtime_cls
from openhands.runtime.runtime import Runtime
from openhands.security import SecurityAnalyzer, options
from openhands.storage.files import FileStore


class AgentSession:
    """Represents a session with an Agent

    Attributes:
        controller: The AgentController instance for controlling the agent.
    """

    sid: str
    event_stream: EventStream
    file_store: FileStore
    controller: AgentController | None = None
    runtime: Runtime | None = None
    security_analyzer: SecurityAnalyzer | None = None
    _closed: bool = False
    loop: asyncio.AbstractEventLoop | None = None

    def __init__(self, sid: str, file_store: FileStore):
        """Initializes a new instance of the Session class

        Parameters:
        - sid: The session ID
        - file_store: Instance of the FileStore
        """

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
        agent_configs: dict[str, AgentConfig] | None = None,
        status_message_callback: Optional[Callable] = None,
    ):
        """Starts the Agent session
        Parameters:
        - runtime_name: The name of the runtime associated with the session
        - config:
        - agent:
        - max_iterations:
        - max_budget_per_task:
        - agent_to_llm_config:
        - agent_configs:
        """
        if self.controller or self.runtime:
            raise RuntimeError(
                'Session already started. You need to close this session and start a new one.'
            )

        asyncio.get_event_loop().run_in_executor(
            None,
            self._start_thread,
            runtime_name,
            config,
            agent,
            max_iterations,
            max_budget_per_task,
            agent_to_llm_config,
            agent_configs,
            status_message_callback,
        )

    def _start_thread(self, *args):
        try:
            asyncio.run(self._start(*args), debug=True)
        except RuntimeError:
            logger.info('Session Finished')

    async def _start(
        self,
        runtime_name: str,
        config: AppConfig,
        agent: Agent,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
        status_message_callback: Optional[Callable] = None,
    ):
        self.loop = asyncio.get_running_loop()
        self._create_security_analyzer(config.security.security_analyzer)
        self._create_runtime(runtime_name, config, agent, status_message_callback)
        self._create_controller(
            agent,
            config.security.confirmation_mode,
            max_iterations,
            max_budget_per_task=max_budget_per_task,
            agent_to_llm_config=agent_to_llm_config,
            agent_configs=agent_configs,
        )
        self.event_stream.add_event(
            ChangeAgentStateAction(AgentState.INIT), EventSource.USER
        )
        if self.controller:
            self.controller.agent_task = self.controller.start_step_loop()
            await self.controller.agent_task  # type: ignore

    async def close(self):
        """Closes the Agent session"""

        if self._closed:
            return
        if self.controller is not None:
            end_state = self.controller.get_state()
            end_state.save_to_session(self.sid, self.file_store)
            await self.controller.close()
        if self.runtime is not None:
            self.runtime.close()
        if self.security_analyzer is not None:
            await self.security_analyzer.close()

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

        self._closed = True

    def _create_security_analyzer(self, security_analyzer: str | None):
        """Creates a SecurityAnalyzer instance that will be used to analyze the agent actions

        Parameters:
        - security_analyzer: The name of the security analyzer to use
        """

        if security_analyzer:
            logger.debug(f'Using security analyzer: {security_analyzer}')
            self.security_analyzer = options.SecurityAnalyzers.get(
                security_analyzer, SecurityAnalyzer
            )(self.event_stream)

    def _create_runtime(
        self,
        runtime_name: str,
        config: AppConfig,
        agent: Agent,
        status_message_callback: Optional[Callable] = None,
    ):
        """Creates a runtime instance

        Parameters:
        - runtime_name: The name of the runtime associated with the session
        - config:
        - agent:
        """

        if self.runtime is not None:
            raise RuntimeError('Runtime already created')

        logger.info(f'Initializing runtime `{runtime_name}` now...')
        runtime_cls = get_runtime_cls(runtime_name)

        try:
            self.runtime = runtime_cls(
                config=config,
                event_stream=self.event_stream,
                sid=self.sid,
                plugins=agent.sandbox_plugins,
                status_message_callback=status_message_callback,
            )
        except Exception as e:
            logger.error(f'Runtime initialization failed: {e}', exc_info=True)
            raise

        if self.runtime is not None:
            logger.debug(
                f'Runtime initialized with plugins: {[plugin.name for plugin in self.runtime.plugins]}'
            )
        else:
            logger.warning('Runtime initialization failed')

    def _create_controller(
        self,
        agent: Agent,
        confirmation_mode: bool,
        max_iterations: int,
        max_budget_per_task: float | None = None,
        agent_to_llm_config: dict[str, LLMConfig] | None = None,
        agent_configs: dict[str, AgentConfig] | None = None,
    ):
        """Creates an AgentController instance

        Parameters:
        - agent:
        - confirmation_mode: Whether to use confirmation mode
        - max_iterations:
        - max_budget_per_task:
        - agent_to_llm_config:
        - agent_configs:
        """

        if self.controller is not None:
            raise RuntimeError('Controller already created')
        if self.runtime is None:
            raise RuntimeError(
                'Runtime must be initialized before the agent controller'
            )

        msg = (
            '\n--------------------------------- OpenHands Configuration ---------------------------------\n'
            f'LLM: {agent.llm.config.model}\n'
            f'Base URL: {agent.llm.config.base_url}\n'
        )
        if agent.llm.config.draft_editor:
            msg += (
                f'Draft editor LLM (for file editing): {agent.llm.config.draft_editor.model}\n'
                f'Draft editor LLM (for file editing) Base URL: {agent.llm.config.draft_editor.base_url}\n'
            )
        msg += (
            f'Agent: {agent.name}\n'
            f'Runtime: {self.runtime.__class__.__name__}\n'
            f'Plugins: {agent.sandbox_plugins}\n'
            '-------------------------------------------------------------------------------------------'
        )
        logger.info(msg)

        self.controller = AgentController(
            sid=self.sid,
            event_stream=self.event_stream,
            agent=agent,
            max_iterations=int(max_iterations),
            max_budget_per_task=max_budget_per_task,
            agent_to_llm_config=agent_to_llm_config,
            agent_configs=agent_configs,
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
            logger.info(f'State could not be restored: {e}')
        logger.info('Agent controller initialized.')
