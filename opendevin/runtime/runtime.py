import asyncio
import atexit
import copy
import json
import os
from abc import abstractmethod
from typing import Any, Optional

from opendevin.core.config import AppConfig, SandboxConfig
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events import EventSource, EventStream, EventStreamSubscriber
from opendevin.events.action import (
    Action,
    ActionConfirmationStatus,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from opendevin.events.event import Event
from opendevin.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    NullObservation,
    Observation,
    UserRejectObservation,
)
from opendevin.events.serialization.action import ACTION_TYPE_TO_CLASS
from opendevin.runtime.plugins import JupyterRequirement, PluginRequirement
from opendevin.runtime.tools import RuntimeTool
from opendevin.runtime.utils.async_utils import async_to_sync
from opendevin.storage import FileStore


def _default_env_vars(sandbox_config: SandboxConfig) -> dict[str, str]:
    ret = {}
    for key in os.environ:
        if key.startswith('SANDBOX_ENV_'):
            sandbox_key = key.removeprefix('SANDBOX_ENV_')
            ret[sandbox_key] = os.environ[key]
    if sandbox_config.enable_auto_lint:
        ret['ENABLE_AUTO_LINT'] = 'true'
    return ret


class Runtime:
    """The runtime is how the agent interacts with the external environment.
    This includes a bash sandbox, a browser, and filesystem interactions.

    sid is the session id, which is used to identify the current user session.
    """

    _instance = None
    _initialization_lock = asyncio.Lock()
    _initialized = False

    sid: str
    file_store: FileStore
    DEFAULT_ENV_VARS: dict[str, str]

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
    ):
        if not hasattr(self, 'initialized'):
            self.sid = sid
            self.event_stream = event_stream
            self.event_stream.subscribe(EventStreamSubscriber.RUNTIME, self.on_event)
            self.plugins = plugins if plugins is not None and len(plugins) > 0 else []

            self.config = copy.deepcopy(config)
            self.DEFAULT_ENV_VARS = _default_env_vars(config.sandbox)
            self.initialized = False
            atexit.register(self.close_sync)
            logger.info(f'Runtime `{sid}` config:\n{self.config}')

    async def ainit(self, env_vars: dict[str, str] | None = None) -> None:
        """
        Initialize the runtime (asynchronously).

        This method should be called after the runtime's constructor.
        """
        if self.DEFAULT_ENV_VARS:
            logger.debug(f'Adding default env vars: {self.DEFAULT_ENV_VARS}')
            await self.add_env_vars(self.DEFAULT_ENV_VARS)
        if env_vars is not None:
            logger.debug(f'Adding provided env vars: {env_vars}')
            await self.add_env_vars(env_vars)

    @async_to_sync
    def close(self):
        pass

    def close_sync(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, use asyncio.run()
            asyncio.run(self.close())
        else:
            # There is a running event loop, create a task
            if loop.is_running():
                loop.create_task(self.close())
            else:
                loop.run_until_complete(self.close())

    # ====================================================================
    # Methods we plan to deprecate when we move to new EventStreamRuntime
    # ====================================================================

    def init_runtime_tools(
        self,
        runtime_tools: list[RuntimeTool],
        runtime_tools_config: Optional[dict[RuntimeTool, Any]] = None,
    ) -> None:
        # TODO: deprecate this method when we move to the new EventStreamRuntime
        raise NotImplementedError('This method is not implemented in the base class.')

    # ====================================================================

    async def add_env_vars(self, env_vars: dict[str, str]) -> None:
        # Add env vars to the IPython shell (if Jupyter is used)
        if any(isinstance(plugin, JupyterRequirement) for plugin in self.plugins):
            code = 'import os\n'
            for key, value in env_vars.items():
                # Note: json.dumps gives us nice escaping for free
                code += f'os.environ["{key}"] = {json.dumps(value)}\n'
            code += '\n'
            obs = await self.run_ipython(IPythonRunCellAction(code))
            logger.info(f'Added env vars to IPython: code={code}, obs={obs}')

        # Add env vars to the Bash shell
        cmd = ''
        for key, value in env_vars.items():
            # Note: json.dumps gives us nice escaping for free
            cmd += f'export {key}={json.dumps(value)}; '
        if not cmd:
            return
        cmd = cmd.strip()
        if cmd == '':
            return
        logger.debug(f'Adding env var: {cmd}')
        obs = await self.run(CmdRunAction(cmd))
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(
                f'Failed to add env vars [{env_vars}] to environment: {obs.content}'
            )

    async def on_event(self, event: Event) -> None:
        if isinstance(event, Action):
            # set timeout to default if not set
            if event.timeout is None:
                event.timeout = self.config.sandbox.timeout
            assert event.timeout is not None
            observation = await self.run_action(event)
            observation._cause = event.id  # type: ignore[attr-defined]
            source = event.source if event.source else EventSource.AGENT
            await self.event_stream.add_event(observation, source)  # type: ignore[arg-type]

    async def run_action(self, action: Action) -> Observation:
        """Run an action and return the resulting observation.
        If the action is not runnable in any runtime, a NullObservation is returned.
        If the action is not supported by the current runtime, an ErrorObservation is returned.
        """
        if not action.runnable:
            return NullObservation('')
        if (
            hasattr(action, 'is_confirmed')
            and action.is_confirmed == ActionConfirmationStatus.AWAITING_CONFIRMATION
        ):
            return NullObservation('')
        action_type = action.action  # type: ignore[attr-defined]
        if action_type not in ACTION_TYPE_TO_CLASS:
            return ErrorObservation(f'Action {action_type} does not exist.')
        if not hasattr(self, action_type):
            return ErrorObservation(
                f'Action {action_type} is not supported in the current runtime.'
            )
        if (
            hasattr(action, 'is_confirmed')
            and action.is_confirmed == ActionConfirmationStatus.REJECTED
        ):
            return UserRejectObservation(
                'Action has been rejected by the user! Waiting for further user input.'
            )
        observation = await getattr(self, action_type)(action)
        return observation

    @abstractmethod
    async def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        raise NotImplementedError('This method is not implemented in the base class.')

    # ====================================================================
    # Implement these methods in the subclass
    # ====================================================================

    @abstractmethod
    async def run(self, action: CmdRunAction) -> Observation:
        pass

    @abstractmethod
    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        pass

    @abstractmethod
    async def read(self, action: FileReadAction) -> Observation:
        pass

    @abstractmethod
    async def write(self, action: FileWriteAction) -> Observation:
        pass

    @abstractmethod
    async def browse(self, action: BrowseURLAction) -> Observation:
        pass

    @abstractmethod
    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        pass
