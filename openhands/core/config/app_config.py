import os
import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from typing import ClassVar

from openhands.core import logger
from openhands.core.config.agent_config import AgentConfig
from openhands.core.config.config_utils import (
    OH_DEFAULT_AGENT,
    OH_MAX_ITERATIONS,
    UndefinedString,
    get_field_info,
)
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.sandbox_config import SandboxConfig
from openhands.core.config.security_config import SecurityConfig


@dataclass
class AppConfig:
    """Configuration for the app.

    Attributes:
        llms: A dictionary of name -> LLM configuration. Default config is under 'llm' key.
        agents: A dictionary of name -> Agent configuration. Default config is under 'agent' key.
        default_agent: The name of the default agent to use.
        sandbox: The sandbox configuration.
        runtime: The runtime environment.
        file_store: The file store to use.
        file_store_path: The path to the file store.
        workspace_base: The base path for the workspace. Defaults to ./workspace as an absolute path.
        workspace_mount_path: The path to mount the workspace. This is set to the workspace base by default.
        workspace_mount_path_in_sandbox: The path to mount the workspace in the sandbox. Defaults to /workspace.
        workspace_mount_rewrite: The path to rewrite the workspace mount path to.
        cache_dir: The path to the cache directory. Defaults to /tmp/cache.
        run_as_openhands: Whether to run as openhands.
        max_iterations: The maximum number of iterations.
        max_budget_per_task: The maximum budget allowed per task, beyond which the agent will stop.
        e2b_api_key: The E2B API key.
        disable_color: Whether to disable color. For terminals that don't support color.
        debug: Whether to enable debugging.
        enable_cli_session: Whether to enable saving and restoring the session when run from CLI.
        file_uploads_max_file_size_mb: Maximum file size for uploads in megabytes. 0 means no limit.
        file_uploads_restrict_file_types: Whether to restrict file types for file uploads. Defaults to False.
        file_uploads_allowed_extensions: List of allowed file extensions for uploads. ['.*'] means all extensions are allowed.
    """

    llms: dict[str, LLMConfig] = field(default_factory=dict)
    agents: dict = field(default_factory=dict)
    default_agent: str = OH_DEFAULT_AGENT
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    runtime: str = 'eventstream'
    file_store: str = 'memory'
    file_store_path: str = '/tmp/file_store'
    # TODO: clean up workspace path after the removal of ServerRuntime
    workspace_base: str = os.path.join(os.getcwd(), 'workspace')
    workspace_mount_path: str | None = (
        UndefinedString.UNDEFINED  # this path should always be set when config is fully loaded
    )  # when set to None, do not mount the workspace
    workspace_mount_path_in_sandbox: str = '/workspace'
    workspace_mount_rewrite: str | None = None
    cache_dir: str = '/tmp/cache'
    run_as_openhands: bool = True
    max_iterations: int = OH_MAX_ITERATIONS
    max_budget_per_task: float | None = None
    e2b_api_key: str = ''
    disable_color: bool = False
    jwt_secret: str = uuid.uuid4().hex
    debug: bool = False
    enable_cli_session: bool = False
    runloop_api_key: str | None = None
    file_uploads_max_file_size_mb: int = 0
    file_uploads_restrict_file_types: bool = False
    file_uploads_allowed_extensions: list[str] = field(default_factory=lambda: ['.*'])

    defaults_dict: ClassVar[dict] = {}

    def get_llm_config(self, name='llm') -> LLMConfig:
        """Llm is the name for default config (for backward compatibility prior to 0.8)"""
        if name in self.llms:
            return self.llms[name]
        if name is not None and name != 'llm':
            logger.openhands_logger.warning(
                f'llm config group {name} not found, using default config'
            )
        if 'llm' not in self.llms:
            self.llms['llm'] = LLMConfig()
        return self.llms['llm']

    def set_llm_config(self, value: LLMConfig, name='llm'):
        self.llms[name] = value

    def get_agent_config(self, name='agent') -> AgentConfig:
        """Agent is the name for default config (for backward compability prior to 0.8)"""
        if name in self.agents:
            return self.agents[name]
        if 'agent' not in self.agents:
            self.agents['agent'] = AgentConfig()
        return self.agents['agent']

    def set_agent_config(self, value: AgentConfig, name='agent'):
        self.agents[name] = value

    def get_agent_to_llm_config_map(self) -> dict[str, LLMConfig]:
        """Get a map of agent names to llm configs."""
        return {name: self.get_llm_config_from_agent(name) for name in self.agents}

    def get_llm_config_from_agent(self, name='agent') -> LLMConfig:
        agent_config: AgentConfig = self.get_agent_config(name)
        llm_config_name = agent_config.llm_config
        return self.get_llm_config(llm_config_name)

    def get_agent_configs(self) -> dict[str, AgentConfig]:
        return self.agents

    def __post_init__(self):
        """Post-initialization hook, called when the instance is created with only default values."""
        AppConfig.defaults_dict = self.defaults_to_dict()

    def defaults_to_dict(self) -> dict:
        """Serialize fields to a dict for the frontend, including type hints, defaults, and whether it's optional."""
        result = {}
        for f in fields(self):
            field_value = getattr(self, f.name)

            # dataclasses compute their defaults themselves
            if is_dataclass(type(field_value)):
                result[f.name] = field_value.defaults_to_dict()
            else:
                result[f.name] = get_field_info(f)
        return result

    def __str__(self):
        attr_str = []
        for f in fields(self):
            attr_name = f.name
            attr_value = getattr(self, f.name)

            if attr_name in [
                'e2b_api_key',
                'github_token',
                'jwt_secret',
            ]:
                attr_value = '******' if attr_value else None

            attr_str.append(f'{attr_name}={repr(attr_value)}')

        return f"AppConfig({', '.join(attr_str)}"

    def __repr__(self):
        return self.__str__()
