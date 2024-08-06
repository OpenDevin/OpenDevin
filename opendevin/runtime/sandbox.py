import asyncio
import copy
import json
import os
from abc import ABC, abstractmethod

from opendevin.core.config import SandboxConfig
from opendevin.core.schema import CancellableStream
from opendevin.runtime.plugins.mixin import PluginMixin
from opendevin.runtime.utils.async_utils import async_to_sync


class Sandbox(ABC, PluginMixin):
    _env: dict[str, str] = {}
    is_initial_session: bool = True

    def __init__(self, config: SandboxConfig):
        self.config = copy.deepcopy(config)
        self.initialize_plugins: bool = config.initialize_plugins
        self._initialization_complete = asyncio.Event()

    async def initialize(self):
        if not self._initialization_complete.is_set():
            async with asyncio.Lock():
                if not self._initialization_complete.is_set():
                    await self._setup_environment()
                    self._initialization_complete.set()
        await self._initialization_complete.wait()

    async def ainit(self):
        # not used by all runtimes
        pass

    async def _setup_environment(self):
        if isinstance(self.config.env, dict):
            self._env = self.config.env.copy()
        for key, value in self._env.items():
            if key:
                await self.add_to_env_async(key, value)
        if self.config.enable_auto_lint:
            self.add_to_env('ENABLE_AUTO_LINT', 'true')

        try:
            for key, value in os.environ.items():
                if key.startswith('SANDBOX_ENV_'):
                    sandbox_key = key.removeprefix('SANDBOX_ENV_')
                    if sandbox_key:
                        await self.add_to_env_async(sandbox_key, value)
        except Exception:
            pass

        if self.config.enable_auto_lint:
            await self.add_to_env_async('ENABLE_AUTO_LINT', 'true')

    @async_to_sync
    def add_to_env(self, key: str, value: str):
        return self.add_to_env_async(key, value)

    async def add_to_env_async(self, key: str, value: str):
        self._env[key] = value
        # Note: json.dumps gives us nice escaping for free
        await self.execute_async(f'export {key}={json.dumps(value)}')

    @abstractmethod
    @async_to_sync
    def execute(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        pass

    @abstractmethod
    async def execute_async(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    @async_to_sync
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        pass

    @abstractmethod
    async def copy_to_async(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ):
        pass

    @abstractmethod
    async def get_working_directory(self) -> str:
        pass
