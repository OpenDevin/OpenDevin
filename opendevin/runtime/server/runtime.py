import copy
from typing import Any, Optional

from opendevin.core.config import AppConfig
from opendevin.core.exceptions import BrowserInitException
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from opendevin.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    IPythonRunCellObservation,
    Observation,
)
from opendevin.events.stream import EventStream
from opendevin.runtime import (
    DockerSSHBox,
    E2BBox,
    LocalBox,
    Sandbox,
)
from opendevin.runtime.browser.browser_env import BrowserEnv
from opendevin.runtime.plugins import JupyterRequirement, PluginRequirement
from opendevin.runtime.runtime import Runtime
from opendevin.runtime.tools import RuntimeTool
from opendevin.storage.local import LocalFileStore

from ..browser import browse
from .files import read_file, write_file


class ServerRuntime(Runtime):
    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        sandbox: Sandbox | None = None,
    ):
        self.config = copy.deepcopy(config)
        super().__init__(config, event_stream, sid, plugins)
        self.file_store = LocalFileStore(config.workspace_base)
        if sandbox is None:
            self.sandbox = self.create_sandbox(sid, config.sandbox.box_type)
            self._is_external_sandbox = False
        else:
            self.sandbox = sandbox
            self._is_external_sandbox = True
        self.browser: BrowserEnv | None = None
        logger.info(f'ServerRuntime `{sid}` config:\n{self.config}')

    def create_sandbox(self, sid: str = 'default', box_type: str = 'ssh') -> Sandbox:
        if box_type == 'local':
            return LocalBox(
                config=self.config.sandbox, workspace_base=self.config.workspace_base
            )
        elif box_type == 'ssh':
            return DockerSSHBox(
                config=self.config.sandbox,
                persist_sandbox=self.config.persist_sandbox,
                workspace_mount_path=self.config.workspace_mount_path,
                sandbox_workspace_dir=self.config.workspace_mount_path_in_sandbox,
                cache_dir=self.config.cache_dir,
                run_as_devin=self.config.run_as_devin,
                ssh_hostname=self.config.ssh_hostname,
                ssh_password=self.config.ssh_password,
                ssh_port=self.config.ssh_port,
                sid=sid,
            )
        elif box_type == 'e2b':
            return E2BBox(
                config=self.config.sandbox,
                e2b_api_key=self.config.e2b_api_key,
            )
        else:
            raise ValueError(f'Invalid sandbox type: {box_type}')

    async def ainit(self, env_vars: dict[str, str] | None = None):
        # init sandbox plugins
        await self.sandbox.init_plugins_async(self.plugins)

        # MUST call super().ainit() to initialize both default env vars
        # AND the ones in env vars!
        await super().ainit(env_vars)

        await self.sandbox.ainit()

        # init sandbox plugins
        # if self.plugins and len(self.plugins) > 0:
        if any(isinstance(plugin, JupyterRequirement) for plugin in self.plugins):
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code=f'import os; os.chdir("{self.config.workspace_mount_path_in_sandbox}")'
                )
            )
            logger.info(
                f'Switch to working directory {self.config.workspace_mount_path_in_sandbox} in IPython. Output: {obs.content}'
            )

    async def close(self):
        if hasattr(self, '_is_external_sandbox') and not self._is_external_sandbox:
            await self.sandbox.close()
        if hasattr(self, 'browser') and self.browser is not None:
            self.browser.close()

    def init_runtime_tools(
        self,
        runtime_tools: list[RuntimeTool],
        runtime_tools_config: Optional[dict[RuntimeTool, Any]] = None,
    ) -> None:
        # if browser in runtime_tools, init it
        if RuntimeTool.BROWSER in runtime_tools:
            if runtime_tools_config is None:
                runtime_tools_config = {}
            browser_env_config = runtime_tools_config.get(RuntimeTool.BROWSER, {})
            try:
                self.browser = BrowserEnv(**browser_env_config)
            except BrowserInitException:
                logger.warn(
                    'Failed to start browser environment, web browsing functionality will not work'
                )

    async def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        self.sandbox.copy_to(host_src, sandbox_dest, recursive)

    async def run(self, action: CmdRunAction) -> Observation:
        return await self._run_command(action.command)

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        write_result = await self._run_command(
            f"cat > /tmp/opendevin_jupyter_temp.py <<'EOL'\n{action.code}\nEOL"
        )
        if isinstance(write_result, ErrorObservation):
            return write_result

        # run the code
        execute_result = await self._run_command(
            'cat /tmp/opendevin_jupyter_temp.py | execute_cli'
        )
        if isinstance(execute_result, ErrorObservation):
            return execute_result

        output = execute_result.content

        if 'pip install' in action.code:
            print(output)
            package_names = action.code.split(' ', 2)[-1]
            is_single_package = ' ' not in package_names

            if 'Successfully installed' in output:
                restart_kernel = 'import IPython\nIPython.Application.instance().kernel.do_shutdown(True)'
                if (
                    'Note: you may need to restart the kernel to use updated packages.'
                    in output
                ):
                    await self._run_command(
                        (
                            "cat > /tmp/opendevin_jupyter_temp.py <<'EOL'\n"
                            f'{restart_kernel}\n'
                            'EOL'
                        )
                    )
                    obs = await self._run_command(
                        'cat /tmp/opendevin_jupyter_temp.py | execute_cli'
                    )
                    output = '[Package installed successfully]'
                    if "{'status': 'ok', 'restart': True}" != obs.content.strip():
                        print(obs.content)
                        output += (
                            '\n[But failed to restart the kernel to load the package]'
                        )
                    else:
                        output += (
                            '\n[Kernel restarted successfully to load the package]'
                        )

                    # re-init the kernel after restart
                    if action.kernel_init_code:
                        await self._run_command(
                            (
                                f"cat > /tmp/opendevin_jupyter_init.py <<'EOL'\n"
                                f'{action.kernel_init_code}\n'
                                'EOL'
                            ),
                        )
                        await self._run_command(
                            'cat /tmp/opendevin_jupyter_init.py | execute_cli',
                        )
            elif (
                is_single_package
                and f'Requirement already satisfied: {package_names}' in output
            ):
                output = '[Package already installed]'
        return IPythonRunCellObservation(content=output, code=action.code)

    async def read(self, action: FileReadAction) -> Observation:
        # TODO: use self.file_store
        assert self.sandbox is not None
        working_dir = await self.sandbox.get_working_directory()
        return await read_file(
            action.path,
            working_dir,
            self.config.workspace_base,
            self.config.workspace_mount_path_in_sandbox,
            action.start,
            action.end,
        )

    async def write(self, action: FileWriteAction) -> Observation:
        # TODO: use self.file_store
        assert self.sandbox is not None
        working_dir = await self.sandbox.get_working_directory()
        return await write_file(
            action.path,
            working_dir,
            self.config.workspace_base,
            self.config.workspace_mount_path_in_sandbox,
            action.content,
            action.start,
            action.end,
        )

    async def browse(self, action: BrowseURLAction) -> Observation:
        return await browse(action, self.browser)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return await browse(action, self.browser)

    async def _run_command(self, command: str) -> Observation:
        assert self.sandbox is not None
        try:
            result = await self.sandbox.execute_async(command)
            if isinstance(result, tuple) and len(result) == 2:
                exit_code, output = result

            if 'pip install' in command:
                package_names = command.split(' ', 2)[-1]
                is_single_package = ' ' not in package_names
                print(output)
                if 'Successfully installed' in output:
                    output = '[Package installed successfully]'
                elif (
                    is_single_package
                    and f'Requirement already satisfied: {package_names}' in output
                ):
                    output = '[Package already installed]'
            return CmdOutputObservation(
                command_id=-1, content=str(output), command=command, exit_code=exit_code
            )
        except UnicodeDecodeError:
            return ErrorObservation('Command output could not be decoded as utf-8')
