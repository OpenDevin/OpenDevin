import os
import time
from typing import Protocol

from pexpect.exceptions import EOF, TIMEOUT

from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import CancellableStream
from opendevin.runtime.plugins.requirement import PluginRequirement
from opendevin.runtime.utils.async_utils import async_to_sync


class SandboxProtocol(Protocol):
    # https://stackoverflow.com/questions/51930339/how-do-i-correctly-add-type-hints-to-mixin-classes

    @property
    def initialize_plugins(self) -> bool: ...

    def source_bashrc(self): ...

    async def source_bashrc_async(self): ...

    def init_plugins(self, requirements: list[PluginRequirement]): ...

    async def init_plugins_async(self, requirements: list[PluginRequirement]): ...

    def execute(
        self, cmd: str, stream: bool = False
    ) -> tuple[int, str | CancellableStream]: ...

    async def execute_async(
        self, cmd: str, stream: bool = False
    ) -> tuple[int, str | CancellableStream]: ...

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False): ...

    async def copy_to_async(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ): ...


def _handle_stream_output(output: CancellableStream, plugin_name: str):
    total_output = ''
    exit_code_value = None
    start_time = time.time()
    minor_timeout = 10
    timeout = 30
    last_output_time = start_time

    try:
        for line in output:
            line = line.rstrip()
            if line.strip():
                logger.info(f'>>> {line}')
                total_output += line + '\n'
            last_output_time = time.time()

            if line.endswith('[PEXPECT]$') or '[PEXPECT]$' in line:
                logger.debug('Detected [PEXPECT]$ prompt, ending stream.')
                break

            if time.time() - start_time > timeout:
                logger.warning(f'Execution timed out after {timeout} seconds.')
                break

            if time.time() - last_output_time > minor_timeout:
                logger.warning(f'No output for {minor_timeout} seconds, ending stream.')
                break

    except (EOF, TIMEOUT) as e:
        logger.warning(f'Stream interrupted: {e}')
    finally:
        try:
            exit_code_value = output.exit_code()
        except Exception as e:
            logger.error(f'Failed to get exit code: {e}')
            exit_code_value = -1
        output.close()

    if exit_code_value is not None and exit_code_value != 0:
        raise RuntimeError(
            f'Failed to initialize plugin {plugin_name} with exit code {exit_code_value} and output: {total_output.strip()}'
        )


class PluginMixin:
    """Mixin for Sandbox to support plugins."""

    # @async_to_sync
    def source_bashrc(self: SandboxProtocol):
        return self.source_bashrc_async()

    async def source_bashrc_async(self: SandboxProtocol):
        exit_code, output = await self.execute_async('source /opendevin/bash.bashrc')
        if exit_code == 0:
            logger.info('Sourced /opendevin/bash.bashrc')
        else:
            raise RuntimeError(
                f'Failed to source /opendevin/bash.bashrc! Exit code {exit_code} and output: {output}'
            )
        exit_code, output = await self.execute_async('source ~/.bashrc')
        if exit_code == 0:
            logger.info('Sourced ~/.bashrc')
        else:
            raise RuntimeError(
                f'Failed to source ~/.bashrc! Exit code {exit_code} and output: {output}'
            )

    @async_to_sync
    def init_plugins(self: SandboxProtocol, requirements: list[PluginRequirement]):
        return self.init_plugins_async(requirements)

    async def init_plugins_async(
        self: SandboxProtocol, requirements: list[PluginRequirement]
    ):
        """Load plugins into the sandbox."""
        if hasattr(self, 'plugin_initialized') and self.plugin_initialized:
            logger.info('Plugins already initialized, skipping.')
            return

        # Check if the sandbox is initialized
        if hasattr(self, 'initialized') and not self.initialized:
            logger.info('Sandbox not yet initialized. Initializing now...')
            if hasattr(self, 'ainit'):
                await self.ainit()
            else:
                logger.warning('Sandbox has no initialize method, skipping.')

        if self.initialize_plugins:
            logger.info('Initializing plugins in the sandbox.')

            # clean-up ~/.bashrc and touch ~/.bashrc. Do not use "&&"!
            exit_code, output = await self.execute_async('rm -f ~/.bashrc')
            if exit_code == 0:
                exit_code, output = await self.execute_async('touch ~/.bashrc')
            if exit_code != 0:
                logger.warning(
                    f'Failed to clean-up ~/.bashrc with exit code {exit_code} and output: {output}'
                )

            for index, requirement in enumerate(requirements, 1):
                logger.info(
                    f'Initializing plugin {index}/{len(requirements)}: {requirement.name}'
                )

                # source bashrc file when plugin loads
                logger.info(f'Sourcing for {requirement.name}.')
                await self.source_bashrc_async()
                logger.info(f'Sourcing done for {requirement.name}.')

                # copy over the files
                await self.copy_to_async(
                    requirement.host_src, requirement.sandbox_dest, recursive=True
                )
                logger.info(
                    f'Copied files from [{requirement.host_src}] to [{requirement.sandbox_dest}] inside sandbox.'
                )

                abs_path_to_bash_script = os.path.join(
                    requirement.sandbox_dest,
                    requirement.name,
                    requirement.bash_script_path,
                )
                logger.info(
                    f'Initializing plugin [{requirement.name}] by executing [{abs_path_to_bash_script}] in the sandbox.'
                )
                exit_code, output = await self.execute_async(
                    abs_path_to_bash_script, stream=False
                )
                if isinstance(output, CancellableStream):
                    logger.info('CancellableStream processing output')
                    # total_output = ''
                    # for line in output:
                    # Removes any trailing whitespace, including \n and \r\n
                    #     line = line.rstrip()
                    #     if 'Requirement already satisfied: ' not in line:
                    #         logger.info(f'>>> {line.strip()}')
                    # Avoid text from lines running into each other
                    #     total_output += line + ' '
                    # _exit_code = output.exit_code()
                    # output.close()
                    # if _exit_code != 0:
                    #     raise RuntimeError(
                    #         f'Failed to initialize plugin {requirement.name} with exit code {_exit_code} and output: {total_output.strip()}'
                    #     )
                    # logger.debug(f'Output: {total_output.strip()}')
                    _handle_stream_output(output, requirement.name)
                else:
                    if exit_code != 0:
                        raise RuntimeError(
                            f'Failed to initialize plugin {requirement.name} with exit code {exit_code} and output: {output.strip()}'
                        )
                logger.info(f'Plugin {requirement.name} initialized successfully.')
        else:
            logger.info('Skipping plugin initialization in the sandbox.')

        if len(requirements) > 0:
            await self.source_bashrc_async()

        setattr(self, '_plugin_initialized', True)
        if isinstance(output, CancellableStream):
            output.close()  # Ensure the stream is closed
