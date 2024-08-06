import json
import os
import tarfile
from glob import glob

from e2b import Sandbox as E2BSandbox
from e2b.sandbox.exception import (
    TimeoutException,
)

from opendevin.core.config import SandboxConfig
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import CancellableStream
from opendevin.runtime.sandbox import Sandbox
from opendevin.runtime.utils.async_utils import async_to_sync


class E2BBox(Sandbox):
    closed = False
    _cwd: str = '/home/user'

    def __init__(
        self,
        config: SandboxConfig,
        e2b_api_key: str,
        template: str = 'open-devin',
    ):
        super().__init__(config)
        self.sandbox = E2BSandbox(
            api_key=e2b_api_key,
            template=template,
            # It's possible to stream stdout and stderr from sandbox and from each process
            on_stderr=lambda x: logger.info(f'E2B sandbox stderr: {x}'),
            on_stdout=lambda x: logger.info(f'E2B sandbox stdout: {x}'),
            cwd=self._cwd,  # Default workdir inside sandbox
        )
        logger.info(f'Started E2B sandbox with ID "{self.sandbox.id}"')

    @property
    def filesystem(self):
        return self.sandbox.filesystem

    def _archive(self, host_src: str, recursive: bool = False):
        if recursive:
            assert os.path.isdir(
                host_src
            ), 'Source must be a directory when recursive is True'
            files = glob(host_src + '/**/*', recursive=True)
            srcname = os.path.basename(host_src)
            tar_filename = os.path.join(os.path.dirname(host_src), srcname + '.tar')
            with tarfile.open(tar_filename, mode='w') as tar:
                for file in files:
                    tar.add(
                        file, arcname=os.path.relpath(file, os.path.dirname(host_src))
                    )
        else:
            assert os.path.isfile(
                host_src
            ), 'Source must be a file when recursive is False'
            srcname = os.path.basename(host_src)
            tar_filename = os.path.join(os.path.dirname(host_src), srcname + '.tar')
            with tarfile.open(tar_filename, mode='w') as tar:
                tar.add(host_src, arcname=srcname)
        return tar_filename

    @async_to_sync
    def execute(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        return self.execute_async(cmd, stream, timeout)  # type: ignore

    async def execute_async(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> tuple[int, str | CancellableStream]:
        timeout = timeout if timeout is not None else self.config.timeout
        process = await self.sandbox.process.start(cmd, env_vars=self._env)
        try:
            process_output = await process.wait(timeout=timeout)
        except TimeoutException:
            logger.info('Command timed out, killing process...')
            await process.kill()
            return -1, f'Command: "{cmd}" timed out'

        logs = [m.line for m in process_output.messages]
        logs_str = '\n'.join(logs)
        if process.exit_code is None:
            return -1, logs_str

        assert process_output.exit_code is not None
        return process_output.exit_code, logs_str

    @async_to_sync
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        """Copies a local file or directory to the sandbox."""
        return self.copy_to_async(host_src, sandbox_dest, recursive)

    async def copy_to_async(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ):
        tar_filename = self._archive(host_src, recursive)

        # Prepend the sandbox destination with our sandbox cwd
        sandbox_dest = os.path.join(self._cwd, sandbox_dest.removeprefix('/'))

        with open(tar_filename, 'rb') as tar_file:
            # Upload the archive to /home/user (default destination that always exists)
            uploaded_path = await self.sandbox.upload_file(tar_file)

            # Check if sandbox_dest exists. If not, create it.
            process = await self.sandbox.process.start_and_wait(
                f'test -d {sandbox_dest}'
            )
            if process.exit_code != 0:
                self.sandbox.filesystem.make_dir(sandbox_dest)

            # Extract the archive into the destination and delete the archive
            process = await self.sandbox.process.start_and_wait(
                f'sudo tar -xf {uploaded_path} -C {sandbox_dest} && sudo rm {uploaded_path}'
            )
            if process.exit_code != 0:
                raise RuntimeError(
                    f'Failed to extract {uploaded_path} to {sandbox_dest}: {process.stderr}'
                )

        # Delete the local archive
        os.remove(tar_filename)

    @async_to_sync
    async def add_to_env(self, key: str, value: str):
        return await self.add_to_env_async(key, value)

    async def add_to_env_async(self, key: str, value: str):
        exit_code, _ = await self.execute_async(f'export {key}={json.dumps(value)}')
        if exit_code == 0:
            self._env[key] = value
        else:
            raise RuntimeError(f'Failed to set environment variable {key}')

    @async_to_sync
    def close(self):
        return self.aclose()

    async def aclose(self):
        await self.sandbox.close()

    async def get_working_directory(self) -> str:
        return self.sandbox.cwd
