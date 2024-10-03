import logging
import os
import tempfile
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import tenacity
from runloop_api_client import APIStatusError, Runloop
from runloop_api_client.types.devbox_create_params import LaunchParameters

from openhands.core.config import AppConfig
from openhands.events import EventStream
from openhands.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.events.observation.files import FileReadObservation, FileWriteObservation
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime import Runtime
from openhands.runtime.utils.files import read_lines


class RunloopRuntime(Runtime):
    """
    The interface for connecting to the Runloop provided cloud Runtime.
    The RunloopRuntime provides a Runtime compliant implementation for using
    a Runloop Devbox as the OpenHands runtime.
    """

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_message_callback: Callable | None = None,
    ):
        super().__init__(
            config, event_stream, sid, plugins, env_vars, status_message_callback
        )

        assert config.runloop_api_key, 'Runloop API key is required'
        self.config = config

        self.api_client = Runloop(bearer_token=config.runloop_api_key)
        self.devbox = self.api_client.devboxes.create(
            name=sid,
            setup_commands=[f'mkdir -p {config.workspace_mount_path_in_sandbox}'],
            prebuilt='openhands',
            launch_parameters=LaunchParameters(
                keep_alive_time_seconds=config.sandbox.timeout,
            ),
        )
        self.shell_name = 'openhands'

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(90),
        wait=tenacity.wait_fixed(0.75),
    )
    def _wait_until_alive(self):
        """Pull devbox status until it is running"""
        if self.devbox.status == 'running':
            return

        devbox = self.api_client.devboxes.retrieve(id=self.devbox.id)
        if devbox.status != 'running':
            raise ConnectionRefusedError('Devbox is not running')

        initialization_cmd = f'cd {self.config.workspace_mount_path_in_sandbox}; '

        self.api_client.devboxes.execute_sync(
            id=self.devbox.id,
            command=initialization_cmd,
            shell_name=self.shell_name,
        )

        # Devbox is connected and running
        self.devbox = devbox

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        raise NotImplementedError

    def read(self, action: FileReadAction) -> Observation:
        self._wait_until_alive()

        try:
            file_contents = self.api_client.devboxes.read_file_contents(
                id=self.devbox.id, file_path=action.path
            )
            return FileReadObservation(
                content=''.join(
                    read_lines(file_contents.split('\n'), action.start, action.end)
                ),
                path=action.path,
            )
        except APIStatusError as e:
            return ErrorObservation(
                content=e.message,
            )

    def write(self, action: FileWriteAction) -> Observation:
        self._wait_until_alive()

        contents: str = action.content
        try:
            self.api_client.devboxes.write_file(
                id=self.devbox.id, file_path=action.path, contents=contents
            )

            return FileWriteObservation(
                content='',
                path=action.path,
            )

        except APIStatusError as e:
            return ErrorObservation(
                content=e.message,
            )

        except Exception as e:
            return ErrorObservation(
                content=str(e),
            )

    def browse(self, action: BrowseURLAction) -> Observation:
        raise NotImplementedError

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        raise NotImplementedError

    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        self._wait_until_alive()

        if not os.path.exists(host_src):
            raise FileNotFoundError(f'file {host_src} does not exist')

        mkdir_resp = self.api_client.devboxes.execute_sync(
            id=self.devbox.id, command=f'mkdir -p {sandbox_dest}'
        )
        if mkdir_resp.exit_status != 0:
            raise Exception(f'error creating directory: {mkdir_resp.stdout}')

        if recursive:
            # For recursive copy, create a zip file
            # Create uuid
            tmp_zip_file_path = '/tmp/tmp.zip'
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
                temp_zip_path = temp_zip.name

                with ZipFile(temp_zip_path, 'w') as zipf:
                    for root, _, files in os.walk(host_src):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(
                                file_path, os.path.dirname(host_src)
                            )
                            zipf.write(file_path, arcname)
                self.api_client.devboxes.upload_file(
                    id=self.devbox.id,
                    file=Path(temp_zip_path),
                    path=tmp_zip_file_path,
                )

                self.api_client.devboxes.execute_sync(
                    id=self.devbox.id,
                    command=f'unzip {tmp_zip_file_path} -d {sandbox_dest} && rm {tmp_zip_file_path}',
                )

        else:
            self.api_client.devboxes.execute_sync(
                id=self.devbox.id, command=f'mkdir -p {sandbox_dest}'
            )

            host_path = Path(host_src)

            if host_path.is_dir():
                raise ValueError('recursive copy is required for directories')

            self.api_client.devboxes.upload_file(
                id=self.devbox.id,
                file=host_path,
                path=f'{sandbox_dest}/{host_path.name}',
            )

    def list_files(self, path: str | None = None) -> list[str]:
        self._wait_until_alive()

        try:
            result = self.api_client.devboxes.execute_sync(
                id=self.devbox.id, command=f'ls {path}'
            )
            return result.stdout.split('\n')
        except APIStatusError as e:
            logging.error(f'Error listing files: {e}')
            raise e
        except Exception as e:
            logging.error(f'Error listing files: {e}')
            raise e

    def run(self, action: CmdRunAction) -> Observation:
        self._wait_until_alive()
        try:
            command = ''
            if action.keep_prompt:
                # If "keep_prompt", echo standard ps1 before continuing
                prompt = '$' if self.config.run_as_openhands else '#'
                user = 'openhands' if self.config.run_as_openhands else 'root'
                command = f'echo -n "{user}@$(hostname):$(pwd) {prompt} " && '

            # Escape single quotes
            escaped_action = action.command.replace("'", "'\"'\"'")

            command = command + escaped_action

            if self.config.run_as_openhands:
                formatted_command = f"eval '{command}'"
            else:
                formatted_command = f"sudo /bin/bash -c '{command}'"

            result = self.api_client.devboxes.execute_sync(
                id=self.devbox.id,
                command=formatted_command,
                shell_name=self.shell_name,
            )

            return CmdOutputObservation(
                content=result.stdout.replace('\n', '\r\n'),
                command_id=action.id,
                command=action.command,
                exit_code=result.exit_status,
            )

        except APIStatusError as e:
            return ErrorObservation(
                content=e.message,
            )
