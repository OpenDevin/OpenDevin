import asyncio
import atexit
import json
import logging
import os
import re
import shlex
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from glob import glob
from queue import Empty, Queue
from typing import Any, Dict, Optional, Tuple, Union  # type: ignore[unused-import]

import aiodocker
import docker
from aiodocker.containers import Exec
from aiodocker.exceptions import DockerError
from pexpect import pxssh
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from opendevin.core.config import SandboxConfig
from opendevin.core.const.guide_url import TROUBLESHOOTING_URL
from opendevin.core.logger import opendevin_logger as logger
from opendevin.core.schema import CancellableStream
from opendevin.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from opendevin.runtime.plugins.requirement import PluginRequirement
from opendevin.runtime.sandbox import Sandbox
from opendevin.runtime.utils import find_available_tcp_port, split_bash_commands
from opendevin.runtime.utils.async_utils import async_to_sync
from opendevin.runtime.utils.image_agnostic import get_od_sandbox_image

logger.setLevel(logging.DEBUG)
PEXPECT_PROMPT = '[PEXPECT]$'


class SSHExecCancellableStream(CancellableStream):
    def __init__(self, ssh, cmd, timeout):
        super().__init__(self.read_output())
        self.ssh = ssh
        self.cmd = cmd
        self.timeout = timeout if timeout is not None else 120
        self.output_queue: Queue[str] = Queue()
        self.thread = threading.Thread(target=self._read_output_thread)
        self.thread.daemon = True
        self.thread.start()
        self.eof_reached = False
        self.closed = False

    def close(self):
        self.closed = True
        self.thread.join(timeout=1)

    def exit_code(self):
        assert self.ssh is not None
        marker = f'EXIT_CODE_MARKER_{uuid.uuid4().hex}'
        self.ssh.sendline(f'echo "{marker}$?{marker}"')

        if not self.ssh.prompt(timeout=self.timeout):
            return None  # Timeout occurred

        output = self.ssh.before
        match = re.search(f'{marker}(\\d+){marker}', output)

        if match:
            try:
                return int(match.group(1))
            except ValueError:
                # Log the unexpected format
                logger.error(f'Unexpected exit code format: {match.group(1)}')
                return None
        else:
            # If we can't find our marked exit code, log the output and return None
            logger.error(f'Could not find exit code in output: {output}')
            return None

    def _read_output_thread(self):
        while not self.closed:
            try:
                new_output = self.ssh.read_nonblocking(size=512, timeout=1)
                if new_output:
                    self.output_queue.put(new_output)
                    if PEXPECT_PROMPT in new_output:
                        self.eof_reached = True
                        break
            except pxssh.TIMEOUT:
                pass
            except pxssh.EOF:
                self.eof_reached = True
                break
            except Exception as e:
                logger.error(f'Error reading output: {e}')
                self.eof_reached = True
                break

    def read_output(self):
        while not self.closed:
            try:
                yield self.output_queue.get(timeout=0.1)
            except Empty:
                if self.eof_reached:
                    break
                # Check if the command has finished
                if self.ssh.prompt(timeout=0.1):
                    break

            # If the queue is empty and EOF is reached, break the loop
            if self.output_queue.empty() and self.eof_reached:
                break


class DockerSSHBox(Sandbox):
    _instance = None
    _initialization_lock = asyncio.Lock()

    instance_id: str
    container_image: str
    container_name_prefix = 'opendevin-sandbox-'
    container_name: str
    container: aiodocker.containers.DockerContainer
    docker_client: aiodocker.Docker

    _ssh_password: str
    _ssh_port: int
    ssh: pxssh.pxssh | None = None
    _cleanup_done: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        config: SandboxConfig,
        persist_sandbox: bool,
        workspace_mount_path: str,
        sandbox_workspace_dir: str,
        cache_dir: str,
        run_as_devin: bool,
        ssh_hostname: str = 'host.docker.internal',
        ssh_password: str | None = None,
        ssh_port: int = 22,
        sid: str | None = None,
    ):
        if not hasattr(self, 'initialized'):
            super().__init__(config)
            self.initialized = False
            self.ssh = None
            self._sshbox_init_complete = asyncio.Event()

            self.cache_dir = cache_dir
            self.config = config
            self.initialize_plugins: bool = config.initialize_plugins
            self.persist_sandbox = persist_sandbox
            self.run_as_devin = run_as_devin
            self.sandbox_workspace_dir = sandbox_workspace_dir
            self.ssh_hostname = ssh_hostname
            self.ssh_port = ssh_port
            self.sid = sid
            self.timeout = config.timeout
            self.use_host_network = config.use_host_network
            self.workspace_mount_path = workspace_mount_path

            # set up random user password
            if self.persist_sandbox:
                if not self.run_as_devin:
                    raise RuntimeError(
                        'Persistent sandbox is currently designed for opendevin user only. Please set run_as_devin=True in your config.toml'
                    )
                self.instance_id = 'persisted'
            else:
                self.instance_id = (sid or '') + str(uuid.uuid4())

            if self.persist_sandbox:
                if not ssh_password:
                    raise RuntimeError(
                        'Please add ssh_password to your config.toml or add -e SSH_PASSWORD to your docker run command'
                    )
                self._ssh_password = ssh_password
            else:
                self._ssh_password = str(uuid.uuid4())

            self.is_initial_session = True

            self.container_image = config.container_image
            self.container_name = self.container_name_prefix + self.instance_id

            # Initialize _env with SANDBOX_ENV_ prefixed variables
            self._env = {
                k[12:]: v for k, v in os.environ.items() if k.startswith('SANDBOX_ENV_')
            }
            if isinstance(config.env, dict):
                self._env.update(config.env)

            self._cleanup_done = False
            atexit.register(self.sync_cleanup)

    @async_to_sync
    def initialize(self):
        return self.ainit()

    async def ainit(self):
        if not self._sshbox_init_complete.is_set():
            async with self._initialization_lock:
                if not self._sshbox_init_complete.is_set():
                    if self.run_as_devin:
                        logger.info(
                            'SSHBox is running as opendevin user with USER_ID=1000 in the sandbox'
                        )
                    else:
                        logger.info(
                            'SSHBox is running as root user with USER_ID=0 in the sandbox'
                        )
                    try:
                        await self._setup_docker_client()
                        await asyncio.sleep(1)

                        # TODO rewrite once get_od_sandbox_image also uses aiodocker!
                        # Create a temporary docker.Docker object
                        temp_docker_client = docker.from_env()
                        try:
                            self.container_image = await asyncio.to_thread(
                                get_od_sandbox_image,
                                self.container_image,
                                temp_docker_client,
                            )
                        finally:
                            # Ensure we close the temporary client
                            temp_docker_client.close()

                        await self._setup_container()
                        await asyncio.sleep(1)
                        await self._setup_ssh_and_user()

                        self.initialized = True
                        logger.info('SSHBox initialization complete')
                        self._sshbox_init_complete.set()
                    except Exception as e:
                        logger.error(f'SSHBox initialization failed: {e}')
                        # Optionally, clean up any partially initialized resources here
                        await self.aclose()
                        raise
        else:
            await self._sshbox_init_complete.wait()

        if not self.initialized:
            raise RuntimeError('SSHBox initialization failed')

    async def _setup_docker_client(self):
        try:
            self.docker_client = aiodocker.Docker()
            # Check if Docker daemon is accessible
            info = await self.docker_client.system.info()
            logger.info(f"Connected to Docker daemon. Version: {info['ServerVersion']}")
        except Exception as ex:
            logger.exception(
                f'Error creating aiodocker client or connecting to Docker daemon. Please check that Docker is running and visit `{TROUBLESHOOTING_URL}` for more debugging information.',
                exc_info=False,
            )
            raise ex

    def format_env_value(self, value):
        if isinstance(value, str):
            # Use shlex.quote for strings to handle all special characters
            return shlex.quote(value)
        elif isinstance(value, (int, float, bool)):
            # Convert to string directly
            return str(value).lower()
        # Use JSON for complex types, then quote the result
        return shlex.quote(json.dumps(value))

    @async_to_sync
    def add_to_env(self, key: str, value: str):
        return self.add_to_env_async(key, value)

    async def add_to_env_async(self, key: str, value: str):
        assert self.ssh is not None, 'SSH session is not initialized'
        formatted_value = self.format_env_value(value)

        # Set the environment variable in the SSH session
        self.send_line(f'export {key}={formatted_value}')
        self.ssh.prompt()

        # Verify that the environment variable was set
        self.send_line(f'echo ${key}')
        self.ssh.prompt()
        output = self.ssh.before.strip()

        if output == value:
            self._env[key] = value
            os.environ[key] = value
        else:
            raise RuntimeError(f'Failed to set environment variable {key}: {output}')

    async def run_command(self, command: str, use_os: Union[bool, None] = False):
        if use_os and os.environ.items():
            environment = dict(os.environ)
        elif self._env:
            environment = dict(self._env)
        else:
            environment = None
        return await self.container_exec_run(
            ['/bin/bash', '-c', command],
            environment=environment,
        )

    async def container_exec_run(
        self,
        cmd: Union[str, list[str]],
        workdir: Union[str, None] = None,
        environment: Union[dict[str, str], None] = None,
    ) -> Tuple[int, str]:
        """Executes a command in the container and returns the exit code and output.

        This method uses the aiodocker library's Exec class to execute a command
        within the Docker container. It waits for the command to complete and
        returns a tuple containing the exit code and the command's output.

        Args:
            cmd (Union[str, list[str]]): The command to execute. Can be a string
                or a list of strings.
            workdir (Union[str, None], optional): The working directory for the
                command. Defaults to None.
            environment (Union[dict[str, str], None], optional): Environment
                variables to set for the command. Defaults to None.

        Returns:
            Tuple[int, str]: A tuple containing the exit code and the command's
                output.
        """

        # Ensure the command is a list
        if isinstance(cmd, str):
            cmd = ['/bin/sh', '-c', cmd]

        # Create the exec instance
        exec_instance: Exec = await self.container.exec(
            cmd=cmd,
            stdout=True,
            stderr=True,
            workdir=workdir or self.sandbox_workspace_dir,
            environment=environment or dict(self._env) if self._env else None,
        )

        # Start the exec instance and collect the output
        output = b''
        async with exec_instance.start() as stream:
            while True:
                chunk = await stream.read_out()
                if chunk:
                    output += chunk.data
                else:
                    break

        # Get the exit code
        inspect_data = await exec_instance.inspect()
        exit_code = inspect_data.get('ExitCode')

        # Decode output, handling potential encoding errors
        try:
            decoded_output = output.decode('utf-8')
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try decoding with 'latin-1'
            decoded_output = output.decode('latin-1')

        return exit_code, decoded_output

    async def _setup_container(self):
        assert self.docker_client is not None, 'aiodocker client is not initialized'

        try:
            existing_container = await self.docker_client.containers.get(
                self.container_name
            )
            if existing_container:
                if self.persist_sandbox and not self.is_initial_session:
                    logger.info('Using existing Docker container')
                    self.container = existing_container
                    await self.start_docker_container()
                else:
                    logger.info('Replacing existing container')
                    try:
                        await existing_container.delete(force=True)
                    except DockerError as delete_error:
                        logger.error(
                            f'Failed to delete existing container: {delete_error}'
                        )
                        # Decide whether to raise this error or continue with creating a new container
                    await self.restart_docker_container()
            else:
                logger.info(
                    'No existing container found, creating new Docker container'
                )
                await self.restart_docker_container()
        except DockerError as e:
            if e.status == 404:
                logger.info('Creating new Docker container')
                await self.restart_docker_container()
            else:
                logger.error(f'Unexpected Docker error: {e}')
                raise
        except Exception as e:
            logger.error(f'Unexpected error during container setup: {e}')
            raise
        else:
            self.is_initial_session = False

    async def _setup_environment(self):
        try:
            assert self.ssh is not None, 'SSH session is not initialized'
            logger.info('Setting up sandbox vars')
            for key, value in self._env.items():
                if key.startswith('SANDBOX_ENV_'):
                    sandbox_key = key.removeprefix('SANDBOX_ENV_')
                    await self.add_to_env_async(sandbox_key, value)
                else:
                    await self.add_to_env_async(key, value)

            logger.info('Setting up tmp folder')
            self.send_line('mkdir -p /tmp')
            self.ssh.prompt()
            logger.info('Setting up git user')
            self.send_line('git config --global user.name "OpenDevin"')
            self.ssh.prompt()
            logger.info('Setting up git email')
            self.send_line('git config --global user.email "opendevin@all-hands.dev"')
            self.ssh.prompt()
        except Exception as e:
            logger.exception(f'Error during initialization: {e}')
            raise e
        logger.info('Environment setup complete')
        time.sleep(1)

    async def _setup_user(self):
        username = 'opendevin' if self.run_as_devin else 'root'
        logger.info(f'Setting up user {username}')
        await asyncio.sleep(2)
        # Make users sudoers passwordless
        # TODO(sandbox): add this line in the Dockerfile for next minor version of docker image
        result = await self.container_exec_run(
            ['/bin/bash', '-c', r"echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"],
        )
        exit_code, logs = result
        if exit_code != 0:
            raise RuntimeError(
                f'Failed to make all users passwordless sudoers in sandbox: {logs}'
            )

        # Check if the opendevin user exists
        exit_code, logs = await self.container_exec_run(
            ['/bin/bash', '-c', 'id -u opendevin'],
        )
        if exit_code == 0 and 'no such user' not in logs:
            # User exists, delete it
            logger.info('Deleting existing opendevin user...')
            exit_code, logs = await self.container_exec_run(
                ['/bin/bash', '-c', 'userdel -r opendevin'],
            )
            if exit_code != 0:
                logger.error(f'Failed to remove opendevin user in sandbox: {logs}')
            else:
                logger.info('Successfully deleted existing opendevin user.')

        if self.run_as_devin:
            # The opendevin home folder is already part of the sandbox.
            # Just try to clear the logs folder here.
            exit_code, logs = await self.container_exec_run(
                ['/bin/bash', '-c', 'test -d /home/opendevin/logs'],
            )
            if exit_code == 0:
                logger.info('Deleting existing logs directory...')
                exit_code, logs = await self.container_exec_run(
                    ['/bin/bash', '-c', 'rm -rf /home/opendevin/logs'],  # 2>/dev/null
                )
                if exit_code != 0:
                    # This is not a fatal error, just a warning
                    logger.warning(
                        f'Failed to remove existing opendevin logs directory in sandbox: {logs}'
                    )

            # Create the opendevin user
            exit_code, logs = await self.container_exec_run(
                [
                    '/bin/bash',
                    '-c',
                    f'useradd -rm -d /home/opendevin -s /bin/bash -g root -G sudo -u {self.config.user_id} opendevin 2>/dev/null && id opendevin',
                ],
            )
            if exit_code != 0:
                raise RuntimeError(
                    f'Failed to create opendevin user in sandbox: {logs}'
                )

            exit_code, logs = await self.container_exec_run(
                [
                    '/bin/bash',
                    '-c',
                    f"echo 'opendevin:{self._ssh_password}' | chpasswd",
                ],
            )
            if exit_code != 0:
                raise RuntimeError(f'Failed to set password in sandbox: {logs}')

            # chown the home directory
            exit_code, logs = await self.container_exec_run(
                ['/bin/bash', '-c', 'chown opendevin:root /home/opendevin'],
            )
            if exit_code != 0:
                raise RuntimeError(
                    f'Failed to chown home directory for opendevin in sandbox: {logs}'
                )
            # check the miniforge3 directory exist
            exit_code, logs = await self.container_exec_run(
                [
                    '/bin/bash',
                    '-c',
                    '[ -d "/opendevin/miniforge3" ] && exit 0 || exit 1',
                ],
            )
            if exit_code != 0:
                if exit_code == 1:
                    raise RuntimeError(
                        'OPENDEVIN_PYTHON_INTERPRETER is not usable. Please pull the latest Docker image: docker pull ghcr.io/opendevin/sandbox:main'
                    )
                else:
                    raise RuntimeError(
                        f'An error occurred while checking if miniforge3 directory exists: {logs}'
                    )
            exit_code, logs = await self.container_exec_run(
                [
                    '/bin/bash',
                    '-c',
                    f'chown opendevin:root {self.sandbox_workspace_dir}',
                ],
            )
            if exit_code != 0:
                # This is not a fatal error, just a warning
                logger.warning(
                    f'Failed to chown workspace directory for opendevin in sandbox: {logs}. But this should be fine if the {self.sandbox_workspace_dir=} is mounted by the app docker container.'
                )

            # Add a check to ensure the user was created successfully
            exit_code, logs = await self.container_exec_run(
                ['/bin/bash', '-c', 'id opendevin'],
            )
            if exit_code != 0:
                raise RuntimeError(f'Failed to create or verify opendevin user: {logs}')
        else:
            exit_code, logs = await self.container_exec_run(
                # change password for root
                ['/bin/bash', '-c', f"echo 'root:{self._ssh_password}' | chpasswd"],
            )
            if exit_code != 0:
                raise RuntimeError(
                    f'Failed to set password for root in sandbox: {logs}'
                )
        exit_code, logs = await self.container_exec_run(
            ['/bin/bash', '-c', "echo 'opendevin-sandbox' > /etc/hostname"],
        )

        logger.info('User setup in sandbox completed')

    async def _check_user_setup(self):
        """Checks if the user setup was successful by verifying user existence and permissions."""
        if self.run_as_devin:
            try:
                # Check if the user exists
                exit_code, logs = await self.container_exec_run(['id', 'opendevin'])
                if exit_code != 0:
                    raise RuntimeError(f'User "opendevin" does not exist: {logs}')

                # Check if the user is in the correct groups (sudo in this case)
                exit_code, logs = await self.container_exec_run(
                    ['/bin/bash', '-c', 'groups opendevin | grep sudo']
                )
                if exit_code != 0:
                    raise RuntimeError(
                        f'User "opendevin" is not in the sudo group: {logs}'
                    )

                logger.info('User setup verification successful.')

            except RuntimeError as e:
                logger.error(f'User setup verification failed: {e}')
                raise

    # Use the retry decorator, with a maximum of 5 attempts and a fixed wait time of 5 seconds between attempts
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(5),
        retry=retry_if_exception_type(pxssh.ExceptionPxssh),
    )
    async def _setup_ssh(self):
        try:
            await asyncio.sleep(2)
            await self.wait_for_ssh_ready()
            await asyncio.sleep(2)
            await self._ssh_login()
            await asyncio.sleep(2)
            await self.start_ssh_session()
            await asyncio.sleep(2)
            await self._setup_environment()
        except pxssh.ExceptionPxssh as e:
            logger.error(
                f'SSH session failed, attempting to restart container.\nError: {e}'
            )
            if not self.persist_sandbox:
                await self.remove_docker_container()
                await self.restart_container()
            raise

    async def _setup_ssh_and_user(self):
        await self._setup_user()
        await self._check_user_setup()
        retry_count = 0
        max_retries = 3
        while retry_count < max_retries:
            try:
                await self._setup_ssh()
                break
            except pxssh.ExceptionPxssh as e:
                retry_count += 1
                logger.warning(
                    f'SSH setup failed (attempt {retry_count}/{max_retries}): {e}'
                )
                if retry_count == max_retries:
                    raise RuntimeError('Failed to set up SSH after multiple attempts')
                await asyncio.sleep(5)

    async def wait_for_ssh_ready(self):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                exit_code, _ = await self.container_exec_run(
                    ['sh', '-c', "service ssh status | grep 'is running'"]
                )
                if exit_code == 0:
                    logger.info('SSH service is running')
                    # Add a small delay after confirming SSH is running
                    await asyncio.sleep(2)
                    return
            except Exception as e:
                logger.warning(f'Error checking SSH status: {e}')

            logger.info(
                f'Waiting for SSH service to start (attempt {attempt + 1}/{max_attempts})'
            )
            await asyncio.sleep(2)

        raise RuntimeError('SSH service failed to start in time')

    async def start_ssh_session(self):
        await asyncio.sleep(2)
        logger.info('Starting SSH session')
        assert self.ssh is not None, 'SSH session is not initialized'
        # Fix: https://github.com/pexpect/pexpect/issues/669
        self.send_line("bind 'set enable-bracketed-paste off'")
        self.ssh.prompt()

        # cd to workspace
        self.send_line(f'cd {self.sandbox_workspace_dir}')
        self.ssh.prompt()
        logger.info('SSH session started')

    async def restart_container(self):
        logger.info('Restarting container...')
        await asyncio.sleep(1)
        await self._setup_container()
        await asyncio.sleep(1)
        await self._setup_user()
        await asyncio.sleep(1)
        logger.info('Container startup successful.')

    async def _ssh_login(self):
        try:
            if not hasattr(self, '_ssh_debug_logged'):
                hostname = self.ssh_hostname
                username = 'opendevin' if self.run_as_devin else 'root'
                if self.persist_sandbox:
                    password_msg = 'using your SSH password'
                else:
                    password_msg = f"using the password '{self._ssh_password}'"
                logger.info('Connecting to SSH session...')
                hostname_to_log = hostname.replace('host.docker.internal', 'localhost')
                ssh_cmd = f'`ssh -v -p {self._ssh_port} {username}@{hostname_to_log}`'
                logger.info(
                    f'You can debug the SSH connection by running: {ssh_cmd} {password_msg}'
                )
                self._ssh_debug_logged = True

            logger.info(
                f"Attempting SSH login to {self.ssh_hostname}:{self._ssh_port} as {'opendevin' if self.run_as_devin else 'root'}"
            )

            login_timeout = (
                max(self.config.timeout, 20) if hasattr(self, 'config.timeout') else 60
            )

            logger.info(' -> Creating pxssh instance')
            self.ssh = pxssh.pxssh(
                echo=False,
                timeout=login_timeout,
                encoding='utf-8',
                codec_errors='replace',
            )
            assert self.ssh is not None

            logger.info(f' -> Logging in to {self.ssh_hostname}...')
            self.ssh.login(
                self.ssh_hostname,
                'opendevin' if self.run_as_devin else 'root',
                self._ssh_password,
                port=self._ssh_port,
                login_timeout=login_timeout,
                quiet=False,
            )

            logger.info('Connected to SSH session')
        except pxssh.ExceptionPxssh as e:
            logger.exception(f'Login failed: {e}', exc_info=False)
            if 'connection refused' in str(e).lower():
                logger.error('SSH connection refused')
            raise e

    def get_exec_cmd(self, cmd: str) -> list[str]:
        if self.run_as_devin:
            return ['su', 'opendevin', '-c', cmd]
        else:
            return ['/bin/bash', '-c', cmd]

    def _send_interrupt(
        self,
        cmd: str,
        prev_output: str = '',
        ignore_last_output: bool = False,
    ) -> tuple[int, str | CancellableStream]:
        assert self.ssh is not None
        logger.exception(
            f'Command "{cmd}" timed out, killing process...', exc_info=False
        )
        # send a SIGINT to the process
        self.ssh.sendintr()
        self.ssh.prompt()
        command_output = prev_output
        if not ignore_last_output:
            command_output += '\n' + self.ssh.before
        return (
            -1,
            f'Command: "{cmd}" timed out. Sent SIGINT to the process: {command_output}',
        )

    def send_line(self, cmd: str):
        if self.ssh is not None:
            self.ssh.sendline(cmd)

    @async_to_sync
    def execute(
        self, cmd: str, stream: bool = False, timeout: int | None = None
    ) -> Union[Tuple[int, str], Tuple[int, CancellableStream]]:
        return self.execute_async(cmd, stream, timeout)  # type: ignore

    async def execute_async(
        self,
        cmd: str,
        stream: bool = False,
        timeout: int | None = None,
        # ) -> Union[Tuple[int, str], Tuple[int, CancellableStream]]:
    ) -> tuple[int, str | CancellableStream]:
        # Ensure initialization is complete
        if not self.initialized:
            await self.ainit()

        assert self.ssh is not None
        timeout = timeout or self.config.timeout
        timeout = 60 if timeout is None or int(timeout) < 60 else int(timeout)

        commands = split_bash_commands(cmd)

        if len(commands) > 1:
            all_output = ''
            for command in commands:
                exit_code, output = await self.execute_async(command, stream, timeout)  # type: ignore
                if all_output:
                    all_output += '\r\n'
                all_output += str(output)
                if exit_code != 0:
                    return exit_code, all_output
                if PEXPECT_PROMPT in output:
                    logger.debug(
                        'Detected [PEXPECT]$ prompt, ending command execution.'
                    )
                    break
            return 0, all_output

        # Prepare environment variables
        env_exports = ' '.join(
            f'export {key}={self.format_env_value(value)}'
            for key, value in self._env.items()
        )
        if env_exports.strip():
            full_cmd = f'({env_exports}) && {cmd}'
        else:
            full_cmd = f'{cmd}'
        self.send_line(full_cmd)
        if stream:
            return 0, SSHExecCancellableStream(self.ssh, full_cmd, timeout)

        success = self.ssh.prompt(timeout=timeout)
        if not success:
            return self._send_interrupt(full_cmd)  # type: ignore
        command_output = self.ssh.before

        # once out, make sure that we have *every* output, we while loop until we get an empty output
        while True:
            self.send_line('\n')
            timeout_not_reached = self.ssh.prompt(timeout=1)
            if not timeout_not_reached:
                logger.debug('TIMEOUT REACHED')
                break
            output = self.ssh.before
            if isinstance(output, str) and output.strip() == '':
                break
            command_output += output
            if PEXPECT_PROMPT in output:
                logger.debug('Detected [PEXPECT]$ prompt, ending command execution.')
                break
        command_output = command_output.removesuffix('\r\n')

        # get the exit code
        self.send_line('echo $?')
        self.ssh.prompt(timeout=timeout)
        exit_code_str = self.ssh.before.strip()
        _start_time = time.time()
        while not exit_code_str:
            self.ssh.prompt(timeout=timeout)
            exit_code_str = self.ssh.before.strip()
            if time.time() - _start_time > timeout:
                return self._send_interrupt(  # type: ignore
                    cmd, command_output, ignore_last_output=True
                )
            if PEXPECT_PROMPT in exit_code_str:
                logger.debug('Detected [PEXPECT]$ prompt, ending command execution.')
                break
        cleaned_exit_code_str = exit_code_str.replace('echo $?', '').strip().split()[0]

        try:
            exit_code = int(cleaned_exit_code_str)
        except ValueError:
            logger.error(f'Invalid exit code: {cleaned_exit_code_str}')
            # Handle the invalid exit code appropriately (e.g., raise an exception or set a default value)
            exit_code = -1  # or some other appropriate default value

        return exit_code, command_output

    @async_to_sync
    def copy_to(self, host_src: str, sandbox_dest: str, recursive: bool = False):
        return self.copy_to_async(host_src, sandbox_dest, recursive)

    async def copy_to_async(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ):
        if not os.path.exists(host_src):
            raise FileNotFoundError(f'Source file {host_src} does not exist')

        # mkdir -p sandbox_dest if it doesn't exist
        exit_code, logs = await self.container_exec_run(  # type: ignore
            ['/bin/bash', '-c', f'mkdir -p {sandbox_dest}'],
            workdir=self.sandbox_workspace_dir,
            environment=self._env,
        )
        if exit_code != 0:
            raise RuntimeError(
                f'Failed to create directory {sandbox_dest} in sandbox: {logs}'
            )

        # use temp directory to store the tar file to avoid
        # conflict of filename when running multi-processes
        with tempfile.TemporaryDirectory() as tmp_dir:
            if recursive:
                assert os.path.isdir(
                    host_src
                ), 'Source must be a directory when recursive is True'
                files = glob(host_src + '/**/*', recursive=True)
                srcname = os.path.basename(host_src)
                tar_filename = os.path.join(tmp_dir, srcname + '.tar')
                with tarfile.open(tar_filename, mode='w') as tar:
                    for file in files:
                        tar.add(
                            file,
                            arcname=os.path.relpath(file, os.path.dirname(host_src)),
                        )
            else:
                assert os.path.isfile(
                    host_src
                ), 'Source must be a file when recursive is False'
                srcname = os.path.basename(host_src)
                tar_filename = os.path.join(tmp_dir, srcname + '.tar')
                with tarfile.open(tar_filename, mode='w') as tar:
                    tar.add(host_src, arcname=srcname)

            with open(tar_filename, 'rb') as f:
                data = f.read()

            await self.container.put_archive(os.path.dirname(sandbox_dest), data)

    async def start_docker_container(self):
        try:
            container = await self.docker_client.containers.get(self.container_name)
            # if container['State']['Status'] != 'running':
            if container._container['State']['Status'] != 'running':
                await container.start()
                logger.info('Container started')
            await self.wait_for_container_ready()
        except Exception:
            logger.exception('Failed to start container')

    async def remove_docker_container(self):
        try:
            container = await self.docker_client.containers.get(self.container_name)
            if not container:
                return

            if await self.is_container_running():
                await container.stop()
                await asyncio.sleep(2)
                logger.info('Container stopped')

            try:
                await container.delete(force=True)
            except DockerError as e:
                if e.status == 404:
                    # If gone we don't care here
                    pass
                else:
                    raise

            # Wait for the container to be fully removed
            await asyncio.sleep(1)
            elapsed = 0
            while elapsed < self.timeout:
                try:
                    container_info = await container.show()
                    if container_info['State']['Status'] == 'exited':
                        break
                except DockerError:
                    # Container no longer exists
                    break
                await asyncio.sleep(2)
                elapsed += 2

            if elapsed >= self.timeout:
                logger.warning('Timeout waiting for container to be removed')
            else:
                logger.info('Container removed')
        except DockerError:
            pass

    def _create_host_config(self, use_host_network: Optional[bool]) -> Dict[str, Any]:
        host_config: Dict[str, Any] = {
            'Binds': [
                f'{host_path}:{container_path}:rw'
                for host_path, container_path in self.volumes.items()
            ],
        }

        if use_host_network:
            host_config['NetworkMode'] = 'host'
        else:
            host_config['NetworkMode'] = 'host'
            host_config['PortBindings'] = {
                f'{self._ssh_port}/tcp': [{'HostPort': str(self._ssh_port)}]
            }

        return host_config

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(DockerError),
    )
    async def restart_docker_container(self):
        try:
            # Attempt to find an existing container with the same name
            existing_container = await self.docker_client.containers.get(
                self.container_name
            )
        except DockerError as e:
            if e.status == 404:
                # Container not found, proceed with creation
                existing_container = None
            else:
                raise  # Re-raise other Docker errors

        logger.info(f'Checking for Docker image: {self.container_image}')
        try:
            await self.docker_client.images.get(self.container_image)
            logger.info(f'Image {self.container_image} found locally')
        except DockerError as img_error:
            if img_error.status == 404:
                logger.warning(f'Image {self.container_image} not found. Pulling...')
                try:
                    await self.docker_client.images.pull(self.container_image)
                    logger.info(f'Successfully pulled image {self.container_image}')
                except DockerError as pull_error:
                    logger.error(
                        f'Failed to pull image {self.container_image}: {pull_error}'
                    )
                    raise

        try:
            if existing_container:
                logger.warning(f'Replacing existing container: {self.container_name}')
                await existing_container.delete(force=True)

            self._ssh_port = find_available_tcp_port()
            logger.info(f'Using port {self._ssh_port}')

            use_host_network = self.use_host_network
            is_github_actions = os.environ.get('GITHUB_ACTIONS') == 'true'
            if is_github_actions:
                use_host_network = True
                logger.info(
                    'Detected GitHub Actions environment, enabling host network mode.'
                )

            if not use_host_network:
                # FIXME: This is a temporary workaround for Windows where host network mode has bugs.
                # FIXME: Docker Desktop for Mac OS has experimental support for host network mode
                logger.warning(
                    (
                        'Using port forwarding till the enable host network mode of Docker is out of experimental mode.\n'
                        'Check the 897th issue on https://github.com/OpenDevin/OpenDevin/issues/ for more information.'
                    )
                )

            host_config = self._create_host_config(use_host_network)
            container_config = {
                'Image': self.container_image,
                'Cmd': [
                    '/usr/sbin/sshd',
                    '-D',
                    '-p',
                    str(self._ssh_port),
                    '-o',
                    'PermitRootLogin=yes',
                ],
                'WorkingDir': self.sandbox_workspace_dir,
                'HostConfig': host_config,
            }

            # Use create_or_replace for idempotent container management
            logger.info(f'Mounting volumes: {self.volumes} now...')
            self.container = await self.docker_client.containers.create_or_replace(
                config=container_config,
                name=self.container_name,
            )
            logger.debug(f'Container created with ID: {self.container.id}')

            assert self.container is not None
            self.container_id = self.container.id

            logger.info('Container created, starting now...')
            await self.container.start()
            await asyncio.sleep(3)
            await self.wait_for_container_ready()

        except DockerError as ex:
            logger.error(f'DockerError occurred: {ex.status} - {ex.message}')
            if ex.status == 404:
                raise
            elif 'Ports are not available' in str(ex):
                logger.warning(
                    f'Port {self._ssh_port} is not available. Retrying with a new port.'
                )
                raise ex
            else:
                logger.exception(
                    f'Failed to start container. Docker error: {ex.status} - {ex.message}',
                    exc_info=True,
                )
                raise ex
        except Exception as ex:
            logger.exception(f'Failed to start container: {str(ex)}', exc_info=True)
            raise ex

    async def wait_for_container_ready(self):
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                container = await self.docker_client.containers.get(self.container_name)
                container_info = await container.show()
                state = container_info['State']

                if state['Running'] is True and state['Paused'] is False:
                    logger.info('Container is running')
                    return
                if state['Running'] is False:
                    logger.error(f'Container is not running. State: {state}')
                    logs = await container.log(stdout=True, stderr=True)
                    logger.error(f'Container logs: {logs}')
                    raise RuntimeError(f'Container failed to start. State: {state}')
            except DockerError as e:
                logger.warning(f'Error getting container info, waiting... Error: {e}')

            await asyncio.sleep(1)
            logger.info(
                f'Waiting for container to start (attempt {attempt + 1}/{max_attempts})'
            )

        raise RuntimeError('Failed to start container within the timeout period')

    async def list_containers(self, **kwargs):
        """List Docker containers using aiodocker."""
        try:
            containers = await self.docker_client.containers.list(**kwargs)
            return containers
        except Exception as e:
            logger.error(f'Error listing containers: {e}')
            return []

    async def list_docker_images(self, **kwargs) -> list:
        """List Docker images using aiodocker."""
        try:
            images = await self.docker_client.images.list(**kwargs)
            return images
        except Exception as e:
            logger.error(f'Error listing Docker images: {e}')
            return []

    async def get_working_directory(self):
        exit_code, result = await self.execute('pwd')
        if exit_code != 0:
            raise RuntimeError('Failed to get working directory')
        return str(result).strip()

    async def is_container_running(self):
        try:
            container = await self.docker_client.containers.get(self.container_name)
            if container._container['State']['Status'] == 'running':
                self.container = container
                return True
            return False
        except DockerError:
            return False

    @property
    def volumes(self):
        return {
            self.workspace_mount_path: self.sandbox_workspace_dir,
            self.cache_dir: (
                '/home/opendevin/.cache' if self.run_as_devin else '/root/.cache'
            ),
        }

    # clean up the container, cannot do it in __del__ because the python interpreter is already shutting down
    @async_to_sync
    def close(self):
        return self.aclose()

    async def aclose(self):
        if self._cleanup_done:
            return
        await self._cleanup()

    async def _cleanup(self):
        if self._cleanup_done or not hasattr(self, 'docker_client'):
            return
        try:
            containers = await self.docker_client.containers.list(all=True)
            for container in containers:
                try:
                    container_info = await container.show()
                    container_name = container_info['Name'].lstrip('/')
                    if container_name.startswith(self.container_name_prefix):
                        await container.stop()
                        await asyncio.sleep(1)
                        await container.delete(force=True)
                        await asyncio.sleep(1)
                        logger.info(f'Container {container_name} removed')
                except DockerError as e:
                    logger.error(f'Error removing container: {e}')

            # Close the Docker client
            await self.docker_client.close()

        except Exception as e:
            logger.error(f'Error during cleanup: {e}', exc_info=False)
        finally:
            self._cleanup_done = True
            atexit.unregister(self.sync_cleanup)
            DockerSSHBox._instance = None

    def sync_cleanup(self):
        if self._cleanup_done:
            return
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._cleanup())
        except Exception as e:
            logger.error(f'Error during sync cleanup: {e}', exc_info=True)
        finally:
            # Close the loop if we created a new one
            if loop != asyncio.get_event_loop():
                loop.close()

    def __del__(self):
        if hasattr(self, '_cleanup_done') and not self._cleanup_done:
            self.sync_cleanup()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sync_cleanup()


if __name__ == '__main__':
    try:
        ssh_box = DockerSSHBox(
            config=SandboxConfig(),
            run_as_devin=False,
            workspace_mount_path='/path/to/workspace',
            cache_dir='/path/to/cache',
            sandbox_workspace_dir='/sandbox',
            persist_sandbox=False,
        )
    except Exception as e:
        logger.exception('Failed to start Docker container: %s', e)
        sys.exit(1)
    logger.info(
        "Interactive Docker container started. Type 'exit' or use Ctrl+C to exit."
    )

    # Initialize required plugins
    plugins: list[PluginRequirement] = [AgentSkillsRequirement(), JupyterRequirement()]
    ssh_box.init_plugins(plugins)
    logger.info(
        '--- AgentSkills COMMAND DOCUMENTATION ---\n'
        f'{AgentSkillsRequirement().documentation}\n'
        '---'
    )

    sys.stdout.flush()
    try:
        while True:
            try:
                user_input = input('$ ')
            except EOFError:
                logger.info('Exiting...')
                break
            if user_input.lower() == 'exit':
                logger.info('Exiting...')
                break
            exit_code, output = ssh_box.execute(user_input)
            logger.info('exit code: %d', exit_code)
            logger.info(output)
            sys.stdout.flush()
    except KeyboardInterrupt:
        logger.info('Exiting...')
    finally:
        ssh_box.close()
