import os
import tempfile
import threading
from typing import Callable
from zipfile import ZipFile

import docker
import requests
import tenacity

from openhands.core.config import AppConfig
from openhands.core.logger import DEBUG
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
    ActionConfirmationStatus,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from openhands.events.action.action import Action
from openhands.events.observation import (
    ErrorObservation,
    NullObservation,
    Observation,
    UserRejectObservation,
)
from openhands.events.serialization import event_to_dict, observation_from_dict
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
from openhands.runtime.builder import DockerRuntimeBuilder
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime import Runtime
from openhands.runtime.utils import find_available_tcp_port
from openhands.runtime.utils.request import send_request_with_retry
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.utils.tenacity_stop import stop_if_should_exit


class LogBuffer:
    """Synchronous buffer for Docker container logs.

    This class provides a thread-safe way to collect, store, and retrieve logs
    from a Docker container. It uses a list to store log lines and provides methods
    for appending, retrieving, and clearing logs.
    """

    def __init__(self, container: docker.models.containers.Container):
        self.client_ready = False
        self.init_msg = 'Runtime client initialized.'

        self.buffer: list[str] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.log_generator = container.logs(stream=True, follow=True)
        self.log_stream_thread = threading.Thread(target=self.stream_logs)
        self.log_stream_thread.daemon = True
        self.log_stream_thread.start()

    def append(self, log_line: str):
        with self.lock:
            self.buffer.append(log_line)

    def get_and_clear(self) -> list[str]:
        with self.lock:
            logs = list(self.buffer)
            self.buffer.clear()
            return logs

    def stream_logs(self):
        """
        Stream logs from the Docker container in a separate thread.

        This method runs in its own thread to handle the blocking
        operation of reading log lines from the Docker SDK's synchronous generator.
        """
        try:
            for log_line in self.log_generator:
                if self._stop_event.is_set():
                    break
                if log_line:
                    decoded_line = log_line.decode('utf-8').rstrip()
                    self.append(decoded_line)
                    if self.init_msg in decoded_line:
                        self.client_ready = True
        except Exception as e:
            logger.error(f'Error streaming docker logs: {e}')

    def __del__(self):
        if self.log_stream_thread.is_alive():
            logger.warn(
                "LogBuffer was not properly closed. Use 'log_buffer.close()' for clean shutdown."
            )
            self.close(timeout=5)

    def close(self, timeout: float = 5.0):
        self._stop_event.set()
        self.log_stream_thread.join(timeout)


class EventStreamRuntime(Runtime):
    """This runtime will subscribe the event stream.
    When receive an event, it will send the event to runtime-client which run inside the docker environment.

    Args:
        config (AppConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    container_name_prefix = 'openhands-runtime-'

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_message_callback: Callable | None = None,
        attach_to_existing: bool = False,
    ):
        self.config = config
        self._host_port = 30000  # initial dummy value
        self._container_port = 30001  # initial dummy value
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        self.session = requests.Session()
        self.status_message_callback = status_message_callback

        self.send_status_message('STATUS$STARTING_RUNTIME')
        self.docker_client: docker.DockerClient = self._init_docker_client()
        self.base_container_image = self.config.sandbox.base_container_image
        self.runtime_container_image = self.config.sandbox.runtime_container_image
        self.container_name = self.container_name_prefix + sid
        self.container = None
        self.action_semaphore = threading.Semaphore(1)  # Ensure one action at a time

        self.runtime_builder = DockerRuntimeBuilder(self.docker_client)

        # Buffer for container logs
        self.log_buffer: LogBuffer | None = None

        if self.config.sandbox.runtime_extra_deps:
            logger.debug(
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}'
            )

        self.skip_container_logs = (
            os.environ.get('SKIP_CONTAINER_LOGS', 'false').lower() == 'true'
        )
        if self.runtime_container_image is None:
            if self.base_container_image is None:
                raise ValueError(
                    'Neither runtime container image nor base container image is set'
                )
            logger.info('Preparing container, this might take a few minutes...')
            self.send_status_message('STATUS$STARTING_CONTAINER')
            self.runtime_container_image = build_runtime_image(
                self.base_container_image,
                self.runtime_builder,
                extra_deps=self.config.sandbox.runtime_extra_deps,
                force_rebuild=self.config.sandbox.force_rebuild_runtime,
            )

        if not attach_to_existing:
            self._init_container(
                sandbox_workspace_dir=self.config.workspace_mount_path_in_sandbox,  # e.g. /workspace
                mount_dir=self.config.workspace_mount_path,  # e.g. /opt/openhands/_test_workspace
                plugins=plugins,
            )
        else:
            self._attach_to_container()

        # will initialize both the event stream and the env vars
        super().__init__(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_message_callback,
            attach_to_existing,
        )

        logger.info('Waiting for client to become ready...')
        self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        self._wait_until_alive()

        self.setup_initial_env()

        logger.info(
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}'
        )
        self.send_status_message(' ')

    @staticmethod
    def _init_docker_client() -> docker.DockerClient:
        try:
            return docker.from_env()
        except Exception as ex:
            logger.error(
                'Launch docker client failed. Please make sure you have installed docker and started docker desktop/daemon.'
            )
            raise ex

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5) | stop_if_should_exit(),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    )
    def _init_container(
        self,
        sandbox_workspace_dir: str,
        mount_dir: str | None = None,
        plugins: list[PluginRequirement] | None = None,
    ):
        try:
            logger.info('Preparing to start container...')
            self.send_status_message('STATUS$PREPARING_CONTAINER')
            plugin_arg = ''
            if plugins is not None and len(plugins) > 0:
                plugin_arg = (
                    f'--plugins {" ".join([plugin.name for plugin in plugins])} '
                )

            self._host_port = self._find_available_port()
            self._container_port = (
                self._host_port
            )  # in future this might differ from host port
            self.api_url = (
                f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
            )

            use_host_network = self.config.sandbox.use_host_network
            network_mode: str | None = 'host' if use_host_network else None
            port_mapping: dict[str, list[dict[str, str]]] | None = (
                None
                if use_host_network
                else {
                    f'{self._container_port}/tcp': [{'HostPort': str(self._host_port)}]
                }
            )

            if use_host_network:
                logger.warn(
                    'Using host network mode. If you are using MacOS, please make sure you have the latest version of Docker Desktop and enabled host network feature: https://docs.docker.com/network/drivers/host/#docker-desktop'
                )

            # Combine environment variables
            environment = {
                'port': str(self._container_port),
                'PYTHONUNBUFFERED': 1,
            }
            if self.config.debug or DEBUG:
                environment['DEBUG'] = 'true'

            logger.debug(f'Workspace Base: {self.config.workspace_base}')
            if mount_dir is not None and sandbox_workspace_dir is not None:
                # e.g. result would be: {"/home/user/openhands/workspace": {'bind': "/workspace", 'mode': 'rw'}}
                volumes = {mount_dir: {'bind': sandbox_workspace_dir, 'mode': 'rw'}}
                logger.debug(f'Mount dir: {mount_dir}')
            else:
                logger.warn(
                    'Warning: Mount dir is not set, will not mount the workspace directory to the container!\n'
                )
                volumes = None
            logger.debug(f'Sandbox workspace: {sandbox_workspace_dir}')

            if self.config.sandbox.browsergym_eval_env is not None:
                browsergym_arg = (
                    f'--browsergym-eval-env {self.config.sandbox.browsergym_eval_env}'
                )
            else:
                browsergym_arg = ''

            self.container = self.docker_client.containers.run(
                self.runtime_container_image,
                command=(
                    f'/openhands/micromamba/bin/micromamba run -n openhands '
                    f'poetry run '
                    f'python -u -m openhands.runtime.client.client {self._container_port} '
                    f'--working-dir "{sandbox_workspace_dir}" '
                    f'{plugin_arg}'
                    f'--username {"openhands" if self.config.run_as_openhands else "root"} '
                    f'--user-id {self.config.sandbox.user_id} '
                    f'{browsergym_arg}'
                ),
                network_mode=network_mode,
                ports=port_mapping,
                working_dir='/openhands/code/',  # do not change this!
                name=self.container_name,
                detach=True,
                environment=environment,
                volumes=volumes,
            )
            self.log_buffer = LogBuffer(self.container)
            logger.info(f'Container started. Server url: {self.api_url}')
            self.send_status_message('STATUS$CONTAINER_STARTED')
        except Exception as e:
            logger.error(
                f'Error: Instance {self.container_name} FAILED to start container!\n'
            )
            logger.exception(e)
            self.close(close_client=False)
            raise e

    def _attach_to_container(self):
        container = self.docker_client.containers.get(self.container_name)
        self.log_buffer = LogBuffer(container)
        self.container = container
        self._container_port = 0
        for port in container.attrs['NetworkSettings']['Ports']:
            self._container_port = int(port.split('/')[0])
            break
        self._host_port = self._container_port
        self.api_url = f'{self.config.sandbox.local_runtime_url}:{self._container_port}'
        print(
            'attached to container:',
            self.container_name,
            self._container_port,
            self.api_url,
        )

    def _refresh_logs(self):
        logger.debug('Getting container logs...')

        assert (
            self.log_buffer is not None
        ), 'Log buffer is expected to be initialized when container is started'

        logs = self.log_buffer.get_and_clear()
        if logs:
            formatted_logs = '\n'.join([f'    |{log}' for log in logs])
            logger.info(
                '\n'
                + '-' * 35
                + 'Container logs:'
                + '-' * 35
                + f'\n{formatted_logs}'
                + '\n'
                + '-' * 80
            )

    @tenacity.retry(
        stop=tenacity.stop_after_delay(120) | stop_if_should_exit(),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=20),
        reraise=(ConnectionRefusedError,),
    )
    def _wait_until_alive(self):
        self._refresh_logs()
        if not (self.log_buffer and self.log_buffer.client_ready):
            raise RuntimeError('Runtime client is not ready.')

        response = send_request_with_retry(
            self.session,
            'GET',
            f'{self.api_url}/alive',
            retry_exceptions=[ConnectionRefusedError],
            timeout=300,  # 5 minutes gives the container time to be alive 🧟‍♂️
        )
        if response.status_code == 200:
            return
        else:
            msg = f'Action execution API is not alive. Response: {response}'
            logger.error(msg)
            raise RuntimeError(msg)

    def close(self, close_client: bool = True, rm_all_containers: bool = True):
        """Closes the EventStreamRuntime and associated objects

        Parameters:
        - close_client (bool): Whether to close the DockerClient
        - rm_all_containers (bool): Whether to remove all containers with the 'openhands-sandbox-' prefix
        """

        if self.log_buffer:
            self.log_buffer.close()

        if self.session:
            self.session.close()

        try:
            containers = self.docker_client.containers.list(all=True)
            for container in containers:
                try:
                    # If the app doesn't shut down properly, it can leave runtime containers on the system. This ensures
                    # that all 'openhands-sandbox-' containers are removed as well.
                    if rm_all_containers and container.name.startswith(
                        self.container_name_prefix
                    ):
                        container.remove(force=True)
                    elif container.name == self.container_name:
                        if not self.skip_container_logs:
                            logs = container.logs(tail=1000).decode('utf-8')
                            logger.debug(
                                f'==== Container logs on close ====\n{logs}\n==== End of container logs ===='
                            )
                        container.remove(force=True)
                except docker.errors.APIError:
                    pass
                except docker.errors.NotFound:
                    pass
        except docker.errors.NotFound:  # yes, this can happen!
            pass

        if close_client:
            self.docker_client.close()

    def run_action(self, action: Action) -> Observation:
        # set timeout to default if not set
        if action.timeout is None:
            action.timeout = self.config.sandbox.timeout

        with self.action_semaphore:
            if not action.runnable:
                return NullObservation('')
            if (
                hasattr(action, 'is_confirmed')
                and action.is_confirmed
                == ActionConfirmationStatus.AWAITING_CONFIRMATION
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

            self._refresh_logs()

            assert action.timeout is not None

            try:
                response = send_request_with_retry(
                    self.session,
                    'POST',
                    f'{self.api_url}/execute_action',
                    json={'action': event_to_dict(action)},
                    timeout=action.timeout,
                )
                if response.status_code == 200:
                    output = response.json()
                    obs = observation_from_dict(output)
                    obs._cause = action.id  # type: ignore[attr-defined]
                else:
                    logger.debug(f'action: {action}')
                    logger.debug(f'response: {response}')
                    error_message = response.text
                    logger.error(f'Error from server: {error_message}')
                    obs = ErrorObservation(f'Action execution failed: {error_message}')
            except requests.Timeout:
                logger.error('No response received within the timeout period.')
                obs = ErrorObservation(
                    f'Action execution timed out after {action.timeout} seconds.'
                )
            except Exception as e:
                logger.error(f'Error during action execution: {e}')
                obs = ErrorObservation(f'Action execution failed: {str(e)}')
            self._refresh_logs()
            return obs

    def run(self, action: CmdRunAction) -> Observation:
        return self.run_action(action)

    def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        return self.run_action(action)

    def read(self, action: FileReadAction) -> Observation:
        return self.run_action(action)

    def write(self, action: FileWriteAction) -> Observation:
        return self.run_action(action)

    def browse(self, action: BrowseURLAction) -> Observation:
        return self.run_action(action)

    def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return self.run_action(action)

    # ====================================================================
    # Implement these methods (for file operations) in the subclass
    # ====================================================================

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        if not os.path.exists(host_src):
            raise FileNotFoundError(f'Source file {host_src} does not exist')

        self._refresh_logs()
        try:
            if recursive:
                # For recursive copy, create a zip file
                with tempfile.NamedTemporaryFile(
                    suffix='.zip', delete=False
                ) as temp_zip:
                    temp_zip_path = temp_zip.name

                with ZipFile(temp_zip_path, 'w') as zipf:
                    for root, _, files in os.walk(host_src):
                        for file in files:
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(
                                file_path, os.path.dirname(host_src)
                            )
                            zipf.write(file_path, arcname)

                upload_data = {'file': open(temp_zip_path, 'rb')}
            else:
                # For single file copy
                upload_data = {'file': open(host_src, 'rb')}

            params = {'destination': sandbox_dest, 'recursive': str(recursive).lower()}

            response = send_request_with_retry(
                self.session,
                'POST',
                f'{self.api_url}/upload_file',
                files=upload_data,
                params=params,
                timeout=300,
            )
            if response.status_code == 200:
                return
            else:
                error_message = response.text
                raise Exception(f'Copy operation failed: {error_message}')

        except requests.Timeout:
            raise TimeoutError('Copy operation timed out')
        except Exception as e:
            raise RuntimeError(f'Copy operation failed: {str(e)}')
        finally:
            if recursive:
                os.unlink(temp_zip_path)
            logger.info(f'Copy completed: host:{host_src} -> runtime:{sandbox_dest}')
            self._refresh_logs()

    def list_files(self, path: str | None = None) -> list[str]:
        """List files in the sandbox.

        If path is None, list files in the sandbox's initial working directory (e.g., /workspace).
        """
        self._refresh_logs()
        try:
            data = {}
            if path is not None:
                data['path'] = path

            response = send_request_with_retry(
                self.session,
                'POST',
                f'{self.api_url}/list_files',
                json=data,
                timeout=30,  # 30 seconds because the container should already be alive
            )
            if response.status_code == 200:
                response_json = response.json()
                assert isinstance(response_json, list)
                return response_json
            else:
                error_message = response.text
                raise Exception(f'List files operation failed: {error_message}')
        except requests.Timeout:
            raise TimeoutError('List files operation timed out')
        except Exception as e:
            raise RuntimeError(f'List files operation failed: {str(e)}')

    def copy_from(self, path: str) -> bytes:
        """Zip all files in the sandbox and return as a stream of bytes."""
        self._refresh_logs()
        try:
            params = {'path': path}
            response = send_request_with_retry(
                self.session,
                'GET',
                f'{self.api_url}/download_files',
                params=params,
                stream=True,
                timeout=30,
            )
            if response.status_code == 200:
                data = response.content
                return data
            else:
                error_message = response.text
                raise Exception(f'Copy operation failed: {error_message}')
        except requests.Timeout:
            raise TimeoutError('Copy operation timed out')
        except Exception as e:
            raise RuntimeError(f'Copy operation failed: {str(e)}')

    def _is_port_in_use_docker(self, port):
        containers = self.docker_client.containers.list()
        for container in containers:
            container_ports = container.ports
            if str(port) in str(container_ports):
                return True
        return False

    def _find_available_port(self, max_attempts=5):
        port = 39999
        for _ in range(max_attempts):
            port = find_available_tcp_port(30000, 39999)
            if not self._is_port_in_use_docker(port):
                return port
        # If no port is found after max_attempts, return the last tried port
        return port

    def send_status_message(self, message: str):
        """Sends a status message if the callback function was provided."""
        if self.status_message_callback:
            self.status_message_callback(message)
