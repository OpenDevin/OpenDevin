import os
import tempfile
import threading
from typing import Callable, Optional
from zipfile import ZipFile

import requests
from requests.exceptions import Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.events.action import (
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
)
from openhands.events.serialization import event_to_dict, observation_from_dict
from openhands.events.serialization.action import ACTION_TYPE_TO_CLASS
from openhands.runtime.builder.remote import RemoteRuntimeBuilder
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.runtime import Runtime
from openhands.runtime.utils.request import (
    DEFAULT_RETRY_EXCEPTIONS,
    is_404_error,
    send_request_with_retry,
)
from openhands.runtime.utils.runtime_build import build_runtime_image
from openhands.utils.tenacity_stop import stop_if_should_exit


class RemoteRuntime(Runtime):
    """This runtime will connect to a remote oh-runtime-client."""

    port: int = 60000  # default port for the remote runtime client

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_message_callback: Optional[Callable] = None,
        attach_to_existing: bool = False,
    ):
        self.config = config
        self.status_message_callback = status_message_callback

        if self.config.sandbox.api_key is None:
            raise ValueError(
                'API key is required to use the remote runtime. '
                'Please set the API key in the config (config.toml) or as an environment variable (SANDBOX_API_KEY).'
            )
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': self.config.sandbox.api_key})
        self.action_semaphore = threading.Semaphore(1)

        if self.config.workspace_base is not None:
            logger.warning(
                'Setting workspace_base is not supported in the remote runtime.'
            )

        self.runtime_builder = RemoteRuntimeBuilder(
            self.config.sandbox.remote_runtime_api_url, self.config.sandbox.api_key
        )
        self.runtime_id: str | None = None
        self.runtime_url: str | None = None

        self.sid = sid

        self._start_or_attach_to_runtime(plugins, attach_to_existing)

        # Initialize the eventstream and env vars
        super().__init__(
            config,
            event_stream,
            sid,
            plugins,
            env_vars,
            status_message_callback,
            attach_to_existing,
        )
        self._wait_until_alive()
        self.setup_initial_env()

    def _start_or_attach_to_runtime(
        self, plugins: list[PluginRequirement] | None, attach_to_existing: bool = False
    ):
        existing_runtime = self._check_existing_runtime()
        if existing_runtime:
            logger.info(f'Using existing runtime with ID: {self.runtime_id}')
        elif attach_to_existing:
            raise RuntimeError('Could not find existing runtime to attach to.')
        else:
            self.send_status_message('STATUS$STARTING_CONTAINER')
            if self.config.sandbox.runtime_container_image is None:
                logger.info(
                    f'Building remote runtime with base image: {self.config.sandbox.base_container_image}'
                )
                self._build_runtime()
            else:
                logger.info(
                    f'Running remote runtime with image: {self.config.sandbox.runtime_container_image}'
                )
                self.container_image = self.config.sandbox.runtime_container_image
            self._start_runtime(plugins)
        assert (
            self.runtime_id is not None
        ), 'Runtime ID is not set. This should never happen.'
        assert (
            self.runtime_url is not None
        ), 'Runtime URL is not set. This should never happen.'
        self.send_status_message('STATUS$WAITING_FOR_CLIENT')
        self._wait_until_alive()

    def _check_existing_runtime(self) -> bool:
        try:
            response = send_request_with_retry(
                self.session,
                'GET',
                f'{self.config.sandbox.remote_runtime_api_url}/runtime/{self.sid}',
                timeout=5,
            )
        except Exception as e:
            logger.debug(f'Error while looking for remote runtime: {e}')
            return False

        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            if status == 'running':
                self._parse_runtime_response(response)
                return True
            elif status == 'stopped':
                logger.info('Found existing remote runtime, but it is stopped')
                return False
            elif status == 'paused':
                logger.info('Found existing remote runtime, but it is paused')
                self._parse_runtime_response(response)
                self._resume_runtime()
                return True
            else:
                logger.error(f'Invalid response from runtime API: {data}')
                return False
        else:
            logger.info('Could not find existing remote runtime')
            return False

    def _build_runtime(self):
        logger.debug(f'RemoteRuntime `{self.sid}` config:\n{self.config}')
        response = send_request_with_retry(
            self.session,
            'GET',
            f'{self.config.sandbox.remote_runtime_api_url}/registry_prefix',
            timeout=30,
        )
        response_json = response.json()
        registry_prefix = response_json['registry_prefix']
        os.environ['OH_RUNTIME_RUNTIME_IMAGE_REPO'] = (
            registry_prefix.rstrip('/') + '/runtime'
        )
        logger.info(
            f'Runtime image repo: {os.environ["OH_RUNTIME_RUNTIME_IMAGE_REPO"]}'
        )

        if self.config.sandbox.runtime_extra_deps:
            logger.info(
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}'
            )

        # Build the container image
        self.container_image = build_runtime_image(
            self.config.sandbox.base_container_image,
            self.runtime_builder,
            extra_deps=self.config.sandbox.runtime_extra_deps,
            force_rebuild=self.config.sandbox.force_rebuild_runtime,
        )

        response = send_request_with_retry(
            self.session,
            'GET',
            f'{self.config.sandbox.remote_runtime_api_url}/image_exists',
            params={'image': self.container_image},
            timeout=30,
        )
        if response.status_code != 200 or not response.json()['exists']:
            raise RuntimeError(f'Container image {self.container_image} does not exist')

    def _start_runtime(self, plugins: list[PluginRequirement] | None):
        # Prepare the request body for the /start endpoint
        plugin_arg = ''
        if plugins is not None and len(plugins) > 0:
            plugin_arg = f'--plugins {" ".join([plugin.name for plugin in plugins])} '
        browsergym_arg = (
            f'--browsergym-eval-env {self.config.sandbox.browsergym_eval_env}'
            if self.config.sandbox.browsergym_eval_env is not None
            else ''
        )
        start_request = {
            'image': self.container_image,
            'command': (
                f'/openhands/micromamba/bin/micromamba run -n openhands '
                'poetry run '
                f'python -u -m openhands.runtime.client.client {self.port} '
                f'--working-dir {self.config.workspace_mount_path_in_sandbox} '
                f'{plugin_arg}'
                f'--username {"openhands" if self.config.run_as_openhands else "root"} '
                f'--user-id {self.config.sandbox.user_id} '
                f'{browsergym_arg}'
            ),
            'working_dir': '/openhands/code/',
            'environment': {'DEBUG': 'true'} if self.config.debug else {},
            'runtime_id': self.sid,
        }

        # Start the sandbox using the /start endpoint
        response = send_request_with_retry(
            self.session,
            'POST',
            f'{self.config.sandbox.remote_runtime_api_url}/start',
            json=start_request,
            timeout=300,
        )
        if response.status_code != 201:
            raise RuntimeError(f'Failed to start sandbox: {response.text}')
        self._parse_runtime_response(response)
        logger.info(
            f'Sandbox started. Runtime ID: {self.runtime_id}, URL: {self.runtime_url}'
        )

    def _resume_runtime(self):
        response = send_request_with_retry(
            self.session,
            'POST',
            f'{self.config.sandbox.remote_runtime_api_url}/resume',
            json={'runtime_id': self.runtime_id},
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(f'Failed to resume sandbox: {response.text}')
        logger.info(f'Sandbox resumed. Runtime ID: {self.runtime_id}')

    def _parse_runtime_response(self, response: requests.Response):
        start_response = response.json()
        self.runtime_id = start_response['runtime_id']
        self.runtime_url = start_response['url']
        if 'session_api_key' in start_response:
            self.session.headers.update(
                {'X-Session-API-Key': start_response['session_api_key']}
            )

    @retry(
        stop=stop_after_attempt(60) | stop_if_should_exit(),
        wait=wait_fixed(2),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    def _wait_until_alive(self):
        logger.info(f'Waiting for runtime to be alive at url: {self.runtime_url}')
        response = send_request_with_retry(
            self.session,
            'GET',
            f'{self.runtime_url}/alive',
            # Retry 404 errors for the /alive endpoint
            # because the runtime might just be starting up
            # and have not registered the endpoint yet
            retry_fns=[is_404_error],
            # leave enough time for the runtime to start up
            timeout=600,
        )
        if response.status_code != 200:
            msg = f'Runtime is not alive yet (id={self.runtime_id}). Status: {response.status_code}.'
            logger.warning(msg)
            raise RuntimeError(msg)

    def close(self, timeout: int = 10):
        if self.config.sandbox.keep_remote_runtime_alive:
            self.session.close()
            return
        if self.runtime_id:
            try:
                response = send_request_with_retry(
                    self.session,
                    'POST',
                    f'{self.config.sandbox.remote_runtime_api_url}/stop',
                    json={'runtime_id': self.runtime_id},
                    timeout=timeout,
                )
                if response.status_code != 200:
                    logger.error(f'Failed to stop sandbox: {response.text}')
                else:
                    logger.info(f'Sandbox stopped. Runtime ID: {self.runtime_id}')
            except Exception as e:
                raise e
            finally:
                self.session.close()

    def run_action(self, action: Action) -> Observation:
        if action.timeout is None:
            action.timeout = self.config.sandbox.timeout
        with self.action_semaphore:
            if not action.runnable:
                return NullObservation('')
            action_type = action.action  # type: ignore[attr-defined]
            if action_type not in ACTION_TYPE_TO_CLASS:
                return ErrorObservation(f'Action {action_type} does not exist.')
            if not hasattr(self, action_type):
                return ErrorObservation(
                    f'Action {action_type} is not supported in the current runtime.'
                )

            assert action.timeout is not None

            try:
                logger.info('Executing action')
                request_body = {'action': event_to_dict(action)}
                logger.debug(f'Request body: {request_body}')
                response = send_request_with_retry(
                    self.session,
                    'POST',
                    f'{self.runtime_url}/execute_action',
                    json=request_body,
                    timeout=action.timeout,
                    retry_exceptions=list(
                        filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                    ),
                    # Retry 404 errors for the /execute_action endpoint
                    # because the runtime might just be starting up
                    # and have not registered the endpoint yet
                    retry_fns=[is_404_error],
                )
                if response.status_code == 200:
                    output = response.json()
                    obs = observation_from_dict(output)
                    obs._cause = action.id  # type: ignore[attr-defined]
                    return obs
                else:
                    error_message = response.text
                    logger.error(f'Error from server: {error_message}')
                    obs = ErrorObservation(f'Action execution failed: {error_message}')
            except Timeout:
                logger.error('No response received within the timeout period.')
                obs = ErrorObservation('Action execution timed out')
            except Exception as e:
                logger.error(f'Error during action execution: {e}')
                obs = ErrorObservation(f'Action execution failed: {str(e)}')
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

    def copy_to(
        self, host_src: str, sandbox_dest: str, recursive: bool = False
    ) -> None:
        if not os.path.exists(host_src):
            raise FileNotFoundError(f'Source file {host_src} does not exist')

        try:
            if recursive:
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
                upload_data = {'file': open(host_src, 'rb')}

            params = {'destination': sandbox_dest, 'recursive': str(recursive).lower()}

            response = send_request_with_retry(
                self.session,
                'POST',
                f'{self.runtime_url}/upload_file',
                files=upload_data,
                params=params,
                retry_exceptions=list(
                    filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                ),
                timeout=300,
            )
            if response.status_code == 200:
                logger.info(
                    f'Copy completed: host:{host_src} -> runtime:{sandbox_dest}. Response: {response.text}'
                )
                return
            else:
                error_message = response.text
                raise Exception(f'Copy operation failed: {error_message}')
        except TimeoutError:
            raise TimeoutError('Copy operation timed out')
        except Exception as e:
            raise RuntimeError(f'Copy operation failed: {str(e)}')
        finally:
            if recursive:
                os.unlink(temp_zip_path)
            logger.info(f'Copy completed: host:{host_src} -> runtime:{sandbox_dest}')

    def list_files(self, path: str | None = None) -> list[str]:
        try:
            data = {}
            if path is not None:
                data['path'] = path

            response = send_request_with_retry(
                self.session,
                'POST',
                f'{self.runtime_url}/list_files',
                json=data,
                retry_exceptions=list(
                    filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                ),
                timeout=30,
            )
            if response.status_code == 200:
                response_json = response.json()
                assert isinstance(response_json, list)
                return response_json
            else:
                error_message = response.text
                raise Exception(f'List files operation failed: {error_message}')
        except TimeoutError:
            raise TimeoutError('List files operation timed out')
        except Exception as e:
            raise RuntimeError(f'List files operation failed: {str(e)}')

    def copy_from(self, path: str) -> bytes:
        """Zip all files in the sandbox and return as a stream of bytes."""
        self._wait_until_alive()
        try:
            params = {'path': path}
            response = send_request_with_retry(
                self.session,
                'GET',
                f'{self.runtime_url}/download_files',
                params=params,
                timeout=30,
                retry_exceptions=list(
                    filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                ),
            )
            if response.status_code == 200:
                return response.content
            else:
                error_message = response.text
                raise Exception(f'Copy operation failed: {error_message}')
        except requests.Timeout:
            raise TimeoutError('Copy operation timed out')
        except Exception as e:
            raise RuntimeError(f'Copy operation failed: {str(e)}')

    def send_status_message(self, message: str):
        """Sends a status message if the callback function was provided."""
        if self.status_message_callback:
            self.status_message_callback(message)
