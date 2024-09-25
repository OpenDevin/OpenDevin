import os
import tempfile
import threading
import uuid
from typing import Callable, Optional
from zipfile import ZipFile

import requests
from requests.exceptions import Timeout
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
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
    send_request,
)
from openhands.runtime.utils.runtime_build import build_runtime_image


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
    ):
        self.config = config
        if self.config.sandbox.api_hostname == 'localhost':
            self.config.sandbox.api_hostname = 'api.all-hands.dev/v0/runtime'
            logger.warning(
                'Using localhost as the API hostname is not supported in the RemoteRuntime. Please set a proper hostname.\n'
                'Setting it to default value: api.all-hands.dev/v0/runtime'
            )
        self.api_url = f'https://{self.config.sandbox.api_hostname.rstrip("/")}'

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
            self.api_url, self.config.sandbox.api_key
        )
        self.runtime_id: str | None = None
        self.runtime_url: str | None = None

        self.instance_id = (
            sid + str(uuid.uuid4()) if sid is not None else str(uuid.uuid4())
        )
        if self.config.sandbox.runtime_container_image is not None:
            raise ValueError(
                'Setting runtime_container_image is not supported in the remote runtime.'
            )
        self.container_image: str = self.config.sandbox.base_container_image
        self.container_name = 'oh-remote-runtime-' + self.instance_id
        logger.debug(f'RemoteRuntime `{sid}` config:\n{self.config}')
        response = send_request(self.session, 'GET', f'{self.api_url}/registry_prefix')
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
            self.container_image,
            self.runtime_builder,
            extra_deps=self.config.sandbox.runtime_extra_deps,
        )

        # Use the /image_exists endpoint to check if the image exists
        response = send_request(
            self.session,
            'GET',
            f'{self.api_url}/image_exists',
            params={'image': self.container_image},
        )
        if response.status_code != 200 or not response.json()['exists']:
            raise RuntimeError(f'Container image {self.container_image} does not exist')

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
                f'/openhands/miniforge3/bin/mamba run --no-capture-output -n base '
                'PYTHONUNBUFFERED=1 poetry run '
                f'python -u -m openhands.runtime.client.client {self.port} '
                f'--working-dir {self.config.workspace_mount_path_in_sandbox} '
                f'{plugin_arg}'
                f'--username {"openhands" if self.config.run_as_openhands else "root"} '
                f'--user-id {self.config.sandbox.user_id} '
                f'{browsergym_arg}'
            ),
            'working_dir': '/openhands/code/',
            'name': self.container_name,
            'environment': {'DEBUG': 'true'} if self.config.debug else {},
        }

        # Start the sandbox using the /start endpoint
        response = send_request(
            self.session, 'POST', f'{self.api_url}/start', json=start_request
        )
        if response.status_code != 201:
            raise RuntimeError(f'Failed to start sandbox: {response.text}')
        start_response = response.json()
        self.runtime_id = start_response['runtime_id']
        self.runtime_url = start_response['url']

        logger.info(
            f'Sandbox started. Runtime ID: {self.runtime_id}, URL: {self.runtime_url}'
        )

        # Initialize the eventstream and env vars
        super().__init__(
            config, event_stream, sid, plugins, env_vars, status_message_callback
        )

        logger.info(
            f'Runtime initialized with plugins: {[plugin.name for plugin in self.plugins]}'
        )
        logger.info(f'Runtime initialized with env vars: {env_vars}')
        assert (
            self.runtime_id is not None
        ), 'Runtime ID is not set. This should never happen.'
        assert (
            self.runtime_url is not None
        ), 'Runtime URL is not set. This should never happen.'

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    def _wait_until_alive(self):
        logger.info('Waiting for sandbox to be alive...')
        response = send_request(
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

    def close(self):
        if self.runtime_id:
            try:
                response = send_request(
                    self.session,
                    'POST',
                    f'{self.api_url}/stop',
                    json={'runtime_id': self.runtime_id},
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

            self._wait_until_alive()

            assert action.timeout is not None

            try:
                logger.debug('Executing action')
                request_body = {'action': event_to_dict(action)}
                logger.debug(f'Request body: {request_body}')
                response = send_request(
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

        self._wait_until_alive()
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

            response = send_request(
                self.session,
                'POST',
                f'{self.runtime_url}/upload_file',
                files=upload_data,
                params=params,
                retry_exceptions=list(
                    filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                ),
            )
            if response.status_code == 200:
                logger.debug(
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
            logger.debug(f'Copy completed: host:{host_src} -> runtime:{sandbox_dest}')

    def list_files(self, path: str | None = None) -> list[str]:
        self._wait_until_alive()
        try:
            data = {}
            if path is not None:
                data['path'] = path

            response = send_request(
                self.session,
                'POST',
                f'{self.runtime_url}/list_files',
                json=data,
                retry_exceptions=list(
                    filter(lambda e: e != TimeoutError, DEFAULT_RETRY_EXCEPTIONS)
                ),
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
