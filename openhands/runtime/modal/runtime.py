import tempfile
import threading
import uuid
from typing import Callable, Generator

import modal
import requests
import tenacity

from openhands.core.config import AppConfig
from openhands.core.logger import openhands_logger as logger
from openhands.events import EventStream
from openhands.runtime.client.runtime import EventStreamRuntime, LogBuffer
from openhands.runtime.plugins import PluginRequirement
from openhands.runtime.utils.runtime_build import put_source_tarball_to_dir


# Modal's log generator returns strings, but the upstream LogBuffer expects bytes.
def bytes_shim(string_generator) -> Generator[bytes, None, None]:
    for line in string_generator:
        yield line.encode('utf-8')


class ModalLogBuffer(LogBuffer):
    """Synchronous buffer for Modal sandbox logs.

    This class provides a thread-safe way to collect, store, and retrieve logs
    from a Modal sandbox. It uses a list to store log lines and provides methods
    for appending, retrieving, and clearing logs.
    """

    def __init__(self, sandbox: modal.Sandbox):
        self.client_ready = False
        self.init_msg = 'Runtime client initialized.'

        self.buffer: list[str] = []
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.log_generator = bytes_shim(sandbox.stderr)
        self.log_stream_thread = threading.Thread(target=self.stream_logs)
        self.log_stream_thread.daemon = True
        self.log_stream_thread.start()


class ModalRuntime(EventStreamRuntime):
    """This runtime will subscribe the event stream.

    When receive an event, it will send the event to runtime-client which run inside the Modal sandbox environment.

    Args:
        config (AppConfig): The application configuration.
        event_stream (EventStream): The event stream to subscribe to.
        sid (str, optional): The session ID. Defaults to 'default'.
        plugins (list[PluginRequirement] | None, optional): List of plugin requirements. Defaults to None.
        env_vars (dict[str, str] | None, optional): Environment variables to set. Defaults to None.
    """

    container_name_prefix = 'openhands-sandbox-'

    def __init__(
        self,
        config: AppConfig,
        event_stream: EventStream,
        sid: str = 'default',
        plugins: list[PluginRequirement] | None = None,
        env_vars: dict[str, str] | None = None,
        status_message_callback: Callable | None = None,
    ):
        assert config.modal_api_token_id, 'Modal API token id is required'
        assert config.modal_api_token_secret, 'Modal API token secret is required'

        self.config = config

        self.modal_client = modal.Client.from_credentials(
            config.modal_api_token_id, config.modal_api_token_secret
        )
        self.app = modal.App.lookup(
            'openhands', create_if_missing=True, client=self.modal_client
        )

        # workspace_base cannot be used because we can't bind mount into a sandbox.
        if self.config.workspace_base is not None:
            logger.warning(
                'Setting workspace_base is not supported in the modal runtime.'
            )

        # This value is arbitrary as it's private to the container
        self.container_port = 3000

        self.session = requests.Session()
        self.instance_id = (
            sid + '_' + str(uuid.uuid4()) if sid is not None else str(uuid.uuid4())
        )
        self.status_message_callback = status_message_callback

        self.send_status_message('STATUS$STARTING_RUNTIME')
        self.base_container_image_id = self.config.sandbox.base_container_image
        self.runtime_container_image_id = self.config.sandbox.runtime_container_image
        self.action_semaphore = threading.Semaphore(1)  # Ensure one action at a time

        logger.info(f'ModalRuntime `{self.instance_id}`')

        # Buffer for container logs
        self.log_buffer: LogBuffer | None = None

        if self.config.sandbox.runtime_extra_deps:
            logger.debug(
                f'Installing extra user-provided dependencies in the runtime image: {self.config.sandbox.runtime_extra_deps}'
            )

        self.image = self._get_image_definition(
            self.base_container_image_id,
            self.runtime_container_image_id,
            self.config.sandbox.runtime_extra_deps,
        )

        self.sandbox = self._init_sandbox(
            sandbox_workspace_dir=self.config.workspace_mount_path_in_sandbox,
            plugins=plugins,
        )

        # Will initialize both the event stream and the env vars
        self.init_base_runtime(
            config, event_stream, sid, plugins, env_vars, status_message_callback
        )

        logger.info('Waiting for client to become ready...')
        self.send_status_message('STATUS$WAITING_FOR_CLIENT')

        self._wait_until_alive()
        self.setup_initial_env()

        logger.info(
            f'Container initialized with plugins: {[plugin.name for plugin in self.plugins]}'
        )
        self.send_status_message(' ')

    def _get_image_definition(
        self,
        base_container_image_id: str | None,
        runtime_container_image_id: str | None,
        runtime_extra_deps: str | None,
    ) -> modal.Image:
        if runtime_container_image_id:
            return modal.Image.from_registry(runtime_container_image_id).run_commands(
                """
# Disable bracketed paste
# https://github.com/pexpect/pexpect/issues/669
echo "set enable-bracketed-paste off" >> /etc/inputrc && \\
echo 'export INPUTRC=/etc/inputrc' >> /etc/bash.bashrc
                """.strip()
            )
        elif base_container_image_id:
            if 'ubuntu' in base_container_image_id and (
                base_container_image_id.endswith(':latest')
                or base_container_image_id.endswith(':24.04')
            ):
                libgl_package = 'libgl1'
            else:
                libgl_package = 'libgl1-mesa-glx'

            extra_deps_line = ''
            if runtime_extra_deps:
                extra_deps_line = f'\n{runtime_extra_deps} && \\'

            # Build the source into a tarball so we can hash it efficiently
            # and avoid rebuilding the image unless the source changes.
            self.build_dir_handler = tempfile.TemporaryDirectory()
            build_dir = self.build_dir_handler.__enter__()

            tarball_path, package_version = put_source_tarball_to_dir(build_dir)

            # Commands lifted from `Dockerfile.j2`
            return (
                modal.Image.from_registry(base_container_image_id)
                .apt_install(
                    'wget',
                    'sudo',
                    'apt-utils',
                    libgl_package,
                    'libasound2-plugins',
                    'git',
                )
                # Remove UID 1000 if it's called pn--this fixes the nikolaik image for ubuntu users
                .run_commands(
                    'if getent passwd 1000 | grep -q pn; then userdel pn; fi',
                    'mkdir -p /openhands/logs && mkdir -p /openhands/poetry',
                )
                .env({'POETRY_VIRTUALENVS_PATH': '/openhands/poetry'})
                .run_commands(
                    """
if [ ! -d /openhands/miniforge3 ]; then \
wget --progress=bar:force -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
bash Miniforge3.sh -b -p /openhands/miniforge3 && \
rm Miniforge3.sh && \
chmod -R g+w /openhands/miniforge3 && \
bash -c ". /openhands/miniforge3/etc/profile.d/conda.sh && conda config --set changeps1 False && conda config --append channels conda-forge"; \
fi
""".strip(),
                    '/openhands/miniforge3/bin/mamba install conda-forge::poetry python=3.11 -y',
                )
                .copy_local_file(tarball_path, '/openhands/openhands_ai.tar.gz')
                .run_commands(
                    f'cd /openhands && tar -xzf openhands_ai.tar.gz && mv openhands_ai-{package_version} code'
                )
                .workdir('/openhands/code')
                .run_commands(
                    f"""
# Configure Poetry and create virtual environment
/openhands/miniforge3/bin/mamba run -n base poetry config virtualenvs.path /openhands/poetry && \\
/openhands/miniforge3/bin/mamba run -n base poetry env use python3.11 && \\
# Install project dependencies
/openhands/miniforge3/bin/mamba run -n base poetry install --only main,runtime --no-interaction --no-root && \\
# Update and install additional tools
apt-get update && \\
/openhands/miniforge3/bin/mamba run -n base poetry run pip install playwright && \\
/openhands/miniforge3/bin/mamba run -n base poetry run playwright install --with-deps chromium && \\
# Set environment variables
export OH_INTERPRETER_PATH=$(/openhands/miniforge3/bin/mamba run -n base poetry run python -c "import sys; print(sys.executable)") && \\
export OH_VENV_PATH=$(/openhands/miniforge3/bin/mamba run -n base poetry env info --path) && \\
# Install extra dependencies if specified {extra_deps_line}
# Clear caches
/openhands/miniforge3/bin/mamba run -n base poetry cache clear --all . && \\
# Set permissions
chmod -R g+rws /openhands/poetry && \\
mkdir -p /openhands/workspace && chmod -R g+rws,o+rw /openhands/workspace && \\
# Clean up
apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \\
/openhands/miniforge3/bin/mamba clean --all
""".strip(),
                    """
# Add the Poetry virtual environment to the bashrc
echo "export OH_INTERPRETER_PATH=\\"$OH_INTERPRETER_PATH\\"" >> /etc/bash.bashrc && \\
echo "export OH_VENV_PATH=\\"$OH_VENV_PATH\\"" >> /etc/bash.bashrc && \\
# Activate the Poetry virtual environment
echo 'source "$OH_VENV_PATH/bin/activate"' >> /etc/bash.bashrc && \\
# Use the Poetry virtual environment's Python interpreter
echo 'alias python="$OH_INTERPRETER_PATH"' >> /etc/bash.bashrc && \\
# Disable bracketed paste
# https://github.com/pexpect/pexpect/issues/669
echo "set enable-bracketed-paste off" >> /etc/inputrc && \\
echo 'export INPUTRC=/etc/inputrc' >> /etc/bash.bashrc
""".strip(),
                )
            )
        else:
            raise ValueError(
                'Neither runtime container image nor base container image is set'
            )

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    )
    def _init_sandbox(
        self,
        sandbox_workspace_dir: str,
        plugins: list[PluginRequirement] | None = None,
    ) -> modal.Sandbox:
        try:
            logger.info('Preparing to start container...')
            self.send_status_message('STATUS$PREPARING_CONTAINER')
            plugin_args = []
            if plugins is not None and len(plugins) > 0:
                plugin_args.append('--plugins')
                plugin_args.extend([plugin.name for plugin in plugins])

            # Combine environment variables
            environment: dict[str, str | None] = {
                'port': str(self.container_port),
                'PYTHONUNBUFFERED': '1',
            }
            if self.config.debug:
                environment['DEBUG'] = 'true'

            browsergym_args = []
            if self.config.sandbox.browsergym_eval_env is not None:
                browsergym_args = [
                    '-browsergym-eval-env',
                    self.config.sandbox.browsergym_eval_env,
                ]

            env_secret = modal.Secret.from_dict(environment)

            logger.debug(f'Sandbox workspace: {sandbox_workspace_dir}')
            sandbox_start_cmd: list[str] = [
                '/openhands/miniforge3/bin/mamba',
                'run',
                '--no-capture-output',
                '-n',
                'base',
                'poetry',
                'run',
                'python',
                '-u',
                '-m',
                'openhands.runtime.client.client',
                str(self.container_port),
                '--working-dir',
                sandbox_workspace_dir,
                *plugin_args,
                '--username',
                'openhands' if self.config.run_as_openhands else 'root',
                '--user-id',
                str(self.config.sandbox.user_id),
                *browsergym_args,
            ]

            sandbox = modal.Sandbox.create(
                *sandbox_start_cmd,
                secrets=[env_secret],
                workdir='/openhands/code',
                encrypted_ports=[self.container_port],
                image=self.image,
                app=self.app,
                client=self.modal_client,
                timeout=60 * 60,
            )

            tunnel = sandbox.tunnels()[self.container_port]
            self.api_url = tunnel.url

            self.log_buffer = ModalLogBuffer(sandbox)
            logger.info(f'Container started. Server url: {self.api_url}')
            self.send_status_message('STATUS$CONTAINER_STARTED')
            return sandbox
        except Exception as e:
            logger.error(
                f'Error: Instance {self.instance_id} FAILED to start container!\n'
            )
            logger.exception(e)
            self.close()
            raise e

    def close(self):
        """Closes the ModalRuntime and associated objects."""
        # if self.temp_dir_handler:
        # self.temp_dir_handler.__exit__(None, None, None)

        if self.log_buffer:
            self.log_buffer.close()

        if self.session:
            self.session.close()

        if self.sandbox:
            self.sandbox.terminate()
