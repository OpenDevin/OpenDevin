"""Test the EventStreamRuntime, which connects to the RuntimeClient running in the sandbox."""

import asyncio
import json
import os
import tempfile
import time
from unittest.mock import patch

import pytest
from pytest import TempPathFactory

from opendevin.core.config import AppConfig, SandboxConfig, load_from_env
from opendevin.core.logger import opendevin_logger as logger
from opendevin.events import EventStream
from opendevin.events.action import (
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from opendevin.events.observation import (
    BrowserOutputObservation,
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
)
from opendevin.runtime.client.runtime import EventStreamRuntime
from opendevin.runtime.plugins import AgentSkillsRequirement, JupyterRequirement
from opendevin.runtime.runtime import Runtime
from opendevin.storage import get_file_store


@pytest.fixture(autouse=True)
def print_method_name(request):
    print('\n########################################################################')
    print(f'Running test: {request.node.name}')
    print('########################################################################')
    yield


@pytest.fixture
def temp_dir(tmp_path_factory: TempPathFactory) -> str:
    return str(tmp_path_factory.mktemp('test_runtime'))


TEST_RUNTIME = os.getenv('TEST_RUNTIME', 'both')
PY3_FOR_TESTING = '/opendevin/miniforge3/bin/mamba run -n base python3'


# Depending on TEST_RUNTIME, feed the appropriate box class(es) to the test.
def get_box_classes():
    runtime = TEST_RUNTIME
    if runtime.lower() == 'eventstream':
        return [EventStreamRuntime]
    else:
        return [EventStreamRuntime]


# This assures that all tests run together per runtime, not alternating between them,
# which cause errors (especially outside GitHub actions).
@pytest.fixture(scope='module', params=get_box_classes())
def box_class(request):
    time.sleep(2)
    return request.param


# TODO: We will change this to `run_as_user` when `ServerRuntime` is deprecated.
# since `EventStreamRuntime` supports running as an arbitrary user.
@pytest.fixture(scope='module', params=[True, False])
def run_as_devin(request):
    time.sleep(1)
    return request.param


@pytest.fixture(scope='module', params=[True, False])
def enable_auto_lint(request):
    time.sleep(1)
    return request.param


@pytest.fixture(
    scope='module', params=['nikolaik/python-nodejs:python3.11-nodejs22', 'debian:11']
)
def container_image(request):
    time.sleep(1)
    return request.param


async def _load_runtime(
    temp_dir,
    box_class,
    run_as_devin: bool = True,
    enable_auto_lint: bool = False,
    container_image: str | None = None,
    browsergym_eval_env: str | None = None,
) -> Runtime:
    sid = 'test'
    cli_session = 'main_test'
    # AgentSkills need to be initialized **before** Jupyter
    # otherwise Jupyter will not access the proper dependencies installed by AgentSkills
    plugins = [AgentSkillsRequirement(), JupyterRequirement()]
    config = AppConfig(
        workspace_base=temp_dir,
        workspace_mount_path=temp_dir,
        sandbox=SandboxConfig(
            use_host_network=True,
            browsergym_eval_env=browsergym_eval_env,
        ),
    )
    load_from_env(config, os.environ)
    config.run_as_devin = run_as_devin
    config.sandbox.enable_auto_lint = enable_auto_lint

    file_store = get_file_store(config.file_store, config.file_store_path)
    event_stream = EventStream(cli_session, file_store)

    if container_image is not None:
        config.sandbox.container_image = container_image

    if box_class == EventStreamRuntime:
        # NOTE: we will use the default container image specified in the config.sandbox
        # if it is an official od_runtime image.
        cur_container_image = config.sandbox.container_image
        if 'od_runtime' not in cur_container_image and cur_container_image not in {
            'xingyaoww/od-eval-miniwob:v1.0'
        }:  # a special exception list
            cur_container_image = 'nikolaik/python-nodejs:python3.11-nodejs22'
            logger.warning(
                f'`{config.sandbox.container_image}` is not an od_runtime image. Will use `{cur_container_image}` as the container image for testing.'
            )

        runtime = EventStreamRuntime(
            config=config,
            event_stream=event_stream,
            sid=sid,
            plugins=plugins,
            # NOTE: we probably don't have a default container image `/sandbox` for the event stream runtime
            # Instead, we will pre-build a suite of container images with OD-runtime-cli installed.
            container_image=cur_container_image,
        )
        await runtime.ainit()

    else:
        raise ValueError(f'Invalid box class: {box_class}')
    await asyncio.sleep(1)
    return runtime


@pytest.mark.asyncio
async def test_env_vars_os_environ(temp_dir, box_class, run_as_devin):
    with patch.dict(os.environ, {'SANDBOX_ENV_FOOBAR': 'BAZ'}):
        runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

        obs: CmdOutputObservation = await runtime.run_action(
            CmdRunAction(command='env')
        )
        print(obs)

        obs: CmdOutputObservation = await runtime.run_action(
            CmdRunAction(command='echo $FOOBAR')
        )
        print(obs)
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert (
            obs.content.strip().split('\n\r')[0].strip() == 'BAZ'
        ), f'Output: [{obs.content}] for {box_class}'

        await runtime.close()
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_env_vars_runtime_add_env_vars(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)
    await runtime.add_env_vars({'QUUX': 'abc"def'})

    obs: CmdOutputObservation = await runtime.run_action(
        CmdRunAction(command='echo $QUUX')
    )
    print(obs)
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert (
        obs.content.strip().split('\r\n')[0].strip() == 'abc"def'
    ), f'Output: [{obs.content}] for {box_class}'

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_env_vars_runtime_add_empty_dict(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    prev_obs = await runtime.run_action(CmdRunAction(command='env'))
    assert prev_obs.exit_code == 0, 'The exit code should be 0.'
    print(prev_obs)

    await runtime.add_env_vars({})

    obs = await runtime.run_action(CmdRunAction(command='env'))
    assert obs.exit_code == 0, 'The exit code should be 0.'
    print(obs)
    assert (
        obs.content == prev_obs.content
    ), 'The env var content should be the same after adding an empty dict.'

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_env_vars_runtime_add_multiple_env_vars(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)
    await runtime.add_env_vars({'QUUX': 'abc"def', 'FOOBAR': 'xyz'})

    obs: CmdOutputObservation = await runtime.run_action(
        CmdRunAction(command='echo $QUUX $FOOBAR')
    )
    print(obs)
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert (
        obs.content.strip().split('\r\n')[0].strip() == 'abc"def xyz'
    ), f'Output: [{obs.content}] for {box_class}'

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_env_vars_runtime_add_env_vars_overwrite(temp_dir, box_class):
    with patch.dict(os.environ, {'SANDBOX_ENV_FOOBAR': 'BAZ'}):
        runtime = await _load_runtime(temp_dir, box_class)
        await runtime.add_env_vars({'FOOBAR': 'xyz'})

        obs: CmdOutputObservation = await runtime.run_action(
            CmdRunAction(command='echo $FOOBAR')
        )
        print(obs)
        assert obs.exit_code == 0, 'The exit code should be 0.'
        assert (
            obs.content.strip().split('\r\n')[0].strip() == 'xyz'
        ), f'Output: [{obs.content}] for {box_class}'

        await runtime.close()
        await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_bash_command_pexcept(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    # We set env var PS1="\u@\h:\w $"
    # and construct the PEXCEPT prompt base on it.
    # When run `env`, bad implementation of CmdRunAction will be pexcepted by this
    # and failed to pexcept the right content, causing it fail to get error code.
    obs = await runtime.run_action(CmdRunAction(command='env'))

    # For example:
    # 02:16:13 - opendevin:DEBUG: client.py:78 - Executing command: env
    # 02:16:13 - opendevin:DEBUG: client.py:82 - Command output: PYTHONUNBUFFERED=1
    # CONDA_EXE=/opendevin/miniforge3/bin/conda
    # [...]
    # LC_CTYPE=C.UTF-8
    # PS1=\u@\h:\w $
    # 02:16:13 - opendevin:DEBUG: client.py:89 - Executing command for exit code: env
    # 02:16:13 - opendevin:DEBUG: client.py:92 - Exit code Output:
    # CONDA_DEFAULT_ENV=base

    # As long as the exit code is 0, the test will pass.
    assert isinstance(
        obs, CmdOutputObservation
    ), 'The observation should be a CmdOutputObservation.'
    assert obs.exit_code == 0, 'The exit code should be 0.'

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_simple_cmd_ipython_and_fileop(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    # Test run command
    action_cmd = CmdRunAction(command='ls -l')
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'total 0' in obs.content

    # Test run ipython
    test_code = "print('Hello, `World`!\\n')"
    action_ipython = IPythonRunCellAction(code=test_code)
    logger.info(action_ipython, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_ipython)
    assert isinstance(obs, IPythonRunCellObservation)

    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.content.strip() == 'Hello, `World`!'

    # Test read file (file should not exist)
    action_read = FileReadAction(path='hello.sh')
    logger.info(action_read, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_read)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, ErrorObservation)
    assert 'File not found' in obs.content

    # Test write file
    action_write = FileWriteAction(content='echo "Hello, World!"', path='hello.sh')
    logger.info(action_write, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_write)
    assert isinstance(obs, FileWriteObservation)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert obs.content == ''
    # event stream runtime will always use absolute path
    assert obs.path == '/workspace/hello.sh'

    # Test read file (file should exist)
    action_read = FileReadAction(path='hello.sh')
    logger.info(action_read, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_read)
    assert isinstance(
        obs, FileReadObservation
    ), 'The observation should be a FileReadObservation.'
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert obs.content == 'echo "Hello, World!"\n'
    assert obs.path == '/workspace/hello.sh'

    # clean up
    action = CmdRunAction(command='rm -rf hello.sh')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_simple_browse(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    # Test browse
    action_cmd = CmdRunAction(
        command=f'{PY3_FOR_TESTING} -m http.server 8000 > server.log 2>&1 &'
    )
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert '[1]' in obs.content

    action_cmd = CmdRunAction(command='sleep 5 && cat server.log')
    logger.info(action_cmd, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_cmd)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action_browse = BrowseURLAction(url='http://localhost:8000')
    logger.info(action_browse, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_browse)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, BrowserOutputObservation)
    assert 'http://localhost:8000' in obs.url
    assert not obs.error
    assert obs.open_pages_urls == ['http://localhost:8000/']
    assert obs.active_page_index == 0
    assert obs.last_browser_action == 'goto("http://localhost:8000")'
    assert obs.last_browser_action_error == ''
    assert 'Directory listing for /' in obs.content
    assert 'server.log' in obs.content

    # clean up
    action = CmdRunAction(command='rm -rf server.log')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_browsergym_eval_env(temp_dir):
    runtime = await _load_runtime(
        temp_dir,
        # only supported in event stream runtime
        box_class=EventStreamRuntime,
        run_as_devin=False,  # need root permission to access file
        container_image='xingyaoww/od-eval-miniwob:v1.0',
        browsergym_eval_env='browsergym/miniwob.choose-list',
    )
    from opendevin.runtime.browser.browser_env import (
        BROWSER_EVAL_GET_GOAL_ACTION,
        BROWSER_EVAL_GET_REWARDS_ACTION,
    )

    # Test browse
    action = BrowseInteractiveAction(browser_actions=BROWSER_EVAL_GET_GOAL_ACTION)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, BrowserOutputObservation)
    assert not obs.error
    assert 'Select' in obs.content
    assert 'from the list and click Submit' in obs.content

    # Make sure the browser can produce observation in eva[l
    action = BrowseInteractiveAction(browser_actions='noop()')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert (
        obs.url.strip()
        == 'file:///miniwob-plusplus/miniwob/html/miniwob/choose-list.html'
    )

    # Make sure the rewards are working
    action = BrowseInteractiveAction(browser_actions=BROWSER_EVAL_GET_REWARDS_ACTION)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert json.loads(obs.content) == [0.0]

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_single_multiline_command(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='echo \\\n -e "foo"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert 'foo' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_multiline_echo(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='echo -e "hello\nworld"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert 'hello\r\nworld' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_runtime_whitespace(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='echo -e "\\n\\n\\n"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert '\r\n\r\n\r\n' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_multiple_multiline_commands(temp_dir, box_class, run_as_devin):
    cmds = [
        'ls -l',
        'echo -e "hello\nworld"',
        """
echo -e "hello it\\'s me"
""".strip(),
        """
echo \\
    -e 'hello' \\
    -v
""".strip(),
        """
echo -e 'hello\\nworld\\nare\\nyou\\nthere?'
""".strip(),
        """
echo -e 'hello
world
are
you\\n
there?'
""".strip(),
        """
echo -e 'hello
world "
'
""".strip(),
    ]
    joined_cmds = '\n'.join(cmds)

    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    action = CmdRunAction(command=joined_cmds)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'

    assert 'total 0' in obs.content
    assert 'hello\r\nworld' in obs.content
    assert "hello it\\'s me" in obs.content
    assert 'hello -v' in obs.content
    assert 'hello\r\nworld\r\nare\r\nyou\r\nthere?' in obs.content
    assert 'hello\r\nworld\r\nare\r\nyou\r\n\r\nthere?' in obs.content
    assert 'hello\r\nworld "\r\n' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_no_ps2_in_output(temp_dir, box_class, run_as_devin):
    """Test that the PS2 sign is not added to the output of a multiline command."""
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    action = CmdRunAction(command='echo -e "hello\nworld"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert 'hello\r\nworld' in obs.content
    assert '>' not in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_multiline_command_loop(temp_dir, box_class):
    # https://github.com/OpenDevin/OpenDevin/issues/3143

    runtime = await _load_runtime(temp_dir, box_class)

    init_cmd = """
mkdir -p _modules && \
for month in {01..04}; do
    for day in {01..05}; do
        touch "_modules/2024-${month}-${day}-sample.md"
    done
done
echo "created files"
"""
    action = CmdRunAction(command=init_cmd)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert 'created files' in obs.content

    follow_up_cmd = """
for file in _modules/*.md; do
    new_date=$(echo $file | sed -E 's/2024-(01|02|03|04)-/2024-/;s/2024-01/2024-08/;s/2024-02/2024-09/;s/2024-03/2024-10/;s/2024-04/2024-11/')
    mv "$file" "$new_date"
done
echo "success"
"""
    action = CmdRunAction(command=follow_up_cmd)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert 'success' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_cmd_run(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    action = CmdRunAction(command='ls -l')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'total 0' in obs.content

    action = CmdRunAction(command='mkdir test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    action = CmdRunAction(command='ls -l')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    if run_as_devin:
        assert 'opendevin' in obs.content
    else:
        assert 'root' in obs.content
    assert 'test' in obs.content

    action = CmdRunAction(command='touch test/foo.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    action = CmdRunAction(command='ls -l test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'foo.txt' in obs.content

    # clean up: this is needed, since CI will not be
    # run as root, and this test may leave a file
    # owned by root
    action = CmdRunAction(command='rm -rf test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_run_as_user_correct_home_dir(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    action = CmdRunAction(command='cd ~ && pwd')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    if run_as_devin:
        assert '/home/opendevin' in obs.content
    else:
        assert '/root' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_multi_cmd_run_in_single_line(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='pwd && ls -l')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert '/workspace' in obs.content
    assert 'total 0' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_stateful_cmd(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='mkdir test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'

    action = CmdRunAction(command='cd test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'

    action = CmdRunAction(command='pwd')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0, 'The exit code should be 0.'
    assert '/workspace/test' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_failed_cmd(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='non_existing_command')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code != 0, 'The exit code should not be 0 for a failed command.'

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_ipython_multi_user(temp_dir, box_class, run_as_devin):
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    # Test run ipython
    # get username
    test_code = "import os; print(os.environ['USER'])"
    action_ipython = IPythonRunCellAction(code=test_code)
    logger.info(action_ipython, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_ipython)
    assert isinstance(obs, IPythonRunCellObservation)

    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if run_as_devin:
        assert 'opendevin' in obs.content
    else:
        assert 'root' in obs.content

    # print pwd
    test_code = 'import os; print(os.getcwd())'
    action_ipython = IPythonRunCellAction(code=test_code)
    logger.info(action_ipython, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_ipython)
    assert isinstance(obs, IPythonRunCellObservation)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.content.strip() == '/workspace'

    # write a file
    test_code = "with open('test.txt', 'w') as f: f.write('Hello, world!')"
    action_ipython = IPythonRunCellAction(code=test_code)
    logger.info(action_ipython, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_ipython)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.strip() == '[Code executed successfully with no output]'

    # check file owner via bash
    action = CmdRunAction(command='ls -alh test.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    if run_as_devin:
        # -rw-r--r-- 1 opendevin root 13 Jul 28 03:53 test.txt
        assert 'opendevin' in obs.content.split('\r\n')[0]
        assert 'root' in obs.content.split('\r\n')[0]
    else:
        # -rw-r--r-- 1 root root 13 Jul 28 03:53 test.txt
        assert 'root' in obs.content.split('\r\n')[0]

    # clean up
    action = CmdRunAction(command='rm -rf test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_ipython_simple(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    # Test run ipython
    # get username
    test_code = 'print(1)'
    action_ipython = IPythonRunCellAction(code=test_code)
    logger.info(action_ipython, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action_ipython)
    assert isinstance(obs, IPythonRunCellObservation)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.content.strip() == '1'

    await runtime.close()
    await asyncio.sleep(1)


async def _test_ipython_agentskills_fileop_pwd_impl(
    runtime: EventStreamRuntime, enable_auto_lint: bool
):
    # remove everything in /workspace
    action = CmdRunAction(command='rm -rf /workspace/*')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='mkdir test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    action = IPythonRunCellAction(code="create_file('hello.py')")
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
        '[File: /workspace/hello.py (1 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|\n'
        '(this is the end of the file)\n'
        '[File hello.py created.]\n'
    ).strip().split('\n')

    action = CmdRunAction(command='cd test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # This should create a file in the current working directory
    # i.e., /workspace/test/hello.py instead of /workspace/hello.py
    action = IPythonRunCellAction(code="create_file('hello.py')")
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
        '[File: /workspace/test/hello.py (1 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|\n'
        '(this is the end of the file)\n'
        '[File hello.py created.]\n'
    ).strip().split('\n')

    if enable_auto_lint:
        # edit file, but make a mistake in indentation
        action = IPythonRunCellAction(
            code="insert_content_at_line('hello.py', 1, '  print(\"hello world\")')"
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = await runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        assert isinstance(obs, IPythonRunCellObservation)
        assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
            """
[Your proposed edit has introduced new syntax error(s). Please understand the errors and retry your edit command.]
ERRORS:
/workspace/test/hello.py:1:3: E999 IndentationError: unexpected indent
[This is how your edit would have looked if applied]
-------------------------------------------------
(this is the beginning of the file)
1|  print("hello world")
(this is the end of the file)
-------------------------------------------------

[This is the original code before your edit]
-------------------------------------------------
(this is the beginning of the file)
1|
(this is the end of the file)
-------------------------------------------------
Your changes have NOT been applied. Please fix your edit command and try again.
You either need to 1) Specify the correct start/end line arguments or 2) Correct your edit code.
DO NOT re-run the same failed edit command. Running it again will lead to the same error.
"""
        ).strip().split('\n')

    # edit file with correct indentation
    action = IPythonRunCellAction(
        code="insert_content_at_line('hello.py', 1, 'print(\"hello world\")')"
    )
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
        """
[File: /workspace/test/hello.py (1 lines total after edit)]
(this is the beginning of the file)
1|print("hello world")
(this is the end of the file)
[File updated (edited at line 1). Please review the changes and make sure they are correct (correct indentation, no duplicate lines, etc). Edit the file again if necessary.]
"""
    ).strip().split('\n')

    action = CmdRunAction(command='rm -rf /workspace/*')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_ipython_agentskills_fileop_pwd(
    temp_dir, box_class, run_as_devin, enable_auto_lint
):
    """Make sure that cd in bash also update the current working directory in ipython."""

    runtime = await _load_runtime(
        temp_dir, box_class, run_as_devin, enable_auto_lint=enable_auto_lint
    )
    await _test_ipython_agentskills_fileop_pwd_impl(runtime, enable_auto_lint)

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_ipython_agentskills_fileop_pwd_with_userdir(temp_dir, box_class):
    """Make sure that cd in bash also update the current working directory in ipython.

    Handle special case where the pwd is provided as "~", which should be expanded using os.path.expanduser
    on the client side.
    """

    runtime = await _load_runtime(
        temp_dir,
        box_class,
        run_as_devin=False,
    )

    action = CmdRunAction(command='cd ~')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='mkdir test && ls -la')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    action = IPythonRunCellAction(code="create_file('hello.py')")
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
        '[File: /root/hello.py (1 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|\n'
        '(this is the end of the file)\n'
        '[File hello.py created.]\n'
    ).strip().split('\n')

    action = CmdRunAction(command='cd test')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # This should create a file in the current working directory
    # i.e., /workspace/test/hello.py instead of /workspace/hello.py
    action = IPythonRunCellAction(code="create_file('hello.py')")
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, IPythonRunCellObservation)
    assert obs.content.replace('\r\n', '\n').strip().split('\n') == (
        '[File: /root/test/hello.py (1 lines total)]\n'
        '(this is the beginning of the file)\n'
        '1|\n'
        '(this is the end of the file)\n'
        '[File hello.py created.]\n'
    ).strip().split('\n')

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_bash_python_version(temp_dir, box_class):
    """Make sure Python is available in bash."""

    runtime = await _load_runtime(temp_dir, box_class)

    action = CmdRunAction(command='which python')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0

    action = CmdRunAction(command='python --version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    # Should not error out

    action = CmdRunAction(command='pip --version')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert obs.exit_code == 0
    # Should not error out

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_ipython_package_install(temp_dir, box_class, run_as_devin):
    """Make sure that cd in bash also update the current working directory in ipython."""
    runtime = await _load_runtime(temp_dir, box_class, run_as_devin)

    # It should error out since pymsgbox is not installed
    action = IPythonRunCellAction(code='import pymsgbox')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert "ModuleNotFoundError: No module named 'pymsgbox'" in obs.content

    # Install pymsgbox in Jupyter
    action = IPythonRunCellAction(code='%pip install pymsgbox==1.0.9')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert (
        'Successfully installed pymsgbox-1.0.9' in obs.content
        or '[Package installed successfully]' in obs.content
    )

    action = IPythonRunCellAction(code='import pymsgbox')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    # import should not error out
    assert obs.content.strip() == '[Code executed successfully with no output]'

    await runtime.close()
    await asyncio.sleep(1)


def _create_test_file(host_temp_dir):
    # Single file
    with open(os.path.join(host_temp_dir, 'test_file.txt'), 'w') as f:
        f.write('Hello, World!')


@pytest.mark.asyncio
async def test_copy_single_file(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    with tempfile.TemporaryDirectory() as host_temp_dir:
        _create_test_file(host_temp_dir)
        await runtime.copy_to(
            os.path.join(host_temp_dir, 'test_file.txt'), '/workspace'
        )

    action = CmdRunAction(command='ls -alh /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'test_file.txt' in obs.content

    action = CmdRunAction(command='cat /workspace/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'Hello, World!' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


def _create_test_dir_with_files(host_temp_dir):
    os.mkdir(os.path.join(host_temp_dir, 'test_dir'))
    with open(os.path.join(host_temp_dir, 'test_dir', 'file1.txt'), 'w') as f:
        f.write('File 1 content')
    with open(os.path.join(host_temp_dir, 'test_dir', 'file2.txt'), 'w') as f:
        f.write('File 2 content')


@pytest.mark.asyncio
async def test_copy_directory_recursively(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    with tempfile.TemporaryDirectory() as host_temp_dir:
        # We need a separate directory, since temp_dir is mounted to /workspace
        _create_test_dir_with_files(host_temp_dir)
        await runtime.copy_to(
            os.path.join(host_temp_dir, 'test_dir'), '/workspace', recursive=True
        )

    action = CmdRunAction(command='ls -alh /workspace')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'test_dir' in obs.content
    assert 'file1.txt' not in obs.content
    assert 'file2.txt' not in obs.content

    action = CmdRunAction(command='ls -alh /workspace/test_dir')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'file1.txt' in obs.content
    assert 'file2.txt' in obs.content

    action = CmdRunAction(command='cat /workspace/test_dir/file1.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'File 1 content' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_copy_to_non_existent_directory(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    with tempfile.TemporaryDirectory() as host_temp_dir:
        _create_test_file(host_temp_dir)
        await runtime.copy_to(
            os.path.join(host_temp_dir, 'test_file.txt'), '/workspace/new_dir'
        )

    action = CmdRunAction(command='cat /workspace/new_dir/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'Hello, World!' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_overwrite_existing_file(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    # touch a file in /workspace
    action = CmdRunAction(command='touch /workspace/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    action = CmdRunAction(command='cat /workspace/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'Hello, World!' not in obs.content

    with tempfile.TemporaryDirectory() as host_temp_dir:
        _create_test_file(host_temp_dir)
        await runtime.copy_to(
            os.path.join(host_temp_dir, 'test_file.txt'), '/workspace'
        )

    action = CmdRunAction(command='cat /workspace/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'Hello, World!' in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_copy_non_existent_file(temp_dir, box_class):
    runtime = await _load_runtime(temp_dir, box_class)

    with pytest.raises(FileNotFoundError):
        await runtime.copy_to(
            os.path.join(temp_dir, 'non_existent_file.txt'),
            '/workspace/should_not_exist.txt',
        )

    action = CmdRunAction(command='ls /workspace/should_not_exist.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code != 0  # File should not exist

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_keep_prompt(temp_dir):
    # only EventStreamRuntime supports keep_prompt
    runtime = await _load_runtime(
        temp_dir, box_class=EventStreamRuntime, run_as_devin=False
    )

    action = CmdRunAction(command='touch /workspace/test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'root@' in obs.content

    action = CmdRunAction(command='cat /workspace/test_file.txt', keep_prompt=False)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    assert 'root@' not in obs.content

    await runtime.close()
    await asyncio.sleep(1)


@pytest.mark.asyncio
async def test_git_operation(box_class):
    # do not mount workspace, since workspace mount by tests will be owned by root
    # while the user_id we get via os.getuid() is different from root
    # which causes permission issues
    runtime = await _load_runtime(
        temp_dir=None,
        box_class=box_class,
        # Need to use non-root user to expose issues
        run_as_devin=True,
    )

    # this will happen if permission of runtime is not properly configured
    # fatal: detected dubious ownership in repository at '/workspace'

    # check the ownership of the current directory
    action = CmdRunAction(command='ls -alh .')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0
    # drwx--S--- 2 opendevin root   64 Aug  7 23:32 .
    # drwxr-xr-x 1 root      root 4.0K Aug  7 23:33 ..
    for line in obs.content.split('\r\n'):
        if ' ..' in line:
            # parent directory should be owned by root
            assert 'root' in line
            assert 'opendevin' not in line
        elif ' .' in line:
            # current directory should be owned by opendevin
            # and its group should be root
            assert 'opendevin' in line
            assert 'root' in line

    # make sure all git operations are allowed
    action = CmdRunAction(command='git init')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # create a file
    action = CmdRunAction(command='echo "hello" > test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # git add
    action = CmdRunAction(command='git add test_file.txt')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # git diff
    action = CmdRunAction(command='git diff')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    # git commit
    action = CmdRunAction(command='git commit -m "test commit"')
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = await runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert isinstance(obs, CmdOutputObservation)
    assert obs.exit_code == 0

    await runtime.close()

    await runtime.close()
    await asyncio.sleep(1)
