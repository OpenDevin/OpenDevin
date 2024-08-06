import asyncio
import os
import shutil
import subprocess

import pytest

from opendevin.controller.state.state import State
from opendevin.core.config import AppConfig, SandboxConfig, load_from_env
from opendevin.core.main import run_controller
from opendevin.core.schema import AgentState
from opendevin.events.action import (
    AgentFinishAction,
    AgentRejectAction,
)
from opendevin.events.observation.browse import BrowserOutputObservation
from opendevin.events.observation.delegate import AgentDelegateObservation
from opendevin.runtime import get_runtime_cls

TEST_RUNTIME = os.getenv('TEST_RUNTIME')
assert TEST_RUNTIME in ['eventstream', 'server']
_ = get_runtime_cls(TEST_RUNTIME)  # make sure it does not raise an error

CONFIG = AppConfig(
    max_iterations=int(os.getenv('MAX_ITERATIONS', 15)),
    max_budget_per_task=int(os.getenv('MAX_BUDGET_PER_TASK', 15)),
    runtime=TEST_RUNTIME,
    default_agent=os.getenv('DEFAULT_AGENT'),
    workspace_base=os.getenv('WORKSPACE_BASE'),
    workspace_mount_path=os.getenv('WORKSPACE_MOUNT_PATH'),
    sandbox=SandboxConfig(
        box_type=os.getenv('SANDBOX_BOX_TYPE', 'ssh'),
        use_host_network=True,
    ),
)
load_from_env(CONFIG, os.environ)

print('\nPaths used:')
print(f'workspace_base: {CONFIG.workspace_base}')
print(f'workspace_mount_path: {CONFIG.workspace_mount_path}')
print(f'workspace_mount_path_in_sandbox: {CONFIG.workspace_mount_path_in_sandbox}')
print(f'CONFIG: {CONFIG}')


def get_number_of_prompts(test_name: str):
    mock_dir = os.path.join(
        os.environ['SCRIPT_DIR'],
        'mock',
        f'{TEST_RUNTIME}_runtime',
        os.environ['DEFAULT_AGENT'],
        test_name,
    )
    prompt_files = [file for file in os.listdir(mock_dir) if file.startswith('prompt_')]
    return len(prompt_files)


def validate_final_state(final_state: State | None, test_name: str):
    assert final_state is not None
    assert final_state.agent_state == AgentState.STOPPED
    assert final_state.last_error is None
    # number of LLM conversations should be the same as number of prompt/response
    # log files under mock/[agent]/[test_name] folder. If not, it means there are
    # redundant prompt/response log files checked into the repository.
    num_of_conversations = get_number_of_prompts(test_name)
    assert num_of_conversations > 0
    # we mock the cost of every conversation to be 1 USD
    assert final_state.metrics.accumulated_cost == num_of_conversations
    if final_state.history.has_delegation():
        assert final_state.iteration > final_state.local_iteration
    else:
        assert final_state.local_iteration == final_state.iteration
        assert final_state.iteration > 0


@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') == 'BrowsingAgent',
    reason='BrowsingAgent is a specialized agent',
)
@pytest.mark.skipif(
    (
        os.getenv('DEFAULT_AGENT') == 'CodeActAgent'
        or os.getenv('DEFAULT_AGENT') == 'CodeActSWEAgent'
    )
    and os.getenv('SANDBOX_BOX_TYPE', '').lower() != 'ssh',
    reason='CodeActAgent/CodeActSWEAgent only supports ssh sandbox which is stateful',
)
@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') == 'ManagerAgent',
    reason='Manager agent is not capable of finishing this in reasonable steps yet',
)
def test_write_simple_script(current_test_name: str) -> None:
    task = "Write a shell script 'hello.sh' that prints 'hello'. Do not ask me for confirmation at any point."

    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)

    # Verify the script file exists
    assert CONFIG.workspace_base is not None
    script_path = os.path.join(CONFIG.workspace_base, 'hello.sh')
    assert os.path.exists(script_path), 'The file "hello.sh" does not exist'

    # Run the script and capture the output
    result = subprocess.run(['bash', script_path], capture_output=True, text=True)

    # Verify the output from the script
    assert (
        result.stdout.strip() == 'hello'
    ), f'Expected output "hello", but got "{result.stdout.strip()}"'


@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') == 'BrowsingAgent',
    reason='BrowsingAgent is a specialized agent',
)
@pytest.mark.skipif(
    (
        os.getenv('DEFAULT_AGENT') == 'CodeActAgent'
        or os.getenv('DEFAULT_AGENT') == 'CodeActSWEAgent'
    )
    and os.getenv('SANDBOX_BOX_TYPE', '').lower() != 'ssh',
    reason='CodeActAgent/CodeActSWEAgent only supports ssh sandbox which is stateful',
)
@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') == 'PlannerAgent',
    reason='We only keep basic tests for PlannerAgent',
)
@pytest.mark.skipif(
    os.getenv('SANDBOX_BOX_TYPE') == 'local',
    reason='local sandbox shows environment-dependent absolute path for pwd command',
)
def test_edits(current_test_name: str):
    # Copy workspace artifacts to workspace_base location
    source_dir = os.path.join(os.path.dirname(__file__), 'workspace/test_edits/')
    files = os.listdir(source_dir)
    for file in files:
        dest_file = os.path.join(CONFIG.workspace_base, file)
        if os.path.exists(dest_file):
            os.remove(dest_file)
        shutil.copy(os.path.join(source_dir, file), dest_file)

    # Execute the task
    task = 'Fix typos in bad.txt. Do not ask me for confirmation at any point.'
    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)

    # Verify bad.txt has been fixed
    text = """This is a stupid typo.
Really?
No more typos!
Enjoy!
"""
    with open(os.path.join(CONFIG.workspace_base, 'bad.txt'), 'r') as f:
        content = f.read()
    assert content.strip() == text.strip(), f'Unexpected content: {content}'


@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') != 'CodeActAgent'
    and os.getenv('DEFAULT_AGENT') != 'CodeActSWEAgent',
    reason='currently only CodeActAgent and CodeActSWEAgent have IPython (Jupyter) execution by default',
)
@pytest.mark.skipif(
    os.getenv('SANDBOX_BOX_TYPE') != 'ssh',
    reason='Currently, only ssh sandbox supports stateful tasks',
)
def test_ipython(current_test_name: str):
    # Execute the task
    task = "Use Jupyter IPython to write a text file containing 'hello world' to '/workspace/test.txt'. Do not ask me for confirmation at any point."
    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)

    # Verify the file exists
    file_path = os.path.join(CONFIG.workspace_base, 'test.txt')
    assert os.path.exists(file_path), 'The file "test.txt" does not exist'

    # Verify the file contains the expected content
    with open(file_path, 'r') as f:
        content = f.read()
    assert (
        content.strip() == 'hello world'
    ), f'Expected content "hello world", but got "{content.strip()}"'


@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') != 'ManagerAgent',
    reason='Currently, only ManagerAgent supports task rejection',
)
@pytest.mark.skipif(
    os.getenv('SANDBOX_BOX_TYPE') == 'local',
    reason='FIXME: local sandbox does not capture stderr',
)
def test_simple_task_rejection(current_test_name: str):
    # Give an impossible task to do: cannot write a commit message because
    # the workspace is not a git repo
    task = 'Write a git commit message for the current staging area. Do not ask me for confirmation at any point.'
    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)
    assert isinstance(final_state.history.get_last_action(), AgentRejectAction)


@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') != 'CodeActAgent'
    and os.getenv('DEFAULT_AGENT') != 'CodeActSWEAgent',
    reason='currently only CodeActAgent and CodeActSWEAgent have IPython (Jupyter) execution by default',
)
@pytest.mark.skipif(
    os.getenv('SANDBOX_BOX_TYPE') != 'ssh',
    reason='Currently, only ssh sandbox supports stateful tasks',
)
def test_ipython_module(current_test_name: str):
    # Execute the task
    task = "Install and import pymsgbox==1.0.9 and print it's version in /workspace/test.txt. Do not ask me for confirmation at any point."
    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)

    # Verify the file exists
    file_path = os.path.join(CONFIG.workspace_base, 'test.txt')
    assert os.path.exists(file_path), 'The file "test.txt" does not exist'

    # Verify the file contains the expected content
    with open(file_path, 'r') as f:
        content = f.read()
        print(content)
    assert (
        content.strip().split(' ')[-1] == '1.0.9'
    ), f'Expected content "1.0.9", but got "{content.strip()}"'


@pytest.mark.skipif(
    os.getenv('TEST_IN_CI') != 'true',
    reason='This test is only for CI',
)
@pytest.mark.skipif(
    os.getenv('DEFAULT_AGENT') != 'BrowsingAgent'
    and os.getenv('DEFAULT_AGENT') != 'CodeActAgent',
    reason='currently only BrowsingAgent and CodeActAgent are capable of searching the internet',
)
@pytest.mark.skipif(
    (
        os.getenv('DEFAULT_AGENT') == 'CodeActAgent'
        or os.getenv('DEFAULT_AGENT') == 'CodeActSWEAgent'
    )
    and os.getenv('SANDBOX_BOX_TYPE', '').lower() != 'ssh',
    reason='CodeActAgent/CodeActSWEAgent only supports ssh sandbox which is stateful',
)
def test_browse_internet(http_server, current_test_name: str):
    # Execute the task
    task = 'Browse localhost:8000, and tell me the ultimate answer to life. Do not ask me for confirmation at any point.'
    final_state: State | None = asyncio.run(
        run_controller(CONFIG, task, exit_on_message=True)
    )
    validate_final_state(final_state, current_test_name)

    # last action
    last_action = final_state.history.get_last_action()
    assert isinstance(last_action, AgentFinishAction)

    # last observation
    last_observation = final_state.history.get_last_observation()
    assert isinstance(
        last_observation, (BrowserOutputObservation, AgentDelegateObservation)
    )
    if isinstance(last_observation, BrowserOutputObservation):
        assert 'OpenDevin is all you need!' in last_observation.content
    elif isinstance(last_observation, AgentDelegateObservation):
        assert 'OpenDevin is all you need!' in last_observation.outputs['content']
