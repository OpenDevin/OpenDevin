import io
import os
import re
import shutil
import subprocess
import tempfile
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from threading import Thread

import pytest
from litellm import completion

from opendevin.llm.llm import message_separator

script_dir = os.environ.get('SCRIPT_DIR')
project_root = os.environ.get('PROJECT_ROOT')
workspace_path = os.environ.get('WORKSPACE_BASE')
test_runtime = os.environ.get('TEST_RUNTIME')
MOCK_ROOT_DIR = os.path.join(
    script_dir,
    'mock',
    f'{test_runtime}_runtime',
    os.environ.get('DEFAULT_AGENT'),
)

assert script_dir is not None, 'SCRIPT_DIR environment variable is not set'
assert project_root is not None, 'PROJECT_ROOT environment variable is not set'
assert workspace_path is not None, 'WORKSPACE_BASE environment variable is not set'
assert test_runtime is not None, 'TEST_RUNTIME environment variable is not set'


class SecretExit(Exception):
    pass


@pytest.hookimpl(tryfirst=True)
def pytest_exception_interact(node, call, report):
    if isinstance(call.excinfo.value, SecretExit):
        report.outcome = 'failed'
        report.longrepr = (
            'SecretExit: Exiting due to an error without revealing secrets.'
        )
        call.excinfo = None


def filter_out_symbols(input):
    # remove shell hostname patterns (e.g., will change between each run)
    # opendevin@379c7fce40b4:/workspace $
    input = re.sub(r'(opendevin|root)@.*(:/.*)', r'\1[DUMMY_HOSTNAME]\2', input)

    # handle sha256 hashes
    # sha256=4ecf8be428f55981e2a188f510ba5f9022bed88f5fb404d7d949f44382201e3d
    input = re.sub(r'sha256=[a-z0-9]+', 'sha256=[DUMMY_HASH]', input)

    # remove newlines and whitespace
    input = re.sub(r'\\n|\\r\\n|\\r|\s+', '', input)

    # remove all non-alphanumeric characters
    input = re.sub(r'[^a-zA-Z0-9]', '', input)
    return input


def get_log_id(prompt_log_name):
    match = re.search(r'prompt_(\d+).log', prompt_log_name)
    if match:
        return match.group(1)


def apply_prompt_and_get_mock_response(test_name: str, messages: str, id: int) -> str:
    """Apply the mock prompt, and find mock response based on id.
    If there is no matching response file, return None.

    Note: this function blindly replaces existing prompt file with the given
    input without checking the contents.
    """
    mock_dir = os.path.join(MOCK_ROOT_DIR, test_name)
    prompt_file_path = os.path.join(mock_dir, f'prompt_{"{0:03}".format(id)}.log')
    resp_file_path = os.path.join(mock_dir, f'response_{"{0:03}".format(id)}.log')
    try:
        # load response
        with open(resp_file_path, 'r') as resp_file:
            response = resp_file.read()
        # apply prompt
        with open(prompt_file_path, 'w') as prompt_file:
            prompt_file.write(messages)
            prompt_file.write('\n')
        return response
    except FileNotFoundError:
        return None


def get_mock_response(test_name: str, messages: str, id: int) -> str:
    """Find mock response based on prompt. Prompts are stored under nested
    folders under mock folder. If prompt_{id}.log matches,
    then the mock response we're looking for is at response_{id}.log.

    Note: we filter out all non-alphanumerical characters, otherwise we would
    see surprising mismatches caused by linters and minor discrepancies between
    different platforms.

    We could have done a slightly more efficient string match with the same time
    complexity (early-out upon first character mismatch), but it is unnecessary
    for tests. Empirically, different prompts of the same task usually only
    differ near the end of file, so the comparison would be more efficient if
    we start from the end of the file, but again, that is unnecessary and only
    makes test code harder to understand.
    """
    mock_dir = os.path.join(MOCK_ROOT_DIR, test_name)
    prompt = filter_out_symbols(messages)
    prompt_file_path = os.path.join(mock_dir, f'prompt_{"{0:03}".format(id)}.log')
    resp_file_path = os.path.join(mock_dir, f'response_{"{0:03}".format(id)}.log')
    # Open the prompt file and compare its contents
    with open(prompt_file_path, 'r') as f:
        file_content = filter_out_symbols(f.read())
        if file_content.strip() == prompt.strip():
            # Read the response file and return its content
            with open(resp_file_path, 'r') as resp_file:
                return resp_file.read()
        else:
            # print the mismatched lines
            print('Mismatched Prompt File path', prompt_file_path)
            print('---' * 10)
            # Create a temporary file to store messages
            with tempfile.NamedTemporaryFile(
                delete=False, mode='w', encoding='utf-8'
            ) as tmp_file:
                tmp_file_path = tmp_file.name
                tmp_file.write(messages)

            try:
                # Use diff command to compare files and capture the output
                result = subprocess.run(
                    ['diff', '-u', prompt_file_path, tmp_file_path],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print('Diff:')
                    print(result.stdout)
                else:
                    print('No differences found.')
            finally:
                # Clean up the temporary file
                os.remove(tmp_file_path)

            print('---' * 10)


def mock_user_response(*args, test_name, **kwargs):
    """The agent will ask for user input using `input()` when calling `asyncio.run(main(task))`.
    This function mocks the user input by providing the response from the mock response file.

    It will read the `user_responses.log` file in the test directory and set as
    STDIN input for the agent to read.
    """
    user_response_file = os.path.join(
        script_dir,
        'mock',
        os.environ.get('DEFAULT_AGENT'),
        test_name,
        'user_responses.log',
    )
    if not os.path.exists(user_response_file):
        return ''
    with open(user_response_file, 'r') as f:
        ret = f.read().rstrip()
    ret += '\n'
    return ret


def mock_completion(*args, test_name, **kwargs):
    global cur_id
    messages = kwargs['messages']
    message_str = ''
    for message in messages:
        for m in message['content']:
            if m['type'] == 'text':
                message_str += message_separator + m['text']
    # this assumes all response_(*).log filenames are in numerical order, starting from one
    cur_id += 1
    if os.environ.get('FORCE_APPLY_PROMPTS') == 'true':
        mock_response = apply_prompt_and_get_mock_response(
            test_name, message_str, cur_id
        )
    else:
        mock_response = get_mock_response(test_name, message_str, cur_id)
    if mock_response is None:
        raise SecretExit('Mock response for prompt is not found')
    response = completion(**kwargs, mock_response=mock_response)
    return response


@pytest.fixture
def current_test_name(request):
    return request.node.name


@pytest.fixture(autouse=True)
def patch_completion(monkeypatch, request):
    test_name = request.node.name
    # Mock LLM completion
    monkeypatch.setattr(
        'opendevin.llm.llm.litellm_completion',
        partial(mock_completion, test_name=test_name),
    )

    # Mock LLM completion cost (1 USD per conversation)
    monkeypatch.setattr(
        'opendevin.llm.llm.litellm_completion_cost',
        lambda completion_response, **extra_kwargs: 1,
    )

    # Mock user input (only for tests that have user_responses.log)
    user_responses_str = mock_user_response(test_name=test_name)
    if user_responses_str:
        user_responses = io.StringIO(user_responses_str)
        monkeypatch.setattr('sys.stdin', user_responses)


@pytest.fixture
def http_server():
    web_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.chdir(web_dir)
    handler = SimpleHTTPRequestHandler

    # Start the server
    server = HTTPServer(('localhost', 8000), handler)
    thread = Thread(target=server.serve_forever)
    thread.setDaemon(True)
    thread.start()
    time.sleep(1)

    print('HTTP server started...')
    yield server

    # Stop the server
    server.shutdown()
    thread.join()


def set_up():
    global cur_id
    cur_id = 0
    assert workspace_path is not None, 'workspace_path is not set'

    # Remove and recreate the workspace_path
    if os.path.exists(workspace_path):
        try:
            print(f'[DEBUG] Removing workspace directory: {workspace_path}')
            shutil.rmtree(workspace_path)
        except Exception:
            pass
    os.makedirs(workspace_path, exist_ok=True)


@pytest.fixture(autouse=True)
def resource_setup():
    try:
        original_cwd = os.getcwd()
    except FileNotFoundError:
        print(
            '[DEBUG] Original working directory does not exist. Using /tmp as fallback.'
        )
        original_cwd = '/tmp'
        os.chdir('/tmp')

    try:
        set_up()
        yield
    finally:
        try:
            print(f'[DEBUG] Final working directory: {os.getcwd()}')
        except FileNotFoundError:
            print('[DEBUG] Final working directory does not exist')

        if os.path.exists(workspace_path):
            try:
                shutil.rmtree(workspace_path)
                os.makedirs(workspace_path, exist_ok=True)
            except Exception:
                pass

        # Try to change back to the original directory
        try:
            os.chdir(original_cwd)
            print(f'[DEBUG] Changed back to original directory: {original_cwd}')
        except Exception:
            os.chdir('/tmp')
