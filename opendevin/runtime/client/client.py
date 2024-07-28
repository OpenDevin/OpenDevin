"""
This is the main file for the runtime client.
It is responsible for executing actions received from OpenDevin backend and producing observations.

NOTE: this will be executed inside the docker sandbox.

If you already have pre-build docker image yet you changed the code in this file OR dependencies, you need to rebuild the docker image to update the source code.

You should add SANDBOX_UPDATE_SOURCE_CODE=True to any `python XXX.py` command you run to update the source code.
"""

import argparse
import asyncio
import os
import re
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import pexpect
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from uvicorn import run

from opendevin.core.logger import opendevin_logger as logger
from opendevin.events.action import (
    Action,
    BrowseInteractiveAction,
    BrowseURLAction,
    CmdRunAction,
    FileReadAction,
    FileWriteAction,
    IPythonRunCellAction,
)
from opendevin.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    FileReadObservation,
    FileWriteObservation,
    IPythonRunCellObservation,
    Observation,
)
from opendevin.events.serialization import event_from_dict, event_to_dict
from opendevin.runtime.browser import browse
from opendevin.runtime.browser.browser_env import BrowserEnv
from opendevin.runtime.plugins import (
    ALL_PLUGINS,
    JupyterPlugin,
    Plugin,
)
from opendevin.runtime.server.files import insert_lines, read_lines
from opendevin.runtime.utils import split_bash_commands


class ActionRequest(BaseModel):
    action: dict


class RuntimeClient:
    """RuntimeClient is running inside docker sandbox.
    It is responsible for executing actions received from OpenDevin backend and producing observations.
    """

    def __init__(
        self, plugins_to_load: list[Plugin], work_dir: str, username: str, user_id: int
    ) -> None:
        self.plugins_to_load = plugins_to_load
        self.username = username
        self.user_id = user_id
        self.pwd = work_dir  # current PWD
        self._init_user(self.username, self.user_id)
        self._init_bash_shell(self.pwd, self.username)
        self.lock = asyncio.Lock()
        self.plugins: dict[str, Plugin] = {}
        self.browser = BrowserEnv()

    async def ainit(self):
        for plugin in self.plugins_to_load:
            await plugin.initialize(self.username)
            self.plugins[plugin.name] = plugin
            logger.info(f'Initializing plugin: {plugin.name}')

        # This is a temporary workaround
        # TODO: refactor AgentSkills to be part of JupyterPlugin
        # AFTER ServerRuntime is deprecated
        if 'agent_skills' in self.plugins and 'jupyter' in self.plugins:
            obs = await self.run_ipython(
                IPythonRunCellAction(
                    code=(
                        'import sys\n'
                        'sys.path.insert(0, "/opendevin/code")\n'
                        # TODO: figure out how to pass added env var to IPython?
                        'from opendevin.runtime.plugins.agent_skills.agentskills import *'
                    )
                )
            )
            logger.info(f'AgentSkills initialized: {obs}')

    def _init_user(self, username: str, user_id: int) -> None:
        """Create user if not exists."""
        # Skip root since it is already created
        if username == 'root':
            return

        # Add sudoer
        sudoer_line = r"echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"
        output = subprocess.run(sudoer_line, shell=True, capture_output=True)
        if output.returncode != 0:
            raise RuntimeError(f'Failed to add sudoer: {output.stderr.decode()}')
        logger.debug(f'Added sudoer successfully. Output: [{output.stdout.decode()}]')

        # Add user
        output = subprocess.run(
            (
                f'useradd -rm -d /home/{username} -s /bin/bash '
                f'-g root -G sudo -g root -G sudo -u {user_id} {username}'
            ),
            shell=True,
            capture_output=True,
        )
        if output.returncode != 0:
            raise RuntimeError(
                f'Failed to create user {username}: {output.stderr.decode()}'
            )
        logger.debug(
            f'Added user {username} successfully. Output: [{output.stdout.decode()}]'
        )

    def _init_bash_shell(self, work_dir: str, username: str) -> None:
        self.shell = pexpect.spawn(
            f'su - {username}',
            encoding='utf-8',
            echo=False,
        )
        self.__bash_PS1 = r'[PEXPECT_BEGIN] \u@\h:\w [PEXPECT_END]'

        # This should NOT match "PS1=\u@\h:\w [PEXPECT]$" when `env` is executed
        self.__bash_expect_regex = (
            r'\[PEXPECT_BEGIN\] ([a-z0-9_-]*)@([a-zA-Z0-9.-]*):(.+) \[PEXPECT_END\]'
        )

        self.shell.sendline(f'export PS1="{self.__bash_PS1}"; export PS2=""')
        self.shell.expect(self.__bash_expect_regex)

        self.shell.sendline(f'cd {work_dir}')
        self.shell.expect(self.__bash_expect_regex)
        logger.debug(
            f'Bash initialized. Working directory: {work_dir}. Output: {self.shell.before}'
        )

    def _get_bash_prompt_and_update_pwd(self):
        ps1 = self.shell.after

        # begin at the last occurence of '[PEXPECT_BEGIN]'.
        # In multi-line bash commands, the prompt will be repeated
        # and the matched regex captures all of them
        # - we only want the last one (newest prompt)
        _begin_pos = ps1.rfind('[PEXPECT_BEGIN]')
        if _begin_pos != -1:
            ps1 = ps1[_begin_pos:]

        # parse the ps1 to get username, hostname, and working directory
        matched = re.match(self.__bash_expect_regex, ps1)
        assert (
            matched is not None
        ), f'Failed to parse bash prompt: {ps1}. This should not happen.'
        username, hostname, working_dir = matched.groups()
        self.pwd = working_dir

        # re-assemble the prompt
        prompt = f'{username}@{hostname}:{working_dir} '
        if username == 'root':
            prompt += '#'
        else:
            prompt += '$'
        return prompt + ' '

    def _execute_bash(
        self,
        command: str,
        keep_prompt: bool = True,
        timeout: int = 10,
    ) -> tuple[str, int]:
        logger.debug(f'Executing command: {command}')
        self.shell.sendline(command)
        self.shell.expect(self.__bash_expect_regex, timeout=timeout)

        output = self.shell.before
        if keep_prompt:
            output += '\r\n' + self._get_bash_prompt_and_update_pwd()
        logger.debug(f'Command output: {output}')

        # Get exit code
        self.shell.sendline('echo $?')
        logger.debug(f'Executing command for exit code: {command}')
        self.shell.expect(self.__bash_expect_regex, timeout=timeout)
        _exit_code_output = self.shell.before
        logger.debug(f'Exit code Output: {_exit_code_output}')
        exit_code = int(_exit_code_output.strip())
        return output, exit_code

    async def run_action(self, action) -> Observation:
        action_type = action.action
        observation = await getattr(self, action_type)(action)
        observation._parent = action.id
        return observation

    async def run(self, action: CmdRunAction) -> CmdOutputObservation:
        try:
            commands = split_bash_commands(action.command)
            all_output = ''
            for command in commands:
                output, exit_code = self._execute_bash(command)
                if all_output:
                    # previous output already exists with prompt "user@hostname:working_dir #""
                    # we need to add the command to the previous output,
                    # so model knows the following is the output of another action)
                    all_output = all_output.rstrip() + ' ' + command + '\r\n'

                all_output += str(output) + '\r\n'
                if exit_code != 0:
                    break
            return CmdOutputObservation(
                command_id=-1,
                content=all_output.rstrip('\r\n'),
                command=action.command,
                exit_code=exit_code,
            )
        except UnicodeDecodeError:
            raise RuntimeError('Command output could not be decoded as utf-8')

    async def run_ipython(self, action: IPythonRunCellAction) -> Observation:
        if 'jupyter' in self.plugins:
            _jupyter_plugin: JupyterPlugin = self.plugins['jupyter']  # type: ignore

            # This is used to make AgentSkills in Jupyter aware of the
            # current working directory in Bash
            reset_jupyter_pwd_code = (
                f'import os; os.environ["JUPYTER_PWD"] = "{self.pwd}"\n\n'
            )
            action.code = reset_jupyter_pwd_code + action.code

            obs: IPythonRunCellObservation = await _jupyter_plugin.run(action)

            obs.code = obs.code.replace(
                reset_jupyter_pwd_code, ''
            )  # clean up the injected code
            return obs
        else:
            raise RuntimeError(
                'JupyterRequirement not found. Unable to run IPython action.'
            )

    def get_working_directory(self):
        result, exit_code = self._execute_bash('pwd', keep_prompt=False)
        if exit_code != 0:
            raise RuntimeError('Failed to get working directory')
        return result.strip()

    def _resolve_path(self, path: str, working_dir: str) -> str:
        filepath = Path(path)
        if not filepath.is_absolute():
            return str(Path(working_dir) / filepath)
        return str(filepath)

    async def read(self, action: FileReadAction) -> Observation:
        # NOTE: the client code is running inside the sandbox,
        # so there's no need to check permission
        working_dir = self.get_working_directory()
        filepath = self._resolve_path(action.path, working_dir)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = read_lines(file.readlines(), action.start, action.end)
        except FileNotFoundError:
            return ErrorObservation(
                f'File not found: {filepath}. Your current working directory is {working_dir}.'
            )
        except UnicodeDecodeError:
            return ErrorObservation(f'File could not be decoded as utf-8: {filepath}.')
        except IsADirectoryError:
            return ErrorObservation(
                f'Path is a directory: {filepath}. You can only read files'
            )

        code_view = ''.join(lines)
        return FileReadObservation(path=filepath, content=code_view)

    async def write(self, action: FileWriteAction) -> Observation:
        working_dir = self.get_working_directory()
        filepath = self._resolve_path(action.path, working_dir)

        insert = action.content.split('\n')
        try:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            mode = 'w' if not os.path.exists(filepath) else 'r+'
            try:
                with open(filepath, mode, encoding='utf-8') as file:
                    if mode != 'w':
                        all_lines = file.readlines()
                        new_file = insert_lines(
                            insert, all_lines, action.start, action.end
                        )
                    else:
                        new_file = [i + '\n' for i in insert]

                    file.seek(0)
                    file.writelines(new_file)
                    file.truncate()
            except FileNotFoundError:
                return ErrorObservation(f'File not found: {filepath}')
            except IsADirectoryError:
                return ErrorObservation(
                    f'Path is a directory: {filepath}. You can only write to files'
                )
            except UnicodeDecodeError:
                return ErrorObservation(
                    f'File could not be decoded as utf-8: {filepath}'
                )
        except PermissionError:
            return ErrorObservation(f'Malformed paths not permitted: {filepath}')
        return FileWriteObservation(content='', path=filepath)

    async def browse(self, action: BrowseURLAction) -> Observation:
        return await browse(action, self.browser)

    async def browse_interactive(self, action: BrowseInteractiveAction) -> Observation:
        return await browse(action, self.browser)

    def close(self):
        self.shell.close()
        self.browser.close()


# def test_run_commond():
#     client = RuntimeClient()
#     command = CmdRunAction(command='ls -l')
#     obs = client.run_action(command)
#     print(obs)

# def test_shell(message):
#     shell = pexpect.spawn('/bin/bash', encoding='utf-8')
#     shell.expect(r'[$#] ')
#     print(f'Received command: {message}')
#     shell.sendline(message)
#     shell.expect(r'[$#] ')
#     output = shell.before.strip().split('\r\n', 1)[1].strip()
#     print(f'Output: {output}')
#     shell.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('port', type=int, help='Port to listen on')
    parser.add_argument('--working-dir', type=str, help='Working directory')
    parser.add_argument('--plugins', type=str, help='Plugins to initialize', nargs='+')
    parser.add_argument(
        '--username', type=str, help='User to run as', default='opendevin'
    )
    parser.add_argument('--user-id', type=int, help='User ID to run as', default=1000)
    # example: python client.py 8000 --working-dir /workspace --plugins JupyterRequirement
    args = parser.parse_args()

    plugins_to_load: list[Plugin] = []
    if args.plugins:
        for plugin in args.plugins:
            if plugin not in ALL_PLUGINS:
                raise ValueError(f'Plugin {plugin} not found')
            plugins_to_load.append(ALL_PLUGINS[plugin]())  # type: ignore

    client: RuntimeClient | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global client
        client = RuntimeClient(
            plugins_to_load,
            work_dir=args.working_dir,
            username=args.username,
            user_id=args.user_id,
        )
        await client.ainit()
        yield
        # Clean up & release the resources
        client.close()

    app = FastAPI(lifespan=lifespan)

    @app.middleware('http')
    async def one_request_at_a_time(request: Request, call_next):
        assert client is not None
        async with client.lock:
            response = await call_next(request)
        return response

    @app.post('/execute_action')
    async def execute_action(action_request: ActionRequest):
        assert client is not None
        try:
            action = event_from_dict(action_request.action)
            if not isinstance(action, Action):
                raise HTTPException(status_code=400, detail='Invalid action type')
            observation = await client.run_action(action)
            return event_to_dict(observation)
        except Exception as e:
            logger.error(f'Error processing command: {str(e)}')
            raise HTTPException(status_code=500, detail=str(e))

    @app.get('/alive')
    async def alive():
        return {'status': 'ok'}

    logger.info(f'Starting action execution API on port {args.port}')
    print(f'Starting action execution API on port {args.port}')
    run(app, host='0.0.0.0', port=args.port)
