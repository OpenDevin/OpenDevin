from typing import List

from opendevin.observation import CmdOutputObservation
from opendevin.sandbox.sandbox import DockerInteractive


class BackgroundCommand:
    def __init__(self, id: int, command: str, dir: str):
        self.command = command
        self.id = id
        self.shell = DockerInteractive(id=str(id), workspace_dir=dir)
        self.shell.execute_in_background(command)

    def get_logs(self) -> str:
        # TODO: get an exit code if process is exited
        return self.shell.read_logs()


class CommandManager:
    def __init__(self, dir):
        self.cur_id = 0
        self.directory = dir
        self.background_commands = {}
        self.shell = DockerInteractive(id="default", workspace_dir=dir)

    def run_command(self, command: str, background=False) -> CmdOutputObservation:
        if background:
            return self._run_background(command)
        else:
            return self._run_immediately(command)

    def _run_immediately(self, command: str) -> CmdOutputObservation:
        exit_code, output = self.shell.execute(command)
        return CmdOutputObservation(
            content=output, command_id=self.cur_id, command=command, exit_code=exit_code
        )

    def _run_background(self, command: str) -> CmdOutputObservation:
        bg_cmd = BackgroundCommand(self.cur_id, command, self.directory)
        self.cur_id += 1
        self.background_commands[bg_cmd.id] = bg_cmd
        return CmdOutputObservation(
            content=f"Background command started.  To stop it, send a `kill` action with id {bg_cmd.id}",
            command_id=bg_cmd.id,
            command=command,
            exit_code=0,
        )

    def kill_command(self, id: int):
        # TODO: get log events before killing
        self.background_commands[id].shell.close()
        del self.background_commands[id]

    def get_background_obs(self) -> List[CmdOutputObservation]:
        obs = []
        for _id, cmd in self.background_commands.items():
            output = cmd.get_logs()
            obs.append(
                CmdOutputObservation(
                    content=output, command_id=_id, command=cmd.command
                )
            )
        return obs
