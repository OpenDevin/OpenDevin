import os
from dataclasses import dataclass

from opendevin.observation import Observation

from .base import ExecutableAction

# This is the path where the workspace is mounted in the container
# The LLM sometimes returns paths with this prefix, so we need to remove it
PATH_PREFIX = "/workspace/"


def resolve_path(base_path, file_path):
    if file_path.startswith(PATH_PREFIX):
        file_path = file_path[len(PATH_PREFIX) :]
    return os.path.join(base_path, file_path)


@dataclass
class FileReadAction(ExecutableAction):
    path: str
    base_path: str = ""

    def run(self, *args, **kwargs) -> Observation:
        path = resolve_path(self.base_path, self.path)
        with open(path, "r") as file:
            return Observation(file.read())

    @property
    def message(self) -> str:
        return f"Reading file: {self.path}"


@dataclass
class FileWriteAction(ExecutableAction):
    path: str
    contents: str
    base_path: str = ""

    def run(self, *args, **kwargs) -> Observation:
        path = resolve_path(self.base_path, self.path)
        with open(path, "w") as file:
            file.write(self.contents)
        return Observation(f"File written to {path}")

    @property
    def message(self) -> str:
        return f"Writing file: {self.path}"
