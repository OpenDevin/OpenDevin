from dataclasses import dataclass


@dataclass
class FilePermissions:
    read: bool = True
    write: bool = True
    execute: bool = False
