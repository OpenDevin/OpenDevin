from typing import List

from opendevin.runtime.utils.async_utils import async_to_sync
from opendevin.storage.files import FileStore


class E2BFileStore(FileStore):
    def __init__(self, filesystem):
        self.filesystem = filesystem

    @async_to_sync
    async def write(self, path: str, contents: str) -> None:
        await self.filesystem.write(path, contents)

    async def write_async(self, path: str, contents: str) -> None:
        await self.filesystem.write(path, contents)

    @async_to_sync
    async def read(self, path: str) -> str:
        return await self.filesystem.read(path)

    async def read_async(self, path: str) -> str:
        return await self.filesystem.read(path)

    @async_to_sync
    async def list(self, path: str) -> List[str]:
        return await self.filesystem.list(path)

    async def list_async(self, path: str) -> List[str]:
        return await self.filesystem.list(path)

    @async_to_sync
    async def delete(self, path: str) -> None:
        await self.filesystem.delete(path)

    async def delete_async(self, path: str) -> None:
        await self.filesystem.delete(path)
