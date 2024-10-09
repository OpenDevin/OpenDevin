from dataclasses import dataclass
from typing import Optional
from oh.file.file_info import FileInfo
from oh.storage.item_filter_abc import ItemFilterABC


@dataclass
class FileFilter(ItemFilterABC):
    path_prefix: Optional[str] = None
    path_delimiter: Optional[str] = "/"

    def filter(self, item: FileInfo) -> bool:
        path = item.path
        path_prefix = self.path_prefix
        if path_prefix:
            if not path.startswith(path_prefix):
                return False
            path = path[len(path_prefix) :]
        if self.path_delimiter and self.path_delimiter in path:
            return False
        return True
