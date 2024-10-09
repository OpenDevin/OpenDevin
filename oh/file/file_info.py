from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from oh.file.file_permissions import FilePermissions


@dataclass
class FileInfo:
    path: str
    updated_at: datetime
    permissions: FilePermissions
    size_in_bytes: Optional[int] = None
    mime_type: Optional[str] = None

    def is_dir(self) -> bool:
        return self.path.endswith("/")
