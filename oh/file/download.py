from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterable, Optional, Union
from oh.file.file_info import FileInfo


@dataclass
class Download:
    """
    Class representing a file for download. May include either:
      * A download url (e.g.: An S3 Signed Download URL)
      * A path to a file
      * A content stream
    """

    file_info: FileInfo
    download_url: Optional[str] = None
    path: Optional[Path] = None
    content_stream: AsyncIterable[Union[str, bytes, memoryview]] = None
