import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opendevin.events.serialization.event import EventSource


@dataclass
class Event:
    @property
    def message(self) -> str | None:
        if hasattr(self, '_message'):
            return self._message  # type: ignore [attr-defined]
        return ''

    @property
    def id(self) -> int | None:
        if hasattr(self, '_id'):
            return self._id  # type: ignore [attr-defined]
        return -1

    @property
    def timestamp(self) -> datetime.datetime | None:
        if hasattr(self, '_timestamp'):
            return self._timestamp  # type: ignore [attr-defined]
        return None

    @property
    def source(self) -> 'EventSource' | None:
        if hasattr(self, '_source'):
            return self._source  # type: ignore [attr-defined]
        return None

    @property
    def cause(self) -> int | None:
        if hasattr(self, '_cause'):
            return self._cause  # type: ignore [attr-defined]
        return None
