from enum import Enum


class CommandStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


FINISHED_STATUSES = (
    CommandStatus.CANCELLED,
    CommandStatus.COMPLETED,
    CommandStatus.ERROR,
)
