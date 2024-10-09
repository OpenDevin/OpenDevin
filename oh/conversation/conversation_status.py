from enum import Enum


class ConversationStatus(Enum):
    CREATING = "CREATING"
    READY = "READY"
    DESTROYING = "DESTROYING"
    DESTROYED = "DESTROYED"
    ERROR = "ERROR"
