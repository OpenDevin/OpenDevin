from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from opendevin.events.event import Event

class ActionConfirmationStatus(str, Enum):
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    AWAITING_CONFIRMATION = 'awaiting_confirmation'

class ActionSecurityRisk(int, Enum):
    UNKNOWN = -1
    LOW = 0
    MEDIUM = 1
    HIGH = 2

@dataclass
class Action(Event):
    runnable: ClassVar[bool] = False