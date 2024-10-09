from abc import ABC
from datetime import datetime
from uuid import UUID


class AnnouncementDetailABC(ABC):
    """
    Class representing the detials of some event that occurred within and OpenHands Process that may be
    interest externally. For example: CommandCompleted
    """
