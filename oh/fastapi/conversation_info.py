from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel

from oh.conversation.conversation_status import ConversationStatus


class ConversationInfo(BaseModel):
    id: UUID
    status: ConversationStatus
    created_at: datetime
    updated_at: datetime
    type: Literal["ConversationInfo"] = "ConversationInfo"
