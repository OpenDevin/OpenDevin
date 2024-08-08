from pydantic.dataclasses import dataclass
from pydantic import BaseModel, Field
from typing import Optional

@dataclass
class LLM:
    vendor: str
    model: str

class Event(BaseModel):
    metadata: Optional[dict] = Field(default_factory=dict, description="Metadata associated with the event")


class Function(BaseModel):
    name: str
    arguments: dict


class ToolCall(Event):
    id: str
    type: str
    function: Function


class Message(Event):
    role: str
    content: Optional[str]
    tool_calls: Optional[list[ToolCall]] = None
    
    def __rich_repr__(self):
        # Print on separate line
        yield "role", self.role
        yield "content", self.content
        yield "tool_calls", self.tool_calls


class ToolOutput(Event):
    role: str
    content: str
    tool_call_id: Optional[str]

    _tool_call: Optional[ToolCall]