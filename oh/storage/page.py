from dataclasses import dataclass
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


@dataclass
class Page(Generic[T]):
    """A page of command info"""

    results: List[T]
    next_page_id: Optional[str] = None
