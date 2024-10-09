import base64
from copy import deepcopy
from dataclasses import dataclass, field
from itertools import count
import itertools
from re import search
from typing import Dict, Generic, Optional
from uuid import UUID, uuid4
from oh.storage.page import Page
from oh.storage.storage_abc import F, T, StorageABC
from oh.storage.storage_error import StorageError


@dataclass
class MemStorage(StorageABC[T, F], Generic[T, F]):
    """In memory storage"""

    data: Dict[UUID, T] = field(default_factory=dict)
    max_page_size: int = 100

    async def create(self, item: T):
        id = uuid4()
        item.id = id
        self.data[id] = deepcopy(item)
        return id

    async def read(self, id: UUID) -> Optional[T]:
        item = self.data.get(id)
        if item:
            return deepcopy(item)

    async def update(self, item: T):
        id = item.id
        if id not in self.data:
            raise StorageError(f"missing:{id}")
        self.data[id] = deepcopy(item)

    async def destroy(self, id: UUID) -> bool:
        self.data.pop(id, None) is not None

    async def search(self, filter: F, page_id: Optional[str] = None) -> Page[T]:
        items = iter(self.data.values())
        if filter:
            items = (item for item in items if filter.filter(item))

        # For this purpose we simply incode an offset in base64. We encode it to insulate against
        # folks thinking that a number is a part of the spec - the page key may not be a number
        # in other impls!
        skip = 0
        if page_id:
            skip = int(base64.b64decode(page_id))
        sum(1 for _ in itertools.islice(items, skip))

        results = list(itertools.islice(items, self.max_page_size))

        # Check if there are  more results...
        next_page_id = None
        try:
            next(items)
            next_page_id = base64.b64encode(str(skip + self.max_page_size))
        except StopIteration:
            next_page_id = None

        return Page(results, next_page_id)

    async def count(self, filter: F) -> int:
        return count(
            item for item in self.data.values() if filter is None or filter.filter(item)
        )
