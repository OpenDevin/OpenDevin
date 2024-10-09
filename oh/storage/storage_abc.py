from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar
from uuid import UUID

from oh.storage.item_abc import ItemABC
from oh.storage.item_filter_abc import ItemFilterABC
from oh.storage.page import Page

T = TypeVar("T", bound=ItemABC)
F = TypeVar("F", bound=ItemFilterABC)


class StorageABC(ABC, Generic[T, F]):
    """
    General storage interface
    """

    @abstractmethod
    async def create(self, item: T) -> UUID:
        """Create a new item."""

    @abstractmethod
    async def read(self, id: UUID) -> Optional[T]:
        """Get an item"""

    @abstractmethod
    async def update(self, id: UUID, item: T) -> Optional[T]:
        """Upate and item. Return true of the item existed and was updated"""

    @abstractmethod
    async def destroy(self, id: UUID) -> bool:
        """
        Destroy an item. Return true if the item existed and was deleted
        """

    @abstractmethod
    async def search(
        self, filter: Optional[F] = None, page_id: Optional[str] = None
    ) -> Page[T]:
        """Get a page of results"""

    @abstractmethod
    async def count(self, filter: Optional[F] = None) -> int:
        """Get the number of results of results"""
