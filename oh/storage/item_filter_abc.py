from abc import ABC
from typing import Generic, TypeVar

T = TypeVar("T")


class ItemFilterABC(ABC, Generic[T]):

    def filter(self, item: T) -> bool:
        """Filter the item in question"""
