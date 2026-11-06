"""Small reusable registry primitive for extensible framework components."""

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Named registry with normalized keys and duplicate protection."""

    def __init__(self, name: str):
        self.name = name
        self._items: dict[str, T] = {}

    @staticmethod
    def normalize_key(key: str) -> str:
        """Return the canonical lookup key for a registry entry."""
        return str(key).strip().lower()

    def register(self, key: str, item: T, *, replace: bool = False) -> T:
        """Add an item under ``key`` and return it, rejecting duplicates."""
        normalized = self.normalize_key(key)
        if not normalized:
            raise ValueError(f"Cannot register an empty key in {self.name}")
        if normalized in self._items and not replace:
            raise KeyError(f"'{normalized}' is already registered in {self.name}")
        self._items[normalized] = item
        return item

    def get(self, key: str) -> T:
        """Return the item registered under ``key``."""
        normalized = self.normalize_key(key)
        try:
            return self._items[normalized]
        except KeyError as exc:
            raise KeyError(f"Unknown key '{key}' in {self.name}; available={self.keys()}") from exc

    def contains(self, key: str) -> bool:
        """Whether ``key`` has a registered item."""
        return self.normalize_key(key) in self._items

    def keys(self) -> list[str]:
        """Return the registered keys in insertion order."""
        return list(self._items)

    def items(self) -> Iterator[tuple[str, T]]:
        """Iterate over ``(key, item)`` pairs."""
        return iter(self._items.items())
