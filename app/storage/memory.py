from __future__ import annotations

from typing import Dict, List, Any

from .base import BaseContextStore


class MemoryContextStore(BaseContextStore):
    """In-memory хранилище."""

    def __init__(self, max_history: int = 10):
        self._store: Dict[int, List[Dict[str, Any]]] = {}
        self._max_history = max_history

    def get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        return self._store.get(chat_id, [])

    def append_message(self, chat_id: int, message: Dict[str, Any]) -> None:
        history = self._store.setdefault(chat_id, [])
        history.append(message)
        if len(history) > self._max_history:
            self._store[chat_id] = history[-self._max_history :]

    def reset(self, chat_id: int) -> None:
        self._store.pop(chat_id, None)

    def has_history(self, chat_id: int) -> bool:
        return chat_id in self._store and len(self._store[chat_id]) > 0
