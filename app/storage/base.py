from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseContextStore(ABC):
    """Базовый интерфейс для хранилища контекста (память, Redis, БД и т.п.)."""

    @abstractmethod
    def get_history(self, chat_id: int) -> List[Dict[str, Any]]:
        """Вернуть историю сообщений для конкретного чата."""
        raise NotImplementedError

    @abstractmethod
    def append_message(self, chat_id: int, message: Dict[str, Any]) -> None:
        """Добавить сообщение в историю чата."""
        raise NotImplementedError

    @abstractmethod
    def reset(self, chat_id: int) -> None:
        """Сбросить историю для чата."""
        raise NotImplementedError

    def has_history(self, chat_id: int) -> bool:
        """Возвращает True, если у чата есть хотя бы одно сообщение."""
        history = self.get_history(chat_id)
        return bool(history)
