from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, TypedDict, List, Optional


class ImagePart(TypedDict):
    data: bytes
    mime_type: str


class AudioPart(TypedDict):
    data: bytes
    mime_type: str


class FilePart(TypedDict):
    data: bytes
    mime_type: str
    name: str | None


class ChatMessage(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str
    images: List[ImagePart]
    files: List[FilePart]
    audios: List[AudioPart]


class LLMClient(ABC):
    """Абстрактный клиент LLM."""

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> Optional[str]:
        """
        messages – список сообщений в "универсальном" формате:
        {
            "role": "user"/"assistant"/"system",
            "content": "...",
            "images": [{"data": bytes, "mime_type": "image/jpeg"}, ...]
        }
        """
        raise NotImplementedError


class LLMError(Exception):
    """Базовая ошибка для всех проблем."""


class LLMQuotaExceededError(LLMError):
    """ Превышен лимит использования модели (квота/тариф)."""


class LLMOverloadedError(LLMError):
    """Модель перегружена или временно недоступна."""
