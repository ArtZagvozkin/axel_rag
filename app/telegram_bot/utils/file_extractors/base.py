from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Protocol

from config import logger


class FileTextExtractor(ABC):
    """
    Базовый интерфейс для извлечения текста из файла.
    """

    @abstractmethod
    def supports(self, *, ext: str, mime_type: str | None) -> bool:
        """
        Возвращает True, если этот экстрактор умеет работать с данным расширением / MIME.
        ext – уже нормализованный (без точки, lower).
        """
        raise NotImplementedError

    @abstractmethod
    def extract_text(self, data: bytes, name: str | None, mime_type: str | None) -> Optional[str]:
        """
        Возвращает извлечённый текст или None, если ничего полезного вытащить не удалось.
        """
        raise NotImplementedError


def normalize_ext(name: str | None) -> str:
    if not name:
        return ""
    lower = name.lower()
    if "." not in lower:
        return ""
    return lower.rsplit(".", 1)[-1]


# Глобальный реестр экстракторов
_EXTRACTORS: list[FileTextExtractor] = []


def register_extractor(extractor: FileTextExtractor) -> None:
    logger.debug("Registering file text extractor: %s", extractor.__class__.__name__)
    _EXTRACTORS.append(extractor)


def extract_text_from_file(data: bytes, mime_type: str | None, name: str | None) -> tuple[Optional[str], bool]:
    """
    Унифицированная точка входа.

    Возвращает (text, handled):
      - text: извлечённый текст или пояснение для пользователя, или None
      - handled: True, если файл был обработан *каким-то* экстрактором
    """
    ext = normalize_ext(name)
    mt = (mime_type or "").lower()

    for extractor in _EXTRACTORS:
        if extractor.supports(ext=ext, mime_type=mt):
            logger.info(
                "Using %s for file '%s' (ext=%s, mime=%s)",
                extractor.__class__.__name__,
                name,
                ext,
                mt,
            )
            text = extractor.extract_text(data, name, mime_type)
            return text, True

    return None, False
