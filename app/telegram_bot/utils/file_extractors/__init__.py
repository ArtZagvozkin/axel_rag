from __future__ import annotations

from typing import Dict, Any, List, Optional

from config import logger
from .base import extract_text_from_file

from . import docx
from . import pptx
from . import xlsx
from . import pdf


def enrich_message_with_file_text(user_message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Пробегается по user_message['files'], вытаскивает текст из поддерживаемых
    форматов и добавляет его в content. Если у файла есть имя, то перед
    контентом файла добавляется его название.

    Файлы, из которых удалось извлечь текст, убираются из списка files,
    чтобы не слать байты в LLM.
    """
    files: List[Dict[str, Any]] = user_message.get("files") or []
    if not files:
        return user_message

    base_content = (user_message.get("content") or "").rstrip()
    text_blocks: List[str] = []
    remaining_files: List[Dict[str, Any]] = []

    for f in files:
        data = f.get("data")
        if not isinstance(data, (bytes, bytearray)):
            remaining_files.append(f)
            continue

        mime_type = f.get("mime_type")
        name = f.get("name")

        text, handled = extract_text_from_file(data, mime_type, name)
        if not handled:
            # формат не поддержан этим уровнем - оставляем файл как есть
            remaining_files.append(f)
            continue

        if not text:
            # обработчик сработал, но текста нет - ничего не добавляем
            continue

        if name:
            block = f"Файл: {name}\n{text}"
        else:
            block = text

        text_blocks.append(block)

    # если ничего не вытащили – ничего не меняем
    if not text_blocks:
        return user_message

    # собираем новый content
    if base_content:
        new_content = base_content + "\n\n\n" + "\n\n\n".join(text_blocks)
    else:
        new_content = "\n\n\n".join(text_blocks)

    user_message["content"] = new_content

    if remaining_files:
        user_message["files"] = remaining_files
    else:
        user_message.pop("files", None)

    return user_message
