from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Any, Dict, Tuple
import mimetypes

from telegram import Message, File

from config import (
    DEFAULT_IMAGE_PROMPT,
    DEFAULT_FILE_PROMPT,
    DEFAULT_AUDIO_PROMPT,
    DEFAULT_VIDEO_PROMPT,
    logger,
)
from llm.base import ChatMessage


class ContentKind(Enum):
    NONE = auto()
    TEXT = auto()
    IMAGE = auto()
    FILE = auto()
    VOICE = auto()
    AUDIO = auto()
    VIDEO = auto()


@dataclass
class ParsedContent:
    text: Optional[str] = None

    image_bytes: Optional[bytes] = None
    image_mime_type: Optional[str] = None

    file_bytes: Optional[bytes] = None
    file_mime_type: Optional[str] = None
    file_name: Optional[str] = None

    audio_bytes: Optional[bytes] = None
    audio_mime_type: Optional[str] = None

    video_bytes: Optional[bytes] = None
    video_mime_type: Optional[str] = None


def log_message_overview(message: Message) -> None:
    """
    Логирует сводную информацию о входящем сообщении Telegram:
    какие типы контента присутствуют.
    """
    from_user = message.from_user
    overview = {
        "message_id": message.message_id,
        "chat_id": message.chat_id,
        "from_id": getattr(from_user, "id", None),
        "username": getattr(from_user, "username", None),
        "has_text": bool(message.text),
        "has_photo": bool(message.photo),
        "has_document": bool(message.document),
        "has_voice": bool(message.voice),
        "has_audio": bool(message.audio),
        "has_video": bool(message.video),
        "has_video_note": bool(message.video_note),
        "has_sticker": bool(message.sticker),
        "has_poll": bool(message.poll),
        "has_location": bool(message.location),
        "has_contact": bool(message.contact),
    }

    logger.info("Incoming Telegram message overview: %s", overview)


def _detect_message_kind(message: Message) -> Tuple[ContentKind, Dict[str, Any]]:
    """
    Определяет тип контента и вытаскивает только "метаданные":
    - что за тип (IMAGE/FILE/VOICE/AUDIO/VIDEO/TEXT/NONE)
    - какой объект файла использовать
    - подпись, имя файла, mime-type
    """
    log_message_overview(message)

    meta: Dict[str, Any] = {
        "file": None,
        "text": None,
        "mime_type": None,
        "file_name": None,
    }

    # Приоритет: сначала медиа/файлы, потом чистый текст
    if message.photo:
        logger.info("Image (Photo) detected")
        meta["file"] = message.photo[-1]
        meta["text"] = message.caption
        meta["mime_type"] = "image/jpeg"
        return ContentKind.IMAGE, meta

    if (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        logger.info("Image (Document) detected")
        meta["file"] = message.document
        meta["text"] = message.caption
        meta["mime_type"] = message.document.mime_type
        return ContentKind.IMAGE, meta

    if message.video:
        logger.info(
            "Video detected: duration=%s mime_type=%s",
            message.video.duration,
            message.video.mime_type,
        )
        meta["file"] = message.video
        meta["text"] = message.caption
        meta["mime_type"] = message.video.mime_type or "video/mp4"
        return ContentKind.VIDEO, meta

    if message.video_note:
        logger.info(
            "Video note (circle) detected: duration=%s length=%s",
            message.video_note.duration,
            message.video_note.length,
        )
        meta["file"] = message.video_note
        meta["text"] = message.caption
        meta["mime_type"] = "video/mp4"
        meta["file_name"] = None
        return ContentKind.VIDEO, meta

    if message.document:
        logger.info(
            "Generic file detected: name=%s mime_type=%s",
            message.document.file_name,
            message.document.mime_type,
        )
        meta["file"] = message.document
        meta["text"] = message.caption
        meta["mime_type"] = message.document.mime_type
        meta["file_name"] = message.document.file_name
        return ContentKind.FILE, meta

    if message.voice:
        logger.info(
            "Voice message detected: duration=%s mime_type=%s",
            message.voice.duration,
            message.voice.mime_type,
        )
        meta["file"] = message.voice
        meta["text"] = message.caption
        meta["mime_type"] = message.voice.mime_type or "audio/ogg"
        return ContentKind.VOICE, meta

    if message.audio:
        logger.info(
            "Audio file detected: title=%s mime_type=%s",
            message.audio.file_name or message.audio.title,
            message.audio.mime_type,
        )
        meta["file"] = message.audio
        meta["text"] = message.caption
        meta["mime_type"] = message.audio.mime_type or "audio/mpeg"
        return ContentKind.AUDIO, meta

    if message.text:
        logger.info("Text message detected")
        meta["text"] = message.text
        return ContentKind.TEXT, meta

    logger.warning("Unsupported Telegram message type, no known content fields set")

    return ContentKind.NONE, meta


async def parse_message(message: Message) -> ParsedContent:
    """
    Высокоуровневая функция:
    1) определяет тип сообщения (_detect_message_kind)
    2) при необходимости скачивает файл/картинку/аудио
    3) подставляет дефолтные промпты
    """
    kind, meta = _detect_message_kind(message)
    content = ParsedContent()

    # Чистый текст без файлов
    if kind == ContentKind.TEXT:
        content.text = meta["text"]
        return content

    if kind == ContentKind.NONE:
        return content  # всё None

    # Для всех остальных типов может быть файл
    file_obj: Optional[File | Any] = meta["file"]
    content.text = meta["text"]

    if not file_obj:
        # Странный кейс, но лучше аккуратно обработать
        return content

    tg_file = await file_obj.get_file()
    raw_bytes = bytes(await tg_file.download_as_bytearray())

    if kind == ContentKind.IMAGE:
        content.image_bytes = raw_bytes
        content.image_mime_type = meta["mime_type"] or "image/jpeg"
        if not content.text:
            content.text = DEFAULT_IMAGE_PROMPT
        logger.info("Image downloaded, size: %d bytes", len(content.image_bytes))

    elif kind in (ContentKind.FILE, ContentKind.VIDEO):
        content.file_bytes = raw_bytes
        content.file_mime_type = meta["mime_type"]
        content.file_name = meta["file_name"]
        if not content.text:
            content.text = (
                DEFAULT_VIDEO_PROMPT if kind == ContentKind.VIDEO else DEFAULT_FILE_PROMPT
            )
        logger.info(
            "%s downloaded, name=%s, size=%d bytes",
            "Video" if kind == ContentKind.VIDEO else "File",
            content.file_name,
            len(content.file_bytes),
        )

    elif kind in (ContentKind.VOICE, ContentKind.AUDIO):
        content.audio_bytes = raw_bytes
        content.audio_mime_type = meta["mime_type"]
        if not content.text:
            content.text = DEFAULT_AUDIO_PROMPT
        logger.info("Audio downloaded, size: %d bytes", len(content.audio_bytes))

    return content


def to_chat_message(content: ParsedContent) -> Optional[ChatMessage]:
    """
    Преобразует ParsedContent в ChatMessage (наш универсальный формат).
    """
    if (
        content.text is None
        and not content.image_bytes
        and not content.file_bytes
        and not content.audio_bytes
    ):
        return None

    user_message: ChatMessage = {
        "role": "user",
        "content": content.text or "",
    }

    if content.image_bytes:
        user_message["images"] = [
            {
                "data": content.image_bytes,
                "mime_type": content.image_mime_type or "image/jpeg",
            }
        ]

    if content.file_bytes:
        guessed_mime, _ = mimetypes.guess_type(content.file_name or "")
        mime_type = content.file_mime_type or guessed_mime
        if not mime_type or mime_type == "application/octet-stream":
            mime_type = "text/plain"

        user_message["files"] = [
            {
                "data": content.file_bytes,
                "mime_type": mime_type,
                "name": content.file_name,
            }
        ]

    if content.audio_bytes:
        user_message["audios"] = [
            {
                "data": content.audio_bytes,
                "mime_type": content.audio_mime_type or "audio/ogg",
            }
        ]

    return user_message
