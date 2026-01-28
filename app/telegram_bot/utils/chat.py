from __future__ import annotations

from typing import Optional, Sequence

from telegram import Message, User, Chat, Bot


def is_group_chat(chat: Chat) -> bool:
    return chat.type in ("group", "supergroup")


def get_bot_mention(bot: Bot) -> Optional[str]:
    username = bot.username or ""
    return f"@{username}".lower() if username else None


def strip_bot_mention(content: str, bot_mention: Optional[str]) -> str:
    """Убирает @botname из начала строки, если он там есть."""

    if not bot_mention:
        return content

    lowered = content.lower()
    if lowered.startswith(bot_mention):
        cleaned = content[len(bot_mention):].lstrip(" \t,:;-")
        return cleaned

    return content


def label_group_content(content: str, user: User, chat: Chat) -> str:
    """
    Добавляет к тексту префикс с автором для группового контекста.
    Пример: "@username: я считаю..." или "Имя Фамилия: я считаю..."
    """

    if not content.strip():
        return content

    if user.username:
        display_name = f"@{user.username}"
    else:
        display_name = user.full_name or f"user_{user.id}"

    return f"{display_name}: {content}"


def _collect_entities(message: Message) -> Sequence:
    return list(message.entities or ()) + list(message.caption_entities or ())


def should_respond_to_message(message: Message, bot: Bot, bot_mention: Optional[str]) -> bool:
    """
    В личке отвечаем всегда.
    В группе:
      - отвечаем, если это reply на сообщение бота;
      - или если явно упомянули @botname.
    """
    chat = message.chat

    if not is_group_chat(chat):
        return True

    # reply на сообщение бота
    if (
        message.reply_to_message
        and message.reply_to_message.from_user
        and message.reply_to_message.from_user.id == bot.id
    ):
        return True

    # Упоминание @bot_username
    text = message.text or message.caption or ""
    entities = _collect_entities(message)

    if bot_mention and text:
        for ent in entities:
            if ent.type == "mention":
                mention_text = text[ent.offset: ent.offset + ent.length]
                if mention_text.lower() == bot_mention:
                    return True

    return False


def prepare_content_for_context(
    content: str,
    *,
    user: User,
    chat: Chat,
    bot_mention: Optional[str],
) -> str:
    """
    В личке возвращает текст как есть.
    В группах:
        - убирает @botname в начале;
        - добавляет префикс с автором.
    """
    if not isinstance(content, str):
        return content

    if not is_group_chat(chat):
        return content

    content = strip_bot_mention(content, bot_mention)
    return content


def extract_ask_question(message: Message) -> str:
    """
    Достаёт текст вопроса из команды /ask.
    - режет саму команду /ask или /ask@botname по entities;
    - если текста нет — пробует взять текст из reply;
    - если так и не нашёл, возвращает пустую строку.
    """

    full_text = (message.text or message.caption or "").strip()
    question_text = ""

    if full_text:
        entities = message.entities or []
        command_end = 0

        for ent in entities:
            # Команда должна идти с самого начала сообщения
            if ent.type == "bot_command" and ent.offset == 0:
                command_end = ent.offset + ent.length
                break

        if command_end > 0 and command_end <= len(full_text):
            question_text = full_text[command_end:].strip()
        else:
            if full_text.startswith("/ask"):
                parts = full_text.split(maxsplit=1)
                question_text = parts[1].strip() if len(parts) > 1 else ""

    # Если после /ask текста нет - попробуем взять текст из reply
    if not question_text:
        if message.reply_to_message and (message.reply_to_message.text or message.reply_to_message.caption):
            question_text = (message.reply_to_message.text
                             or message.reply_to_message.caption
                             or "").strip()

    return question_text
