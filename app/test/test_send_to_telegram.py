#!/usr/bin/env python3

"""
Интеграционный тест: отправляет реальные сообщения в Telegram
на TELEGRAM_TEST_CHAT_ID, используя файлы в test/llm_messages/*.md
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from telegram import Bot
from telegram.constants import ParseMode

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import MAX_TELEGRAM_MESSAGE_LEN
from telegram_bot.handlers import send_reply

MESSAGES_DIR = Path(__file__).parent / "llm_messages"


def _get_message_files() -> list[Path]:
    files = sorted(MESSAGES_DIR.glob("*.md"))
    if not files:
        pytest.skip(f"Не найдено ни одного файла в {MESSAGES_DIR}")
    return files


class TelegramTestMessage:
    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id: int = chat_id
        self.calls: list[tuple[str, str | None]] = []

    async def reply_text(self, text: str, parse_mode=None):
        self.calls.append((text, parse_mode))
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=text,
            parse_mode=parse_mode,
        )


@pytest.fixture
def telegram_test_context() -> tuple[Bot, int]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id_str = os.environ.get("TELEGRAM_TEST_CHAT_ID")

    if not token or not chat_id_str:
        pytest.skip("TELEGRAM_BOT_TOKEN или TELEGRAM_TEST_CHAT_ID не заданы в окружении")

    bot = Bot(token=token)
    chat_id = int(chat_id_str)
    return bot, chat_id


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize("path", _get_message_files())
async def test_send_to_telegram(path: Path, telegram_test_context):
    bot, chat_id = telegram_test_context
    content = path.read_text(encoding="utf-8")
    msg = TelegramTestMessage(bot, chat_id)
    prefix = "`" + path.name + "`\n\n"
    content = prefix + content

    await send_reply(msg, content)

    assert msg.calls, f"send_reply должен вызвать reply_text хотя бы один раз (файл: {path.name})"

    for idx, (text, parse_mode) in enumerate(msg.calls):
        assert parse_mode == ParseMode.MARKDOWN_V2, (
            f"Ожидался ParseMode.MARKDOWN_V2, "
            f"но пришёл {parse_mode!r} (файл: {path.name}, chunk #{idx})"
        )
        assert text, f"Пустой chunk недопустим (файл: {path.name}, chunk #{idx})"
        assert len(text) <= MAX_TELEGRAM_MESSAGE_LEN, (
            f"Chunk превышает лимит {MAX_TELEGRAM_MESSAGE_LEN} символов "
            f"(файл: {path.name}, chunk #{idx}, длина={len(text)})"
        )
