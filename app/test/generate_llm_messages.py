#!/usr/bin/env python3

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import SYSTEM_PROMPT, logger
from llm.gemini_client import GeminiClient
from llm.base import ChatMessage


LLM_MESSAGES_DIR = Path(__file__).parent / "llm_messages"
PROMPTS: Dict[str, str] = {
    "bold_text": "Сгенерируй короткий текст, в котором активно используется жирное выделение (**...** или __...__). Разметка должна быть логичной и проверять работу жирного текста в нескольких местах.",
    "italic_text": "Создай небольшой текст, где встречаются примеры курсивного выделения (*...* или _..._) в разных контекстах.",
    "bold_and_italic": "Сделай текст, содержащий комбинации жирного, курсивного и жирно-курсивного (***...***) выделения. Используй смешанные конструкции в нескольких предложениях.",
    "strikethrough": "Сгенерируй абзац, где используется зачёркнутый текст (~~...~~) для демонстрации поддержки этого синтаксиса.",
    "inline_code": "Создай текст с несколькими примерами встроенного кода (`...`) в середине предложений. Используй разные короткие фрагменты кода.",
    "fenced_code_block": "Сгенерируй текст, содержащий один или два блока кода с тройными бэктиками (``` ```). Можно использовать любой язык для подсветки.",
    "headings": "Подготовь текст, включающий все уровни заголовков Markdown от # до ######, по одному примеру каждого уровня.",
    "ordered_list": "Сделай текст, содержащий пронумерованный список из 5–7 пунктов, желательно с вложенностью второго уровня.",
    "unordered_list": "Сгенерируй текст с маркированным списком (-, * или +) с несколькими уровнями вложенности.",
    "mixed_lists": "Создай пример текста, где одновременно используются нумерованные и маркированные списки, в том числе вложенные друг в друга.",
    "blockquote": "Сформируй текст с несколькими уровнями цитирования: одинарным (>) и вложенным (>>).",
    "links": "Создай текст, содержащий несколько обычных ссылок [text](url) и одну reference-style ссылку.",
    "images": "Сгенерируй текст с несколькими Markdown-картинками (![alt](url)), желательно с разными alt-подписями.",
    "horizontal_rules": "Сделай текст, включающий горизонтальные разделители (---, ***, ___).",
    "tables": "Сформируй небольшой текст с Markdown-таблицей, содержащей минимум 3 столбца и 3 строки, включая заголовки.",
    "emoji": "Создай текст с несколькими emoji, вставленными в Markdown-разметку. Используй их в списках и предложениях.",
    "escapes": "Создай текст, демонстрирующий экранирование markdown-символов (\\*, \\_, \\#, \\[]).",
    "math_latex_inline": "Сгенерируй текст с несколькими inline-формулами в стиле LaTeX внутри $...$. Например: $x^2 + y^2 = z^2$.",
    "math_latex_block": "Создай текст с одним блоком формулы LaTeX в $$ ... $$. Формула может быть сложной.",
    "html_in_markdown": "Сгенерируй текст, где смешан обычный Markdown и встроенный HTML-код (например <span>, <b>, <div>).",
    "nested_formatting": "Сделай текст, демонстрирующий вложенные форматы: жирный внутри курсивного, курсив внутри жирного, код внутри списка и т.д.",
    "long_complex_markdown": "Сгенерируй большой комплексный текст, содержащий максимум разных элементов Markdown: заголовки, таблицы, списки, ссылки, картинки, сноски, кодовые блоки, формулы и т.д. Это будет стресс-тест Markdown."
}


async def generate_and_save():
    logger.info("Generating LLM test messages...")

    client = GeminiClient()
    LLM_MESSAGES_DIR.mkdir(parents=True, exist_ok=True)

    for key, prompt_text in PROMPTS.items():
        logger.info("Generating message for prompt '%s'...", key)

        messages: list[ChatMessage] = [
            {"role": "user", "content": prompt_text},
        ]

        text = await client.generate(messages)
        if not text:
            logger.error("LLM returned empty message for prompt '%s'", key)
            continue

        # имя файла формируется из названия промпта
        filename = f"{key}.md"
        out_path = LLM_MESSAGES_DIR / filename

        out_path.write_text(text, encoding="utf-8")
        logger.info("Saved LLM message for '%s' to %s", key, out_path)


async def main():
    await generate_and_save()


if __name__ == "__main__":
    asyncio.run(main())
