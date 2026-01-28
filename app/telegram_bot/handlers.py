from __future__ import annotations

from telegram import Update, Message
from telegram.constants import ParseMode
from telegram.error import TimedOut, BadRequest, TelegramError
from telegram.ext import (
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import (
    logger,
    SYSTEM_PROMPT,
    EMPTY_TRIGGER_FALLBACK_PROMPT,
    DISALLOWED_FILE_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MEDIA_GROUP_TIMEOUT,
)
from storage.base import BaseContextStore
from telegram_bot.utils.markdown import convert_to_md_v2, split_md_v2
from telegram_bot.utils.chat import (
    is_group_chat,
    get_bot_mention,
    label_group_content,
    should_respond_to_message,
    prepare_content_for_context,
    extract_ask_question,
)
from telegram_bot.utils.file_extractors import enrich_message_with_file_text
from telegram_bot.message_adapter import parse_message, to_chat_message
from llm.base import (
    LLMClient,
    LLMError,
    LLMOverloadedError,
    LLMQuotaExceededError,
)

from rag.rag_service import RagService
from config import RAG_ENABLED


async def send_reply(message: Message, text: str) -> None:
    if not text or not text.strip():
        logger.warning("send_reply called with empty text")
        await safe_reply_text(message, "Ответ получился пустым 😔 Попробуй спросить иначе.")
        return

    try:
        md_text = convert_to_md_v2(text)
        chunks = split_md_v2(md_text)
    except Exception:
        logger.exception("MarkdownV2 conversion/split failed")
        await safe_reply_text(message,
            "Я сгенерировал ответ, но не смог корректно его отформатировать для Telegram. "
            "Попробуй переформулировать запрос или сократить его 🙂"
        )
        return

    if not chunks:
        logger.warning("split_md_v2 returned no chunks for non-empty text")
        await safe_reply_text(message, "Ответ получился пустым 😔 Попробуй спросить иначе.")
        return

    for chunk in chunks:
        try:
            await message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN_V2)
        except TelegramError as e:
            logger.exception(
                "TelegramError while sending MarkdownV2.\n"
                "Error: %r\n"
                "Chunk preview: %r",
                e, chunk,
            )
            await safe_reply_text(message,
                "Я подготовил ответ, но Telegram не смог его принять из-за форматирования. "
                "Попробуй задать вопрос ещё раз 🙂"
            )
            break


async def safe_reply_text(message: Message, text: str) -> None:
    """
    Безопасная отправка reply_text:
    - гасим TimedOut (из-за сетевых проблем);
    - логируем TelegramError, чтобы не уронить обработчик.
    """
    try:
        await message.reply_text(text)
    except TimedOut:
        logger.warning(
            "Timed out while sending reply_text for chat %s, message_id=%s",
            message.chat_id,
            message.message_id,
            exc_info=True,
        )
    except TelegramError:
        logger.exception(
            "TelegramError while sending reply_text for chat %s, message_id=%s",
            message.chat_id,
            message.message_id,
        )


def contains_disallowed_files(user_message: dict) -> bool:
    """
    Возвращает True, если хотя бы один файл сообщения имеет запрещённое расширение.
    """
    files = user_message.get("files") or []

    for f in files:
        name = (f.get("name") or "").lower()
        if any(name.endswith(ext) for ext in DISALLOWED_FILE_EXTENSIONS):
            return True

    return False


def contains_oversized_files(user_message: dict) -> bool:
    """
    Возвращает True, если хотя бы один файл больше MAX_FILE_SIZE_BYTES.
    Ориентируемся на длину поля data (байты).
    """
    files = user_message.get("files") or []

    for f in files:
        data = f.get("data")
        if isinstance(data, (bytes, bytearray)) and len(data) > MAX_FILE_SIZE_BYTES:
            return True

    return False


def merge_chat_messages(messages: list[dict]) -> dict:
    """
    Объединяет несколько ChatMessage (одного пользователя) в один:
    - склеивает content через \n\n
    - мёржит images/files/audios.
    """
    merged: dict = {"role": "user"}
    content_parts: list[str] = []
    images: list[dict] = []
    files: list[dict] = []
    audios: list[dict] = []

    for m in messages:
        content = m.get("content")
        if isinstance(content, str) and content.strip():
            content_parts.append(content)

        for key, bucket in (("images", images), ("files", files), ("audios", audios)):
            items = m.get(key) or []
            bucket.extend(items)

    if content_parts:
        merged["content"] = "\n\n".join(content_parts)

    if images:
        merged["images"] = images
    if files:
        merged["files"] = files
    if audios:
        merged["audios"] = audios

    return merged


def create_handlers(llm_client: LLMClient, context_store: BaseContextStore):
    """
    Фабрика хендлеров.
    Внутренние функции-обработчики видят llm_client и context_store через замыкание.
    """

    async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.message
        if message:
            await safe_reply_text(message, "Привет! Чем могу помочь?")

    async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.message

        if not chat or not message:
            logger.warning("Reset called without proper chat/message: %s", update)
            return

        chat_id = chat.id
        context_store.reset(chat_id)

        logger.info("Context reset for chat %s", chat_id)
        await safe_reply_text(message, "Контекст чата очищен 🧹")

    async def process_chat_turn(
        chat_id: int,
        message: Message,
    ) -> None:
        """Общая логика работы с контекстом и LLM."""
        history = context_store.get_history(chat_id)

        messages_for_llm = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + history

        # Запрос к LLM
        try:
            assistant_response = await llm_client.generate(messages_for_llm)
            if not assistant_response:
                logger.error("LLM returned empty text for chat %s", chat_id)
                await safe_reply_text(message, "Не смог получить ответ от модели 😔")
                return

            # Сохраняем ответ ассистента в контекст
            context_store.append_message(
                chat_id,
                {"role": "assistant", "content": assistant_response},
            )

            await send_reply(message, assistant_response)

        except LLMQuotaExceededError:
            logger.warning("LLM quota exceeded (Error 429) for chat %s", chat_id)
            await safe_reply_text(message,
                "Исчерпан доступный лимит запросов к модели. "
                "Лимит скоро обновится. Пожалуйста, попробуй ещё раз чуть позже 🙂"
            )

        except LLMOverloadedError:
            logger.warning("LLM overloaded (Error 503) for chat %s", chat_id)
            await safe_reply_text(message,
                "Сейчас модель перегружена и временно недоступна. "
                "Пожалуйста, попробуй ещё раз чуть позже 🙂"
            )

        except LLMError:
            logger.exception(
                "LLMError while getting response from LLM for chat %s", chat_id
            )
            await safe_reply_text(message,
                "Возникла ошибка при обращении к модели. "
                "Скорее всего проблема на стороне сервиса LLM. "
                "Пожалуйста, попробуй ещё раз чуть позже 🙂"
            )

        except Exception:
            logger.exception(
                "Unexpected error while processing message for chat %s", chat_id
            )
            await safe_reply_text(message, "Произошла непредвиденная ошибка, попробуйте позже.")

    async def process_user_message_common(
        message: Message,
        user_message: dict,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Общая часть обработки уже готового user_message:
        - вытаскиваем текст из файлов;
        - чистим контент;
        - добавляем в контекст;
        - решаем, отвечать или нет;
        - при необходимости вызываем process_chat_turn.
        """
        chat = message.chat
        chat_id = chat.id
        bot = context.bot
        bot_mention = get_bot_mention(bot)
        has_history = context_store.has_history(chat_id)
        user = message.from_user

        # Вытаскиваем текст из поддерживаемых форматов и дополняем content
        user_message = enrich_message_with_file_text(user_message)

        content = user_message.get("content")

        if isinstance(content, str):
            # В группах убираем @botname в начале
            content = prepare_content_for_context(
                content,
                user=user,
                chat=chat,
                bot_mention=bot_mention,
            )

            if not has_history and not content.strip():
                await safe_reply_text(
                    message,
                    "Привет! Напиши, пожалуйста, вопрос или расскажи, что тебя интересует 🙂"
                )
                return

            # Если текст пустой - подставляем fallback-промпт (уже при НЕпустом контексте)
            if not content.strip():
                content = EMPTY_TRIGGER_FALLBACK_PROMPT

            # В группах помечаем автора
            if is_group_chat(chat):
                content = label_group_content(content, user, chat)

            user_message["content"] = content

        # После извлечения текста сырые файлы нам больше не нужны в истории.
        # Это защищает от Unsupported MIME type даже если LLM-клиент случайно
        # попробует отправить files как inline_data.
        if "files" in user_message:
            logger.debug(
                "Dropping 'files' from user_message before saving to context: "
                "chat_id=%s", chat_id
            )
            user_message.pop("files", None)

        # Сохраняем пользовательское сообщение в контекст чата
        context_store.append_message(chat_id, user_message)

        # Нужно ли боту отвечать?
        if not should_respond_to_message(message, bot, bot_mention):
            logger.info(
                "Ignoring message not addressed to bot (but stored in context): "
                "chat_id=%s message_id=%s",
                chat.id, message.message_id,
            )
            return

        logger.info("Processing message in chat %s", chat_id)
        await process_chat_turn(chat_id, message)

    async def process_media_group_job(job_context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Job, который срабатывает спустя MEDIA_GROUP_TIMEOUT после последнего сообщения альбома.
        Здесь у нас есть уже ВСЕ Message с одинаковым media_group_id.
        """
        job = job_context.job
        data = job.data or {}
        mg_key = data.get("media_group_key")
        chat_id = job.chat_id

        if mg_key is None or chat_id is None:
            logger.warning(
                "process_media_group_job called without media_group_key or chat_id: "
                "mg_key=%r chat_id=%r", mg_key, chat_id
            )
            return

        chat_data = job_context.chat_data
        if chat_data is None:
            logger.warning(
                "process_media_group_job: chat_data is None for chat_id=%s", chat_id
            )
            return

        mg_state = chat_data.pop(mg_key, None)
        if not mg_state:
            return

        messages: list[Message] = mg_state.get("messages") or []
        if not messages:
            return

        last_message = messages[-1]
        logger.info(
            "Processing completed media group in chat %s, media_group_id=%s, messages=%s",
            chat_id,
            last_message.media_group_id,
            len(messages),
        )

        parsed_items = []
        try:
            for msg in messages:
                p = await parse_message(msg)
                if p is not None:
                    parsed_items.append(p)
        except TimedOut:
            logger.warning(
                "Timed out while downloading media in media group for chat %s",
                chat_id,
                exc_info=True,
            )
            await safe_reply_text(last_message,
                "Произошла ошибка при скачивании файлов из Telegram 😔\n"
                "Попробуй, пожалуйста, отправить ещё раз "
                "или пришли скриншоты/текстовое описание."
            )
            return
        except BadRequest as e:
            err_text = str(e)
            if "File is too big" in err_text:
                logger.warning(
                    "Telegram reported 'File is too big' in media group for chat %s",
                    chat_id,
                )
                await safe_reply_text(last_message,
                    "Один из файлов в альбоме слишком большой для Telegram-бота 😔\n"
                    "Попробуй сократить файл, описать проблему текстом или отправить скриншот."
                )
                return
            logger.exception(
                "BadRequest while parsing media group for chat %s: %s",
                chat_id,
                err_text,
            )
            await safe_reply_text(last_message,
                "Телеграм вернул ошибку при обработке файлов 😔\n"
                "Попробуй отправить их ещё раз или пришли текстом."
            )
            return
        except TelegramError as e:
            logger.exception(
                "TelegramError while parsing media group for chat %s: %s",
                chat_id, e,
            )
            await safe_reply_text(last_message,
                "Возникла ошибка при обработке файлов 😔\n"
                "Попробуй отправить их ещё раз или пришли текстом."
            )
            return

        if not parsed_items:
            logger.warning(
                "Media group parsing produced no items, chat_id=%s, media_group_id=%s",
                chat_id,
                last_message.media_group_id,
            )
            await safe_reply_text(last_message,
                "Пока я понимаю только текст, изображения, файлы и аудио 🙂"
            )
            return

        # Конвертируем каждый parsed в ChatMessage
        user_messages: list[dict] = []
        for p in parsed_items:
            um = to_chat_message(p)
            if um is not None:
                user_messages.append(um)

        if not user_messages:
            logger.warning(
                "to_chat_message returned no messages for media group, chat_id=%s",
                chat_id,
            )
            await safe_reply_text(
                last_message,
                "Пока я понимаю только текст, изображения, файлы и аудио 🙂",
            )
            return

        # Проверка размеров / запрещённых расширений для каждого сообщения
        for um in user_messages:
            if contains_oversized_files(um):
                logger.info(
                    "Rejecting media group in chat %s due to oversized file",
                    chat_id,
                )
                await safe_reply_text(last_message,
                    "Один из файлов в альбоме слишком большой (более 20 МБ), "
                    "и я не могу передать его в модель 😔\n"
                    "Попробуй сократить файл, описать проблему текстом или отправить скриншот."
                )
                return
            if contains_disallowed_files(um):
                logger.info(
                    "Rejecting media group in chat %s due to disallowed file",
                    chat_id,
                )
                await safe_reply_text(last_message,
                    "Один из файлов в альбоме относится к неподдерживаемым форматам "
                    "(исполняемые файлы, архивы, образы дисков и другие бинарные данные).\n"
                    "Я не могу передать их в модель.\n"
                    "Пожалуйста, пришли текстовое описание или скриншот 🙌"
                )
                return

        # Кладём каждый файл/сообщение как отдельный элемент в контекст:
        #   - без сырых files (чтобы не отправить их в LLM),
        #   - но с теми же полями content/images/audios (если они есть).
        for um in user_messages:
            um_for_context = dict(um)  # неглубокая копия, чтобы не трогать исходный список
            if "files" in um_for_context:
                logger.debug(
                    "Dropping 'files' from media-group message before saving to context: "
                    "chat_id=%s media_group_id=%s",
                    chat_id,
                    last_message.media_group_id,
                )
                um_for_context.pop("files", None)

            context_store.append_message(chat_id, um_for_context)

        # Объединяем всё в один user_message и обрабатываем как единое сообщение.
        # Внутри process_user_message_common ещё раз пройдём enrich_message_with_file_text
        # уже по объединённому сообщению, чтобы текст из всех файлов оказался в одном content.
        combined_user_message = merge_chat_messages(user_messages)
        await process_user_message_common(last_message, combined_user_message, job_context)

    async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Обработчик команды /ask для работы в группах и личке.
        Примеры:
          - /ask почему не работает редирект?
          - /ask@ask_arty_bot почему не работает редирект?
          - ответом на сообщение: /ask
        """
        message = update.message
        chat = update.effective_chat
        user = update.effective_user

        if message is None or user is None or chat is None:
            logger.warning("Ask called without message/user/chat: %s", update)
            return

        chat_id = chat.id
        logger.info("User %s called /ask in chat %s", user.id, chat_id)

        has_history = context_store.has_history(chat_id)
        question_text = extract_ask_question(message)
        if not has_history and not question_text:
            await safe_reply_text(message,
                "Привет! Напиши, пожалуйста, вопрос или расскажи, что тебя интересует 🙂"
            )
            return

        if not question_text.strip():
            question_text = EMPTY_TRIGGER_FALLBACK_PROMPT

        user_message = {
            "role": "user",
            "content": question_text,
        }

        # В группах помечаем, кто задал вопрос
        if is_group_chat(chat):
            user_message["content"] = label_group_content(
                user_message["content"],
                user,
                chat,
            )

        context_store.append_message(chat_id, user_message)

        await process_chat_turn(chat_id, message)

    async def handle_media_group_message(
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """
        Обработка сообщения, входящего в media_group:
        - складываем его в chat_data;
        - переинициализируем таймер;
        - ждём, пока job соберёт всю группу.
        """
        chat_id = message.chat_id
        media_group_id = message.media_group_id
        if media_group_id is None:
            return

        mg_key = f"media_group:{media_group_id}"
        chat_data = context.chat_data

        mg_state = chat_data.get(mg_key)
        if mg_state is None:
            mg_state = chat_data[mg_key] = {"messages": [], "job": None}

        mg_state["messages"].append(message)

        # Перезапускаем job
        job = mg_state.get("job")
        if job is not None:
            job.schedule_removal()

        mg_state["job"] = context.job_queue.run_once(
            process_media_group_job,
            MEDIA_GROUP_TIMEOUT,
            data={"media_group_key": mg_key},
            chat_id=chat_id,  # <-- важно!
        )

    async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        В личке отвечает на любые.
        В группе сохраняет всё в контекст, но отвечает на reply боту или упоминания @botname.
        """

        message = update.message
        user = update.effective_user
        if message is None or user is None:
            logger.warning("Update without message or user: %s", update)
            return

        # Если это часть альбома (media_group) — собираем и уходим.
        if message.media_group_id:
            await handle_media_group_message(message, context)
            return

        chat = message.chat
        chat_id = chat.id

        # Парсим входящее сообщение (текст / медиа → внутренний формат)
        try:
            parsed = await parse_message(message)
        except TimedOut:
            logger.warning(
                "Timed out while downloading media for chat %s, message_id=%s",
                chat_id,
                message.message_id,
                exc_info=True,
            )
            await safe_reply_text(message,
                "Произошла ошибка при скачивании файла из Telegram 😔\n"
                "Попробуй, пожалуйста, отправить ещё раз "
                "или пришли скриншот/текстовое описание."
            )
            return
        except BadRequest as e:
            # Например: telegram.error.BadRequest: File is too big
            err_text = str(e)
            if "File is too big" in err_text:
                logger.warning(
                    "Telegram reported 'File is too big' while downloading media "
                    "for chat %s, message_id=%s",
                    chat_id,
                    message.message_id,
                )
                await safe_reply_text(message,
                    "Файл слишком большой для Telegram-бота 😔\n"
                    "Попробуй сократить файл, описать проблему текстом или отправить скриншот."
                )
                return
            else:
                logger.exception(
                    "BadRequest while parsing message for chat %s, message_id=%s: %s",
                    chat_id,
                    message.message_id,
                    err_text,
                )
                await safe_reply_text(message,
                    "Телеграм вернул ошибку при обработке файла 😔\n"
                    "Попробуй отправить его ещё раз или пришли текстом."
                )
                return
        except TelegramError as e:
            logger.exception(
                "TelegramError while parsing message for chat %s, message_id=%s: %s",
                chat_id,
                message.message_id,
                e,
            )
            await safe_reply_text(message,
                "Возникла ошибка при обработке файла 😔\n"
                "Попробуй отправить его ещё раз или пришли текстом."
            )
            return

        if parsed is None:
            logger.warning("parse_message returned None")
            await safe_reply_text(message, "Пока я понимаю только текст, изображения, файлы и аудио 🙂")
            return

        user_message = to_chat_message(parsed)
        if user_message is None:
            logger.warning("No text or supported media found, exiting")
            await safe_reply_text(message,
                "Пока я понимаю только текст, изображения, файлы и аудио 🙂"
            )
            return

        # Проверка слишком больших файлов
        if contains_oversized_files(user_message):
            logger.info(
                "Rejecting message with oversized file in chat %s, message_id=%s",
                chat_id,
                message.message_id,
            )

            await safe_reply_text(message,
                "Файл слишком большой (более 20 МБ), и я не могу передать его в модель 😔\n"
                "Попробуй сократить файл, описать проблему текстом или отправить скриншот."
            )
            return

        # Проверка запрещённых файлов
        if contains_disallowed_files(user_message):
            logger.info(
                "Rejecting message with disallowed file in chat %s, message_id=%s",
                chat_id,
                message.message_id,
            )

            await safe_reply_text(message,
                "Этот файл относится к неподдерживаемым форматам "
                "(исполняемые файлы, архивы, образы дисков и другие бинарные данные).\n"
                "Я не могу передать их в модель.\n"
                "Пожалуйста, пришли текстовое описание или скриншот 🙌"
            )
            return

        await process_user_message_common(message, user_message, context)

    return [
        CommandHandler("start", start),
        CommandHandler("ask", ask),
        CommandHandler("reset", reset),
        MessageHandler(
            (filters.TEXT & ~filters.COMMAND)
            | filters.PHOTO
            | filters.Document.ALL
            | filters.VOICE
            | filters.AUDIO
            | filters.VIDEO
            | filters.VIDEO_NOTE,
            handle_message,
        ),
    ]
