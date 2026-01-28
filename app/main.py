#!/usr/bin/env python3

from telegram.ext import ApplicationBuilder
from telegram.request import HTTPXRequest
from telegram.error import TimedOut, NetworkError

from config import TELEGRAM_TOKEN, MAX_HISTORY, LLM_PROVIDER, RAG_ENABLED, logger
from telegram_bot.handlers import create_handlers
from storage.memory import MemoryContextStore
from llm.gemini_client import GeminiClient
from llm.grok_client import GroqClient

from rag.rag_service import RagService


def build_llm_client():
    if LLM_PROVIDER == "gemini":
        logger.info("Using Gemini LLM provider")
        return GeminiClient()
    if LLM_PROVIDER == "grok":
        logger.info("Using Grok LLM provider")
        return GroqClient()
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def main() -> None:
    request = HTTPXRequest(
        connect_timeout=10.0,
        read_timeout=30.0,
        write_timeout=30.0,
        pool_timeout=10.0,
    )

    application = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .request(request)
        .build()
    )

    llm_client = build_llm_client()
    context_store = MemoryContextStore(max_history=MAX_HISTORY)

    handlers = create_handlers(llm_client, context_store)
    for h in handlers:
        application.add_handler(h)
    
    rag = RagService() if RAG_ENABLED else None

    handlers = create_handlers(llm_client, context_store, rag)

    logger.info("Starting Telegram bot polling...")
    try:
        application.run_polling()
    except TimedOut:
        logger.error(
            "Telegram TimedOut during bot startup or operation",
            exc_info=True,
        )
    except NetworkError:
        logger.error(
            "A network-related error occurred while the bot was running.",
            exc_info=True,
        )
    except Exception:
        logger.exception(
            "Fatal error while running application.run_polling()",
            exc_info=True,
        )


if __name__ == "__main__":
    main()
