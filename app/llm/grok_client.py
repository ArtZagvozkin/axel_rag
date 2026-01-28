from __future__ import annotations

import asyncio
from typing import List, Optional

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, logger
from .base import (
    LLMClient,
    ChatMessage,
    LLMError,
    LLMOverloadedError,
    LLMQuotaExceededError,
)


class GroqClient(LLMClient):
    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError("GROQ_API_KEY is empty")
        self._client = Groq(api_key=api_key)
        self._model = model

    async def generate(self, messages: List[ChatMessage]) -> Optional[str]:
        loop = asyncio.get_running_loop()
        groq_messages = self._convert_messages(messages)

        def _request() -> str:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=groq_messages,
                temperature=0.2,
            )
            return (resp.choices[0].message.content or "").strip()

        logger.info("Sending request to Groq: model=%s", self._model)
        try:
            return await loop.run_in_executor(None, _request)

        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate limit" in msg or "quota" in msg:
                raise LLMQuotaExceededError("Groq quota/rate limit") from e

            if "503" in msg or "overload" in msg or "unavailable" in msg:
                raise LLMOverloadedError("Groq overloaded/unavailable") from e

            logger.exception("Unexpected error while calling Groq")
            raise LLMError(f"Groq error: {e}") from e

    @staticmethod
    def _convert_messages(messages: List[ChatMessage]):
        out = []
        for m in messages:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            if role not in ("system", "user", "assistant"):
                role = "user"
            out.append({"role": role, "content": content})
        return out
