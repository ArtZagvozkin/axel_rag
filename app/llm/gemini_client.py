from __future__ import annotations

import asyncio
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import errors as genai_errors

from config import GEMINI_API_KEY, GEMINI_MODEL, logger
from .base import (
    LLMClient,
    ChatMessage,
    LLMError,
    LLMOverloadedError,
    LLMQuotaExceededError,
)


class GeminiClient(LLMClient):
    def __init__(self, api_key: str = GEMINI_API_KEY, model: str = GEMINI_MODEL):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def generate(self, messages: List[ChatMessage]) -> Optional[str]:
        """
        Основной метод: принимает "универсальные" сообщения, конвертирует
        в формат Gemini и возвращает текст ответа.
        """
        contents = self._convert_messages_for_gemini(messages)
        loop = asyncio.get_running_loop()

        def _request() -> Optional[str]:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=contents,
            )
            return getattr(resp, "text", None)

        logger.info("Sending request to Gemini")
        try:
            result = await loop.run_in_executor(None, _request)
            logger.info("Response from Gemini received")
            return result

        except genai_errors.ClientError as e:
            status, body = self._extract_http_info(e)

            if status == 429:
                logger.warning(
                    "Gemini quota exceeded (429 RESOURCE_EXHAUSTED): status=%s, body=%r",
                    status, body,
                )
                raise LLMQuotaExceededError("Gemini quota exceeded (429)") from e

            logger.warning(
                "Gemini ClientError (status=%s): %s, body=%r",
                status, e, body,
            )
            raise LLMError(f"Gemini client error (status={status})") from e

        except genai_errors.ServerError as e:
            status, body = self._extract_http_info(e)
            logger.warning(
                "Gemini ServerError (status=%s): %s, body=%r",
                status, e, body,
            )

            if status == 503:
                raise LLMOverloadedError("Gemini overloaded (503)") from e

            raise LLMError(f"Gemini server error (status={status})") from e

        except Exception:
            logger.exception("Unexpected error while calling Gemini")
            raise LLMError("Unexpected error while calling Gemini")


    @staticmethod
    def _extract_http_info(exc: BaseException) -> tuple[Optional[int], Any]:
        """
        Достаёт HTTP-статус и тело ответа из исключений genai.
        Работает и для ClientError, и для ServerError.
        """
        resp = getattr(exc, "response", None)
        status = getattr(exc, "status_code", None) or getattr(resp, "status_code", None)

        body_json = getattr(exc, "response_json", None)
        if body_json is not None:
            body = body_json
        else:
            text = getattr(resp, "text", None)
            body = text if text is not None else repr(resp)

        return status, body


    @staticmethod
    def _convert_messages_for_gemini(messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Конвертация в формат Gemini:
        вход:
        {
            "role": "user"/"assistant"/"system",
            "content": "...",
            "images": [{"data": bytes, "mime_type": "image/jpeg"}, ...]
            "files":  [{"data": bytes, "mime_type": "application/pdf", "name": "file.pdf"}, ...]
            "audios": [{"data": bytes, "mime_type": "audio/ogg"}, ...]
        }

        выход:
        {
            "role": "user"/"model",
            "parts": [...]
        }
        """
        converted: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")

            if role == "assistant":
                role = "model"
            elif role == "system":
                # У Gemini нет system – передаём как user
                role = "user"

            parts: List[Dict[str, Any]] = []

            content = msg.get("content")
            if content:
                parts.append({"text": content})

            images = msg.get("images") or []
            for img in images:
                data = bytes(img["data"])
                mime_type = img.get("mime_type", "image/jpeg")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    }
                )

            audios = msg.get("audios") or []
            for a in audios:
                data = bytes(a["data"])
                mime_type = a.get("mime_type", "audio/ogg")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    }
                )

            files = msg.get("files") or []
            for f in files:
                data = bytes(f["data"])
                mime_type = f.get("mime_type", "application/octet-stream")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": data,
                        }
                    }
                )

            if parts:
                converted.append({"role": role, "parts": parts})

        return converted
