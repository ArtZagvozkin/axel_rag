from __future__ import annotations

import io
from typing import Optional

from config import logger
from .base import FileTextExtractor, register_extractor

try:
    from docx import Document
except ImportError:
    Document = None


class DocxExtractor(FileTextExtractor):
    DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    def supports(self, *, ext: str, mime_type: str | None) -> bool:
        mt = (mime_type or "").lower()
        return ext == "docx" or mt == self.DOCX_MIME

    def extract_text(self, data: bytes, name: str | None, mime_type: str | None) -> Optional[str]:
        if Document is None:
            logger.warning("python-docx is not installed; cannot extract DOCX text")
            return None

        doc = Document(io.BytesIO(data))
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(paragraphs).strip()
        return text or None


register_extractor(DocxExtractor())
