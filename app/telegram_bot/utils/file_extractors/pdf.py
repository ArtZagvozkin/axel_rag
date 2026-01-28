from __future__ import annotations

import io
from typing import Optional, List

from config import logger
from .base import FileTextExtractor, register_extractor

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover
    PdfReader = None


class PdfExtractor(FileTextExtractor):
    PDF_MIME = "application/pdf"

    def supports(self, *, ext: str, mime_type: str | None) -> bool:
        mt = (mime_type or "").lower()
        return ext == "pdf" or mt == self.PDF_MIME

    def extract_text(self, data: bytes, name: str | None, mime_type: str | None) -> Optional[str]:
        if PdfReader is None:
            logger.warning("pypdf is not installed; cannot extract PDF text")
            return None

        reader = PdfReader(io.BytesIO(data))
        pages_text: List[str] = []

        for page in reader.pages:
            t = (page.extract_text() or "").strip()
            if t:
                pages_text.append(t)

        text = "\n\n".join(pages_text).strip()
        return text or None


register_extractor(PdfExtractor())
