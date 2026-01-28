from __future__ import annotations

import io
from typing import Optional, List

from config import logger
from .base import FileTextExtractor, register_extractor

try:
    import openpyxl
except ImportError:  # pragma: no cover
    openpyxl = None


class XlsxExtractor(FileTextExtractor):
    XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    def supports(self, *, ext: str, mime_type: str | None) -> bool:
        mt = (mime_type or "").lower()
        return ext == "xlsx" or mt == self.XLSX_MIME

    def extract_text(self, data: bytes, name: str | None, mime_type: str | None) -> Optional[str]:
        if openpyxl is None:
            logger.warning("openpyxl is not installed; cannot extract XLSX text")
            return None

        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        chunks: List[str] = []

        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(v) for v in row if v not in (None, "")]
                if cells:
                    chunks.append(" | ".join(cells))

        text = "\n".join(chunks).strip()
        return text or None


register_extractor(XlsxExtractor())
