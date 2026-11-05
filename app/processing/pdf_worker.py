"""Извлекает текст в отдельном процессе; Linux ограничивает память до распаковки PDF."""

import json
import logging
import sys
from pathlib import Path

from pydantic import JsonValue
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.constants import MAX_DOCUMENT_PAGES, PAGE_SEPARATOR
from app.core.errors import ApplicationError
from app.models import Page


def extract_pages(path: Path, max_page_chars: int, max_document_chars: int) -> list[Page]:
    with path.open("rb") as source:
        reader = PdfReader(source)
        if reader.is_encrypted:
            raise ApplicationError("encrypted_pdf", "Зашифрованные PDF не поддерживаются.")
        if not 1 <= len(reader.pages) <= MAX_DOCUMENT_PAGES:
            raise ApplicationError("page_limit", "Допускается PDF от 1 до 100 страниц.")
        pages: list[Page] = []
        offset = 0
        for number, original in enumerate(reader.pages, start=1):
            text = original.extract_text() or ""
            if not text.strip():
                raise ApplicationError(
                    "page_without_text",
                    f"На странице {number} нет извлекаемого текста. OCR не поддерживается.",
                )
            if len(text) > max_page_chars or offset + len(text) > max_document_chars:
                raise ApplicationError(
                    "text_limit", "Превышен допустимый объём извлечённого текста."
                )
            pages.append(Page(number=number, text=text, char_start=offset))
            offset += len(text) + len(PAGE_SEPARATOR)
        return pages


def main() -> None:
    logging.disable(logging.CRITICAL)
    request = json.load(sys.stdin)
    result: dict[str, JsonValue]
    if sys.platform == "linux":
        import resource

        memory_bytes = request["memory_mb"] * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    try:
        pages = extract_pages(
            Path(request["path"]),
            request["max_page_chars"],
            request["max_document_chars"],
        )
        result = {"pages": [page.model_dump() for page in pages]}
    except ApplicationError as exc:
        result = {"error": {"code": exc.code, "message": exc.message}}
    except MemoryError:
        result = {"error": {"code": "pdf_resource_limit", "message": "PDF превышает лимит памяти."}}
    except (PdfReadError, ValueError, OSError, RecursionError):
        result = {"error": {"code": "invalid_pdf", "message": "Не удалось прочитать исходный PDF."}}
    sys.stdout.write(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
