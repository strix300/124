"""Извлекает текст PDF постранично; OCR намеренно отсутствует."""

import json
import subprocess
import sys
from pathlib import Path

from pydantic import TypeAdapter

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.models import Page


def parse_pdf(path: Path, settings: Settings) -> list[Page]:
    """Изолирует распаковку недоверенного PDF в процессе с лимитами времени и памяти."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "app.processing.pdf_worker"],
            input=json.dumps(
                {
                    "path": str(path.resolve()),
                    "memory_mb": settings.pdf_memory_limit_mb,
                    "max_page_chars": settings.max_page_chars,
                    "max_document_chars": settings.max_document_chars,
                }
            ),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=settings.pdf_parse_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ApplicationError("pdf_timeout", "Превышено время разбора PDF.") from exc
    except OSError as exc:
        raise ApplicationError(
            "parser_unavailable", "Не удалось запустить разбор PDF.", 503
        ) from exc
    if result.returncode:
        raise ApplicationError("pdf_resource_limit", "PDF не удалось разобрать в пределах лимитов.")
    try:
        payload = json.loads(result.stdout)
        if "error" in payload:
            raise ApplicationError(payload["error"]["code"], payload["error"]["message"])
        return TypeAdapter(list[Page]).validate_python(payload["pages"])
    except (ValueError, KeyError, TypeError) as exc:
        raise ApplicationError("parser_failed", "Некорректный результат разбора PDF.", 500) from exc
