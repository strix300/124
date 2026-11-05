"""Проверяет реальную доставку Celery и наблюдаемую ошибку разбора PDF."""

import os
import time
from io import BytesIO
from uuid import uuid4

import httpx
import pytest

from app.core.config import Settings

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RAG_RUN_STACK_TESTS") != "1",
        reason="Требуются запущенные API, Redis, Qdrant и Celery worker",
    ),
]


def test_scanned_page_failure_is_visible_through_api() -> None:
    writer = pytest.importorskip("pypdf").PdfWriter()
    writer.add_blank_page(width=300, height=300)
    source = BytesIO()
    writer.write(source)
    with httpx.Client(
        base_url=os.getenv("RAG_TEST_API_URL", "http://localhost:8000"),
        timeout=10,
        headers={"Authorization": f"Bearer {Settings().api_key.get_secret_value()}"},
    ) as client:
        upload = client.post(
            "/documents",
            data={"title": f"test-{uuid4()}"},
            files={"file": ("blank.pdf", source.getvalue(), "application/pdf")},
        )
        assert upload.status_code == 202
        submission = upload.json()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            task = client.get(submission["status_url"]).json()
            if task["status"] == "failed":
                break
            time.sleep(0.5)
        assert task["status"] == "failed"
        assert task["error"]["code"] == "page_without_text"
        assert client.delete(f"/documents/{submission['doc_id']}").status_code == 204
