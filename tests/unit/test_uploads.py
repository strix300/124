"""Проверяет независимость заголовка, дедупликации и безопасной записи файла."""

from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.infrastructure.files import FileStore
from app.services.documents import DocumentService, document_identity


def test_title_is_part_of_document_identity() -> None:
    assert document_identity("same_hash", "Договор") == document_identity("same_hash", "Договор")
    assert document_identity("same_hash", "Договор") != document_identity("same_hash", "Приложение")


@pytest.mark.parametrize(
    "content,code",
    [
        (b"", "empty_upload"),
        (b"hello", "unsupported_format"),
        (b"%PDF-" + b"x" * 100, "upload_too_large"),
    ],
)
def test_invalid_upload_does_not_leave_temporary_files(
    tmp_path: Path,
    content: bytes,
    code: str,
) -> None:
    files = FileStore(Settings(storage_dir=tmp_path, max_upload_bytes=20))
    with pytest.raises(ApplicationError) as error:
        files.stage(BytesIO(content))
    assert error.value.code == code
    assert list(tmp_path.iterdir()) == []


def test_replay_after_crash_before_catalog_creation_registers_original_task(tmp_path: Path) -> None:
    """Повтор использует зарезервированный ID даже до появления каталога."""
    from contextlib import nullcontext

    settings = Settings(storage_dir=tmp_path)
    store = Mock()
    store.get_document.return_value = None
    store.create_document.side_effect = lambda document: document
    state = Mock()
    state.get.return_value = None
    state.document_lease.return_value = nullcontext(Mock())
    original_task = uuid4()
    state.bind_request.side_effect = lambda key, signature, doc_id, task_id: (doc_id, original_task)
    enqueue = Mock()
    service = DocumentService(settings, store, state, FileStore(settings), enqueue)
    response = service.submit(BytesIO(b"%PDF-1.7\nsource"), "Договор", "retry-key")
    assert response.task_id == original_task
    assert state.register.call_args.args[0].task_id == original_task
    assert enqueue.call_args.args[2] == original_task
    assert FileStore(settings).path(response.doc_id).exists()


@pytest.mark.parametrize("size,status", [(20, 200), (1024, 413)])
def test_multipart_limit_closes_spooled_files(
    monkeypatch: pytest.MonkeyPatch,
    size: int,
    status: int,
) -> None:
    """Поток без Content-Length ограничивается с освобождением временных файлов."""
    import asyncio
    import tempfile
    from collections.abc import AsyncIterator
    from typing import BinaryIO

    import httpx
    from fastapi import FastAPI, UploadFile

    from app.api.upload_limit import UploadLimitMiddleware

    opened: list[BinaryIO] = []

    def tracked_spool(max_size: int) -> BinaryIO:
        spool = tempfile.SpooledTemporaryFile(max_size=max_size)
        opened.append(spool)
        return spool

    monkeypatch.setattr("starlette.formparsers.SpooledTemporaryFile", tracked_spool)
    application = FastAPI()
    application.add_middleware(UploadLimitMiddleware, max_bytes=256)

    @application.post("/documents")
    def upload(file: UploadFile) -> dict[str, int]:
        return {"size": len(file.file.read())}

    body = (
        b'--test\r\nContent-Disposition: form-data; name="file"; filename="test.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n" + b"x" * size + b"\r\n--test--\r\n"
    )

    async def chunks() -> AsyncIterator[bytes]:
        for offset in range(0, len(body), 128):
            yield body[offset : offset + 128]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/documents",
                content=chunks(),
                headers={"Content-Type": "multipart/form-data; boundary=test"},
            )

    response = asyncio.run(request())
    assert response.status_code == status
    assert opened and all(spool.closed for spool in opened)
    if status == 413:
        assert response.json()["error"]["code"] == "upload_too_large"
