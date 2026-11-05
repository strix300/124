"""Проверяет неоднозначный результат публикации на последней разрешённой попытке."""

from contextlib import nullcontext
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.errors import ServiceUnavailable
from app.models import DocumentRecord, TaskRecord


@pytest.mark.parametrize("published", [True, False])
def test_last_attempt_checks_publication_before_marking_failed(
    monkeypatch: pytest.MonkeyPatch,
    published: bool,
) -> None:
    pytest.importorskip("celery", reason="Требуется Celery")
    pytest.importorskip("qdrant_client", reason="Требуется qdrant-client")
    from app.workers import tasks

    settings = Settings()
    document = DocumentRecord(
        doc_id=uuid4(),
        doc_title="Договор",
        content_hash="hash",
        task_id=uuid4(),
        pending_generation=uuid4(),
        index_profile=settings.index_profile(),
    )
    record = TaskRecord(
        task_id=document.task_id,
        doc_id=document.doc_id,
        generation_id=document.pending_generation,
        attempt=settings.task_max_attempts - 1,
    )
    committed = document.model_copy(
        update={
            "active_generation": document.pending_generation if published else None,
            "chunk_count": 2 if published else None,
        }
    )
    state = Mock()
    state.get.return_value = record
    state.document_lease.return_value = nullcontext(Mock())
    store = Mock()
    store.get_document.side_effect = [document, committed]
    indexing = Mock()
    indexing.run.side_effect = ServiceUnavailable("Qdrant")
    monkeypatch.setattr(tasks, "Settings", lambda: settings)
    monkeypatch.setattr(tasks, "TaskState", lambda settings: state)
    monkeypatch.setattr(tasks, "QdrantStore", lambda settings: store)
    monkeypatch.setattr(tasks, "FileStore", Mock())
    monkeypatch.setattr(tasks, "IndexingService", Mock(return_value=indexing))

    tasks.index_document.push_request(id=str(record.task_id), retries=0)
    try:
        if published:
            tasks.index_document.run(str(document.doc_id), str(document.pending_generation))
        else:
            with pytest.raises(ServiceUnavailable):
                tasks.index_document.run(str(document.doc_id), str(document.pending_generation))
    finally:
        tasks.index_document.pop_request()

    assert record.status == ("succeeded" if published else "failed")
    assert record.attempt == settings.task_max_attempts
    assert (record.error is None) is published
    assert state.save.call_args.args[0].status == record.status


@pytest.mark.parametrize("redis_down", [False, True])
def test_continued_storage_failure_still_schedules_retry(redis_down: bool) -> None:
    from app.workers.tasks import _retry_indexing

    settings = Settings()
    record = TaskRecord(task_id=uuid4(), doc_id=uuid4(), generation_id=uuid4(), attempt=3)
    state = Mock()
    if redis_down:
        state.save.side_effect = ServiceUnavailable("Redis")
    store = Mock()
    store.get_document.side_effect = ServiceUnavailable("Qdrant")
    task = Mock()
    task.request.retries = 0
    scheduled = RuntimeError("retry scheduled")
    task.retry.side_effect = scheduled

    with pytest.raises(RuntimeError, match="retry scheduled"):
        _retry_indexing(task, settings, state, store, record, ServiceUnavailable("Qdrant"))

    task.retry.assert_called_once()
    assert record.status == "retrying"


def test_publication_check_retries_are_bounded() -> None:
    from app.workers.tasks import _retry_indexing

    settings = Settings()
    task = Mock()
    task.request.retries = settings.task_max_attempts
    store = Mock()
    store.get_document.side_effect = ServiceUnavailable("Qdrant")
    record = TaskRecord(task_id=uuid4(), doc_id=uuid4(), generation_id=uuid4(), attempt=3)
    with pytest.raises(ServiceUnavailable):
        _retry_indexing(task, settings, Mock(), store, record, ServiceUnavailable("Qdrant"))
    task.retry.assert_not_called()
    assert record.status == "retrying"
