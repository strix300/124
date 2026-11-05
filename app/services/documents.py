"""Принимает документы с раздельной дедупликацией и HTTP-идемпотентностью."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID, uuid4, uuid5

from app.contracts import DocumentPage, DocumentSummary, Submission
from app.core.config import Settings
from app.core.constants import MAX_DOCUMENT_TITLE_CHARS, POINT_NAMESPACE, TERMINAL_STATUSES
from app.core.errors import ApplicationError
from app.infrastructure.files import FileStore
from app.infrastructure.task_state import TaskState
from app.models import DocumentRecord, TaskRecord

if TYPE_CHECKING:
    from app.infrastructure.qdrant import QdrantStore


def document_identity(content_hash: str, title: str) -> UUID:
    """Различает логические документы с одинаковым содержимым, но разными заголовками."""
    signature = hashlib.sha256((content_hash + "\0" + title).encode("utf-8")).hexdigest()
    return uuid5(POINT_NAMESPACE, signature)


class DocumentService:
    """Координирует сохранение файла, каталог и отправку известного задания."""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        state: TaskState,
        files: FileStore,
        enqueue: Callable[[UUID, UUID, UUID], None],
    ) -> None:
        self.settings = settings
        self.store = store
        self.state = state
        self.files = files
        self.enqueue = enqueue

    def submit(self, stream: BinaryIO, title: str, idempotency_key: str | None) -> Submission:
        """Возвращает 202 только после подтверждённой отправки задания в брокер."""
        if not title.strip() or len(title) > MAX_DOCUMENT_TITLE_CHARS:
            raise ApplicationError(
                "invalid_title",
                "Заголовок должен содержать от 1 до 500 символов.",
            )
        staged, content_hash = self.files.stage(stream)
        doc_id = document_identity(content_hash, title)
        try:
            with self.state.document_lease(
                doc_id,
                wait=self.settings.document_lock_wait_seconds,
            ) as lease:
                document = self.store.get_document(doc_id)
                is_new = document is None
                if document is None:
                    document = DocumentRecord(
                        doc_id=doc_id,
                        doc_title=title,
                        content_hash=content_hash,
                        task_id=uuid4(),
                        pending_generation=uuid4(),
                        index_profile=self.settings.index_profile(),
                    )
                if document.deleting:
                    raise ApplicationError("document_deleting", "Документ удаляется.", 409)
                task_id = document.task_id
                if idempotency_key:
                    key = hashlib.sha256(idempotency_key.encode()).hexdigest()
                    _, task_id = self.state.bind_request(key, str(doc_id), doc_id, task_id)
                    if is_new:
                        if self.state.get(task_id) is not None:
                            raise ApplicationError(
                                "document_deleted",
                                "Документ этого запроса уже удалён.",
                                410,
                            )
                        document.task_id = task_id
                    elif task_id != document.task_id:
                        return Submission(
                            doc_id=doc_id,
                            task_id=task_id,
                            status_url=f"/tasks/{task_id}",
                        )
                lease.ensure_owned()
                self.files.commit(staged, doc_id)
                document = self.store.create_document(document)
                task = self.state.get(document.task_id)
                if document.active_generation is None and task is None:
                    task = TaskRecord(
                        task_id=document.task_id,
                        doc_id=doc_id,
                        generation_id=document.pending_generation,
                    )
                    self.state.register(task)
                if task is not None and task.status == "queued":
                    self.enqueue(doc_id, document.pending_generation, document.task_id)
                return Submission(
                    doc_id=doc_id,
                    task_id=document.task_id,
                    status_url=f"/tasks/{document.task_id}",
                )
        finally:
            staged.unlink(missing_ok=True)

    def delete(self, doc_id: UUID) -> None:
        """Удаляет файл и все поколения после завершения фонового задания."""
        with self.state.document_lease(
            doc_id,
            wait=self.settings.document_lock_wait_seconds,
        ) as lease:
            document = self.store.get_document(doc_id)
            if document is not None:
                task = self.state.get(document.task_id)
                if task and task.status not in TERMINAL_STATUSES:
                    raise ApplicationError(
                        "indexing_active",
                        "Удаление возможно после завершения индексации.",
                        409,
                    )
                self.store.mark_deleting(doc_id)
            self.files.delete(doc_id)
            lease.ensure_owned()
            self.store.delete_document(doc_id)

    def list(self, limit: int, cursor: UUID | None) -> DocumentPage:
        """Читает одну страницу каталога и одним запросом дополняет её состояниями заданий."""
        records, next_cursor = self.store.list_documents(
            min(limit, self.settings.page_size_max),
            cursor,
        )
        tasks = self.state.get_many([record.task_id for record in records])
        summaries = []
        for document, task in zip(records, tasks, strict=True):
            status = (
                "ready"
                if document.active_generation
                else (task.status if task else "status_expired")
            )
            summaries.append(
                DocumentSummary(
                    doc_id=document.doc_id,
                    doc_title=document.doc_title,
                    uploaded_at=document.uploaded_at,
                    status=status,
                    task_id=document.task_id,
                    page_count=document.page_count,
                    chunk_count=document.chunk_count,
                )
            )
        return DocumentPage(items=summaries, next_cursor=next_cursor)
