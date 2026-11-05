"""Оркестрирует индексацию с публикацией только полностью сохранённого поколения."""

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.core.timing import measure
from app.infrastructure.embeddings import LocalEmbeddings
from app.infrastructure.files import FileStore
from app.infrastructure.qdrant import QdrantStore
from app.infrastructure.task_state import Lease, TaskState
from app.models import DocumentRecord, TaskRecord
from app.processing.chunking import split_pages
from app.processing.parsing import parse_pdf


class IndexingService:
    """Разделяет порядок этапов и их инфраструктурные реализации."""

    def __init__(
        self,
        settings: Settings,
        store: QdrantStore,
        state: TaskState,
        files: FileStore,
    ) -> None:
        self.settings = settings
        self.store = store
        self.state = state
        self.files = files

    def run(self, document: DocumentRecord, task: TaskRecord, lease: Lease) -> None:
        """Повторяет идемпотентные записи, не открывая неполный индекс для чтения."""
        if document.deleting or document.pending_generation != task.generation_id:
            raise ApplicationError(
                "generation_superseded",
                "Документ удалён или поколение заменено.",
                409,
            )
        if document.active_generation == task.generation_id:
            task.completed_chunks = document.chunk_count or 0
            task.total_chunks = document.chunk_count
            return
        if document.index_profile != self.settings.index_profile():
            raise ApplicationError(
                "index_profile_changed",
                "Настройки изменились после создания задания.",
                409,
            )
        with measure(task.timings_ms, "indexing"):
            task.stage = "parsing"
            self.state.save(task)
            with measure(task.timings_ms, "parsing"):
                pages = parse_pdf(self.files.path(document.doc_id), self.settings)
            wait_seconds = self.settings.indexing_inference_wait_seconds
            with self.state.inference_slot(wait_seconds) as inference:
                with LocalEmbeddings(self.settings) as embeddings:
                    with measure(task.timings_ms, "chunking"):
                        chunks = split_pages(
                            pages,
                            document,
                            self.settings,
                            embeddings.count_tokens,
                        )
                    task.total_chunks = len(chunks)
                    task.completed_chunks = 0
                    for start in range(0, len(chunks), self.settings.qdrant_batch_size):
                        lease.ensure_owned()
                        inference.ensure_owned()
                        task.stage = "embedding"
                        self.state.save(task)
                        with measure(task.timings_ms, "embedding"):
                            batch = embeddings.embed_chunks(
                                chunks[start : start + self.settings.qdrant_batch_size],
                            )
                        task.stage = "persisting"
                        with measure(task.timings_ms, "persistence"):
                            self.store.upsert_chunks(batch)
                        task.completed_chunks += len(batch)
                        self.state.save(task)
            task.stage = "publishing"
            self.state.save(task)
            with measure(task.timings_ms, "publication"):
                self.store.verify_generation(chunks)
                lease.ensure_owned()
                self.store.publish(document, len(pages), len(chunks))
