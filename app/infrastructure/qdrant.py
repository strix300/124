"""Изолирует Qdrant, включая публикацию поколений и серверный hybrid search."""

from collections.abc import Callable, Sequence
from functools import partial
from uuid import UUID, uuid4, uuid5

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from app.core.config import Settings
from app.core.constants import DENSE_VECTOR, POINT_NAMESPACE, SPARSE_VECTOR, TRANSIENT_HTTP_STATUSES
from app.core.errors import ApplicationError, ServiceUnavailable
from app.models import Candidate, Chunk, DocumentRecord, EmbeddedChunk, SparseVector


def chunk_id(chunk: Chunk) -> UUID:
    """Повторно получает тот же идентификатор при доставке задания более одного раза."""
    return uuid5(POINT_NAMESPACE, f"{chunk.doc_id}:{chunk.generation_id}:{chunk.chunk_index}")


def _transient(error: BaseException) -> bool:
    if isinstance(error, UnexpectedResponse):
        return error.status_code in TRANSIENT_HTTP_STATUSES
    if isinstance(error, ResponseHandlingException):
        return isinstance(error.source, (httpx.TransportError, TimeoutError, ConnectionError))
    return isinstance(error, (httpx.TransportError, TimeoutError, ConnectionError))


class QdrantStore:
    """Сохраняет документы и возвращает доменные типы вместо объектов SDK."""

    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        self.settings = settings
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            timeout=settings.qdrant_timeout,
        )
        self.collection = settings.qdrant_collection

    def close(self) -> None:
        """Освобождает соединения клиента."""
        self.client.close()

    def _execute[T](self, operation: Callable[[], T]) -> T:
        try:
            for attempt in Retrying(
                retry=retry_if_exception(_transient),
                stop=stop_after_attempt(self.settings.retry_attempts),
                wait=wait_random_exponential(
                    multiplier=self.settings.retry_min_seconds,
                    max=self.settings.retry_max_seconds,
                ),
                reraise=True,
            ):
                with attempt:
                    return operation()
        except (
            UnexpectedResponse,
            ResponseHandlingException,
            httpx.TransportError,
            TimeoutError,
            ConnectionError,
        ) as exc:
            if _transient(exc):
                raise ServiceUnavailable("Qdrant") from exc
            raise ApplicationError(
                "vector_store_error",
                "Ошибка операции с индексом.",
                500,
            ) from exc
        raise RuntimeError("Повтор операции завершился без результата.")

    def initialize(self) -> None:
        """Создаёт схему, не перезаписывая существующую несовместимую коллекцию."""
        if not self._execute(lambda: self.client.collection_exists(self.collection)):

            def create_collection() -> None:
                try:
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config={
                            DENSE_VECTOR: models.VectorParams(
                                size=self.settings.dense_dimensions,
                                distance=models.Distance.COSINE,
                            )
                        },
                        sparse_vectors_config={
                            SPARSE_VECTOR: models.SparseVectorParams(
                                modifier=models.Modifier.IDF,
                            )
                        },
                        shard_number=1,
                    )
                except UnexpectedResponse as exc:
                    if exc.status_code != 409:
                        raise

            self._execute(create_collection)
        info = self._execute(lambda: self.client.get_collection(self.collection))
        vectors = info.config.params.vectors
        sparse = info.config.params.sparse_vectors or {}
        if (
            not isinstance(vectors, dict)
            or DENSE_VECTOR not in vectors
            or vectors[DENSE_VECTOR].size != self.settings.dense_dimensions
            or vectors[DENSE_VECTOR].distance != models.Distance.COSINE
            or SPARSE_VECTOR not in sparse
            or sparse[SPARSE_VECTOR].modifier != models.Modifier.IDF
        ):
            raise ApplicationError(
                "incompatible_index",
                "Конфигурация коллекции несовместима.",
                503,
            )
        for field in (
            "record_type",
            "doc_id",
            "generation_id",
            "pending_generation",
            "index_profile",
        ):
            self._execute(
                partial(
                    self.client.create_payload_index,
                    self.collection,
                    field,
                    models.PayloadSchemaType.KEYWORD,
                    wait=True,
                )
            )
        self._execute(
            lambda: self.client.create_payload_index(
                self.collection,
                "deleting",
                models.PayloadSchemaType.BOOL,
                wait=True,
            )
        )

    def verify_catalog_support(self) -> None:
        """Проверяет payload-only точку в выбранной конфигурации без фиктивных векторов."""
        point_id = str(uuid4())
        self._execute(
            lambda: self.client.upsert(
                self.collection,
                [models.PointStruct(id=point_id, vector={}, payload={"record_type": "probe"})],
                wait=True,
            )
        )
        try:
            points = self._execute(
                lambda: self.client.retrieve(
                    self.collection,
                    [point_id],
                    with_vectors=True,
                )
            )
            if len(points) != 1 or points[0].vector not in ({}, None):
                raise ApplicationError(
                    "catalog_unsupported",
                    "Payload-only каталог не поддержан.",
                    503,
                )
        finally:
            self._execute(
                lambda: self.client.delete(
                    self.collection,
                    models.PointIdsList(points=[point_id]),
                    wait=True,
                )
            )

    def get_document(self, doc_id: UUID) -> DocumentRecord | None:
        """Возвращает запись каталога по ID без поиска по фрагментам."""
        points = self._execute(lambda: self.client.retrieve(self.collection, [str(doc_id)]))
        if not points:
            return None
        return DocumentRecord.model_validate(points[0].payload)

    def create_document(self, document: DocumentRecord) -> DocumentRecord:
        """Сохраняет новый каталог, не сбрасывая уже опубликованное поколение."""
        self._execute(
            lambda: self.client.upsert(
                self.collection,
                [
                    models.PointStruct(
                        id=str(document.doc_id),
                        vector={},
                        payload=document.model_dump(mode="json"),
                    )
                ],
                update_mode=models.UpdateMode.INSERT_ONLY,
                wait=True,
            )
        )
        saved = self.get_document(document.doc_id)
        if saved is None:
            raise ServiceUnavailable("Qdrant")
        return saved

    def list_documents(
        self,
        limit: int,
        cursor: UUID | None,
    ) -> tuple[list[DocumentRecord], UUID | None]:
        """Постранично читает только каталог, не загружая векторы."""
        points, offset = self._execute(
            lambda: self.client.scroll(
                self.collection,
                limit=limit,
                offset=str(cursor) if cursor else None,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="record_type", match=models.MatchValue(value="document")
                        ),
                        models.FieldCondition(key="deleting", match=models.MatchValue(value=False)),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )
        )
        documents = [DocumentRecord.model_validate(point.payload) for point in points]
        return documents, UUID(str(offset)) if offset else None

    def upsert_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Записывает оба вектора и payload каждой точки одной операцией."""
        points = [
            models.PointStruct(
                id=str(chunk_id(item.chunk)),
                vector={
                    DENSE_VECTOR: item.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=item.sparse.indices,
                        values=item.sparse.values,
                    ),
                },
                payload=item.chunk.model_dump(mode="json"),
            )
            for item in chunks
        ]
        self._execute(lambda: self.client.upsert(self.collection, points, wait=True))

    def verify_generation(self, chunks: Sequence[Chunk]) -> None:
        """Проверяет полный набор точек, payload и наличие обоих векторов перед публикацией."""
        if not chunks:
            raise ApplicationError("empty_document", "Документ не содержит фрагментов.")
        expected = {str(chunk_id(chunk)): chunk for chunk in chunks}
        for start in range(0, len(chunks), self.settings.qdrant_batch_size):
            batch = chunks[start : start + self.settings.qdrant_batch_size]
            points = self._execute(
                partial(
                    self.client.retrieve,
                    self.collection,
                    [str(chunk_id(c)) for c in batch],
                    with_payload=True,
                    with_vectors=True,
                )
            )
            if len(points) != len(batch):
                raise ServiceUnavailable("Qdrant: неполное поколение")
            for point in points:
                vectors = point.vector
                dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
                if (
                    Chunk.model_validate(point.payload) != expected[str(point.id)]
                    or not isinstance(vectors, dict)
                    or DENSE_VECTOR not in vectors
                    or SPARSE_VECTOR not in vectors
                    or not isinstance(dense, list)
                    or len(dense) != self.settings.dense_dimensions
                ):
                    raise ApplicationError("invalid_index", "Проверка индекса не пройдена.", 500)
        count = self._execute(
            lambda: self.client.count(
                self.collection,
                count_filter=self._filter(chunks[0].doc_id, chunks[0].generation_id),
                exact=True,
            )
        ).count
        if count != len(chunks):
            raise ApplicationError("invalid_index", "Число фрагментов поколения не совпадает.", 500)

    def publish(self, document: DocumentRecord, page_count: int, chunk_count: int) -> None:
        """Публикует полностью проверенное поколение одним обновлением записи каталога."""
        condition = models.Filter(
            must=[
                models.HasIdCondition(has_id=[str(document.doc_id)]),
                models.FieldCondition(
                    key="pending_generation",
                    match=models.MatchValue(value=str(document.pending_generation)),
                ),
                models.FieldCondition(key="deleting", match=models.MatchValue(value=False)),
            ]
        )
        self._execute(
            lambda: self.client.set_payload(
                self.collection,
                payload={
                    "active_generation": str(document.pending_generation),
                    "active_profile": document.index_profile,
                    "page_count": page_count,
                    "chunk_count": chunk_count,
                },
                points=condition,
                wait=True,
            )
        )
        saved = self.get_document(document.doc_id)
        if saved is None or saved.active_generation != document.pending_generation:
            raise ApplicationError(
                "generation_superseded",
                "Поколение заменено или документ удалён.",
                409,
            )

    def schedule_generation(self, document: DocumentRecord) -> None:
        """Меняет только ожидаемое поколение при контролируемой переиндексации."""
        self._execute(
            lambda: self.client.set_payload(
                self.collection,
                payload={
                    "pending_generation": str(document.pending_generation),
                    "task_id": str(document.task_id),
                    "index_profile": document.index_profile,
                },
                points=[str(document.doc_id)],
                wait=True,
            )
        )

    def delete_document(self, doc_id: UUID) -> None:
        """Удаляет каталог и все поколения после явного запроса удаления."""
        self._execute(
            lambda: self.client.delete(
                self.collection,
                models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="doc_id", match=models.MatchValue(value=str(doc_id))
                            ),
                        ]
                    )
                ),
                wait=True,
            )
        )

    def mark_deleting(self, doc_id: UUID) -> None:
        """Запрещает публикацию и новые вопросы до удаления данных."""
        self._execute(
            lambda: self.client.set_payload(
                self.collection,
                {"deleting": True},
                points=[str(doc_id)],
                wait=True,
            )
        )

    @staticmethod
    def _filter(doc_id: UUID, generation_id: UUID) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(key="record_type", match=models.MatchValue(value="chunk")),
                models.FieldCondition(key="doc_id", match=models.MatchValue(value=str(doc_id))),
                models.FieldCondition(
                    key="generation_id",
                    match=models.MatchValue(value=str(generation_id)),
                ),
            ]
        )

    def retrieve(
        self,
        doc_id: UUID,
        generation_id: UUID,
        dense: list[float],
        sparse: SparseVector,
        diagnostics: bool,
    ) -> tuple[list[Candidate], dict[str, int]]:
        """Выполняет один fusion-запрос; дополнительные ветви измеряют только диагностику."""
        query_filter = self._filter(doc_id, generation_id)
        sparse_query = models.SparseVector(indices=sparse.indices, values=sparse.values)
        result = self._execute(
            lambda: self.client.query_points(
                self.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using=DENSE_VECTOR,
                        filter=query_filter,
                        limit=self.settings.dense_candidates,
                    ),
                    models.Prefetch(
                        query=sparse_query,
                        using=SPARSE_VECTOR,
                        filter=query_filter,
                        limit=self.settings.sparse_candidates,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion(self.settings.fusion)),
                query_filter=query_filter,
                limit=self.settings.fused_candidates,
                with_payload=True,
                with_vectors=False,
            )
        )
        candidates = [
            Candidate(
                fragment_id=UUID(str(point.id)),
                chunk=Chunk.model_validate(point.payload),
                fusion_score=point.score,
            )
            for point in result.points
        ]
        counts = {"fusion_candidates": len(candidates)}
        if diagnostics:
            branches = self._execute(
                lambda: self.client.query_batch_points(
                    self.collection,
                    requests=[
                        models.QueryRequest(
                            query=dense,
                            using=DENSE_VECTOR,
                            filter=query_filter,
                            limit=self.settings.dense_candidates,
                            with_payload=False,
                        ),
                        models.QueryRequest(
                            query=sparse_query,
                            using=SPARSE_VECTOR,
                            filter=query_filter,
                            limit=self.settings.sparse_candidates,
                            with_payload=False,
                        ),
                    ],
                )
            )
            counts.update(
                dense_candidates=len(branches[0].points),
                sparse_candidates=len(branches[1].points),
            )
        return candidates, counts
