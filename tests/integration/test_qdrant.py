"""Проверяет настоящий сервер: payload-only каталог, поколения и обе ветви поиска."""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.models import Chunk, DocumentRecord, EmbeddedChunk, SparseVector

if TYPE_CHECKING:
    from app.infrastructure.qdrant import QdrantStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RAG_RUN_INTEGRATION") != "1",
        reason="Включите RAG_RUN_INTEGRATION=1 и запустите Qdrant",
    ),
]


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    sdk = pytest.importorskip("qdrant_client")
    from app.infrastructure.qdrant import QdrantStore

    settings = Settings(
        qdrant_collection=f"test_{uuid4().hex}",
        dense_dimensions=3,
        retry_attempts=1,
    )
    client = sdk.QdrantClient(url=settings.qdrant_url)
    store = QdrantStore(settings, client)
    store.initialize()
    try:
        yield store
    finally:
        client.delete_collection(settings.qdrant_collection)
        client.close()


@pytest.mark.parametrize("fusion", ["rrf", "dbsf"])
def test_catalog_generation_and_document_isolation(store: QdrantStore, fusion: str) -> None:
    store.settings.fusion = fusion
    store.verify_catalog_support()
    document = DocumentRecord(
        doc_id=uuid4(),
        doc_title="Договор",
        content_hash="hash",
        task_id=uuid4(),
        pending_generation=uuid4(),
        index_profile=store.settings.index_profile(),
    )
    store.create_document(document)
    catalog = store.client.retrieve(store.collection, [str(document.doc_id)], with_vectors=True)[0]
    assert catalog.vector == {}
    text = "Неустойка 0,1%."
    chunk = Chunk(
        doc_id=document.doc_id,
        doc_title=document.doc_title,
        generation_id=document.pending_generation,
        index_profile=document.index_profile,
        chunk_index=0,
        page_from=1,
        page_to=1,
        section=None,
        text=text,
        char_start=0,
        char_end=len(text),
    )
    embedded = EmbeddedChunk(
        chunk=chunk,
        dense=[1, 0, 0],
        sparse=SparseVector(indices=[42], values=[1]),
    )
    store.upsert_chunks([embedded])
    store.upsert_chunks([embedded])
    assert store.get_document(document.doc_id).active_generation is None
    store.verify_generation([chunk])
    store.publish(document, 1, 1)
    before, _ = store.retrieve(
        document.doc_id,
        document.pending_generation,
        [1, 0, 0],
        embedded.sparse,
        False,
    )
    after, counts = store.retrieve(
        document.doc_id,
        document.pending_generation,
        [1, 0, 0],
        embedded.sparse,
        True,
    )
    assert [item.fragment_id for item in before] == [item.fragment_id for item in after]
    assert len(after) == counts["dense_candidates"] == counts["sparse_candidates"] == 1
    other, _ = store.retrieve(
        uuid4(),
        document.pending_generation,
        [1, 0, 0],
        embedded.sparse,
        False,
    )
    assert other == []
    _, empty_counts = store.retrieve(
        document.doc_id,
        document.pending_generation,
        [1, 0, 0],
        SparseVector(indices=[], values=[]),
        True,
    )
    assert empty_counts["sparse_candidates"] == 0
