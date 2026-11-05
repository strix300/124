"""Вычисляет локальные эмбеддинги через LangChain без удалённого inference."""

import gc
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Self

from app.core.config import Settings
from app.core.errors import ApplicationError
from app.models import Chunk, EmbeddedChunk, SparseVector

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_qdrant import FastEmbedSparse
    from transformers import PreTrainedTokenizerBase


def model_directory(settings: Settings, role: str, model: str, revision: str) -> Path:
    """Проверяет происхождение весов; во время запросов загрузка из сети запрещена."""
    directory = settings.model_cache_dir / role
    marker = directory / "rag-model.json"
    if not marker.exists():
        raise ApplicationError("models_missing", "Сначала выполните rag download-models.", 503)
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("model") != model or metadata.get("revision") != revision:
        raise ApplicationError(
            "model_cache_mismatch",
            "Кеш модели не соответствует настройкам.",
            503,
        )
    return directory


def release_memory(device: str) -> None:
    """Освобождает неиспользуемые веса перед загрузкой следующей модели."""
    gc.collect()
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class LocalEmbeddings:
    """Загружает dense и BM25 только на время этапа обработки."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dense: HuggingFaceEmbeddings | None = None
        self.sparse: FastEmbedSparse | None = None
        self.tokenizer: PreTrainedTokenizerBase | None = None

    def __enter__(self) -> Self:
        import torch
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_qdrant import FastEmbedSparse
        from transformers import AutoTokenizer

        directory = model_directory(
            self.settings,
            "dense",
            self.settings.dense_model,
            self.settings.dense_revision,
        )
        torch.set_num_threads(self.settings.inference_threads)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True)
            self.dense = HuggingFaceEmbeddings(
                model_name=str(directory),
                model_kwargs={"device": self.settings.model_device, "local_files_only": True},
                encode_kwargs={
                    "normalize_embeddings": True,
                    "batch_size": self.settings.embedding_batch_size,
                },
            )
            model_limit = self.dense._client.max_seq_length
            if self.settings.dense_max_tokens > model_limit:
                raise ApplicationError(
                    "embedding_token_limit",
                    "Лимит токенов превышает возможности dense-модели.",
                    503,
                )
            self.sparse = FastEmbedSparse(
                model_name=self.settings.sparse_model,
                cache_dir=str(self.settings.model_cache_dir / "sparse"),
                threads=self.settings.inference_threads,
                language=self.settings.bm25_language,
                disable_stemmer=self.settings.bm25_disable_stemmer,
                k=self.settings.bm25_k,
                b=self.settings.bm25_b,
                avg_len=self.settings.bm25_avg_len,
                local_files_only=True,
            )
        except BaseException as exc:
            self.__exit__(None, None, None)
            if isinstance(exc, (OSError, ValueError, RuntimeError)):
                raise ApplicationError(
                    "embedding_model_error",
                    "Не удалось загрузить локальные эмбеддинги.",
                    503,
                ) from exc
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.dense = None
        self.sparse = None
        self.tokenizer = None
        release_memory(self.settings.model_device)

    def count_tokens(self, text: str) -> int:
        """Считает токены тем же токенизатором, который использует dense-модель."""
        if self.tokenizer is None:
            raise RuntimeError("Модель не загружена.")
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Сохраняет одинаковый порядок dense, BM25 и исходных фрагментов."""
        if self.dense is None or self.sparse is None:
            raise RuntimeError("Модель не загружена.")
        texts = [chunk.text for chunk in chunks]
        prefixed = [self.settings.dense_document_prefix + text for text in texts]
        for text in prefixed:
            self._validate_input(text, "chunk_too_long")
        try:
            dense = self.dense.embed_documents(prefixed)
            sparse = self.sparse.embed_documents(texts)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApplicationError(
                "embedding_failed",
                "Не удалось вычислить эмбеддинги документа.",
                503,
            ) from exc
        embedded = []
        for chunk, dense_vector, sparse_vector in zip(chunks, dense, sparse, strict=True):
            self._validate_vector(dense_vector)
            embedded.append(
                EmbeddedChunk(
                    chunk=chunk,
                    dense=dense_vector,
                    sparse=SparseVector(indices=sparse_vector.indices, values=sparse_vector.values),
                )
            )
        return embedded

    def embed_question(self, question: str) -> tuple[list[float], SparseVector]:
        """Применяет query-префикс E5 и отдельный режим BM25 для вопроса."""
        if self.dense is None or self.sparse is None:
            raise RuntimeError("Модель не загружена.")
        text = self.settings.dense_query_prefix + question
        self._validate_input(text, "question_too_long")
        try:
            dense = self.dense.embed_query(text)
            sparse = self.sparse.embed_query(question)
        except (OSError, ValueError, RuntimeError) as exc:
            raise ApplicationError(
                "embedding_failed",
                "Не удалось вычислить эмбеддинги вопроса.",
                503,
            ) from exc
        self._validate_vector(dense)
        return dense, SparseVector(indices=sparse.indices, values=sparse.values)

    def _validate_input(self, text: str, code: str) -> None:
        if self.tokenizer is None:
            raise RuntimeError("Модель не загружена.")
        if (
            len(self.tokenizer.encode(text, add_special_tokens=True))
            > self.settings.dense_max_tokens
        ):
            raise ApplicationError(code, "Текст превышает лимит токенов dense-модели.")

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.settings.dense_dimensions or not all(map(math.isfinite, vector)):
            raise ApplicationError(
                "embedding_dimension",
                "Модель вернула вектор, несовместимый с индексом.",
                503,
            )
