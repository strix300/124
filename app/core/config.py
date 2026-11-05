"""Загружает и проверяет все изменяемые настройки приложения."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.constants import LLM_PROMPT_TOKEN_RESERVE, MAX_QUESTION_CHARS, PIPELINE_VERSION


class Settings(BaseSettings):
    """Содержит подключения, лимиты и параметры исключительно локальных моделей."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    storage_dir: Path = Path(".data/documents")
    model_cache_dir: Path = Path(".data/models")
    api_key: SecretStr = SecretStr("")
    max_active_requests: int = Field(16, ge=1, le=32)
    max_active_questions: int = Field(2, ge=1, le=16)
    max_json_bytes: int = Field(128 * 1024, ge=1024)
    request_body_timeout: float = Field(60, gt=0)
    pdf_parse_timeout: float = Field(60, gt=0)
    pdf_memory_limit_mb: int = Field(1024, ge=128)
    max_upload_bytes: int = Field(50 * 1024 * 1024, gt=0)
    multipart_overhead_bytes: int = Field(1024 * 1024, ge=1024)
    max_page_chars: int = Field(250_000, gt=0)
    max_document_chars: int = Field(10_000_000, gt=0)
    question_max_chars: int = Field(2000, gt=0, le=MAX_QUESTION_CHARS)
    page_size_max: int = Field(100, ge=1, le=1000)

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_timeout: int = Field(20, gt=0)
    qdrant_batch_size: int = Field(32, ge=1)
    broker_url: str = "redis://localhost:6379/0"
    result_backend_url: str = "redis://localhost:6379/1"
    state_redis_url: str = "redis://localhost:6379/2"
    redis_timeout: float = Field(5, gt=0)
    redis_prefix: str = "document-rag"
    terminal_task_ttl: int = Field(7 * 24 * 3600, ge=1)
    idempotency_ttl: int = Field(7 * 24 * 3600, ge=1)
    worker_concurrency: int = Field(1, ge=1)
    task_soft_limit: int = Field(3300, ge=1)
    task_hard_limit: int = Field(3600, ge=1)
    visibility_timeout: int = Field(3900, ge=1)
    task_max_attempts: int = Field(3, ge=1)
    task_retry_max_seconds: int = Field(30, ge=1)
    reconciliation_interval: int = Field(60, ge=1)
    reconciliation_batch_size: int = Field(100, ge=1)
    stale_task_seconds: int = Field(4000, ge=1)

    dense_model: str = "intfloat/multilingual-e5-small"
    dense_revision: str = "main"
    dense_dimensions: int = Field(384, gt=0)
    dense_query_prefix: str = "query: "
    dense_document_prefix: str = "passage: "
    dense_max_tokens: int = Field(512, ge=32)
    embedding_batch_size: int = Field(16, ge=1)
    sparse_model: str = "Qdrant/bm25"
    bm25_language: str = "russian"
    bm25_disable_stemmer: bool = False
    bm25_k: float = Field(1.2, gt=0)
    bm25_b: float = Field(0.75, ge=0, le=1)
    bm25_avg_len: float = Field(256, gt=0)
    chunk_tokens: int = Field(240, ge=16)
    chunk_overlap_tokens: int = Field(35, ge=0)
    dense_candidates: int = Field(40, ge=1)
    sparse_candidates: int = Field(40, ge=1)
    fused_candidates: int = Field(30, ge=1)
    fusion: Literal["rrf", "dbsf"] = "rrf"

    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    reranker_revision: str = "main"
    reranker_input_limit: int = Field(30, ge=1)
    reranker_output_limit: int = Field(5, ge=1)
    reranker_batch_size: int = Field(8, ge=1)
    reranker_max_tokens: int = Field(512, ge=32)
    model_device: Literal["cpu", "cuda", "mps"] = "cuda"
    inference_threads: int = Field(4, ge=1)
    inference_slots: int = Field(1, ge=1)
    inference_wait_seconds: float = Field(30, gt=0)
    indexing_inference_wait_seconds: float = Field(1500, gt=0)
    lock_lease_seconds: int = Field(60, ge=10)
    document_lock_wait_seconds: float = Field(5, ge=0)
    lock_poll_seconds: float = Field(0.1, gt=0)

    ollama_url: str = "http://localhost:11434"
    generation_model: str = "qwen2.5:7b-instruct-q4_K_M"
    verification_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_context_tokens: int = Field(16384, ge=2048)
    llm_output_tokens: int = Field(1200, ge=64)
    llm_timeout: float = Field(300, gt=0)
    llm_keep_alive: str = "5m"
    llm_log_outputs: bool = False
    ollama_num_gpu: int = Field(-1, ge=-1)
    context_max_bytes: int = Field(8000, ge=256)
    answer_deadline_seconds: float = Field(1300, gt=0)
    retry_attempts: int = Field(3, ge=1, le=5)
    retry_min_seconds: float = Field(0.5, gt=0)
    retry_max_seconds: float = Field(4, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    download_timeout: float = Field(3600, gt=0)

    @model_validator(mode="after")
    def check_relationships(self) -> Self:
        """Проверяет совместимость лимитов и запрет внешнего inference."""
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("Перекрытие должно быть меньше размера фрагмента.")
        if self.chunk_tokens >= self.dense_max_tokens:
            raise ValueError("Размер фрагмента должен оставлять место для префикса и спецтокенов.")
        if self.api_key.get_secret_value() and len(self.api_key.get_secret_value()) < 32:
            raise ValueError("API-ключ должен содержать не менее 32 символов.")
        if self.max_active_questions >= self.max_active_requests:
            raise ValueError(
                "Число вопросов должно оставлять свободные места для остальных запросов."
            )
        if not self.reranker_output_limit <= self.reranker_input_limit <= self.fused_candidates:
            raise ValueError("Требуется reranker_output <= reranker_input <= fused_candidates.")
        if self.fused_candidates > self.dense_candidates + self.sparse_candidates:
            raise ValueError("Лимит fusion превышает сумму лимитов ветвей.")
        if not self.task_soft_limit < self.task_hard_limit < self.visibility_timeout:
            raise ValueError("Требуется soft_limit < hard_limit < visibility_timeout.")
        if self.stale_task_seconds <= self.task_hard_limit:
            raise ValueError("Восстановление задания допускается только после hard_limit.")
        endpoint = urlsplit(self.ollama_url)
        local_hosts = {"localhost", "127.0.0.1", "::1", "ollama", "host.docker.internal"}
        if endpoint.hostname not in local_hosts:
            raise ValueError("Ollama должен работать локально или в локальном Compose.")
        if endpoint.scheme != "http" or endpoint.username or endpoint.password:
            raise ValueError("Для локального Ollama требуется HTTP без credentials.")
        for model in (self.generation_model, self.verification_model):
            if "cloud" in model.lower() or "/" in model:
                raise ValueError("Облачные и внешние модели Ollama запрещены.")
        if (
            self.context_max_bytes + self.llm_output_tokens + LLM_PROMPT_TOKEN_RESERVE
            > self.ollama_context_tokens
        ):
            raise ValueError("Контекст не оставляет места для инструкций и проверки ответа.")
        return self

    def index_profile(self) -> str:
        """Фиксирует параметры, изменение которых требует нового индекса."""
        names = (
            "dense_model",
            "dense_revision",
            "dense_dimensions",
            "dense_query_prefix",
            "dense_document_prefix",
            "dense_max_tokens",
            "sparse_model",
            "bm25_language",
            "bm25_disable_stemmer",
            "bm25_k",
            "bm25_b",
            "bm25_avg_len",
            "chunk_tokens",
            "chunk_overlap_tokens",
        )
        parameters = {name: getattr(self, name) for name in names}
        parameters["pipeline_version"] = PIPELINE_VERSION
        parameters["libraries"] = {
            package: version(package)
            for package in (
                "sentence-transformers",
                "transformers",
                "torch",
                "langchain-text-splitters",
            )
        }
        marker = self.model_cache_dir / "dense" / "rag-model.json"
        if marker.exists():
            parameters["dense_weights"] = json.loads(marker.read_text(encoding="utf-8"))
        return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
