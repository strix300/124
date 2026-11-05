"""Команды подготовки локальных моделей и контролируемого обслуживания."""

import argparse
import json
import logging
import secrets
import sys
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from app.core.config import Settings
from app.core.errors import ApplicationError

logger = logging.getLogger(__name__)


def initialize_api_key(path: Path) -> None:
    """Сохраняет ключ в локальном env-файле, не выводя секрет в терминал."""
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    lines = content.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("RAG_API_KEY="):
            if line.removeprefix("RAG_API_KEY=").strip():
                return
            lines[index] = f"RAG_API_KEY={secrets.token_urlsafe(32)}"
            break
    else:
        lines.append(f"RAG_API_KEY={secrets.token_urlsafe(32)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)


def download_models(settings: Settings) -> None:
    """Скачивает публичные веса; приложение никогда не выполняет inference в облаке."""
    from huggingface_hub import HfApi, snapshot_download
    from langchain_qdrant import FastEmbedSparse

    for role, model, revision in (
        ("dense", settings.dense_model, settings.dense_revision),
        ("reranker", settings.reranker_model, settings.reranker_revision),
    ):
        directory = settings.model_cache_dir / role
        marker = directory / "rag-model.json"
        expected = {"model": model, "revision": revision}
        if marker.exists():
            existing = json.loads(marker.read_text(encoding="utf-8"))
            if any(existing.get(key) != value for key, value in expected.items()):
                raise ApplicationError(
                    "model_cache_mismatch",
                    "Для другой модели выберите новый каталог кеша.",
                )
            logger.info("model_cached role=%s", role)
            continue
        commit = HfApi().model_info(model, revision=revision, token=False).sha
        snapshot_download(
            repo_id=model,
            revision=commit,
            local_dir=str(directory),
            token=False,
            ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"],
        )
        marker.write_text(json.dumps({**expected, "commit": commit}), encoding="utf-8")
    FastEmbedSparse(
        model_name=settings.sparse_model,
        cache_dir=str(settings.model_cache_dir / "sparse"),
        language=settings.bm25_language,
        disable_stemmer=settings.bm25_disable_stemmer,
        k=settings.bm25_k,
        b=settings.bm25_b,
        avg_len=settings.bm25_avg_len,
    )
    with httpx.Client(
        base_url=settings.ollama_url,
        timeout=settings.download_timeout,
        trust_env=False,
    ) as client:
        for model in dict.fromkeys([settings.generation_model, settings.verification_model]):
            logger.info("pulling_local_model model=%s", model)
            with client.stream(
                "POST",
                "/api/pull",
                json={"model": model, "stream": True},
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        event = json.loads(line)
                        if "error" in event:
                            raise ApplicationError(
                                "model_download_failed",
                                "Ollama не смог скачать модель.",
                                503,
                            )
    logger.info("models_downloaded")


def initialize(settings: Settings) -> None:
    """Инициализирует одну коллекцию и проверяет настоящий payload-only point."""
    from app.infrastructure.qdrant import QdrantStore

    store = QdrantStore(settings)
    try:
        store.initialize()
        store.verify_catalog_support()
    finally:
        store.close()


def doctor(settings: Settings) -> None:
    """Проверяет наличие весов, Redis и совместимость Qdrant без генерации ответа."""
    from app.infrastructure.embeddings import model_directory
    from app.infrastructure.local_chat import validate_local_model
    from app.infrastructure.task_state import TaskState

    initialize(settings)
    state = TaskState(settings)
    try:
        state.redis.ping()
    finally:
        state.close()
    for role, model, revision in (
        ("dense", settings.dense_model, settings.dense_revision),
        ("reranker", settings.reranker_model, settings.reranker_revision),
    ):
        model_directory(settings, role, model, revision)
    with httpx.Client(
        base_url=settings.ollama_url,
        timeout=settings.llm_timeout,
        trust_env=False,
    ) as client:
        for model in {settings.generation_model, settings.verification_model}:
            validate_local_model(client, model)
    report = {"status": "ok", "inference": "local", "index_profile": settings.index_profile()}
    sys.stdout.write(json.dumps(report) + "\n")


def reindex(settings: Settings, doc_id: UUID) -> None:
    """Запускает новое поколение под блокировкой, сохраняя прежний опубликованный индекс."""
    from app.infrastructure.qdrant import QdrantStore
    from app.infrastructure.task_state import TaskState
    from app.models import TaskRecord
    from app.workers.publishing import enqueue_indexing

    store = QdrantStore(settings)
    state = TaskState(settings)
    try:
        with state.document_lease(doc_id, wait=settings.document_lock_wait_seconds) as lease:
            document = store.get_document(doc_id)
            if document is None or document.deleting:
                raise ApplicationError("document_not_found", "Документ не найден.", 404)
            task = state.get(document.task_id)
            if task and task.status in {"queued", "running", "retrying"}:
                raise ApplicationError("indexing_active", "Предыдущее задание ещё активно.", 409)
            document.pending_generation = uuid4()
            document.task_id = uuid4()
            document.index_profile = settings.index_profile()
            lease.ensure_owned()
            store.schedule_generation(document)
            task = TaskRecord(
                task_id=document.task_id,
                doc_id=doc_id,
                generation_id=document.pending_generation,
            )
            state.register(task)
            enqueue_indexing(doc_id, document.pending_generation, document.task_id)
            sys.stdout.write(task.model_dump_json() + "\n")
    finally:
        store.close()
        state.close()


def main() -> None:
    """Обрабатывает команды без изменения настроек глобального окружения."""
    parser = argparse.ArgumentParser(description="Локальный Document RAG")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("download-models", "init", "doctor"):
        commands.add_parser(name)
    credentials = commands.add_parser("init-api-key")
    credentials.add_argument("--env-file", type=Path, default=Path(".env"))
    maintenance = commands.add_parser("reindex")
    maintenance.add_argument("doc_id", type=UUID)
    arguments = parser.parse_args()
    if arguments.command == "init-api-key":
        initialize_api_key(arguments.env_file)
        return
    settings = Settings()
    logging.basicConfig(level=settings.log_level)
    try:
        if arguments.command == "download-models":
            download_models(settings)
        elif arguments.command == "init":
            initialize(settings)
        elif arguments.command == "doctor":
            doctor(settings)
        else:
            reindex(settings, arguments.doc_id)
    except ApplicationError as exc:
        logger.error("%s: %s", exc.code, exc.message)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
