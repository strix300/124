"""Публикует задания только после регистрации в Redis."""

from uuid import UUID

from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from app.core.errors import ServiceUnavailable
from app.workers.celery_app import celery_app


def enqueue_indexing(doc_id: UUID, generation_id: UUID, task_id: UUID) -> None:
    """Сохраняет ID при неопределённом результате сетевой отправки."""
    try:
        celery_app.send_task(
            "document_rag.index",
            args=[str(doc_id), str(generation_id)],
            task_id=str(task_id),
        )
    except (OperationalError, RedisError, OSError) as exc:
        raise ServiceUnavailable("Redis broker") from exc
