"""Настраивает Celery с Redis broker/backend и ограниченным числом исполнителей."""

from celery import Celery

from app.core.config import Settings

settings = Settings()
celery_app = Celery("document_rag", include=["app.workers.tasks"])
celery_app.conf.update(
    broker_url=settings.broker_url,
    result_backend=settings.result_backend_url,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=settings.worker_concurrency,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_soft_time_limit=settings.task_soft_limit,
    task_time_limit=settings.task_hard_limit,
    result_expires=settings.terminal_task_ttl,
    broker_connection_timeout=settings.redis_timeout,
    redis_socket_timeout=settings.redis_timeout,
    redis_socket_connect_timeout=settings.redis_timeout,
    broker_transport_options={
        "visibility_timeout": settings.visibility_timeout,
        "socket_timeout": settings.redis_timeout,
        "socket_connect_timeout": settings.redis_timeout,
    },
    result_backend_transport_options={"visibility_timeout": settings.visibility_timeout},
    visibility_timeout=settings.visibility_timeout,
    task_publish_retry=False,
    beat_schedule={
        "reconcile-indexing": {
            "task": "document_rag.reconcile",
            "schedule": settings.reconciliation_interval,
        }
    },
)
