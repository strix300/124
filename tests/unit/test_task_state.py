"""Проверяет TTL и атомарные переходы задания на Redis-совместимом движке."""

from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.core.errors import ApplicationError, ServiceUnavailable
from app.infrastructure.task_state import TaskState
from app.models import TaskRecord


def test_active_records_never_expire_and_terminal_records_do() -> None:
    fakeredis = pytest.importorskip("fakeredis", reason="Требуется fakeredis[lua]")
    settings = Settings()
    redis = fakeredis.FakeRedis(decode_responses=True)
    state = TaskState(settings, redis)
    task = TaskRecord(task_id=uuid4(), doc_id=uuid4(), generation_id=uuid4())
    state.register(task)
    key = f"{settings.redis_prefix}:task:{task.task_id}"
    assert redis.ttl(key) == -1
    task.status = "running"
    task.attempt = 1
    state.save(task)
    assert redis.ttl(key) == -1
    task.status = "succeeded"
    state.save(task)
    assert 0 < redis.ttl(key) <= 604800
    task.status = "running"
    state.save(task)
    assert state.get(task.task_id).status == "succeeded"


def test_idempotency_key_rejects_different_request() -> None:
    fakeredis = pytest.importorskip("fakeredis", reason="Требуется fakeredis[lua]")
    state = TaskState(Settings(), fakeredis.FakeRedis(decode_responses=True))
    first = state.bind_request("key", "signature", uuid4(), uuid4())
    assert state.bind_request("key", "signature", uuid4(), uuid4()) == first
    with pytest.raises(ApplicationError, match="Ключ"):
        state.bind_request("key", "different_signature", uuid4(), uuid4())


def test_inference_stays_exclusive_after_redis_lock_is_lost(tmp_path: Path) -> None:
    import fakeredis

    settings = Settings(storage_dir=tmp_path, inference_wait_seconds=0.01, lock_poll_seconds=0.001)
    redis = fakeredis.FakeRedis(decode_responses=True)
    first = TaskState(settings, redis)
    second = TaskState(settings, redis)
    with pytest.raises(ServiceUnavailable):
        with first.inference_slot():
            redis.delete(f"{settings.redis_prefix}:inference:0")
            with pytest.raises(ServiceUnavailable):
                with second.inference_slot():
                    pytest.fail("Потеря Redis-блокировки не должна допускать второй inference")
    with second.inference_slot() as lease:
        lease.ensure_owned()
